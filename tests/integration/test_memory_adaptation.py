"""Tests for memory-driven adaptation (Components 1 + 2).

Component 1: behavioral distillation — L2 records los_break_count +
los_break_hotspots per episode.
Component 2: Conductor warm-start reads the behavioral stance and deploys
the hold-on-LOS-break run-down SELECTIVELY (only vs cover-dancers).

Covers:
  - schema v2 round-trips behavioral fields
  - behavioral_stance deploys for high-LOS-break Jerrys, not low/unseen
  - los_break_count is tracked during an episode
  - the end-to-end loop: distill in episode N → stance deploys in N+1
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.conductor import Conductor
from src.hunter.agent.memory.l1 import L1Memory
from src.hunter.agent.memory.l2_lookup import L2Lookup, StrategicStance
from src.persistence.redis.client import FakeRedis, RedisClient
from src.persistence.sqlite.client import SQLiteClient, SQLiteConfig
from src.persistence.sqlite.l2_store import EpisodeSummary, L2Store
from src.utils.types import Action


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "behav.db"
    return L2Store(SQLiteClient(SQLiteConfig(db_path=db)))


# ---- Component 1: schema + distillation -------------------------------

def test_schema_v2_round_trips_behavioral_fields(store):
    s = EpisodeSummary(
        map_fingerprint_fine="abc", map_fingerprint_coarse="ab",
        jerry_fingerprint="jf", outcome="survived",
        total_ticks=300, total_jerry_reward=15.0,
        los_break_count=12, los_break_hotspots=[(5, 5, 8), (6, 5, 4)],
        time_in_cover_fraction=0.62, oscillation_score=0.4,
    )
    store.insert(s)
    got = store.get_by_id(s.episode_id)
    assert got.los_break_count == 12
    assert got.los_break_hotspots == [(5, 5, 8), (6, 5, 4)]
    assert got.time_in_cover_fraction == pytest.approx(0.62)
    assert got.oscillation_score == pytest.approx(0.4)


def test_schema_version_is_2(store):
    assert store.client.schema_version == 2


def test_behavioral_fields_default_to_zero(store):
    """An episode summary built without behavioral fields defaults cleanly
    (backward compatible with pre-Component-1 distillation)."""
    s = EpisodeSummary(
        map_fingerprint_fine="x", map_fingerprint_coarse="x",
        jerry_fingerprint="j", outcome="caught",
        total_ticks=100, total_jerry_reward=1.0,
    )
    store.insert(s)
    got = store.get_by_id(s.episode_id)
    assert got.los_break_count == 0
    assert got.los_break_hotspots == []


# ---- Component 2: behavioral stance -----------------------------------

def test_stance_deploys_for_cover_dancer(store):
    lookup = L2Lookup(store)
    store.insert(EpisodeSummary(
        map_fingerprint_fine="M1", map_fingerprint_coarse="M",
        jerry_fingerprint="dancer", outcome="survived",
        total_ticks=300, total_jerry_reward=20.0,
        los_break_count=14, los_break_hotspots=[(5, 5, 9), (6, 6, 5)],
    ))
    stance = lookup.behavioral_stance("M1", "M", "dancer")
    assert stance.deploy_hold_on_los_break is True
    assert stance.mean_los_breaks == pytest.approx(14.0)
    assert (5, 5) in stance.los_break_hotspots


def test_stance_does_not_deploy_for_normal_evader(store):
    lookup = L2Lookup(store)
    store.insert(EpisodeSummary(
        map_fingerprint_fine="M1", map_fingerprint_coarse="M",
        jerry_fingerprint="normal", outcome="caught",
        total_ticks=120, total_jerry_reward=3.0,
        los_break_count=2, los_break_hotspots=[(1, 1, 2)],
    ))
    stance = lookup.behavioral_stance("M1", "M", "normal")
    assert stance.deploy_hold_on_los_break is False


def test_stance_neutral_for_unseen_jerry(store):
    lookup = L2Lookup(store)
    stance = lookup.behavioral_stance("M1", "M", "never_seen")
    assert stance.is_neutral
    assert stance.deploy_hold_on_los_break is False


def test_stance_threshold_boundary(store):
    """Deploy fires at >= threshold."""
    lookup = L2Lookup(store)
    # Two episodes averaging exactly the default threshold (5.0)
    store.insert(EpisodeSummary(
        map_fingerprint_fine="M1", map_fingerprint_coarse="M",
        jerry_fingerprint="edge", outcome="survived",
        total_ticks=300, total_jerry_reward=10.0, los_break_count=5))
    stance = lookup.behavioral_stance("M1", "M", "edge", deploy_threshold=5.0)
    assert stance.deploy_hold_on_los_break is True
    stance2 = lookup.behavioral_stance("M1", "M", "edge", deploy_threshold=5.1)
    assert stance2.deploy_hold_on_los_break is False


# ---- Component 1: in-episode LOS-break tracking -----------------------

def test_los_break_count_tracked_during_episode():
    """Tom should count LOS-breaks during an episode (saw Jerry, then lost
    sight). We drive a random Jerry and confirm the counter is sane
    (>= 0, and matches manual sight-transition counting)."""
    world = World(WorldConfig(max_ticks=200), seed=3)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=3)
    tom.reset()
    rng = random.Random(11)
    manual_breaks = 0
    prev_see = False
    last_seen_tile = None
    for _ in range(200):
        a = tom(world)
        see = world._tom_can_see_jerry()
        if see:
            last_seen_tile = world.jerry.position
        elif prev_see and last_seen_tile is not None:
            manual_breaks += 1
        prev_see = see
        world.step(tom_action=a, jerry_action=rng.randint(0, 4))
        if not world.jerry.alive:
            break
    assert tom._los_break_count == manual_breaks


# ---- end-to-end loop --------------------------------------------------

def _make_memory_tom(store, lookup, seed):
    l1 = L1Memory(
        client=RedisClient(client=FakeRedis()),
        episode_id=f"e_{random.random()}",
    )
    return ChemicalTom(
        l1=l1, l2_lookup=lookup, l2_store=store,
        conductor=Conductor(), seed=seed,
    )


def test_loop_distill_then_deploy(store):
    """The full Component 1+2 loop, with a SYNTHETIC high-LOS-break episode
    distilled directly (so we don't depend on the sandbox reproducing a
    real cover-dance): after a high-LOS-break episode is in L2, the next
    episode's warm-start should deploy the run-down for that Jerry.
    """
    lookup = L2Lookup(store)

    # Pre-load L2 with a cover-dance signature for jerry 'dancer' on this map.
    world = World(WorldConfig(max_ticks=100), seed=1)
    world.reset()
    from src.hunter.agent.memory.fingerprint import (
        fingerprint_jerry,
        fingerprint_map,
    )
    fine_fp, coarse_fp = fingerprint_map(world.grid)
    # warm_start fingerprints the label ("dancer" -> "label:dancer"), so the
    # stored summary must use the same fingerprinted form to match.
    dancer_fp = fingerprint_jerry(None, label="dancer")
    store.insert(EpisodeSummary(
        map_fingerprint_fine=fine_fp, map_fingerprint_coarse=coarse_fp,
        jerry_fingerprint=dancer_fp, outcome="survived",
        total_ticks=300, total_jerry_reward=20.0,
        los_break_count=14, los_break_hotspots=[(5, 5, 9)],
    ))

    # Episode N+1: warm-start for the SAME jerry should deploy the run-down.
    tom = _make_memory_tom(store, lookup, seed=1)
    tom.reset()
    tom.warm_start_for_episode(world.grid, jerry_policy=None, jerry_label="dancer")
    assert tom.last_stance is not None
    assert tom.last_stance.deploy_hold_on_los_break is True
    assert tom.conductor.runtime_hold_on_los_break is True
    # And the behavior is now actually active on the conductor.
    assert tom.conductor._hold_on_los_break_active() is True


def test_loop_stands_down_for_normal_jerry(store):
    """Warm-start for a low-LOS-break Jerry should NOT deploy the run-down
    — selective deployment, the whole point of the memory loop."""
    lookup = L2Lookup(store)
    world = World(WorldConfig(max_ticks=100), seed=1)
    world.reset()
    from src.hunter.agent.memory.fingerprint import (
        fingerprint_jerry,
        fingerprint_map,
    )
    fine_fp, coarse_fp = fingerprint_map(world.grid)
    normal_fp = fingerprint_jerry(None, label="normal")
    store.insert(EpisodeSummary(
        map_fingerprint_fine=fine_fp, map_fingerprint_coarse=coarse_fp,
        jerry_fingerprint=normal_fp, outcome="caught",
        total_ticks=120, total_jerry_reward=3.0,
        los_break_count=2, los_break_hotspots=[(1, 1, 2)],
    ))
    tom = _make_memory_tom(store, lookup, seed=1)
    tom.reset()
    tom.warm_start_for_episode(world.grid, jerry_policy=None, jerry_label="normal")
    # The stance must be non-neutral (we DID find the episode) but NOT deploy.
    assert tom.last_stance is not None
    assert tom.last_stance.is_neutral is False
    assert tom.last_stance.deploy_hold_on_los_break is False
    assert tom.conductor._hold_on_los_break_active() is False


def test_distillation_writes_los_break_fields(store):
    """distill_at_episode_end should write the tracked LOS-break data into
    the L2 summary."""
    lookup = L2Lookup(store)
    tom = _make_memory_tom(store, lookup, seed=3)
    tom.reset()
    # Inject some tracked LOS-break data as if an episode had produced it.
    from collections import Counter
    tom._los_break_count = 9
    tom._los_break_tiles = Counter({(5, 5): 6, (6, 6): 3})

    world = World(WorldConfig(max_ticks=50), seed=3)
    world.reset()
    tom.distill_at_episode_end(
        world.grid, jerry_policy=None, outcome="survived",
        total_ticks=300, total_jerry_reward=18.0, jerry_label="dancer")

    # Read it back (distill fingerprints the label → query with same form)
    from src.hunter.agent.memory.fingerprint import (
        fingerprint_jerry,
        fingerprint_map,
    )
    fine_fp, _ = fingerprint_map(world.grid)
    dancer_fp = fingerprint_jerry(None, label="dancer")
    results = store.query_fine(map_fp_fine=fine_fp, jerry_fp=dancer_fp, limit=5)
    assert len(results) >= 1
    summ = results[0]
    assert summ.los_break_count == 9
    # hotspots stored, sorted by count desc
    assert (5, 5, 6) in summ.los_break_hotspots
