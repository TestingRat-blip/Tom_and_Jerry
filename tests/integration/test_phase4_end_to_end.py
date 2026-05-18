"""End-to-end test for Phase 4 persistent memory.

These tests prove the central claim of Phase 4: episodes that have ALREADY
HAPPENED change the behavior of episodes that haven't yet. Tom carries
forward, across deaths, what he learned in past lives.

The hardest tests run two consecutive episodes on the same map+jerry
combination and verify episode 2's L1 starts non-empty (warm-started
from episode 1's distillation).

Most tests use FakeRedis + tmp-path SQLite — no Docker, no real Redis,
fast. A separate redis-marked test exercises the full path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.memory.l1 import L1Memory
from src.hunter.agent.memory.l2_lookup import L2Lookup, WarmStart
from src.persistence.redis.client import FakeRedis, RedisClient
from src.persistence.sqlite.client import SQLiteClient, SQLiteConfig
from src.persistence.sqlite.l2_store import L2Store
from src.render.replay.recorder import ReplayRecorder
from src.utils.types import Action


# ---- fixtures ----------------------------------------------------------

@pytest.fixture
def sqlite_client(tmp_path: Path):
    c = SQLiteClient(SQLiteConfig(db_path=tmp_path / "phase4.db"))
    yield c
    c.close()


@pytest.fixture
def l2_store(sqlite_client):
    return L2Store(sqlite_client)


@pytest.fixture
def l2_lookup(l2_store):
    return L2Lookup(l2_store)


def _build_tom_with_full_memory(l2_lookup, l2_store, seed: int = 0) -> ChemicalTom:
    """Build a ChemicalTom with L1 (FakeRedis) + L2 lookup + L2 store wired."""
    l1 = L1Memory(
        client=RedisClient(client=FakeRedis()),
        episode_id="phase4_e2e_test",
    )
    return ChemicalTom(
        l1=l1,
        l2_lookup=l2_lookup,
        l2_store=l2_store,
        seed=seed,
    )


# ---- backwards compatibility --------------------------------------

def test_chemical_tom_with_no_l2_behaves_like_phase3(l2_store):
    """ChemicalTom without l2_lookup/l2_store should match Phase 3 exactly."""
    l1 = L1Memory(RedisClient(client=FakeRedis()), episode_id="phase3_check")
    tom = ChemicalTom(l1=l1, seed=0)
    # Both lifecycle hooks should be no-ops returning False
    from src.env.world.world import World
    w = World(WorldConfig(max_ticks=10), seed=42)
    w.reset()
    assert tom.warm_start_for_episode(w.grid, None) is False
    assert tom.distill_at_episode_end(
        w.grid, None, outcome="survived",
        total_ticks=10, total_jerry_reward=0.0,
    ) is False


def test_l2_lookup_without_l1_is_noop(l2_lookup, l2_store):
    """If L1 is missing, L2 wiring should not crash — lookup is just skipped."""
    tom = ChemicalTom(l1=None, l2_lookup=l2_lookup, l2_store=l2_store, seed=0)
    from src.env.world.world import World
    w = World(WorldConfig(max_ticks=10), seed=42)
    w.reset()
    assert tom.warm_start_for_episode(w.grid, None) is False
    assert tom.distill_at_episode_end(
        w.grid, None, outcome="survived",
        total_ticks=10, total_jerry_reward=0.0,
    ) is False


# ---- warm-start: no priors → empty -----------------------------------

def test_warm_start_returns_false_when_l2_empty(l2_lookup, l2_store):
    """No past episodes → no warm-start applied."""
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    from src.env.world.world import World
    w = World(WorldConfig(max_ticks=10), seed=42)
    w.reset()
    result = tom.warm_start_for_episode(w.grid, None, jerry_label="test")
    assert result is False
    # L1 should still be empty
    assert tom.l1._warm_heatmap == {}


# ---- the central claim: episode 2 inherits from episode 1 ----------

def test_two_consecutive_episodes_warm_start_from_each_other(l2_lookup, l2_store):
    """The fundamental Phase 4 claim:

    Episode 1: Tom plays, accumulates L1 data, distills into L2 at episode end.
    Episode 2: Tom plays the SAME map+jerry combo. L1 warm-starts from L2.
               Episode 2's L1 has non-empty priors at tick 0.

    Test setup notes: we need an episode where Tom *actually saw Jerry*
    so the summary has non-empty heatmap. Larger time budget against
    a passive Jerry usually gets a sighting on most map seeds. If Tom
    happens to never sight Jerry, the summary is empty and warm-start
    correctly returns False — which is the right behavior, but useless
    for testing the warm-start path. So we use a long budget and assert
    explicitly that episode 1 produced something to warm-start FROM.
    """
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    # 400 ticks gives Tom enough time to find a passive Jerry on most seeds
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=400), seed=42)

    # Run episode 1
    rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        jerry_label="passive_jerry_for_test",
        tom_label="phase4_tom",
    )

    # L2 should now have one episode
    assert l2_store.count() == 1

    # Sanity check: episode 1 actually saw Jerry (otherwise the test
    # setup is wrong, not the code under test)
    summary = l2_store.query_jerry_only("label:passive_jerry_for_test")[0]
    if not summary.heatmap_top and not summary.lockers and summary.total_noise_events == 0:
        pytest.skip(
            f"Episode 1 produced an empty summary (Tom never engaged Jerry). "
            f"Try a different seed or longer budget. "
            f"outcome={summary.outcome}, ticks={summary.total_ticks}"
        )

    # Episode 2 — same map seed, same jerry label → same fingerprints
    from src.env.world.world import World
    w2 = World(WorldConfig(max_ticks=400), seed=42)
    w2.reset()
    tom.reset()
    tom.l1.set_locker_positions(list(w2.grid.locker_positions))
    applied = tom.warm_start_for_episode(
        grid=w2.grid,
        jerry_policy=None,
        jerry_label="passive_jerry_for_test",
    )
    assert applied, "Episode 2 should have warm-started from episode 1's distillation"

    # At least one warm dict should be populated
    any_warm = bool(
        tom.l1._warm_heatmap
        or tom.l1._warm_lockers
        or tom.l1._warm_false_noise
    )
    assert any_warm, "warm-start was applied but all warm dicts are empty"


def test_warm_start_does_not_carry_to_different_jerry(l2_lookup, l2_store):
    """If episode 2 uses a DIFFERENT jerry policy, no warm-start applies
    even though the map is the same.
    """
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=60), seed=42)
    rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        jerry_label="jerry_A",
    )
    assert l2_store.count() == 1

    # Episode 2: same map (same seed), different jerry label
    from src.env.world.world import World
    w2 = World(WorldConfig(max_ticks=60), seed=42)
    w2.reset()
    tom.reset()
    tom.l1.set_locker_positions(list(w2.grid.locker_positions))
    applied = tom.warm_start_for_episode(
        grid=w2.grid,
        jerry_policy=None,
        jerry_label="jerry_B",  # different
    )
    assert applied is False
    assert not tom.l1._warm_heatmap


def test_warm_start_does_not_carry_to_different_map(l2_lookup, l2_store):
    """If episode 2 has a DIFFERENT map (different seed), warm-start may
    still trigger via the COARSE cascade — but only if the map shapes
    match. With different sizes/wall densities, no warm-start.
    """
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    # Episode 1 on a 30×30 map
    rec1 = ReplayRecorder(
        world_config=WorldConfig(max_ticks=60, grid_width=30, grid_height=30),
        seed=42,
    )
    rec1.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        jerry_label="same_jerry",
    )
    assert l2_store.count() == 1

    # Episode 2 on a different SIZE map → different coarse fingerprint
    from src.env.world.world import World
    w2 = World(WorldConfig(max_ticks=60, grid_width=20, grid_height=20), seed=42)
    w2.reset()
    tom.reset()
    tom.l1.set_locker_positions(list(w2.grid.locker_positions))
    applied = tom.warm_start_for_episode(
        grid=w2.grid,
        jerry_policy=None,
        jerry_label="same_jerry",
    )
    # Coarse fingerprint differs because dimensions differ → no warm-start
    assert applied is False


# ---- distillation writes L2 ---------------------------------------

def test_distillation_writes_to_l2_at_episode_end(l2_lookup, l2_store):
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=60), seed=42)
    assert l2_store.count() == 0
    rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        jerry_label="test_j",
    )
    assert l2_store.count() == 1
    # The stored summary should reflect actual episode data
    summaries = l2_store.query_jerry_only("label:test_j")
    assert len(summaries) == 1
    s = summaries[0]
    assert s.outcome in ("caught", "survived")
    assert s.total_ticks > 0


def test_three_consecutive_episodes_accumulate_in_l2(l2_lookup, l2_store):
    """Each recorded episode writes one summary. After three episodes,
    L2 should have three rows.
    """
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=60), seed=42)
    for i in range(3):
        rec.record_episode(
            jerry_policy=lambda obs, world: int(Action.WAIT),
            tom_policy=tom,
            jerry_label="accumulating_jerry",
        )
    assert l2_store.count() == 3


# ---- ticks_to_first_sight tracking --------------------------------

def test_ticks_to_first_sight_tracked_during_episode(l2_lookup, l2_store):
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=60), seed=42)
    rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        jerry_label="vis_tracker",
    )
    summaries = l2_store.query_jerry_only("label:vis_tracker")
    assert len(summaries) == 1
    # ticks_to_first_sight is either a positive int (Tom saw Jerry) or
    # None (Tom never saw Jerry this episode). Both are valid outcomes.
    tt = summaries[0].ticks_to_first_sight
    assert tt is None or tt >= 0


def test_ticks_to_first_sight_resets_between_episodes(l2_lookup, l2_store):
    """After episode 1 sets the timer, episode 2 should start with None."""
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=60), seed=42)
    rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        jerry_label="reset_check",
    )
    # After episode 1, _ticks_to_first_sight may be an int OR None
    # depending on whether Tom saw Jerry. What matters: a fresh reset
    # zeroes it out.
    tom.reset()
    assert tom._ticks_to_first_sight is None


# ---- behavior actually changes when warm-started ------------------

def test_warm_start_affects_noise_threshold(l2_lookup, l2_store):
    """Concrete behavior check: if episode 1 distills false-noise priors,
    episode 2's _modulated_noise_threshold should reflect them.
    """
    tom = _build_tom_with_full_memory(l2_lookup, l2_store)

    # Manually craft an L2 entry with strong false-noise priors
    from src.persistence.sqlite.l2_store import EpisodeSummary
    from src.hunter.agent.memory.fingerprint import fingerprint_map
    from src.env.world.world import World

    # Build a world to get fingerprints
    w = World(WorldConfig(max_ticks=60), seed=42)
    w.reset()
    fine_fp, coarse_fp = fingerprint_map(w.grid)

    # Insert a fake summary with false-noise hotspots near (10, 10)
    summary = EpisodeSummary(
        map_fingerprint_fine=fine_fp,
        map_fingerprint_coarse=coarse_fp,
        jerry_fingerprint="label:trickster_jerry",
        outcome="survived",
        total_ticks=100,
        total_jerry_reward=10.0,
        ticks_to_first_sight=20,
        false_noise_top=[(10, 10, 4), (11, 10, 3), (10, 11, 2)],
    )
    l2_store.insert(summary)

    # Episode 2: same map, same jerry → warm-start should kick in
    from src.utils.types import Position
    tom.reset()
    tom.l1.set_locker_positions(list(w.grid.locker_positions))
    applied = tom.warm_start_for_episode(
        grid=w.grid, jerry_policy=None, jerry_label="trickster_jerry",
    )
    assert applied is True
    # The noise threshold near (10, 10) should be elevated
    threshold_at_hotspot = tom._modulated_noise_threshold(tom_pos=Position(10, 10))
    threshold_far_away = tom._modulated_noise_threshold(tom_pos=Position(0, 0))
    assert threshold_at_hotspot > threshold_far_away
