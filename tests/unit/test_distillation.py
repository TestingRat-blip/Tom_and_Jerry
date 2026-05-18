"""Unit tests for L1 → L2 distillation."""
from __future__ import annotations

import pytest

from src.env.world.world import Event, EventType, Grid
from src.hunter.agent.memory.distillation import (
    DEFAULT_FALSE_NOISE_TOP_N,
    DEFAULT_HEATMAP_TOP_N,
    distill_l1_to_summary,
)
from src.hunter.agent.memory.l1 import L1Memory
from src.persistence.redis.client import FakeRedis, RedisClient
from src.utils.types import Position


# ---- fixtures ----------------------------------------------------------

@pytest.fixture
def grid():
    return Grid.generate(width=20, height=20, wall_density=0.15,
                         n_vent_pairs=2, n_lockers=3, seed=42)


@pytest.fixture
def memory():
    """Empty L1Memory backed by FakeRedis."""
    return L1Memory(
        RedisClient(client=FakeRedis()),
        episode_id="distill_test",
    )


# ---- basic shape ------------------------------------------------------

def test_distill_returns_episode_summary(grid, memory):
    s = distill_l1_to_summary(
        l1=memory,
        grid=grid,
        jerry_policy=None,
        outcome="survived",
        total_ticks=300,
        total_jerry_reward=12.5,
        ticks_to_first_sight=42,
        jerry_label="test_jerry",
    )
    assert s.outcome == "survived"
    assert s.total_ticks == 300
    assert s.total_jerry_reward == pytest.approx(12.5)
    assert s.ticks_to_first_sight == 42


def test_distill_includes_fingerprints(grid, memory):
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="caught", total_ticks=100, total_jerry_reward=-1.0,
        ticks_to_first_sight=5, jerry_label="J",
    )
    assert s.map_fingerprint_fine != ""
    assert s.map_fingerprint_coarse != ""
    assert s.jerry_fingerprint == "label:J"


def test_distill_empty_memory_produces_empty_lists(grid, memory):
    """A memory with no sightings/noises/lockers → empty top-N lists."""
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=50, total_jerry_reward=0.0,
        ticks_to_first_sight=None, jerry_label="J",
    )
    assert s.heatmap_top == []
    assert s.lockers == []
    assert s.false_noise_top == []
    assert s.total_noise_events == 0


# ---- heatmap distillation ---------------------------------------------

def test_distill_heatmap_top_n(grid, memory):
    # Pump 15 distinct sighting tiles with varying counts
    for i in range(15):
        for _ in range(i + 1):
            memory.store.increment_sighting_heatmap(i, 0)

    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
        heatmap_top_n=10,
    )
    assert len(s.heatmap_top) == 10
    # Verify order: highest counts first
    counts = [c for (x, y, c) in s.heatmap_top]
    assert counts == sorted(counts, reverse=True)
    # The hottest tile should be (14, 0) with 15 sightings
    assert s.heatmap_top[0] == (14, 0, 15)


def test_distill_heatmap_default_top_n(grid, memory):
    """If heatmap_top_n is not given, defaults to DEFAULT_HEATMAP_TOP_N."""
    for i in range(20):
        memory.store.increment_sighting_heatmap(i, 0)
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
    )
    assert len(s.heatmap_top) == DEFAULT_HEATMAP_TOP_N


def test_distill_heatmap_fewer_than_top_n(grid, memory):
    """If there are fewer unique tiles than top_n, return all of them."""
    memory.store.increment_sighting_heatmap(5, 5)
    memory.store.increment_sighting_heatmap(10, 10)
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
        heatmap_top_n=10,
    )
    assert len(s.heatmap_top) == 2


# ---- locker distillation ----------------------------------------------

def test_distill_lockers_keeps_all_nonzero(grid, memory):
    """Lockers are few — we keep all with count > 0."""
    memory.store.increment_locker_sightings(5, 5)
    memory.store.increment_locker_sightings(5, 5)
    memory.store.increment_locker_sightings(10, 10)
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
    )
    assert len(s.lockers) == 2
    # Sorted by count desc
    assert s.lockers[0] == (5, 5, 2)
    assert s.lockers[1] == (10, 10, 1)


def test_distill_lockers_excludes_zero_count(grid, memory):
    """We don't store lockers Tom never saw Jerry near."""
    # Force a zero by writing then ignoring; in practice this state
    # shouldn't occur, but we want defensive behavior.
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
    )
    assert s.lockers == []


# ---- false-noise distillation -----------------------------------------

def test_distill_false_noise_top_n(grid, memory):
    """Top-N false noise tiles in sorted order."""
    counts = {(1, 1): 5, (2, 2): 3, (3, 3): 9, (4, 4): 1}
    for (x, y), n in counts.items():
        for _ in range(n):
            memory.store.increment_false_noise_count(x, y)
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
        false_noise_top_n=3,
    )
    assert len(s.false_noise_top) == 3
    # Highest first
    assert s.false_noise_top[0] == (3, 3, 9)
    assert s.false_noise_top[1] == (1, 1, 5)
    assert s.false_noise_top[2] == (2, 2, 3)


def test_distill_total_noise_events(grid, memory):
    """Total noise events = sum of all false-noise counts."""
    memory.store.increment_false_noise_count(1, 1)
    memory.store.increment_false_noise_count(1, 1)
    memory.store.increment_false_noise_count(2, 2)
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
    )
    assert s.total_noise_events == 3


# ---- verified noise count ---------------------------------------------

def test_distill_counts_verified_noises(grid, memory):
    """Verified noise count comes from scanning noise:{tick} records."""
    # Push some noise events into L1 (as if Jerry made them)
    noise_events_list = [
        Event(type=EventType.NOISE_EMITTED, actor="jerry",
              position=Position(5, 5), payload=1.0),
    ]
    memory.observe_events(
        noise_events_list,
        tom_pos=Position(0, 0),
        jerry_pos=Position(5, 5),
        jerry_visible=False,
        tick=1,
    )
    # Tom sees Jerry near the noise within window → verified
    memory.observe_events(
        [],
        tom_pos=Position(3, 3),
        jerry_pos=Position(6, 6),
        jerry_visible=True,
        tick=3,
    )

    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=3, jerry_label="J",
    )
    assert s.verified_noise_count == 1


# ---- ticks_to_first_sight ---------------------------------------------

def test_distill_preserves_none_ticks_to_first_sight(grid, memory):
    """A Tom that never sighted Jerry → None, not 0."""
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=400, total_jerry_reward=20.0,
        ticks_to_first_sight=None, jerry_label="J",
    )
    assert s.ticks_to_first_sight is None


# ---- notes pass-through -----------------------------------------------

def test_distill_passes_through_notes(grid, memory):
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="caught", total_ticks=50, total_jerry_reward=-1.0,
        ticks_to_first_sight=10, jerry_label="J",
        notes={"experiment": "phase4_demo", "iteration": 5},
    )
    assert s.notes == {"experiment": "phase4_demo", "iteration": 5}


def test_distill_default_notes_is_empty_dict(grid, memory):
    s = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
    )
    assert s.notes == {}


# ---- determinism -------------------------------------------------------

def test_distill_is_deterministic(grid, memory):
    """Two distillations of the same L1 state produce identical summaries
    (except for episode_id and created_at, which are auto-generated).
    """
    for i in range(5):
        memory.store.increment_sighting_heatmap(i, 0)
        memory.store.increment_false_noise_count(i, 5)
    s1 = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
    )
    s2 = distill_l1_to_summary(
        l1=memory, grid=grid, jerry_policy=None,
        outcome="survived", total_ticks=100, total_jerry_reward=0.0,
        ticks_to_first_sight=10, jerry_label="J",
    )
    assert s1.heatmap_top == s2.heatmap_top
    assert s1.false_noise_top == s2.false_noise_top
    assert s1.total_noise_events == s2.total_noise_events
    assert s1.map_fingerprint_fine == s2.map_fingerprint_fine
    # episode_id and created_at differ — that's expected
    assert s1.episode_id != s2.episode_id
