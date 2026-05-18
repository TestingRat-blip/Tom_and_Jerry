"""Unit tests for L1Memory using FakeRedis.

These run without a Redis instance — fast feedback loop. The integration
tests in tests/integration/test_l1_redis.py exercise the same behavior
against a real Redis to confirm the FakeRedis is a faithful stand-in.
"""
from __future__ import annotations

import pytest

from src.env.world.world import Event, EventType
from src.hunter.agent.memory.l1 import L1Config, L1Memory
from src.persistence.redis.client import FakeRedis, RedisClient
from src.persistence.redis.l1_store import L1Store, NoiseRecord
from src.utils.types import Position


@pytest.fixture
def client():
    """A RedisClient backed by FakeRedis."""
    return RedisClient(client=FakeRedis())


@pytest.fixture
def store(client):
    return L1Store(client, episode_id="test_ep")


@pytest.fixture
def memory(client):
    return L1Memory(client, episode_id="test_ep")


# ---- L1Store unit tests -----------------------------------------------

def test_store_record_and_get_noise(store):
    n = NoiseRecord(tick=5, x=3, y=4, intensity=0.7)
    store.record_noise(n)
    got = store.get_noise(5)
    assert got is not None
    assert got.tick == 5
    assert got.x == 3
    assert got.y == 4
    assert got.intensity == pytest.approx(0.7)
    assert got.verified is False


def test_store_mark_noise_verified(store):
    store.record_noise(NoiseRecord(tick=3, x=1, y=1, intensity=0.5))
    store.mark_noise_verified(3)
    got = store.get_noise(3)
    assert got.verified is True


def test_store_false_noise_count(store):
    assert store.get_false_noise_count(5, 5) == 0
    store.increment_false_noise_count(5, 5)
    store.increment_false_noise_count(5, 5)
    store.increment_false_noise_count(5, 5)
    assert store.get_false_noise_count(5, 5) == 3
    # Different tile, separate counter
    assert store.get_false_noise_count(6, 6) == 0


def test_store_locker_sightings(store):
    store.increment_locker_sightings(10, 10)
    store.increment_locker_sightings(10, 10)
    assert store.get_locker_sightings(10, 10) == 2
    assert store.get_locker_sightings(11, 11) == 0


def test_store_heatmap(store):
    store.increment_sighting_heatmap(5, 5)
    store.increment_sighting_heatmap(5, 5)
    store.increment_sighting_heatmap(7, 5)
    counts = store.all_heatmap_counts()
    assert counts[(5, 5)] == 2
    assert counts[(7, 5)] == 1


def test_store_clear_episode(client, store):
    store.record_noise(NoiseRecord(tick=1, x=0, y=0, intensity=1.0))
    store.increment_false_noise_count(5, 5)
    store.increment_locker_sightings(8, 8)
    store.increment_sighting_heatmap(9, 9)
    store.clear_episode()
    # All getters should return empty / zero now
    assert store.get_noise(1) is None
    assert store.get_false_noise_count(5, 5) == 0
    assert store.get_locker_sightings(8, 8) == 0
    assert store.get_heatmap_count(9, 9) == 0


def test_store_episodes_isolated(client):
    """Two L1Stores with different episode_ids must not see each other's data."""
    s1 = L1Store(client, "ep1")
    s2 = L1Store(client, "ep2")
    s1.increment_false_noise_count(3, 3)
    s2.increment_false_noise_count(3, 3)
    s2.increment_false_noise_count(3, 3)
    assert s1.get_false_noise_count(3, 3) == 1
    assert s2.get_false_noise_count(3, 3) == 2


# ---- L1Memory event integration ---------------------------------------

def _noise_event(x: int, y: int, intensity: float = 1.0, actor: str = "jerry") -> Event:
    return Event(
        type=EventType.NOISE_EMITTED,
        actor=actor,
        position=Position(x, y),
        payload=intensity,
    )


def test_memory_records_jerry_noise(memory):
    events = [_noise_event(5, 5, 1.2, actor="jerry")]
    memory.observe_events(
        events,
        tom_pos=Position(10, 10),
        jerry_pos=Position(5, 5),
        jerry_visible=False,
        tick=1,
    )
    # Pending noise should be tracked but not yet verified/falsified
    assert len(memory._pending) == 1


def test_memory_ignores_tom_own_noise(memory):
    """Tom's footsteps shouldn't pollute his own noise log."""
    events = [_noise_event(3, 3, 1.0, actor="tom")]
    memory.observe_events(
        events, tom_pos=Position(3, 3), jerry_pos=Position(20, 20),
        jerry_visible=False, tick=1,
    )
    assert len(memory._pending) == 0


def test_memory_verifies_noise_with_nearby_sighting(memory):
    """A noise followed within window by a sighting near it → verified."""
    # Tick 1: Jerry makes noise at (5, 5)
    memory.observe_events(
        [_noise_event(5, 5, 1.2)],
        tom_pos=Position(10, 10), jerry_pos=Position(5, 5),
        jerry_visible=False, tick=1,
    )
    # Tick 3: Tom sees Jerry at (6, 6) — within verification radius
    memory.observe_events(
        [], tom_pos=Position(7, 7), jerry_pos=Position(6, 6),
        jerry_visible=True, tick=3,
    )
    # The pending noise should now be verified and dropped from _pending
    assert len(memory._pending) == 0
    # And the false-noise counter for that tile should NOT have ticked up
    assert memory.store.get_false_noise_count(5, 5) == 0


def test_memory_logs_false_noise_when_unverified(memory):
    """A noise NOT followed by a nearby sighting within window → false."""
    cfg = L1Config(verification_window_ticks=5)
    memory.config = cfg

    memory.observe_events(
        [_noise_event(5, 5)], tom_pos=Position(10, 10),
        jerry_pos=Position(5, 5),
        jerry_visible=False, tick=1,
    )
    # Many ticks pass with no sighting near (5, 5)
    for t in range(2, 10):
        memory.observe_events(
            [], tom_pos=Position(10, 10), jerry_pos=Position(20, 20),
            jerry_visible=False, tick=t,
        )
    # The original pending noise should have expired and been counted as false
    assert memory.store.get_false_noise_count(5, 5) >= 1


def test_memory_false_noise_factor_zero_at_start(memory):
    assert memory.false_noise_factor_near(Position(5, 5)) == 0.0


def test_memory_false_noise_factor_grows_with_count(memory):
    # Force several false noises near (5, 5)
    for _ in range(memory.config.false_noise_saturation):
        memory.store.increment_false_noise_count(5, 5)
    factor = memory.false_noise_factor_near(Position(5, 5))
    assert factor == pytest.approx(memory.config.max_false_noise_factor, abs=0.01)


def test_memory_false_noise_factor_saturates(memory):
    """Going past saturation count should not increase the factor."""
    for _ in range(memory.config.false_noise_saturation * 3):
        memory.store.increment_false_noise_count(5, 5)
    factor = memory.false_noise_factor_near(Position(5, 5))
    assert factor <= memory.config.max_false_noise_factor + 0.001


def test_memory_false_noise_factor_only_counts_nearby(memory):
    """A false noise far from `pos` should not influence the factor."""
    # Lots of false noises FAR from the query
    for _ in range(memory.config.false_noise_saturation * 2):
        memory.store.increment_false_noise_count(25, 25)
    factor = memory.false_noise_factor_near(Position(5, 5))
    assert factor == 0.0


def test_memory_locker_sightings_increment_near_lockers(memory):
    memory.set_locker_positions([Position(10, 10), Position(20, 20)])
    # Jerry sighted exactly on the locker
    memory.observe_events(
        [], tom_pos=Position(5, 5), jerry_pos=Position(10, 10),
        jerry_visible=True, tick=1,
    )
    assert memory.locker_suspicion(Position(10, 10)) == 1
    # Sighted within proximity (locker_proximity default is 2)
    memory.observe_events(
        [], tom_pos=Position(5, 5), jerry_pos=Position(11, 10),
        jerry_visible=True, tick=2,
    )
    assert memory.locker_suspicion(Position(10, 10)) == 2


def test_memory_locker_sightings_dont_increment_far_from_lockers(memory):
    memory.set_locker_positions([Position(10, 10)])
    memory.observe_events(
        [], tom_pos=Position(5, 5), jerry_pos=Position(20, 20),
        jerry_visible=True, tick=1,
    )
    assert memory.locker_suspicion(Position(10, 10)) == 0


def test_memory_most_suspicious_locker(memory):
    memory.set_locker_positions([Position(10, 10), Position(20, 20)])
    # Two sightings near locker A, one near locker B
    memory.observe_events(
        [], tom_pos=Position(0, 0), jerry_pos=Position(10, 10),
        jerry_visible=True, tick=1,
    )
    memory.observe_events(
        [], tom_pos=Position(0, 0), jerry_pos=Position(10, 10),
        jerry_visible=True, tick=2,
    )
    memory.observe_events(
        [], tom_pos=Position(0, 0), jerry_pos=Position(20, 20),
        jerry_visible=True, tick=3,
    )
    best = memory.most_suspicious_locker()
    assert best == Position(10, 10)


def test_memory_heatmap_records_sightings(memory):
    memory.observe_events(
        [], tom_pos=Position(0, 0), jerry_pos=Position(5, 5),
        jerry_visible=True, tick=1,
    )
    memory.observe_events(
        [], tom_pos=Position(0, 0), jerry_pos=Position(5, 5),
        jerry_visible=True, tick=2,
    )
    memory.observe_events(
        [], tom_pos=Position(0, 0), jerry_pos=Position(7, 5),
        jerry_visible=True, tick=3,
    )
    hottest = memory.heatmap_hottest(top_n=3)
    # The (5, 5) tile should be hotter than (7, 5)
    assert hottest[0][0] == Position(5, 5)
    assert hottest[0][1] == 2
    assert hottest[1][0] == Position(7, 5)
    assert hottest[1][1] == 1


def test_memory_reset_clears_everything(memory):
    memory.set_locker_positions([Position(10, 10)])
    memory.observe_events(
        [_noise_event(5, 5)],
        tom_pos=Position(0, 0), jerry_pos=Position(10, 10),
        jerry_visible=True, tick=1,
    )
    memory.reset()
    assert memory.store.get_false_noise_count(5, 5) == 0
    assert memory.locker_suspicion(Position(10, 10)) == 0
    assert memory.heatmap_hottest() == []
    assert len(memory._pending) == 0


def test_memory_total_noise_events_accumulates(memory):
    # Make 3 noises that won't get verified
    for tick in range(1, 4):
        memory.observe_events(
            [_noise_event(5 + tick, 5)],
            tom_pos=Position(0, 0), jerry_pos=Position(20, 20),
            jerry_visible=False, tick=tick,
        )
    # Let them age out
    for tick in range(50, 60):
        memory.observe_events(
            [], tom_pos=Position(0, 0), jerry_pos=Position(20, 20),
            jerry_visible=False, tick=tick,
        )
    assert memory.total_noise_events() == 3
