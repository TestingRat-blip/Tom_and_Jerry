"""Integration tests for L1 against a REAL Redis instance.

Marked @pytest.mark.redis so they're skipped by default. Run with:
    pytest -m redis tests/integration/test_l1_redis.py

These tests confirm that FakeRedis (used in unit tests) is a faithful
stand-in for real Redis. If a test passes here but failed against
FakeRedis (or vice versa), one of the two is wrong.

Each test uses a UNIQUE episode_id (uuid) so concurrent test runs and
leftover state from prior runs don't interfere.
"""
from __future__ import annotations

import uuid

import pytest

from src.env.world.world import Event, EventType
from src.hunter.agent.memory.l1 import L1Memory
from src.persistence.redis.client import RedisClient, RedisConfig
from src.persistence.redis.l1_store import L1Store, NoiseRecord
from src.utils.types import Position


pytestmark = pytest.mark.redis


@pytest.fixture
def client():
    """A RedisClient connected to the real test Redis.

    If the connection fails, the test is skipped — assumes the user just
    doesn't have Redis running. We do NOT auto-start it; that's the
    user's responsibility via `docker compose up -d`.
    """
    try:
        c = RedisClient(RedisConfig())
        c.ping()
    except Exception as e:
        pytest.skip(f"Redis not available at default config: {e}")
    return c


@pytest.fixture
def episode_id():
    """Unique per test."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def memory(client, episode_id):
    m = L1Memory(client, episode_id=episode_id)
    yield m
    # Clean up after the test
    m.reset()


def _noise_event(x: int, y: int, intensity: float = 1.0, actor: str = "jerry") -> Event:
    return Event(
        type=EventType.NOISE_EMITTED,
        actor=actor,
        position=Position(x, y),
        payload=intensity,
    )


def test_real_redis_store_round_trip(client, episode_id):
    store = L1Store(client, episode_id)
    store.record_noise(NoiseRecord(tick=7, x=2, y=3, intensity=0.9))
    got = store.get_noise(7)
    assert got is not None
    assert got.tick == 7
    assert got.x == 2
    assert got.intensity == pytest.approx(0.9)
    store.clear_episode()


def test_real_redis_memory_full_flow(memory):
    """End-to-end: noise → sighting → verification, against real Redis."""
    memory.observe_events(
        [_noise_event(5, 5)],
        tom_pos=Position(10, 10), jerry_pos=Position(5, 5),
        jerry_visible=False, tick=1,
    )
    # Tom sees Jerry near the noise on tick 3
    memory.observe_events(
        [], tom_pos=Position(7, 7), jerry_pos=Position(6, 6),
        jerry_visible=True, tick=3,
    )
    # The pending noise should have been verified and removed
    assert len(memory._pending) == 0
    # No false-noise count was recorded
    assert memory.store.get_false_noise_count(5, 5) == 0


def test_real_redis_episode_isolation(client):
    """Two different episode_ids must not see each other's data."""
    s1 = L1Store(client, f"iso_a_{uuid.uuid4().hex[:6]}")
    s2 = L1Store(client, f"iso_b_{uuid.uuid4().hex[:6]}")
    try:
        s1.increment_false_noise_count(3, 3)
        s2.increment_false_noise_count(3, 3)
        s2.increment_false_noise_count(3, 3)
        assert s1.get_false_noise_count(3, 3) == 1
        assert s2.get_false_noise_count(3, 3) == 2
    finally:
        s1.clear_episode()
        s2.clear_episode()


def test_real_redis_clear_episode_removes_all_keys(client):
    ep = f"clear_test_{uuid.uuid4().hex[:6]}"
    store = L1Store(client, ep)
    store.record_noise(NoiseRecord(tick=1, x=0, y=0, intensity=1.0))
    store.increment_false_noise_count(5, 5)
    store.increment_locker_sightings(8, 8)
    store.increment_sighting_heatmap(9, 9)
    store.clear_episode()
    # Scan should find no keys for this episode
    pattern = client.ns("l1", ep, "*")
    keys = list(client.scan_iter(match=pattern))
    assert keys == []
