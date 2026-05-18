"""Tests for L1 warm-start integration and the L3 stub.

L1 warm-start checks:
  - apply_warm_start() pre-seeds priors without touching the store
  - Behavior queries combine warm + in-episode counts
  - Distillation reads only in-episode counters (no double-counting)
  - reset() clears warm-start along with in-episode state

L3 stub checks:
  - Methods exist with expected signatures
  - All operations return safe defaults
  - `enabled` is False
"""
from __future__ import annotations

import pytest

from src.env.world.world import Event, EventType
from src.hunter.agent.memory.l1 import L1Memory
from src.hunter.agent.memory.l2_lookup import WarmStart
from src.hunter.agent.memory.l3 import EpisodeMemory, L3Config, L3Memory
from src.persistence.redis.client import FakeRedis, RedisClient
from src.utils.types import Position


# ---- L1 warm-start integration ----------------------------------------

@pytest.fixture
def memory():
    return L1Memory(
        RedisClient(client=FakeRedis()),
        episode_id="warm_test",
    )


def test_apply_warm_start_does_not_touch_redis_store(memory):
    """Warm-start lives in Python dicts, NOT in Redis. This is the
    architectural invariant that prevents priors from compounding
    across episodes via distillation.
    """
    warm = WarmStart(
        heatmap={(5, 5): 2.5},
        lockers={(10, 10): 1.0},
        false_noise={(7, 7): 3.0},
    )
    memory.apply_warm_start(warm)

    # The Redis store should still be empty
    assert memory.store.all_heatmap_counts() == {}
    assert memory.store.all_locker_sightings() == {}
    assert memory.store.all_false_noise_counts() == {}

    # But the in-memory warm dicts have the priors
    assert memory._warm_heatmap[(5, 5)] == 2.5
    assert memory._warm_lockers[(10, 10)] == 1.0
    assert memory._warm_false_noise[(7, 7)] == 3.0


def test_heatmap_hottest_combines_warm_and_in_episode(memory):
    """Behavior query should sum warm + in-episode counts."""
    memory.apply_warm_start(WarmStart(heatmap={(5, 5): 2.0}))
    # Add an in-episode sighting on the same tile
    memory.store.increment_sighting_heatmap(5, 5)
    hottest = memory.heatmap_hottest(top_n=1)
    assert hottest[0][0] == Position(5, 5)
    assert hottest[0][1] == pytest.approx(3.0)  # 2.0 warm + 1 in-episode


def test_heatmap_hottest_includes_warm_only_tiles(memory):
    """Tiles with priors but no in-episode sightings should still appear."""
    memory.apply_warm_start(WarmStart(heatmap={(5, 5): 2.0, (10, 10): 1.0}))
    hottest = memory.heatmap_hottest(top_n=5)
    coords = [p for p, _ in hottest]
    assert Position(5, 5) in coords
    assert Position(10, 10) in coords


def test_locker_suspicion_combines_warm_and_in_episode(memory):
    memory.apply_warm_start(WarmStart(lockers={(8, 8): 1.5}))
    memory.store.increment_locker_sightings(8, 8)
    susp = memory.locker_suspicion(Position(8, 8))
    assert susp == pytest.approx(2.5)


def test_most_suspicious_locker_includes_warm(memory):
    """Most suspicious locker should consider warm + in-episode together."""
    memory.apply_warm_start(WarmStart(lockers={(8, 8): 2.0, (3, 3): 0.5}))
    # In-episode: (3, 3) bumped twice → 2.0 in-episode, plus 0.5 warm = 2.5
    memory.store.increment_locker_sightings(3, 3)
    memory.store.increment_locker_sightings(3, 3)
    best = memory.most_suspicious_locker()
    # (3, 3) has 2.5; (8, 8) has 2.0 → (3, 3) wins
    assert best == Position(3, 3)


def test_false_noise_factor_combines_warm_and_in_episode(memory):
    """False-noise factor should reflect both prior and current fooling."""
    pos = Position(10, 10)
    # No warm-start, no in-episode: factor should be 0
    assert memory.false_noise_factor_near(pos) == 0.0

    # Add warm-start priors near pos
    memory.apply_warm_start(WarmStart(false_noise={(10, 10): 2.0}))
    factor_warm_only = memory.false_noise_factor_near(pos)
    assert factor_warm_only > 0.0

    # Now also accumulate in-episode counts
    memory.store.increment_false_noise_count(10, 10)
    memory.store.increment_false_noise_count(10, 10)
    factor_combined = memory.false_noise_factor_near(pos)
    assert factor_combined > factor_warm_only


def test_distillation_reads_in_episode_only(memory):
    """total_noise_events() must read in-episode only.

    This is the architectural invariant that prevents priors from
    compounding across episodes. If distillation summarized warm-start
    too, each new L2 entry would include all priors, which would
    re-feed into the next warm-start...
    """
    memory.apply_warm_start(WarmStart(false_noise={(5, 5): 100.0}))
    # No in-episode false noises
    assert memory.total_noise_events() == 0

    memory.store.increment_false_noise_count(5, 5)
    memory.store.increment_false_noise_count(5, 5)
    assert memory.total_noise_events() == 2  # in-episode count only


def test_reset_clears_warm_start(memory):
    memory.apply_warm_start(WarmStart(heatmap={(5, 5): 2.0}))
    memory.store.increment_sighting_heatmap(5, 5)
    assert memory.heatmap_hottest(1)[0][1] == pytest.approx(3.0)

    memory.reset()
    assert memory._warm_heatmap == {}
    assert memory._warm_lockers == {}
    assert memory._warm_false_noise == {}
    assert memory.heatmap_hottest() == []


def test_apply_warm_start_overwrites_previous(memory):
    """Calling apply_warm_start twice should replace, not accumulate."""
    memory.apply_warm_start(WarmStart(heatmap={(5, 5): 2.0}))
    memory.apply_warm_start(WarmStart(heatmap={(7, 7): 1.0}))
    # The (5, 5) prior should be gone
    assert (5, 5) not in memory._warm_heatmap
    assert memory._warm_heatmap[(7, 7)] == 1.0


def test_apply_warm_start_copies_input_dicts(memory):
    """Mutating the source WarmStart after apply_warm_start should not
    affect L1's internal state.
    """
    warm = WarmStart(heatmap={(5, 5): 2.0})
    memory.apply_warm_start(warm)
    warm.heatmap[(5, 5)] = 999.0
    warm.heatmap[(10, 10)] = 42.0
    assert memory._warm_heatmap[(5, 5)] == 2.0
    assert (10, 10) not in memory._warm_heatmap


# ---- L3 stub ----------------------------------------------------------

def test_l3_default_disabled():
    l3 = L3Memory()
    assert l3.enabled is False


def test_l3_index_episode_returns_none():
    l3 = L3Memory()
    result = l3.index_episode("ep_123", "the episode summary text",
                              metadata={"foo": "bar"})
    assert result is None


def test_l3_recall_similar_returns_empty():
    l3 = L3Memory()
    assert l3.recall_similar("some query") == []


def test_l3_recall_for_map_and_jerry_returns_empty():
    l3 = L3Memory()
    assert l3.recall_for_map_and_jerry("map text", "jerry text") == []


def test_l3_count_zero():
    l3 = L3Memory()
    assert l3.count() == 0


def test_l3_clear_is_noop():
    l3 = L3Memory()
    # Should not raise
    l3.clear()


def test_l3_config_defaults():
    cfg = L3Config()
    assert cfg.top_k == 5
    assert cfg.embedding_model == "all-MiniLM-L6-v2"


def test_l3_can_be_constructed_with_custom_config():
    cfg = L3Config(top_k=10, min_similarity=0.8)
    l3 = L3Memory(config=cfg)
    assert l3.config.top_k == 10
    assert l3.config.min_similarity == 0.8


def test_episode_memory_dataclass_shape():
    """L3's data type exists and works as expected."""
    em = EpisodeMemory(summary_id="ep_1", similarity=0.85,
                       metadata={"map": "fp_a"})
    assert em.summary_id == "ep_1"
    assert em.similarity == 0.85
    assert em.metadata == {"map": "fp_a"}


def test_episode_memory_default_metadata_is_empty_dict():
    em = EpisodeMemory(summary_id="ep_1", similarity=0.5)
    assert em.metadata == {}
