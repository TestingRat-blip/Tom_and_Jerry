"""Unit tests for the L2 lookup cascade.

Verifies:
  - Fine matches retrieved
  - Coarse matches retrieved without double-counting
  - Age decay applied (recent weighs more)
  - Tier weights applied (fine weighs more than coarse)
  - Per-tile counts capped at max_prior_count
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.hunter.agent.memory.l2_lookup import (
    DEFAULT_COARSE_WEIGHT,
    DEFAULT_DECAY_BASE,
    DEFAULT_FINE_WEIGHT,
    L2Lookup,
    L2LookupConfig,
    WarmStart,
)
from src.persistence.sqlite.client import SQLiteClient, SQLiteConfig
from src.persistence.sqlite.l2_store import EpisodeSummary, L2Store


# ---- fixtures ----------------------------------------------------------

@pytest.fixture
def client(tmp_path: Path):
    c = SQLiteClient(SQLiteConfig(db_path=tmp_path / "lookup.db"))
    yield c
    c.close()


@pytest.fixture
def store(client):
    return L2Store(client)


@pytest.fixture
def lookup(store):
    return L2Lookup(store)


def _make_summary(
    fine: str = "fine_A",
    coarse: str = "coarse_X",
    jerry: str = "jerry_J",
    created_at: float | None = None,
    heatmap_top: list | None = None,
    lockers: list | None = None,
    false_noise_top: list | None = None,
) -> EpisodeSummary:
    s = EpisodeSummary(
        map_fingerprint_fine=fine,
        map_fingerprint_coarse=coarse,
        jerry_fingerprint=jerry,
        outcome="survived",
        total_ticks=100,
        total_jerry_reward=0.0,
        heatmap_top=heatmap_top or [],
        lockers=lockers or [],
        false_noise_top=false_noise_top or [],
    )
    if created_at is not None:
        s.created_at = created_at
    return s


# ---- empty store -------------------------------------------------------

def test_empty_store_returns_empty_warm_start(lookup):
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert warm.is_empty
    assert warm.total_episodes == 0


def test_no_match_returns_empty_warm_start(lookup, store):
    store.insert(_make_summary(fine="other", coarse="other_c", jerry="other_j"))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert warm.is_empty


# ---- fine matches ------------------------------------------------------

def test_fine_match_contributes_to_warm_start(lookup, store):
    store.insert(_make_summary(
        fine="fine_A",
        heatmap_top=[(5, 5, 3), (10, 10, 1)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert not warm.is_empty
    assert warm.fine_episode_count == 1
    assert warm.coarse_episode_count == 0
    # Age 0 (most recent), fine tier weight = 1.0, decay^0 = 1.0
    # Expected weight = 1.0 * 1.0 = 1.0, so count 3 → 3.0
    assert warm.heatmap[(5, 5)] == pytest.approx(3.0)
    assert warm.heatmap[(10, 10)] == pytest.approx(1.0)


def test_multiple_fine_matches_aggregate(lookup, store):
    # Three episodes — most recent goes first by created_at
    now = time.time()
    store.insert(_make_summary(
        fine="fine_A", created_at=now,
        heatmap_top=[(5, 5, 2)],
    ))
    store.insert(_make_summary(
        fine="fine_A", created_at=now - 100,
        heatmap_top=[(5, 5, 2)],
    ))
    store.insert(_make_summary(
        fine="fine_A", created_at=now - 200,
        heatmap_top=[(5, 5, 2)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert warm.fine_episode_count == 3
    # Counts: 2*(1.0)^0 + 2*(0.95)^1 + 2*(0.95)^2 = 2 + 1.9 + 1.805 = 5.705
    assert warm.heatmap[(5, 5)] == pytest.approx(2.0 + 1.9 + 1.805, abs=0.01)


# ---- age decay --------------------------------------------------------

def test_recent_episode_weighs_more_than_old(lookup, store):
    now = time.time()
    # Same tile, same count, but two different ages
    store.insert(_make_summary(
        fine="fine_A", created_at=now,           # age 0
        heatmap_top=[(5, 5, 1)],
    ))
    store.insert(_make_summary(
        fine="fine_A", created_at=now - 1000,    # age 1
        heatmap_top=[(7, 7, 1)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    # Recent should weigh more
    assert warm.heatmap[(5, 5)] > warm.heatmap[(7, 7)]
    # Specifically: 1.0 vs 0.95
    assert warm.heatmap[(5, 5)] == pytest.approx(1.0)
    assert warm.heatmap[(7, 7)] == pytest.approx(DEFAULT_DECAY_BASE)


# ---- coarse fallback --------------------------------------------------

def test_coarse_only_match(lookup, store):
    """When no fine match exists, coarse matches are used at coarse weight."""
    store.insert(_make_summary(
        fine="different_fine",  # different fine
        coarse="coarse_X",       # same coarse
        heatmap_top=[(5, 5, 10)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert warm.fine_episode_count == 0
    assert warm.coarse_episode_count == 1
    # Coarse tier weight = 0.4, age 0 → weight 0.4, count 10 → 4.0
    assert warm.heatmap[(5, 5)] == pytest.approx(0.4 * 10)


def test_fine_and_coarse_combine_without_double_counting(lookup, store):
    """An episode matching the fine fingerprint should NOT also be
    counted at the coarse layer.
    """
    # This episode matches both fine AND coarse — but should only
    # contribute once (at the fine layer)
    store.insert(_make_summary(
        fine="fine_A",
        coarse="coarse_X",
        heatmap_top=[(5, 5, 1)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    # Only the fine layer should contribute
    assert warm.fine_episode_count == 1
    assert warm.coarse_episode_count == 0
    # Weight should be 1.0 * 1 = 1.0, NOT 1.0 * 1 + 0.4 * 1 = 1.4
    assert warm.heatmap[(5, 5)] == pytest.approx(1.0)


def test_fine_dominates_when_both_available(lookup, store):
    """A fine match should produce a larger contribution than a coarse
    match for the same age and count.
    """
    # Two episodes at the same age, same count:
    # - One matches fine (gets fine weight)
    # - One matches coarse only (gets coarse weight)
    now = time.time()
    store.insert(_make_summary(
        fine="fine_A", coarse="coarse_X", created_at=now,
        heatmap_top=[(5, 5, 1)],
    ))
    store.insert(_make_summary(
        fine="other_fine", coarse="coarse_X", created_at=now,
        heatmap_top=[(7, 7, 1)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    # (5,5) is the fine match — weight 1.0
    # (7,7) is the coarse match — weight 0.4
    assert warm.heatmap[(5, 5)] > warm.heatmap[(7, 7)]
    assert warm.heatmap[(5, 5)] == pytest.approx(1.0)
    assert warm.heatmap[(7, 7)] == pytest.approx(0.4)


# ---- jerry isolation ---------------------------------------------------

def test_different_jerry_excluded(lookup, store):
    """A summary with a different jerry fingerprint must not contribute."""
    store.insert(_make_summary(
        fine="fine_A", jerry="OTHER_JERRY",
        heatmap_top=[(5, 5, 10)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert warm.is_empty


# ---- saturation cap ---------------------------------------------------

def test_warm_start_capped_at_max_prior_count(lookup, store):
    """Many high-count past episodes should not produce an unbounded prior."""
    now = time.time()
    # Insert 30 episodes all with the same (5, 5) sighting of count 100
    for i in range(30):
        store.insert(_make_summary(
            fine="fine_A", created_at=now - i,
            heatmap_top=[(5, 5, 100)],
        ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    # The cap is 10 (default). Without it the value would be enormous.
    assert warm.heatmap[(5, 5)] <= 10.0
    assert warm.heatmap[(5, 5)] == pytest.approx(10.0)


# ---- lockers and false_noise aggregate similarly ---------------------

def test_lockers_aggregate(lookup, store):
    store.insert(_make_summary(
        fine="fine_A",
        lockers=[(3, 3, 2), (8, 8, 1)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert warm.lockers[(3, 3)] == pytest.approx(2.0)
    assert warm.lockers[(8, 8)] == pytest.approx(1.0)


def test_false_noise_aggregate(lookup, store):
    store.insert(_make_summary(
        fine="fine_A",
        false_noise_top=[(15, 15, 5), (20, 20, 2)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    assert warm.false_noise[(15, 15)] == pytest.approx(5.0)
    assert warm.false_noise[(20, 20)] == pytest.approx(2.0)


# ---- config overrides --------------------------------------------------

def test_custom_decay_changes_age_weighting(store):
    """A very low decay base should make old episodes contribute less."""
    custom = L2LookupConfig(decay_base=0.5)
    lookup = L2Lookup(store, config=custom)

    now = time.time()
    store.insert(_make_summary(
        fine="fine_A", created_at=now,
        heatmap_top=[(5, 5, 1)],
    ))
    store.insert(_make_summary(
        fine="fine_A", created_at=now - 100,
        heatmap_top=[(7, 7, 1)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    # (5,5): 1.0 * 1 = 1.0
    # (7,7): 0.5 * 1 = 0.5
    assert warm.heatmap[(5, 5)] == pytest.approx(1.0)
    assert warm.heatmap[(7, 7)] == pytest.approx(0.5)


def test_custom_tier_weights(store):
    """Boost coarse weight enough to exceed default fine weight."""
    custom = L2LookupConfig(fine_weight=0.1, coarse_weight=1.0)
    lookup = L2Lookup(store, config=custom)

    store.insert(_make_summary(
        fine="fine_A", coarse="coarse_X",
        heatmap_top=[(5, 5, 1)],
    ))
    store.insert(_make_summary(
        fine="other", coarse="coarse_X",
        heatmap_top=[(7, 7, 1)],
    ))
    warm = lookup.build_warm_start("fine_A", "coarse_X", "jerry_J")
    # Now (7,7) should weigh MORE than (5,5)
    assert warm.heatmap[(7, 7)] > warm.heatmap[(5, 5)]


# ---- WarmStart shape ----------------------------------------------------

def test_warm_start_is_empty_property():
    empty = WarmStart()
    assert empty.is_empty
    assert empty.total_episodes == 0

    populated = WarmStart(heatmap={(1, 1): 1.0})
    assert not populated.is_empty


def test_warm_start_total_episodes():
    w = WarmStart(fine_episode_count=3, coarse_episode_count=2)
    assert w.total_episodes == 5
