"""Unit tests for SQLite persistence (Phase 4 L2).

Uses tmp_path-scoped databases so tests are fully isolated and don't
require any external services.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.persistence.sqlite.client import SQLiteClient, SQLiteConfig
from src.persistence.sqlite.l2_store import EpisodeSummary, L2Store


# ---- fixtures ----------------------------------------------------------

@pytest.fixture
def client(tmp_path: Path):
    """Fresh SQLite client on a temp DB file."""
    db_path = tmp_path / "test.db"
    c = SQLiteClient(SQLiteConfig(db_path=db_path))
    yield c
    c.close()


@pytest.fixture
def store(client):
    return L2Store(client)


def _make_summary(**overrides) -> EpisodeSummary:
    base = dict(
        map_fingerprint_fine="fine_abc",
        map_fingerprint_coarse="coarse_30x30",
        jerry_fingerprint="jerry_v1_baseline",
        outcome="survived",
        total_ticks=300,
        total_jerry_reward=12.5,
        ticks_to_first_sight=45,
        heatmap_top=[(10, 10, 5), (12, 8, 3)],
        lockers=[(5, 5, 2)],
        false_noise_top=[(20, 20, 4)],
        total_noise_events=18,
        verified_noise_count=3,
        notes={"phase": 4},
    )
    base.update(overrides)
    return EpisodeSummary(**base)


# ---- SQLiteClient -----------------------------------------------------

def test_client_creates_db_file(tmp_path: Path):
    db_path = tmp_path / "subdir" / "auto.db"
    assert not db_path.parent.exists()
    c = SQLiteClient(SQLiteConfig(db_path=db_path))
    try:
        assert db_path.exists()
    finally:
        c.close()


def test_client_applies_schema_version(client):
    assert client.schema_version == 1


def test_client_schema_is_idempotent(tmp_path: Path):
    """Opening the same DB twice shouldn't error or break the schema."""
    db_path = tmp_path / "twice.db"
    c1 = SQLiteClient(SQLiteConfig(db_path=db_path))
    c1.close()
    c2 = SQLiteClient(SQLiteConfig(db_path=db_path))
    assert c2.schema_version == 1
    c2.close()


def test_client_wal_mode_active(client):
    row = client.fetchone("PRAGMA journal_mode")
    # PRAGMA journal_mode returns the mode in a column named 'journal_mode'
    # In sqlite Row form, we access by index since column name varies
    assert "wal" in str(row[0]).lower()


def test_client_foreign_keys_on(client):
    row = client.fetchone("PRAGMA foreign_keys")
    assert int(row[0]) == 1


# ---- L2Store: insert + get -------------------------------------------

def test_insert_and_get_summary(store):
    s = _make_summary()
    store.insert(s)
    got = store.get_by_id(s.episode_id)
    assert got is not None
    assert got.episode_id == s.episode_id
    assert got.outcome == "survived"
    assert got.total_ticks == 300
    assert got.ticks_to_first_sight == 45
    assert got.total_jerry_reward == pytest.approx(12.5)


def test_summary_serializes_json_fields(store):
    """Heatmap and lockers should round-trip through JSON unchanged."""
    s = _make_summary(
        heatmap_top=[(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        lockers=[(10, 10, 1)],
        false_noise_top=[(15, 15, 7), (20, 20, 2)],
        notes={"foo": "bar", "n": 42},
    )
    store.insert(s)
    got = store.get_by_id(s.episode_id)
    assert got.heatmap_top == [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    assert got.lockers == [(10, 10, 1)]
    assert got.false_noise_top == [(15, 15, 7), (20, 20, 2)]
    assert got.notes == {"foo": "bar", "n": 42}


def test_summary_handles_null_ticks_to_first_sight(store):
    """A Tom that never saw Jerry should store NULL, not 0."""
    s = _make_summary(ticks_to_first_sight=None)
    store.insert(s)
    got = store.get_by_id(s.episode_id)
    assert got.ticks_to_first_sight is None


def test_insert_is_idempotent_on_episode_id(store):
    """INSERT OR REPLACE means retrying a write doesn't duplicate."""
    s = _make_summary(total_ticks=100)
    store.insert(s)
    # Same episode_id, different content
    s2 = _make_summary(episode_id=s.episode_id, total_ticks=999)
    store.insert(s2)
    assert store.count() == 1
    got = store.get_by_id(s.episode_id)
    assert got.total_ticks == 999


def test_get_by_id_returns_none_when_missing(store):
    assert store.get_by_id("does_not_exist") is None


# ---- L2Store: queries -------------------------------------------------

def test_query_fine_returns_exact_matches_only(store):
    store.insert(_make_summary(map_fingerprint_fine="A"))
    store.insert(_make_summary(map_fingerprint_fine="B"))
    store.insert(_make_summary(map_fingerprint_fine="A"))
    results = store.query_fine(map_fp_fine="A", jerry_fp="jerry_v1_baseline")
    assert len(results) == 2
    for r in results:
        assert r.map_fingerprint_fine == "A"


def test_query_fine_returns_most_recent_first(store):
    s1 = _make_summary(); s1.created_at = 100.0
    s2 = _make_summary(); s2.created_at = 200.0
    s3 = _make_summary(); s3.created_at = 150.0
    for s in (s1, s2, s3):
        store.insert(s)
    results = store.query_fine("fine_abc", "jerry_v1_baseline")
    # DESC order: 200, 150, 100
    assert [r.created_at for r in results] == [200.0, 150.0, 100.0]


def test_query_fine_respects_jerry_fingerprint(store):
    store.insert(_make_summary(jerry_fingerprint="jerry_A"))
    store.insert(_make_summary(jerry_fingerprint="jerry_B"))
    results = store.query_fine("fine_abc", "jerry_A")
    assert len(results) == 1
    assert results[0].jerry_fingerprint == "jerry_A"


def test_query_fine_respects_limit(store):
    for _ in range(10):
        store.insert(_make_summary())
    results = store.query_fine("fine_abc", "jerry_v1_baseline", limit=3)
    assert len(results) == 3


def test_query_coarse_matches_coarse_fingerprint(store):
    store.insert(_make_summary(
        map_fingerprint_fine="fine_1",
        map_fingerprint_coarse="coarse_X",
    ))
    store.insert(_make_summary(
        map_fingerprint_fine="fine_2",
        map_fingerprint_coarse="coarse_X",
    ))
    store.insert(_make_summary(
        map_fingerprint_fine="fine_3",
        map_fingerprint_coarse="coarse_Y",
    ))
    results = store.query_coarse("coarse_X", "jerry_v1_baseline")
    assert len(results) == 2


def test_query_coarse_can_exclude_a_fine_fingerprint(store):
    """The fine→coarse cascade in Phase 9c needs to exclude already-
    matched fine episodes when running the coarse fallback.
    """
    store.insert(_make_summary(
        map_fingerprint_fine="fine_1",
        map_fingerprint_coarse="coarse_X",
    ))
    store.insert(_make_summary(
        map_fingerprint_fine="fine_2",
        map_fingerprint_coarse="coarse_X",
    ))
    results = store.query_coarse(
        "coarse_X", "jerry_v1_baseline", exclude_fine="fine_1",
    )
    assert len(results) == 1
    assert results[0].map_fingerprint_fine == "fine_2"


def test_query_jerry_only(store):
    store.insert(_make_summary(jerry_fingerprint="jerry_A"))
    store.insert(_make_summary(jerry_fingerprint="jerry_A"))
    store.insert(_make_summary(jerry_fingerprint="jerry_B"))
    a_results = store.query_jerry_only("jerry_A")
    b_results = store.query_jerry_only("jerry_B")
    assert len(a_results) == 2
    assert len(b_results) == 1


# ---- L2Store: counts + maintenance ----------------------------------

def test_count_total(store):
    assert store.count() == 0
    for _ in range(5):
        store.insert(_make_summary())
    assert store.count() == 5


def test_count_for_specific_pair(store):
    store.insert(_make_summary(
        map_fingerprint_fine="A",
        jerry_fingerprint="J1",
    ))
    store.insert(_make_summary(
        map_fingerprint_fine="A",
        jerry_fingerprint="J1",
    ))
    store.insert(_make_summary(
        map_fingerprint_fine="A",
        jerry_fingerprint="J2",
    ))
    assert store.count_for("A", "J1") == 2
    assert store.count_for("A", "J2") == 1
    assert store.count_for("B", "J1") == 0


def test_delete_all(store):
    for _ in range(3):
        store.insert(_make_summary())
    assert store.count() == 3
    deleted = store.delete_all()
    assert deleted == 3
    assert store.count() == 0


def test_delete_older_than(store):
    now = time.time()
    s_old = _make_summary(); s_old.created_at = now - 1000
    s_new = _make_summary(); s_new.created_at = now
    store.insert(s_old)
    store.insert(s_new)
    # Prune anything older than 500 seconds
    deleted = store.delete_older_than(500)
    assert deleted == 1
    assert store.count() == 1
    # The remaining episode should be the new one
    remaining = store.query_jerry_only("jerry_v1_baseline")
    assert remaining[0].episode_id == s_new.episode_id


# ---- EpisodeSummary defaults -----------------------------------------

def test_summary_auto_assigns_episode_id_and_created_at():
    s = EpisodeSummary(
        map_fingerprint_fine="x",
        map_fingerprint_coarse="y",
        jerry_fingerprint="z",
        outcome="caught",
        total_ticks=10,
        total_jerry_reward=0.0,
    )
    assert s.episode_id != ""
    assert len(s.episode_id) >= 16
    assert s.created_at > 0


def test_summary_explicit_episode_id_preserved():
    s = EpisodeSummary(
        episode_id="custom_id",
        map_fingerprint_fine="x",
        map_fingerprint_coarse="y",
        jerry_fingerprint="z",
        outcome="caught",
        total_ticks=10,
        total_jerry_reward=0.0,
    )
    assert s.episode_id == "custom_id"
