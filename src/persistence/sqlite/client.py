"""SQLite client for Tom_and_Jerry persistent memory.

This module owns:
  - The default database path
  - Connection construction (with WAL mode, foreign keys on, JSON support)
  - Schema migration on first connect

Project-specific table schemas and queries live in `l2_store.py`. This
file only knows about connection mechanics, mirroring the split between
`redis/client.py` and `redis/l1_store.py`.

WAL mode is on because the trainer (Phase 6) will eventually write
summaries from multiple processes. SQLite handles concurrent reads
fine and serializes writes; WAL means readers don't block writers.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# Default database location. Stored under data/ alongside snapshots and replays.
DEFAULT_DB_PATH = Path("data/persistence/tj_l2.db")


@dataclass(frozen=True, slots=True)
class SQLiteConfig:
    db_path: Path = DEFAULT_DB_PATH
    # WAL mode lets readers and writers proceed in parallel. Safe for
    # the Phase 6 multi-worker case; harmless otherwise.
    journal_mode: str = "WAL"
    # Synchronous=NORMAL is the WAL-mode sweet spot: durable enough for
    # research data, much faster than the default FULL setting.
    synchronous: str = "NORMAL"
    # Foreign keys are off by default in SQLite (legacy); we want them on.
    foreign_keys: bool = True


# Schema version — bump when migrations are added.
SCHEMA_VERSION = 1


class SQLiteClient:
    """Owns a SQLite connection plus first-time schema setup.

    Use as a context manager OR via explicit close():

        with SQLiteClient() as db:
            db.execute(...)

        # or:
        db = SQLiteClient()
        try:
            ...
        finally:
            db.close()
    """

    def __init__(self, config: SQLiteConfig | None = None):
        self.config = config or SQLiteConfig()

        # Make sure the parent directory exists. SQLite will create the
        # .db file itself but won't mkdir for us.
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # `check_same_thread=False` lets the same connection be used
        # across threads. We serialize at a higher level (one writer at
        # a time) so this is safe.
        self._conn = sqlite3.connect(
            self.config.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit — we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row

        self._apply_pragmas()
        self._run_migrations()

    # ---- pragmas / migrations -----------------------------------------

    def _apply_pragmas(self) -> None:
        cur = self._conn.cursor()
        cur.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
        cur.execute(f"PRAGMA synchronous = {self.config.synchronous}")
        if self.config.foreign_keys:
            cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    def _run_migrations(self) -> None:
        """Apply schema changes idempotently.

        The pattern: every migration is its own function `_migrate_to_N`
        that takes us from version N-1 to N. We read the current version
        from `schema_version` and run any pending migrations.
        """
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """)
        cur.execute("SELECT version FROM schema_version")
        row = cur.fetchone()
        current = int(row["version"]) if row is not None else 0

        if current < 1:
            self._migrate_to_1(cur)
            current = 1

        if current < 2:
            self._migrate_to_2(cur)
            current = 2

        if row is None:
            cur.execute("INSERT INTO schema_version (version) VALUES (?)", (current,))
        else:
            cur.execute("UPDATE schema_version SET version = ?", (current,))
        cur.close()

    def _migrate_to_1(self, cur: sqlite3.Cursor) -> None:
        """Initial schema. Adds the episode_summaries table + indexes."""
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episode_summaries (
                episode_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                map_fingerprint_fine TEXT NOT NULL,
                map_fingerprint_coarse TEXT NOT NULL,
                jerry_fingerprint TEXT NOT NULL,
                tom_label TEXT,
                outcome TEXT NOT NULL,
                total_ticks INTEGER NOT NULL,
                total_jerry_reward REAL NOT NULL,
                ticks_to_first_sight INTEGER,
                heatmap_top_json TEXT NOT NULL DEFAULT '[]',
                lockers_json TEXT NOT NULL DEFAULT '[]',
                false_noise_top_json TEXT NOT NULL DEFAULT '[]',
                total_noise_events INTEGER NOT NULL DEFAULT 0,
                verified_noise_count INTEGER NOT NULL DEFAULT 0,
                notes_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        # Indexes for the two query shapes Phase 9c uses
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_fine_jerry
            ON episode_summaries (map_fingerprint_fine, jerry_fingerprint, created_at DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_coarse_jerry
            ON episode_summaries (map_fingerprint_coarse, jerry_fingerprint, created_at DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_jerry
            ON episode_summaries (jerry_fingerprint, created_at DESC)
        """)

    def _migrate_to_2(self, cur: sqlite3.Cursor) -> None:
        """Phase 6 / memory-adaptation: behavioral-signature columns.

        Adds columns capturing HOW Jerry behaved, not just where — the
        data Tom's memory-driven adaptation reads to deploy counters like
        the hold-on-LOS-break run-down. Added via ALTER TABLE so existing
        databases upgrade in place (SQLite ALTER ADD COLUMN is cheap and
        non-destructive; existing rows get the DEFAULT).

          - los_break_count: how many times Jerry broke Tom's line of sight
            (the cover-dance signature — high count = evasive LOS-breaker)
          - los_break_hotspots_json: WHERE LOS-breaks clustered ([x,y,count])
          - time_in_cover_fraction: fraction of ticks Jerry was adjacent to
            a wall / in cover (0..1)
          - oscillation_score: how much Jerry reversed direction (0..1-ish)
        """
        # ALTER TABLE ADD COLUMN can't be wrapped in IF NOT EXISTS, so guard
        # by inspecting existing columns (idempotent for safety).
        cur.execute("PRAGMA table_info(episode_summaries)")
        existing = {r["name"] for r in cur.fetchall()}
        adds = [
            ("los_break_count", "INTEGER NOT NULL DEFAULT 0"),
            ("los_break_hotspots_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("time_in_cover_fraction", "REAL NOT NULL DEFAULT 0.0"),
            ("oscillation_score", "REAL NOT NULL DEFAULT 0.0"),
        ]
        for name, decl in adds:
            if name not in existing:
                cur.execute(
                    f"ALTER TABLE episode_summaries ADD COLUMN {name} {decl}")
        # Index to support "find evasive LOS-breakers for this map+jerry".
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_losbreak
            ON episode_summaries (jerry_fingerprint, los_break_count)
        """)

    # ---- low-level passthroughs ---------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq_of_params) -> sqlite3.Cursor:
        return self._conn.executemany(sql, seq_of_params)

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        cur = self.execute(sql, params)
        try:
            return cur.fetchone()
        finally:
            cur.close()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        cur = self.execute(sql, params)
        try:
            return cur.fetchall()
        finally:
            cur.close()

    def iter_rows(self, sql: str, params: tuple = ()) -> Iterator[sqlite3.Row]:
        """Stream rows without loading them all into memory."""
        cur = self.execute(sql, params)
        try:
            while True:
                row = cur.fetchone()
                if row is None:
                    return
                yield row
        finally:
            cur.close()

    # ---- transactions -------------------------------------------------

    def begin(self) -> None:
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    # ---- lifecycle ----------------------------------------------------

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "SQLiteClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self.fetchone("SELECT version FROM schema_version")
        return int(row["version"]) if row else 0
