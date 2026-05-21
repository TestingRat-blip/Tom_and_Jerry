"""L2 episode-summary store backed by SQLite.

One row per finished episode. Stores aggregated stats Tom can use to
warm-start L1 on subsequent episodes. The schema is defined in
`client.py`'s migrations.

This module knows the schema. The high-level distillation logic that
builds an EpisodeSummary from an L1Memory lives in
`src/hunter/agent/memory/distillation.py` (Batch 9b).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from src.persistence.sqlite.client import SQLiteClient


# Heatmap and locker entries are stored as JSON arrays of [x, y, count].
# Kept as a type alias for readability.
TileCount = tuple[int, int, int]


@dataclass(slots=True)
class EpisodeSummary:
    """One finished episode's compressed memory.

    `episode_id` and `created_at` are auto-filled if you leave them as
    defaults. Everything else must be supplied by the caller (typically
    by `distillation.distill_l1_to_summary`).
    """
    map_fingerprint_fine: str
    map_fingerprint_coarse: str
    jerry_fingerprint: str
    outcome: str            # "caught" | "survived" | "timeout"
    total_ticks: int
    total_jerry_reward: float

    # Optional fields with defaults
    episode_id: str = ""
    created_at: float = 0.0
    tom_label: str = ""
    ticks_to_first_sight: int | None = None
    heatmap_top: list[TileCount] = field(default_factory=list)
    lockers: list[TileCount] = field(default_factory=list)
    false_noise_top: list[TileCount] = field(default_factory=list)
    total_noise_events: int = 0
    verified_noise_count: int = 0
    # Behavioral signatures (schema v2) — HOW Jerry behaved, for memory-
    # driven adaptation. los_break_count is the cover-dance signal.
    los_break_count: int = 0
    los_break_hotspots: list[TileCount] = field(default_factory=list)
    time_in_cover_fraction: float = 0.0
    oscillation_score: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id:
            self.episode_id = uuid.uuid4().hex
        if not self.created_at:
            self.created_at = time.time()


class L2Store:
    """CRUD for episode summaries in SQLite.

    Construct once per process and reuse — the underlying SQLiteClient
    holds a single connection.
    """

    def __init__(self, client: SQLiteClient):
        self.client = client

    # ---- write --------------------------------------------------------

    def insert(self, summary: EpisodeSummary) -> None:
        """Persist a summary. Uses INSERT OR REPLACE on episode_id so
        idempotent retries are safe.
        """
        self.client.execute(
            """
            INSERT OR REPLACE INTO episode_summaries (
                episode_id, created_at,
                map_fingerprint_fine, map_fingerprint_coarse, jerry_fingerprint,
                tom_label, outcome, total_ticks, total_jerry_reward,
                ticks_to_first_sight,
                heatmap_top_json, lockers_json, false_noise_top_json,
                total_noise_events, verified_noise_count,
                los_break_count, los_break_hotspots_json,
                time_in_cover_fraction, oscillation_score,
                notes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.episode_id, summary.created_at,
                summary.map_fingerprint_fine, summary.map_fingerprint_coarse,
                summary.jerry_fingerprint, summary.tom_label,
                summary.outcome, summary.total_ticks, summary.total_jerry_reward,
                summary.ticks_to_first_sight,
                json.dumps([list(t) for t in summary.heatmap_top]),
                json.dumps([list(t) for t in summary.lockers]),
                json.dumps([list(t) for t in summary.false_noise_top]),
                summary.total_noise_events, summary.verified_noise_count,
                summary.los_break_count,
                json.dumps([list(t) for t in summary.los_break_hotspots]),
                summary.time_in_cover_fraction, summary.oscillation_score,
                json.dumps(summary.notes),
            ),
        )

    # ---- read ---------------------------------------------------------

    def get_by_id(self, episode_id: str) -> EpisodeSummary | None:
        row = self.client.fetchone(
            "SELECT * FROM episode_summaries WHERE episode_id = ?",
            (episode_id,),
        )
        return _row_to_summary(row) if row else None

    def query_fine(
        self,
        map_fp_fine: str,
        jerry_fp: str,
        limit: int = 20,
    ) -> list[EpisodeSummary]:
        """Most recent N episodes matching this EXACT map + jerry combo."""
        rows = self.client.fetchall(
            """
            SELECT * FROM episode_summaries
            WHERE map_fingerprint_fine = ? AND jerry_fingerprint = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (map_fp_fine, jerry_fp, limit),
        )
        return [_row_to_summary(r) for r in rows]

    def query_coarse(
        self,
        map_fp_coarse: str,
        jerry_fp: str,
        limit: int = 20,
        exclude_fine: str | None = None,
    ) -> list[EpisodeSummary]:
        """Most recent N episodes matching coarse map signature + jerry.

        Pass `exclude_fine` to omit episodes that already matched the
        fine fingerprint (avoids double-counting in a fine→coarse cascade).
        """
        if exclude_fine is None:
            rows = self.client.fetchall(
                """
                SELECT * FROM episode_summaries
                WHERE map_fingerprint_coarse = ? AND jerry_fingerprint = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (map_fp_coarse, jerry_fp, limit),
            )
        else:
            rows = self.client.fetchall(
                """
                SELECT * FROM episode_summaries
                WHERE map_fingerprint_coarse = ?
                  AND jerry_fingerprint = ?
                  AND map_fingerprint_fine != ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (map_fp_coarse, jerry_fp, exclude_fine, limit),
            )
        return [_row_to_summary(r) for r in rows]

    def query_jerry_only(
        self,
        jerry_fp: str,
        limit: int = 20,
    ) -> list[EpisodeSummary]:
        """All episodes against this jerry, across any map. Useful for
        jerry-policy-level stats (e.g. "how often does this jerry
        survive on average?").
        """
        rows = self.client.fetchall(
            """
            SELECT * FROM episode_summaries
            WHERE jerry_fingerprint = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (jerry_fp, limit),
        )
        return [_row_to_summary(r) for r in rows]

    # ---- counts / introspection ---------------------------------------

    def count(self) -> int:
        row = self.client.fetchone("SELECT COUNT(*) AS c FROM episode_summaries")
        return int(row["c"]) if row else 0

    def count_for(self, map_fp_fine: str, jerry_fp: str) -> int:
        row = self.client.fetchone(
            """
            SELECT COUNT(*) AS c FROM episode_summaries
            WHERE map_fingerprint_fine = ? AND jerry_fingerprint = ?
            """,
            (map_fp_fine, jerry_fp),
        )
        return int(row["c"]) if row else 0

    # ---- maintenance --------------------------------------------------

    def delete_all(self) -> int:
        """Wipe every episode summary. Tests use this; production code
        should not.
        """
        cur = self.client.execute("DELETE FROM episode_summaries")
        return cur.rowcount

    def delete_older_than(self, seconds: float) -> int:
        """Prune episodes older than `seconds` ago. Useful when L2 grows
        unbounded during long training runs.
        """
        cutoff = time.time() - seconds
        cur = self.client.execute(
            "DELETE FROM episode_summaries WHERE created_at < ?",
            (cutoff,),
        )
        return cur.rowcount


def _row_to_summary(row) -> EpisodeSummary:
    """Convert a SQLite Row back into an EpisodeSummary."""
    return EpisodeSummary(
        episode_id=row["episode_id"],
        created_at=float(row["created_at"]),
        map_fingerprint_fine=row["map_fingerprint_fine"],
        map_fingerprint_coarse=row["map_fingerprint_coarse"],
        jerry_fingerprint=row["jerry_fingerprint"],
        tom_label=row["tom_label"] or "",
        outcome=row["outcome"],
        total_ticks=int(row["total_ticks"]),
        total_jerry_reward=float(row["total_jerry_reward"]),
        ticks_to_first_sight=(
            int(row["ticks_to_first_sight"])
            if row["ticks_to_first_sight"] is not None else None
        ),
        heatmap_top=[tuple(t) for t in json.loads(row["heatmap_top_json"])],
        lockers=[tuple(t) for t in json.loads(row["lockers_json"])],
        false_noise_top=[tuple(t) for t in json.loads(row["false_noise_top_json"])],
        total_noise_events=int(row["total_noise_events"]),
        verified_noise_count=int(row["verified_noise_count"]),
        los_break_count=int(_row_get(row, "los_break_count", 0)),
        los_break_hotspots=[
            tuple(t) for t in json.loads(_row_get(row, "los_break_hotspots_json", "[]"))
        ],
        time_in_cover_fraction=float(_row_get(row, "time_in_cover_fraction", 0.0)),
        oscillation_score=float(_row_get(row, "oscillation_score", 0.0)),
        notes=json.loads(row["notes_json"]),
    )


def _row_get(row, key: str, default):
    """Defensive Row accessor: returns default if the column is absent.
    Lets the reader tolerate rows from a DB that hasn't been migrated to v2
    yet (shouldn't happen in normal flow, but safe)."""
    try:
        val = row[key]
    except (IndexError, KeyError):
        return default
    return val if val is not None else default
