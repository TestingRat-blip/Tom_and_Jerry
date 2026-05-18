"""L1 Redis store — key schema and low-level CRUD for per-encounter memory.

L1 memory tracks three categories within a single episode:
  - Noise events: each NOISE_EMITTED Tom heard, with verified/unverified status
  - Locker observations: per-locker count of "saw Jerry within 2 tiles"
  - Sighting heatmap: per-tile count of times Tom saw Jerry there

Key schema (all prefixed with the config namespace, e.g. 'tj:'):
  l1:{ep}:meta               hash with metadata (start_tick, etc.)
  l1:{ep}:noise:{tick}       hash: {x, y, intensity, verified}
  l1:{ep}:noise_index        hash: tile_key → counter (false-noise count per tile)
  l1:{ep}:locker             hash: "x,y" → integer (sighting-near count per locker)
  l1:{ep}:heatmap            hash: "x,y" → integer (sighting count per tile)

`ep` is an episode_id — typically a monotonic counter or UUID. The store
handles all key building; callers never construct raw key strings.

All keys for a given episode TTL together at episode end via clear_episode().

This module is intentionally thin — it knows about Redis semantics but
NOT about Tom's behavior. The high-level API for Tom lives in
src/hunter/agent/memory/l1.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.persistence.redis.client import RedisClient


@dataclass(frozen=True, slots=True)
class NoiseRecord:
    """A single noise event Tom heard.

    `verified=True` means Tom subsequently saw Jerry near the noise's
    location within a configurable window — so this noise was a real
    Jerry-event, not a false alarm. The L1Memory class decides when
    to mark records verified.
    """
    tick: int
    x: int
    y: int
    intensity: float
    verified: bool = False


class L1Store:
    """Redis-backed storage for L1 per-encounter memory.

    Each L1Store instance is bound to a specific (episode_id) so that
    keys don't collide across concurrent episodes. Reset between episodes
    by calling clear_episode().
    """

    def __init__(self, client: RedisClient, episode_id: str):
        self.client = client
        self.episode_id = episode_id

    # ---- key builders --------------------------------------------------

    def _meta_key(self) -> str:
        return self.client.ns("l1", self.episode_id, "meta")

    def _noise_key(self, tick: int) -> str:
        return self.client.ns("l1", self.episode_id, "noise", str(tick))

    def _noise_index_key(self) -> str:
        return self.client.ns("l1", self.episode_id, "noise_index")

    def _locker_key(self) -> str:
        return self.client.ns("l1", self.episode_id, "locker")

    def _heatmap_key(self) -> str:
        return self.client.ns("l1", self.episode_id, "heatmap")

    @staticmethod
    def _tile_field(x: int, y: int) -> str:
        return f"{x},{y}"

    @staticmethod
    def _parse_tile_field(field: str) -> tuple[int, int]:
        x, y = field.split(",")
        return int(x), int(y)

    # ---- noise ---------------------------------------------------------

    def record_noise(self, record: NoiseRecord) -> None:
        """Store a noise event with all its fields."""
        self.client.hset(self._noise_key(record.tick), mapping={
            "x": str(record.x),
            "y": str(record.y),
            "intensity": repr(record.intensity),
            "verified": "1" if record.verified else "0",
        })

    def get_noise(self, tick: int) -> NoiseRecord | None:
        raw = self.client.hgetall(self._noise_key(tick))
        if not raw:
            return None
        return NoiseRecord(
            tick=tick,
            x=int(raw["x"]),
            y=int(raw["y"]),
            intensity=float(raw["intensity"]),
            verified=raw.get("verified", "0") == "1",
        )

    def mark_noise_verified(self, tick: int) -> None:
        """Flip the verified bit on an existing noise event."""
        self.client.hset(self._noise_key(tick), "verified", "1")

    def increment_false_noise_count(self, x: int, y: int) -> int:
        """Add 1 to the per-tile false-noise counter. Returns the new count."""
        return self.client.hincrby(
            self._noise_index_key(), self._tile_field(x, y), 1
        )

    def get_false_noise_count(self, x: int, y: int) -> int:
        v = self.client.hget(self._noise_index_key(), self._tile_field(x, y))
        return int(v) if v is not None else 0

    def all_false_noise_counts(self) -> dict[tuple[int, int], int]:
        """Get the whole noise-index hash as {(x, y): count}."""
        raw = self.client.hgetall(self._noise_index_key())
        return {self._parse_tile_field(f): int(v) for f, v in raw.items()}

    # ---- lockers -------------------------------------------------------

    def increment_locker_sightings(self, x: int, y: int) -> int:
        return self.client.hincrby(
            self._locker_key(), self._tile_field(x, y), 1
        )

    def get_locker_sightings(self, x: int, y: int) -> int:
        v = self.client.hget(self._locker_key(), self._tile_field(x, y))
        return int(v) if v is not None else 0

    def all_locker_sightings(self) -> dict[tuple[int, int], int]:
        raw = self.client.hgetall(self._locker_key())
        return {self._parse_tile_field(f): int(v) for f, v in raw.items()}

    # ---- heatmap -------------------------------------------------------

    def increment_sighting_heatmap(self, x: int, y: int) -> int:
        return self.client.hincrby(
            self._heatmap_key(), self._tile_field(x, y), 1
        )

    def get_heatmap_count(self, x: int, y: int) -> int:
        v = self.client.hget(self._heatmap_key(), self._tile_field(x, y))
        return int(v) if v is not None else 0

    def all_heatmap_counts(self) -> dict[tuple[int, int], int]:
        raw = self.client.hgetall(self._heatmap_key())
        return {self._parse_tile_field(f): int(v) for f, v in raw.items()}

    # ---- lifecycle -----------------------------------------------------

    def set_meta(self, **fields) -> None:
        if not fields:
            return
        self.client.hset(self._meta_key(),
                         mapping={k: str(v) for k, v in fields.items()})

    def get_meta(self) -> dict[str, str]:
        return self.client.hgetall(self._meta_key())

    def clear_episode(self) -> None:
        """Delete all keys for this episode.

        Called at the END of each episode (before reset) to free Redis
        memory. L1 is per-encounter only — distillation into L2 happens
        BEFORE this clear, in Phase 4.
        """
        pattern = self.client.ns("l1", self.episode_id, "*")
        keys_to_delete = list(self.client.scan_iter(match=pattern, count=200))
        if keys_to_delete:
            self.client.delete(*keys_to_delete)
