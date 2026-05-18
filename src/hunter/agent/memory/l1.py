"""L1Memory — high-level per-encounter memory API for Tom.

This is the layer ChemicalTom talks to. It hides the Redis store
behind methods that match the questions Tom actually asks:

  - "Did I just hear a noise? Record it."
  - "Was the noise I heard N ticks ago actually Jerry? Mark it verified."
  - "How likely is a noise from THIS direction to be real, given recent
    history?"
  - "Did I see Jerry near a locker? Bump the suspicion."
  - "Where do I tend to see Jerry most? (sighting hotspot)"

The behavior tree reads `false_noise_factor_near` to depreciate noise-
following when nearby noises have been distractions. Higher factor →
higher effective noise threshold → Tom needs LOUDER noise to investigate.

PHASE 4 EXTENSION — warm-start priors:
At episode begin, L1 can be pre-seeded with priors from L2 via
`apply_warm_start(WarmStart)`. The priors live in parallel dicts
(separate from in-episode counters). Behavior queries (false_noise_factor_near,
heatmap_hottest, locker_suspicion, most_suspicious_locker) read BOTH
warm-start + in-episode and return the combined value. The distillation
pipeline (which produces NEW L2 entries) reads in-episode counters
ONLY — so priors don't compound across episodes.

Warm-start is cleared by `reset()` along with in-episode state. To
apply warm-start: reset(), then apply_warm_start(...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.env.world.world import Event, EventType
from src.persistence.redis.client import RedisClient
from src.persistence.redis.l1_store import L1Store, NoiseRecord
from src.utils.types import Position

if TYPE_CHECKING:
    from src.hunter.agent.memory.l2_lookup import WarmStart


@dataclass(frozen=True, slots=True)
class L1Config:
    """Tunable parameters for L1 memory.

    Defaults are conservative — meaningful but not dominating. Phase 5+
    may evolve these per archetype.
    """
    # How many ticks after a noise to look for a sighting near it.
    # If Tom sees Jerry within this window AND within `verification_radius`
    # of the noise, mark the noise verified.
    verification_window_ticks: int = 12
    verification_radius: int = 5

    # Radius around Tom's position to consider when computing the
    # false-noise factor for the current location.
    false_noise_lookup_radius: int = 6

    # Saturation: how many false noises in the lookup area produce the
    # maximum threshold-multiplier contribution.
    false_noise_saturation: int = 4

    # Maximum factor contribution to the noise threshold multiplier.
    # `factor` value is added to (1.0 + ...) so 0.5 means a +50% threshold
    # bump at saturation.
    max_false_noise_factor: float = 0.5

    # Locker bookkeeping: how close to a locker does Jerry need to be
    # sighted for that locker to count as "suspicious"?
    locker_proximity: int = 2


@dataclass(slots=True)
class _PendingNoise:
    """An unverified noise we're still watching for a sighting."""
    tick: int
    x: int
    y: int
    intensity: float


class L1Memory:
    """The high-level per-encounter memory ChemicalTom uses.

    Usage from ChemicalTom (Batch 8b):
        self.l1 = L1Memory(client, episode_id="ep_001")
        self.l1.observe_events(events, tom_pos, jerry_pos, jerry_visible, tick)
        # then read:
        factor = self.l1.false_noise_factor_near(tom_pos)
    """

    def __init__(
        self,
        client: RedisClient,
        episode_id: str,
        config: L1Config | None = None,
        locker_positions: list[Position] | None = None,
    ):
        self.store = L1Store(client, episode_id)
        self.config = config or L1Config()
        # Set of locker positions on this map — known at episode start.
        # Used to recognize "Jerry sighted near a locker."
        self.locker_positions: set[Position] = set(locker_positions or [])
        # Pending noises awaiting verification, kept in memory (Redis is
        # for finalized state; this is per-tick scratchpad).
        self._pending: list[_PendingNoise] = []

        # Phase 4 warm-start priors. Parallel to the store's counters;
        # behavior queries combine both. Distillation reads store-only.
        self._warm_heatmap: dict[tuple[int, int], float] = {}
        self._warm_lockers: dict[tuple[int, int], float] = {}
        self._warm_false_noise: dict[tuple[int, int], float] = {}

    def reset(self) -> None:
        """Clear all L1 state for a new episode.

        Clears BOTH in-episode counters AND warm-start priors. To use
        warm-start: call reset() first, then apply_warm_start(...).
        """
        self.store.clear_episode()
        self._pending.clear()
        self._warm_heatmap.clear()
        self._warm_lockers.clear()
        self._warm_false_noise.clear()

    def apply_warm_start(self, warm: "WarmStart") -> None:
        """Pre-seed L1 with priors from past episodes.

        Call AFTER reset() and BEFORE the first tick. The warm-start
        priors influence behavior queries (false_noise_factor_near,
        heatmap_hottest, locker_suspicion) but do NOT affect the
        in-episode counters that distillation will read at episode end.

        This separation prevents the bug of priors compounding across
        episodes — each episode learns from past episodes but only
        records its own observations into the next L2 summary.
        """
        # Take copies so the caller's WarmStart can be reused/mutated
        # without affecting our state.
        self._warm_heatmap = dict(warm.heatmap)
        self._warm_lockers = dict(warm.lockers)
        self._warm_false_noise = dict(warm.false_noise)

    def set_locker_positions(self, positions: list[Position]) -> None:
        self.locker_positions = set(positions)

    # ---- observation -----------------------------------------------------

    def observe_events(
        self,
        events: list[Event] | tuple[Event, ...],
        tom_pos: Position,
        jerry_pos: Position,
        jerry_visible: bool,
        tick: int,
    ) -> None:
        """Called once per tick by ChemicalTom. Routes events into L1
        and updates the verification state of pending noises.
        """
        # 1. Record any noise events Tom would have heard
        for ev in events:
            if ev.type == EventType.NOISE_EMITTED and ev.position is not None:
                intensity = float(ev.payload) if ev.payload is not None else 1.0
                # Only record noises Tom is actually aware of (some
                # filtering: skip Tom's OWN noise). Jerry's noise is the
                # interesting case.
                if ev.actor == "tom":
                    continue
                noise = NoiseRecord(
                    tick=tick, x=ev.position.x, y=ev.position.y,
                    intensity=intensity,
                )
                self.store.record_noise(noise)
                self._pending.append(_PendingNoise(
                    tick=tick, x=ev.position.x, y=ev.position.y,
                    intensity=intensity,
                ))

        # 2. If Jerry is visible, update sighting heatmap + try verifying
        # pending noises near him.
        if jerry_visible:
            self.store.increment_sighting_heatmap(jerry_pos.x, jerry_pos.y)
            # Try to mark recent nearby noises as verified
            new_pending: list[_PendingNoise] = []
            for p in self._pending:
                if tick - p.tick > self.config.verification_window_ticks:
                    # Aged out without verification — counts as a false noise
                    self.store.increment_false_noise_count(p.x, p.y)
                    continue
                dist = abs(p.x - jerry_pos.x) + abs(p.y - jerry_pos.y)
                if dist <= self.config.verification_radius:
                    self.store.mark_noise_verified(p.tick)
                    # Verified — drop from pending
                    continue
                new_pending.append(p)
            self._pending = new_pending

            # Locker suspicion: if Jerry is near a known locker, bump it
            for lp in self.locker_positions:
                if jerry_pos.manhattan(lp) <= self.config.locker_proximity:
                    self.store.increment_locker_sightings(lp.x, lp.y)
        else:
            # Without sighting, still age out expired pending noises.
            new_pending: list[_PendingNoise] = []
            for p in self._pending:
                if tick - p.tick > self.config.verification_window_ticks:
                    self.store.increment_false_noise_count(p.x, p.y)
                else:
                    new_pending.append(p)
            self._pending = new_pending

    # ---- queries -------------------------------------------------------

    def false_noise_factor_near(self, pos: Position) -> float:
        """Return a multiplier ∈ [0, max_false_noise_factor] for the
        noise threshold near `pos`. Higher = more false noises in the
        area, so Tom should be more skeptical.

        Combines in-episode false-noise counts (this life) with warm-start
        priors (past lives). Tom is skeptical of areas where EITHER
        previous Toms got fooled OR he's been fooled this episode.
        """
        # Sum in-episode false-noise counts within lookup radius
        total: float = 0.0
        radius = self.config.false_noise_lookup_radius
        for (x, y), count in self.store.all_false_noise_counts().items():
            if abs(x - pos.x) + abs(y - pos.y) <= radius:
                total += count
        # Add warm-start priors within the same radius
        for (x, y), weight in self._warm_false_noise.items():
            if abs(x - pos.x) + abs(y - pos.y) <= radius:
                total += weight

        if total == 0:
            return 0.0
        saturation = self.config.false_noise_saturation
        fraction = min(1.0, total / saturation)
        return fraction * self.config.max_false_noise_factor

    def locker_suspicion(self, locker_pos: Position) -> float:
        """Combined in-episode + warm-start sighting count for this locker."""
        in_ep = self.store.get_locker_sightings(locker_pos.x, locker_pos.y)
        warm = self._warm_lockers.get((locker_pos.x, locker_pos.y), 0.0)
        return in_ep + warm

    def most_suspicious_locker(self) -> Position | None:
        """The locker with the highest combined (in-episode + warm) sighting
        count, or None if no locker has nonzero suspicion from either source.
        """
        combined: dict[tuple[int, int], float] = {}
        for (x, y), c in self.store.all_locker_sightings().items():
            combined[(x, y)] = combined.get((x, y), 0.0) + c
        for (x, y), w in self._warm_lockers.items():
            combined[(x, y)] = combined.get((x, y), 0.0) + w
        if not combined:
            return None
        (best_xy, _) = max(combined.items(), key=lambda kv: kv[1])
        return Position(best_xy[0], best_xy[1])

    def heatmap_hottest(self, top_n: int = 5) -> list[tuple[Position, float]]:
        """Top-N tiles by combined sighting count (in-episode + warm-start).

        Note: return type changed from int counts to float counts in
        Phase 4 because warm-start weights are floats. Callers that
        treat the count as int should cast explicitly.
        """
        combined: dict[tuple[int, int], float] = {}
        for (x, y), c in self.store.all_heatmap_counts().items():
            combined[(x, y)] = combined.get((x, y), 0.0) + c
        for (x, y), w in self._warm_heatmap.items():
            combined[(x, y)] = combined.get((x, y), 0.0) + w
        if not combined:
            return []
        items = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        return [(Position(x, y), c) for (x, y), c in items[:top_n]]

    def total_noise_events(self) -> int:
        """How many noises have been recorded THIS EPISODE.

        Reads in-episode counters ONLY — does not include warm-start.
        Used by distillation to summarize the current episode without
        accidentally double-counting priors.
        """
        return sum(self.store.all_false_noise_counts().values())
