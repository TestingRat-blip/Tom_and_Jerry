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
"""
from __future__ import annotations

from dataclasses import dataclass

from src.env.world.world import Event, EventType
from src.persistence.redis.client import RedisClient
from src.persistence.redis.l1_store import L1Store, NoiseRecord
from src.utils.types import Position


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

    def reset(self) -> None:
        """Clear all L1 state for a new episode."""
        self.store.clear_episode()
        self._pending.clear()

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

        Used by ChemicalTom's _modulated_noise_threshold to depreciate
        noise-following when Tom has been fooled repeatedly nearby.
        """
        # Sum false-noise counts within lookup radius
        total = 0
        for (x, y), count in self.store.all_false_noise_counts().items():
            if abs(x - pos.x) + abs(y - pos.y) <= self.config.false_noise_lookup_radius:
                total += count
        if total == 0:
            return 0.0
        # Saturating linear function: 0 at zero counts, max at saturation count
        saturation = self.config.false_noise_saturation
        fraction = min(1.0, total / saturation)
        return fraction * self.config.max_false_noise_factor

    def locker_suspicion(self, locker_pos: Position) -> int:
        """How many times Jerry has been sighted near this locker this episode."""
        return self.store.get_locker_sightings(locker_pos.x, locker_pos.y)

    def most_suspicious_locker(self) -> Position | None:
        """The locker with the highest sighting count, or None if no lockers
        have been sighted near.
        """
        sightings = self.store.all_locker_sightings()
        if not sightings:
            return None
        (best_xy, _) = max(sightings.items(), key=lambda kv: kv[1])
        return Position(best_xy[0], best_xy[1])

    def heatmap_hottest(self, top_n: int = 5) -> list[tuple[Position, int]]:
        """Return the top-N most-sighted tiles this episode."""
        heat = self.store.all_heatmap_counts()
        if not heat:
            return []
        items = sorted(heat.items(), key=lambda kv: kv[1], reverse=True)
        return [(Position(x, y), c) for (x, y), c in items[:top_n]]

    def total_noise_events(self) -> int:
        """How many noises have been recorded this episode (useful for
        bookkeeping/inspection).
        """
        return sum(self.store.all_false_noise_counts().values())
