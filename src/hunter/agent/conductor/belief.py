"""The Conductor's belief about where Jerry is — Phase 6a.

This module is deliberately SELF-CONTAINED. It has no dependency on the
world, on Tom, or on the Conductor itself. It is a pure data structure +
the rules for maintaining it, so it can be unit-tested in complete
isolation.

The belief is NOT a position estimate. It is a set of *reasons to suspect
Jerry is somewhere* — "suspicion sources" — each tagged with what caused
it and how confident we are, with confidence decaying over time at a
type-specific rate.

  - SIGHTING : Tom actually saw Jerry here. Trustworthy; decays slowly.
  - SCENT    : a scent gradient points here. Diffuse; decays medium.
  - NOISE    : a sound happened here. Fresh only briefly; decays fast.

The Conductor (Phase 6b) queries this belief each tick to decide where to
send Tom and in what mode. Jerry can manipulate the belief by generating
false signals (e.g. noise far from its real position) — that manipulation
is a first-class part of the design, which is why the belief tracks
*reasons* (typed, forgeable) rather than ground truth.

Per ADR-013: nothing in this module ever sees Jerry's true position. It
only ingests observable signals handed to it by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from src.utils.types import Position


class SuspicionType(IntEnum):
    """Why we suspect Jerry is somewhere. Ordered by base trustworthiness
    (higher = more trustworthy), though confidence and decay ultimately
    determine which source wins.
    """
    NOISE = 0      # a sound happened here; fades fast
    SCENT = 1      # scent gradient points here; fades medium
    SIGHTING = 2   # Tom saw Jerry here; fades slow


@dataclass
class SuspicionSource:
    """A single reason to suspect Jerry is at/near `position`.

    confidence is the value at birth; the *current* confidence is computed
    on demand from age and the decay rate (see BeliefConfig). We store the
    birth confidence + birth tick rather than mutating confidence in place,
    so decay is always a pure function of (birth state, current tick).
    """
    position: Position
    type: SuspicionType
    birth_confidence: float
    born_tick: int
    # last_reinforced_tick lets a merged/refreshed source reset its age
    # without losing its identity. Decay is measured from here.
    last_reinforced_tick: int

    def current_confidence(self, now_tick: int, decay_rate: float) -> float:
        """Exponential decay from the last reinforcement.

        decay_rate is per-tick multiplicative (0 < decay_rate < 1).
        age 0 => birth_confidence; each tick multiplies by decay_rate.
        """
        age = max(0, now_tick - self.last_reinforced_tick)
        return self.birth_confidence * (decay_rate ** age)


@dataclass(frozen=True)
class BeliefConfig:
    """Tunable knobs for the belief layer. All the Phase 6 [TBD] decay /
    merge / cap parameters live here so they're in one place for tuning
    (Stage 1) and learning (Stage 2).
    """
    # Per-type per-tick decay multipliers. Smaller = faster decay.
    # Chosen so that:
    #   NOISE    ~ half-life  ~5 ticks   (0.87^5  ~ 0.50)
    #   SCENT    ~ half-life ~12 ticks   (0.944^12 ~ 0.50)
    #   SIGHTING ~ half-life ~25 ticks   (0.973^25 ~ 0.50)
    decay_noise: float = 0.87
    decay_scent: float = 0.944
    decay_sighting: float = 0.973

    # A source is dead (removed) once current confidence drops below this.
    death_floor: float = 0.05

    # Birth confidence per type when a fresh signal arrives. NOISE birth
    # confidence may be further scaled by the signal intensity (payload).
    birth_noise: float = 0.6
    birth_scent: float = 0.5
    birth_sighting: float = 1.0

    # Merge: a new signal of the SAME type within this manhattan radius of
    # an existing source merges into it (refresh) instead of spawning a
    # duplicate. Keeps the belief legible (one noisy Jerry != five sources).
    merge_radius: int = 3

    # Maximum number of live sources retained. Over cap, the weakest
    # (lowest current confidence) is dropped. Bounds memory + keeps
    # behavior legible.
    max_sources: int = 8

    def decay_for(self, t: SuspicionType) -> float:
        if t == SuspicionType.NOISE:
            return self.decay_noise
        if t == SuspicionType.SCENT:
            return self.decay_scent
        return self.decay_sighting

    def birth_for(self, t: SuspicionType) -> float:
        if t == SuspicionType.NOISE:
            return self.birth_noise
        if t == SuspicionType.SCENT:
            return self.birth_scent
        return self.birth_sighting


class SuspicionBelief:
    """The Conductor's live belief: a managed collection of suspicion
    sources. Caller drives it with add_* (when signals arrive) and tick()
    (once per world tick), then queries strongest()/live_sources().

    Time is caller-supplied (now_tick). The belief does not own a clock —
    it's told the current tick on every mutating call. This keeps it pure
    and testable.
    """

    def __init__(self, config: BeliefConfig | None = None):
        self.config = config or BeliefConfig()
        self._sources: list[SuspicionSource] = []

    # ---- ingestion ----------------------------------------------------

    def add_sighting(self, position: Position, now_tick: int) -> None:
        """Tom saw Jerry at `position`. Strongest, slowest-decaying signal."""
        self._add(SuspicionType.SIGHTING, position, now_tick,
                  self.config.birth_for(SuspicionType.SIGHTING))

    def add_noise(self, position: Position, now_tick: int,
                  intensity: float = 1.0) -> None:
        """A sound was emitted at `position`. Fades fast. `intensity`
        (0..1+) scales birth confidence so loud sounds matter more.
        """
        birth = self.config.birth_for(SuspicionType.NOISE) * max(0.0, intensity)
        # Clamp so a very loud noise can't out-trust a sighting at birth.
        birth = min(birth, self.config.birth_for(SuspicionType.SIGHTING))
        self._add(SuspicionType.NOISE, position, now_tick, birth)

    def add_scent(self, position: Position, now_tick: int,
                  strength: float = 1.0) -> None:
        """A scent gradient points toward `position`. Medium decay.
        `strength` scales birth confidence.
        """
        birth = self.config.birth_for(SuspicionType.SCENT) * max(0.0, strength)
        birth = min(birth, self.config.birth_for(SuspicionType.SIGHTING))
        self._add(SuspicionType.SCENT, position, now_tick, birth)

    def _add(self, t: SuspicionType, position: Position, now_tick: int,
             birth_confidence: float) -> None:
        """Add a new source, or merge into a nearby same-type source."""
        # Try to merge into an existing same-type source within merge_radius.
        for src in self._sources:
            if src.type != t:
                continue
            if src.position.manhattan(position) <= self.config.merge_radius:
                # Refresh: move toward the new signal, reset age, take the
                # stronger birth confidence. The suspicion "follows" Jerry.
                src.position = position
                src.birth_confidence = max(src.birth_confidence, birth_confidence)
                src.last_reinforced_tick = now_tick
                return
        # No merge target — create a new source.
        self._sources.append(SuspicionSource(
            position=position,
            type=t,
            birth_confidence=birth_confidence,
            born_tick=now_tick,
            last_reinforced_tick=now_tick,
        ))
        self._enforce_cap(now_tick)

    # ---- time ---------------------------------------------------------

    def tick(self, now_tick: int) -> None:
        """Advance time: drop any source whose current confidence has
        fallen below the death floor. Decay itself is computed on demand
        in current_confidence(), so tick() only prunes.
        """
        floor = self.config.death_floor
        self._sources = [
            s for s in self._sources
            if s.current_confidence(now_tick, self.config.decay_for(s.type)) >= floor
        ]

    # ---- queries ------------------------------------------------------

    def live_sources(self, now_tick: int) -> list[tuple[SuspicionSource, float]]:
        """All live sources paired with their current confidence, sorted
        strongest first. A source is 'live' if its current confidence is
        at or above the death floor.
        """
        floor = self.config.death_floor
        out: list[tuple[SuspicionSource, float]] = []
        for s in self._sources:
            c = s.current_confidence(now_tick, self.config.decay_for(s.type))
            if c >= floor:
                out.append((s, c))
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out

    def strongest(self, now_tick: int) -> tuple[SuspicionSource, float] | None:
        """The single highest-confidence live source, or None if the belief
        is empty. This is the Conductor's primary query.
        """
        live = self.live_sources(now_tick)
        return live[0] if live else None

    def is_empty(self, now_tick: int) -> bool:
        """True if there are no live suspicions (Conductor should patrol)."""
        return self.strongest(now_tick) is None

    def __len__(self) -> int:
        """Raw source count (including not-yet-pruned dead ones).
        Use live_sources() for the live count.
        """
        return len(self._sources)

    def clear(self) -> None:
        """Drop all suspicions (e.g. on episode reset)."""
        self._sources.clear()

    # ---- internal -----------------------------------------------------

    def _enforce_cap(self, now_tick: int) -> None:
        """If over max_sources, drop the weakest by current confidence."""
        if len(self._sources) <= self.config.max_sources:
            return
        self._sources.sort(
            key=lambda s: s.current_confidence(
                now_tick, self.config.decay_for(s.type)),
            reverse=True,
        )
        self._sources = self._sources[:self.config.max_sources]
