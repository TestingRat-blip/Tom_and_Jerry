"""Chemistry configuration.

Five chemicals modulate Tom's moment-to-moment behavior. Unlike drives,
chemicals:
  - Have baseline ZERO (chemistry is absent at rest, spikes on stimulus)
  - Decay exponentially (~half-life of seconds, not minutes)
  - Interact with each other (adrenaline suppresses cortisol, etc.)
  - Use accumulation buffers (events deposit into a buffer that decays
    separately, preventing single-event saturation)

Five chemicals in v1:
  - adrenaline: spike on stimulus; drives prediction horizon, pursuit aggression
  - cortisol:   slow stress build-up; drives "give up" vs "desperate" branching
  - dopamine:   reward signal; reinforces behaviors that preceded a catch
  - oxytocin:   pack bonding (placeholder in v1; baseline-only)
  - serotonin:  baseline confidence; modulates adrenaline ceiling

Per ADR-006, this is re-implemented from Vera's patterns, not imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.env.world.world import EventType


@dataclass(frozen=True, slots=True)
class ChemicalAxisConfig:
    """How one chemical behaves."""
    decay_per_tick: float = 0.92       # multiplicative; 0.92 → 50% in ~8 ticks
    buffer_decay_per_tick: float = 0.80  # accumulation buffer decays faster
    buffer_transfer_rate: float = 0.4   # how much of buffer moves into level each tick
    ceiling: float = 1.0
    floor: float = 0.0


@dataclass(frozen=True, slots=True)
class ChemistryConfig:
    """Per-chemical config plus event mapping plus cross-interactions."""
    # Per-chemical knobs. Defaults tuned to feel right for Phase 2 v1 —
    # adrenaline is fast (sharp spike, fast decay), cortisol is slow (slow
    # build, slow decay), dopamine sharp-then-fades, serotonin nearly static.
    adrenaline: ChemicalAxisConfig = ChemicalAxisConfig(
        decay_per_tick=0.85, buffer_decay_per_tick=0.70, buffer_transfer_rate=0.5,
    )
    cortisol: ChemicalAxisConfig = ChemicalAxisConfig(
        decay_per_tick=0.97, buffer_decay_per_tick=0.92, buffer_transfer_rate=0.2,
    )
    dopamine: ChemicalAxisConfig = ChemicalAxisConfig(
        decay_per_tick=0.88, buffer_decay_per_tick=0.75, buffer_transfer_rate=0.5,
    )
    oxytocin: ChemicalAxisConfig = ChemicalAxisConfig(
        decay_per_tick=0.99, buffer_decay_per_tick=0.95, buffer_transfer_rate=0.1,
    )
    serotonin: ChemicalAxisConfig = ChemicalAxisConfig(
        decay_per_tick=0.995, buffer_decay_per_tick=0.97, buffer_transfer_rate=0.05,
    )

    # Passive trickle — even without events, some chemicals creep up slowly
    cortisol_per_tick_when_no_jerry: float = 0.002  # frustration builds
    serotonin_baseline_pressure: float = 0.001     # confidence slowly recovers toward 0.3

    # Event mappings: (event_type, chemical_name, buffer_delta)
    # Note: deltas go into the BUFFER, not directly to the level. This keeps
    # large/repeated events from saturating chemicals to 1.0 immediately.
    event_deltas: tuple[tuple[int, str, float], ...] = (
        # Sighting Jerry: massive adrenaline spike
        (EventType.TOM_SAW_JERRY, "adrenaline", +0.35),
        # Catching Jerry: dopamine flood, cortisol crash
        (EventType.TOM_CAUGHT_JERRY, "dopamine", +0.8),
        (EventType.TOM_CAUGHT_JERRY, "cortisol", -0.5),
        (EventType.TOM_CAUGHT_JERRY, "serotonin", +0.3),
        # Bumping a wall: mild cortisol bump (frustration)
        (EventType.TOM_BUMPED_WALL, "cortisol", +0.04),
        # Hearing noise: small adrenaline tick (alertness)
        (EventType.NOISE_EMITTED, "adrenaline", +0.03),
    )

    # Cross-chemical interactions, applied EACH TICK after decay + event deltas:
    # (source_chemical, target_chemical, coefficient)
    # Effect: target += source * coefficient * dt
    # Negative coefficient = suppression. Positive = reinforcement.
    interactions: tuple[tuple[str, str, float], ...] = (
        # Adrenaline suppresses cortisol (fight-or-flight masks stress)
        ("adrenaline", "cortisol", -0.05),
        # Serotonin caps adrenaline (confident Tom doesn't get rattled as easily)
        ("serotonin", "adrenaline", -0.03),
        # Dopamine slowly raises serotonin baseline (success builds confidence)
        ("dopamine", "serotonin", +0.02),
    )
