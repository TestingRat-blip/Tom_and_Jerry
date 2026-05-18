"""Drives configuration.

A drive is a slow-changing scalar in [0, 1] representing one axis of
Tom's motivational state. Drives shift based on events that happen in
the world; they decay (or grow) toward baseline when nothing is
happening.

This module defines the SHAPE of how drives respond. The actual state
update logic lives in DriveSystem in drives.py.

Six axes in v1:
  - hunger:      time-since-last-catch pressure; rises on patrol, falls on contact
  - aggression:  willingness to commit to pursuit even at risk of losing scent
  - caution:     willingness to break off pursuit when uncertain
  - curiosity:   willingness to investigate weak/distant stimuli
  - fatigue:     accumulated exhaustion from long pursuits or wall-bumping
  - social_bond: pack cohesion drive (placeholder for v2; baseline-only in v1)

Convention: drives drift toward their baseline at decay_rate per tick.
Events apply deltas which decay back over tens of ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.env.world.world import EventType


@dataclass(frozen=True, slots=True)
class DriveAxisConfig:
    """How one drive axis behaves."""
    baseline: float = 0.3
    decay_rate: float = 0.02     # per-tick pull toward baseline
    floor: float = 0.0
    ceiling: float = 1.0


@dataclass(frozen=True, slots=True)
class DrivesConfig:
    """Per-axis config plus the event → delta mapping.

    Event mapping is a dict from EventType to a dict of {drive_name: delta}.
    Multiple drives can move on the same event. Deltas are applied AFTER
    decay each tick.
    """
    hunger: DriveAxisConfig = DriveAxisConfig(baseline=0.4, decay_rate=0.005)
    aggression: DriveAxisConfig = DriveAxisConfig(baseline=0.5, decay_rate=0.02)
    caution: DriveAxisConfig = DriveAxisConfig(baseline=0.5, decay_rate=0.02)
    curiosity: DriveAxisConfig = DriveAxisConfig(baseline=0.4, decay_rate=0.02)
    fatigue: DriveAxisConfig = DriveAxisConfig(baseline=0.1, decay_rate=0.01)
    social_bond: DriveAxisConfig = DriveAxisConfig(baseline=0.5, decay_rate=0.0)

    # Passive growth — drives that creep up regardless of events
    # (hunger and fatigue grow with time; others are event-driven)
    hunger_per_tick: float = 0.001
    fatigue_per_tick_on_move: float = 0.0005

    # Event-driven deltas. Tuple-of-tuples because dict isn't frozen-friendly.
    # Each entry: (event_type, drive_name, delta)
    event_deltas: tuple[tuple[int, str, float], ...] = (
        # Catch reduces hunger dramatically, builds confidence (aggression up)
        (EventType.TOM_CAUGHT_JERRY, "hunger", -0.6),
        (EventType.TOM_CAUGHT_JERRY, "aggression", +0.2),
        (EventType.TOM_CAUGHT_JERRY, "fatigue", -0.1),

        # Seeing Jerry sharpens aggression and reduces caution
        (EventType.TOM_SAW_JERRY, "aggression", +0.05),
        (EventType.TOM_SAW_JERRY, "caution", -0.02),

        # Bumping a wall is frustrating — fatigue + reduces aggression slightly
        (EventType.TOM_BUMPED_WALL, "fatigue", +0.02),
        (EventType.TOM_BUMPED_WALL, "aggression", -0.01),

        # Hearing noise piques curiosity
        (EventType.NOISE_EMITTED, "curiosity", +0.01),
    )
