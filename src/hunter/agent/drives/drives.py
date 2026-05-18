"""Drive state and per-tick update system.

A Drives instance is just six floats. DriveSystem owns the update logic:
  1. Apply passive growth (hunger, fatigue from movement)
  2. Apply per-tick decay toward baseline
  3. Apply event-driven deltas
  4. Clamp to [floor, ceiling]

Per ADR-006, this is RE-IMPLEMENTED from the patterns Vera uses, not
imported. The two systems should not share code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from src.env.world.world import Event, EventType
from src.hunter.agent.drives.config import DriveAxisConfig, DrivesConfig


@dataclass
class Drives:
    """Tom's drive vector. Six axes, each ∈ [0, 1].

    Mutable on purpose — DriveSystem updates these in place each tick.
    For replay / inspection use snapshot() to get an immutable dict.
    """
    hunger: float = 0.4
    aggression: float = 0.5
    caution: float = 0.5
    curiosity: float = 0.4
    fatigue: float = 0.1
    social_bond: float = 0.5

    def snapshot(self) -> dict[str, float]:
        """Immutable view, safe to embed in replay Frames."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def reset_to_baseline(self, config: DrivesConfig) -> None:
        """Restore baselines for a new episode."""
        self.hunger = config.hunger.baseline
        self.aggression = config.aggression.baseline
        self.caution = config.caution.baseline
        self.curiosity = config.curiosity.baseline
        self.fatigue = config.fatigue.baseline
        self.social_bond = config.social_bond.baseline


# Map drive name → DriveAxisConfig attribute lookup
def _axis_config(config: DrivesConfig, name: str) -> DriveAxisConfig:
    return getattr(config, name)


class DriveSystem:
    """Per-tick drive update.

    Usage:
        drives = Drives()
        system = DriveSystem(DrivesConfig())
        system.reset(drives)  # call at start of episode
        # each tick:
        system.tick(drives, events, agent_moved=did_tom_move_this_tick)
    """

    DRIVE_NAMES: tuple[str, ...] = (
        "hunger", "aggression", "caution", "curiosity", "fatigue", "social_bond",
    )

    def __init__(self, config: DrivesConfig | None = None):
        self.config = config or DrivesConfig()

    def reset(self, drives: Drives) -> None:
        drives.reset_to_baseline(self.config)

    def tick(
        self,
        drives: Drives,
        events: list[Event] | tuple[Event, ...],
        agent_moved: bool = False,
    ) -> None:
        """Advance drives one tick given the events this tick.

        Order:
          1. Passive growth (hunger always, fatigue if moved)
          2. Decay toward baseline
          3. Event-driven deltas
          4. Clamp
        """
        # 1. Passive growth
        drives.hunger += self.config.hunger_per_tick
        if agent_moved:
            drives.fatigue += self.config.fatigue_per_tick_on_move

        # 2. Decay toward baseline
        for name in self.DRIVE_NAMES:
            axis = _axis_config(self.config, name)
            current = getattr(drives, name)
            delta = (axis.baseline - current) * axis.decay_rate
            setattr(drives, name, current + delta)

        # 3. Event-driven deltas (only for events relevant to Tom)
        # Build a fast lookup of (event_type → list of (drive_name, delta))
        # Cached on the system instance.
        if not hasattr(self, "_event_lookup"):
            lookup: dict[int, list[tuple[str, float]]] = {}
            for ev_type, drive_name, delta in self.config.event_deltas:
                lookup.setdefault(int(ev_type), []).append((drive_name, delta))
            self._event_lookup = lookup

        for ev in events:
            # Skip Jerry-side events for Tom's drives
            if ev.actor and ev.actor != "tom" and int(ev.type) not in {
                int(EventType.TOM_SAW_JERRY),  # Tom is the actor implicitly
                int(EventType.TOM_BUMPED_WALL),
                int(EventType.TOM_CAUGHT_JERRY),
            }:
                # NOISE_EMITTED by jerry IS relevant — Tom hears it.
                # Everything else from jerry is internal to jerry.
                if int(ev.type) != int(EventType.NOISE_EMITTED):
                    continue
            for drive_name, delta in self._event_lookup.get(int(ev.type), []):
                current = getattr(drives, drive_name)
                setattr(drives, drive_name, current + delta)

        # 4. Clamp
        for name in self.DRIVE_NAMES:
            axis = _axis_config(self.config, name)
            current = getattr(drives, name)
            if current < axis.floor:
                current = axis.floor
            elif current > axis.ceiling:
                current = axis.ceiling
            setattr(drives, name, current)
