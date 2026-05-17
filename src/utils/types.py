"""Shared types and protocols for Tom_and_Jerry.

This module is the foundation. Every other module imports from here.
Keep it small. Keep it stable. Breaking changes here ripple everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable


class Action(IntEnum):
    """Discrete action space for both Tom and Jerry.

    IntEnum so values are stable for serialization, replays, and
    Gymnasium's Discrete space. Order matters — do NOT reorder without
    invalidating all saved replays and training checkpoints.
    """
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    WAIT = 4
    INTERACT = 5  # context-dependent: enter locker, use vent, throw item


ACTION_COUNT = len(Action)


class TileType(IntEnum):
    """What occupies a grid cell.

    EMPTY tiles are traversable. WALL tiles block movement, sight, and sound.
    VENT and LOCKER are interactive features.
    """
    EMPTY = 0
    WALL = 1
    VENT = 2
    LOCKER = 3


@dataclass(frozen=True, slots=True)
class Position:
    """Integer grid coordinate. Frozen so it's hashable and safe to use
    as a dict key (we will need this for memory keys).
    """
    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def manhattan(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


# Direction deltas, keyed by Action. Used by the world to resolve moves.
ACTION_DELTAS: dict[Action, Position] = {
    Action.NORTH: Position(0, -1),
    Action.SOUTH: Position(0, 1),
    Action.EAST: Position(1, 0),
    Action.WEST: Position(-1, 0),
    Action.WAIT: Position(0, 0),
    Action.INTERACT: Position(0, 0),
}


@runtime_checkable
class ActionSpace(Protocol):
    """Abstract action space. Phase 1 uses DiscreteActionSpace below.

    Defined as a Protocol so a continuous variant can be swapped in
    later (3D port) without rewriting agents. See ADR-007.
    """
    @property
    def n(self) -> int:
        """Cardinality (for discrete) or dimensionality (for continuous)."""
        ...

    def sample(self) -> int:
        """Return a random valid action."""
        ...

    def contains(self, action: int) -> bool:
        """Return True if `action` is valid in this space."""
        ...


@dataclass(frozen=True, slots=True)
class DiscreteActionSpace:
    """Concrete discrete action space backed by the Action IntEnum."""
    _n: int = ACTION_COUNT

    @property
    def n(self) -> int:
        return self._n

    def sample(self) -> int:
        # Lazy import so utils stays dependency-free at module load
        import random
        return random.randint(0, self._n - 1)

    def contains(self, action: int) -> bool:
        return 0 <= action < self._n


@dataclass(slots=True)
class AgentState:
    """Per-agent state tracked by the world.

    Both Tom and Jerry have one of these. The world owns it; agents
    receive observations derived from it.
    """
    position: Position
    facing: Action = Action.NORTH      # last movement direction, for FOV
    in_locker: bool = False
    alive: bool = True
    # Episode-level counters (handy for reward shaping later)
    sound_events_emitted: int = 0
    distance_traveled: int = 0
