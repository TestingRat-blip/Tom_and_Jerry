"""Unit tests for sound, LOS, and scent sensors."""
from __future__ import annotations

import numpy as np
import pytest

from src.env.sensors.los import tiles_visible_from, visible_from
from src.env.sensors.scent import ScentField
from src.env.sensors.sound import SoundEvent, SoundField
from src.env.world.grid import Grid
from src.utils.types import Action, Position, TileType


# ---- helpers -----------------------------------------------------------

def make_empty_grid(width: int = 10, height: int = 10) -> Grid:
    """Build an open grid (only border walls) for predictable sensor testing."""
    g = Grid.generate(width=width, height=height, wall_density=0.0,
                      n_vent_pairs=0, n_lockers=0, seed=0)
    return g


def make_walled_grid() -> Grid:
    """A 10x10 grid with a vertical wall down the middle (column 5).
    Used to verify sensors respect occlusion.
    """
    g = make_empty_grid(10, 10)
    # Drop a vertical wall down the middle, leaving a single gap at y=5
    for y in range(1, 9):
        if y == 5:
            continue
        g.tiles[y, 5] = TileType.WALL
    return g


# ---- SoundField --------------------------------------------------------

def test_sound_at_source_is_loudest():
    g = make_empty_grid()
    sf = SoundField(g)
    sf.emit(SoundEvent(Position(5, 5), intensity=1.0))
    near = sf.heard_at(Position(5, 5))
    far = sf.heard_at(Position(5, 8))
    assert near > far > 0


def test_sound_decays_with_distance():
    g = make_empty_grid()
    sf = SoundField(g)
    sf.emit(SoundEvent(Position(5, 5), intensity=1.0))
    d1 = sf.heard_at(Position(5, 6))
    d2 = sf.heard_at(Position(5, 7))
    d3 = sf.heard_at(Position(5, 8))
    assert d1 > d2 > d3


def test_sound_blocked_by_walls():
    """A sound on one side of a wall should be quieter on the other side
    than the same Euclidean distance in an open grid.
    """
    walled = make_walled_grid()
    open_grid = make_empty_grid()

    sf_walled = SoundField(walled)
    sf_walled.emit(SoundEvent(Position(3, 3), intensity=1.0))
    heard_through_wall = sf_walled.heard_at(Position(7, 3))

    sf_open = SoundField(open_grid)
    sf_open.emit(SoundEvent(Position(3, 3), intensity=1.0))
    heard_in_open = sf_open.heard_at(Position(7, 3))

    # Sound has to route through the gap at y=5 — much longer path
    assert heard_through_wall < heard_in_open


def test_sound_clear_resets():
    g = make_empty_grid()
    sf = SoundField(g)
    sf.emit(SoundEvent(Position(5, 5), intensity=1.0))
    assert sf.heard_at(Position(5, 5)) > 0
    sf.clear()
    assert sf.heard_at(Position(5, 5)) == 0


def test_sound_below_threshold_returns_zero():
    g = make_empty_grid(width=30, height=30)
    sf = SoundField(g)
    sf.emit(SoundEvent(Position(5, 5), intensity=0.5))
    # Very far away — should drop below threshold
    far = sf.heard_at(Position(25, 25), threshold=0.05)
    assert far == 0.0


def test_directional_hearing_points_correctly():
    """A sound source east of the listener should register as 'E'."""
    g = make_empty_grid(width=20, height=20)
    sf = SoundField(g)
    sf.emit(SoundEvent(Position(15, 10), intensity=3.0))
    directions = sf.directional_hearing(Position(10, 10))
    assert directions["E"] > directions["W"]
    assert directions["E"] > directions["N"]
    assert directions["E"] > directions["S"]


def test_directional_hearing_handles_north_source():
    g = make_empty_grid(width=20, height=20)
    sf = SoundField(g)
    sf.emit(SoundEvent(Position(10, 5), intensity=3.0))  # north (lower y)
    directions = sf.directional_hearing(Position(10, 10))
    assert directions["N"] > directions["S"]


def test_attenuation_parameter_changes_decay():
    """Higher attenuation = sound dies faster."""
    g = make_empty_grid(width=20, height=20)
    sf_slow = SoundField(g, attenuation=0.5)   # Tom's ears
    sf_fast = SoundField(g, attenuation=2.0)   # Jerry's ears
    sf_slow.emit(SoundEvent(Position(5, 5), intensity=1.0))
    sf_fast.emit(SoundEvent(Position(5, 5), intensity=1.0))
    assert sf_slow.heard_at(Position(10, 5)) > sf_fast.heard_at(Position(10, 5))


# ---- Line of sight -----------------------------------------------------

def test_los_open_space():
    g = make_empty_grid()
    assert visible_from(g, Position(2, 2), Position(7, 7))


def test_los_self():
    g = make_empty_grid()
    assert visible_from(g, Position(5, 5), Position(5, 5))


def test_los_blocked_by_wall():
    g = make_walled_grid()
    # Across the wall at y=3 (no gap there)
    assert not visible_from(g, Position(3, 3), Position(7, 3))


def test_los_through_gap_in_wall():
    g = make_walled_grid()
    # Through the gap at y=5
    assert visible_from(g, Position(3, 5), Position(7, 5))


def test_los_max_range():
    g = make_empty_grid(width=20, height=20)
    a = Position(2, 2)
    b = Position(15, 15)
    assert not visible_from(g, a, b, max_range=5)
    assert visible_from(g, a, b, max_range=100)


def test_tiles_visible_360():
    g = make_empty_grid()
    visible = tiles_visible_from(g, Position(5, 5), max_range=3)
    # Should see tiles in all four cardinal directions
    assert Position(5, 4) in visible  # N
    assert Position(5, 6) in visible  # S
    assert Position(6, 5) in visible  # E
    assert Position(4, 5) in visible  # W


def test_tiles_visible_with_cone():
    """An agent facing NORTH with a 90° cone should NOT see tiles directly
    behind it.
    """
    g = make_empty_grid()
    visible = tiles_visible_from(
        g, Position(5, 5),
        max_range=3,
        facing=Action.NORTH,
        fov_degrees=90,
    )
    assert Position(5, 3) in visible  # straight ahead
    assert Position(5, 7) not in visible  # directly behind


def test_tiles_visible_respects_walls():
    g = make_walled_grid()
    visible = tiles_visible_from(g, Position(3, 3), max_range=10)
    # Cannot see through the wall at column 5 (gap is at y=5).
    # (7, 3) is directly across the wall — line of sight passes through
    # the wall at (5, 3), so it's blocked.
    assert Position(7, 3) not in visible
    # (7, 2) is also occluded — line from (3,3)→(7,2) crosses (5, 2) which is wall.
    assert Position(7, 2) not in visible
    # Sanity: (5, 5) is the gap, should be visible from (3, 3)
    assert Position(5, 5) in visible


def test_tiles_visible_can_see_through_gap():
    """The flip side — make sure the gap actually works for distant targets."""
    g = make_walled_grid()
    visible = tiles_visible_from(g, Position(3, 5), max_range=10)
    # On the same row as the gap, looking east — wall is at (5, *) except y=5
    assert Position(7, 5) in visible


# ---- Scent -------------------------------------------------------------

def test_scent_deposit_and_read():
    g = make_empty_grid()
    sc = ScentField(g, deposit_amount=1.0)
    sc.deposit_at(Position(5, 5))
    assert sc.strength_at(Position(5, 5)) == pytest.approx(1.0)


def test_scent_decay():
    g = make_empty_grid()
    sc = ScentField(g, decay_per_tick=0.9, deposit_amount=1.0)
    sc.deposit_at(Position(5, 5))
    initial = sc.strength_at(Position(5, 5))
    sc.tick()
    after_one = sc.strength_at(Position(5, 5))
    sc.tick()
    after_two = sc.strength_at(Position(5, 5))
    assert initial > after_one > after_two


def test_scent_floor_snaps_to_zero():
    g = make_empty_grid()
    sc = ScentField(g, decay_per_tick=0.1, deposit_amount=0.05, floor=0.01)
    sc.deposit_at(Position(5, 5))
    sc.tick()  # 0.05 * 0.1 = 0.005, below floor
    assert sc.strength_at(Position(5, 5)) == 0.0


def test_scent_caps_at_one():
    g = make_empty_grid()
    sc = ScentField(g, deposit_amount=1.0)
    for _ in range(10):
        sc.deposit_at(Position(5, 5))
    assert sc.strength_at(Position(5, 5)) == pytest.approx(1.0)


def test_scent_gradient_points_to_strongest():
    g = make_empty_grid()
    sc = ScentField(g, deposit_amount=1.0)
    # Lay a trail east of (5, 5)
    sc.deposit_at(Position(6, 5))
    sc.deposit_at(Position(7, 5))
    grad = sc.gradient_at(Position(5, 5))
    assert grad["E"] > 0
    assert grad["W"] == 0
    assert grad["E"] > grad["N"]


def test_scent_not_deposited_on_walls():
    g = make_walled_grid()
    sc = ScentField(g, deposit_amount=1.0)
    # Wall at (5, 3)
    sc.deposit_at(Position(5, 3))
    assert sc.strength_at(Position(5, 3)) == 0.0


def test_scent_clear():
    g = make_empty_grid()
    sc = ScentField(g, deposit_amount=1.0)
    sc.deposit_at(Position(5, 5))
    sc.clear()
    assert sc.strength_at(Position(5, 5)) == 0.0


def test_scent_oob_returns_zero():
    g = make_empty_grid()
    sc = ScentField(g)
    assert sc.strength_at(Position(-1, -1)) == 0.0
    assert sc.strength_at(Position(1000, 1000)) == 0.0
