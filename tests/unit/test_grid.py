"""Unit tests for the Grid map."""
from __future__ import annotations

import numpy as np
import pytest

from src.env.world.grid import Grid
from src.utils.types import Position, TileType


def test_generate_produces_bordered_grid():
    g = Grid.generate(width=20, height=15, seed=42)
    # All border tiles are walls
    assert (g.tiles[0, :] == TileType.WALL).all()
    assert (g.tiles[-1, :] == TileType.WALL).all()
    assert (g.tiles[:, 0] == TileType.WALL).all()
    assert (g.tiles[:, -1] == TileType.WALL).all()


def test_generate_is_deterministic_with_seed():
    g1 = Grid.generate(seed=123)
    g2 = Grid.generate(seed=123)
    assert np.array_equal(g1.tiles, g2.tiles)
    assert g1.vent_links == g2.vent_links


def test_in_bounds():
    g = Grid.generate(width=10, height=10, seed=0)
    assert g.in_bounds(Position(0, 0))
    assert g.in_bounds(Position(9, 9))
    assert not g.in_bounds(Position(-1, 0))
    assert not g.in_bounds(Position(10, 5))


def test_tile_at_out_of_bounds_is_wall():
    g = Grid.generate(seed=0)
    assert g.tile_at(Position(-1, -1)) == TileType.WALL
    assert g.tile_at(Position(1000, 1000)) == TileType.WALL


def test_is_walkable():
    g = Grid.generate(seed=0)
    # Border is always wall
    assert not g.is_walkable(Position(0, 0))
    # Find an empty tile and confirm it's walkable
    empty = g.random_empty_position()
    assert g.is_walkable(empty)


def test_vents_come_in_pairs():
    g = Grid.generate(n_vent_pairs=3, seed=7)
    # vent_links must be symmetric: if A→B then B→A
    for a, b in g.vent_links.items():
        assert g.vent_links[b] == a
    # Tile type at each vent position is VENT
    for p in g.vent_positions:
        assert g.tile_at(p) == TileType.VENT


def test_vent_destination():
    g = Grid.generate(n_vent_pairs=2, seed=11)
    for vent, paired in g.vent_links.items():
        assert g.vent_destination(vent) == paired
    # Non-vent position returns None
    assert g.vent_destination(Position(0, 0)) is None


def test_locker_placement():
    g = Grid.generate(n_lockers=5, seed=3)
    # We asked for 5 — generator may place fewer if space is tight, but never more
    assert len(g.locker_positions) <= 5
    for p in g.locker_positions:
        assert g.tile_at(p) == TileType.LOCKER


def test_random_empty_position_returns_empty():
    g = Grid.generate(seed=0)
    for _ in range(20):
        p = g.random_empty_position()
        assert g.tile_at(p) == TileType.EMPTY


def test_blocks_sight_only_for_walls():
    g = Grid.generate(seed=0)
    # Walls block sight
    assert g.blocks_sight(Position(0, 0))
    # Vents and lockers do not
    if g.vent_positions:
        assert not g.blocks_sight(g.vent_positions[0])
    if g.locker_positions:
        assert not g.blocks_sight(g.locker_positions[0])


def test_position_arithmetic():
    a = Position(3, 4)
    b = Position(1, 2)
    assert a + b == Position(4, 6)
    assert a.manhattan(b) == 4
    # Frozen — hashable
    assert hash(a) == hash(Position(3, 4))


def test_action_deltas_resolve_correctly():
    from src.utils.types import Action, ACTION_DELTAS
    start = Position(5, 5)
    assert start + ACTION_DELTAS[Action.NORTH] == Position(5, 4)
    assert start + ACTION_DELTAS[Action.SOUTH] == Position(5, 6)
    assert start + ACTION_DELTAS[Action.EAST] == Position(6, 5)
    assert start + ACTION_DELTAS[Action.WEST] == Position(4, 5)
    assert start + ACTION_DELTAS[Action.WAIT] == start


def test_discrete_action_space():
    from src.utils.types import DiscreteActionSpace, ACTION_COUNT
    space = DiscreteActionSpace()
    assert space.n == ACTION_COUNT
    assert space.contains(0)
    assert space.contains(ACTION_COUNT - 1)
    assert not space.contains(-1)
    assert not space.contains(ACTION_COUNT)
    # Sample stays in range
    for _ in range(50):
        assert space.contains(space.sample())


# ---- reachability (spawn-connectivity bug fix) ------------------------

def _split_grid():
    """A 5x3 grid split by a vertical wall at x=2 into two disconnected
    rooms (left x<2, right x>2)."""
    tiles = np.zeros((3, 5), dtype=np.int8)
    tiles[:, 2] = int(TileType.WALL)
    return Grid(width=5, height=3, tiles=tiles)


def test_is_reachable_within_connected_region():
    g = _split_grid()
    # Two tiles in the same (left) room are reachable.
    assert g.is_reachable(Position(1, 0), Position(1, 2))


def test_is_reachable_false_across_partition():
    g = _split_grid()
    # Left room to right room is blocked by the wall column.
    assert not g.is_reachable(Position(1, 1), Position(3, 1))


def test_is_reachable_same_tile():
    g = _split_grid()
    assert g.is_reachable(Position(1, 1), Position(1, 1))


def test_is_reachable_false_for_wall_endpoint():
    g = _split_grid()
    # An endpoint that isn't walkable is never reachable.
    assert not g.is_reachable(Position(1, 1), Position(2, 1))  # (2,1) is wall


def test_generated_spawns_are_always_connected():
    """Regression: _spawn_agents must place Tom and Jerry on mutually
    reachable tiles. Pre-fix, the generator could place Jerry in a pocket
    Tom could not path to (seed 49 was one such case), inflating survival."""
    from src.env.world.world import World, WorldConfig
    for seed in range(60):
        w = World(WorldConfig(), seed=seed)
        w.reset()
        assert w.grid.is_reachable(w.tom.position, w.jerry.position), (
            f"seed {seed}: Tom {w.tom.position} cannot reach Jerry "
            f"{w.jerry.position}"
        )
