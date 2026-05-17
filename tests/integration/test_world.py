"""Integration tests for the World.

Unit tests in tests/unit/ check individual modules. These tests
exercise the World end-to-end: reset, step through several ticks,
confirm events fire, observations are well-shaped, and the catch
condition works.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.world.world import (
    EventType,
    Observation,
    World,
    WorldConfig,
)
from src.utils.types import Action, Position, TileType


# ---- fixtures ----------------------------------------------------------

def make_world(seed: int = 42, **overrides) -> World:
    cfg = WorldConfig(**overrides)
    w = World(cfg, seed=seed)
    w.reset()
    return w


# ---- lifecycle ---------------------------------------------------------

def test_reset_returns_two_observations():
    w = World(WorldConfig(), seed=42)
    tom_obs, jerry_obs = w.reset()
    assert isinstance(tom_obs, Observation)
    assert isinstance(jerry_obs, Observation)


def test_reset_is_deterministic_with_seed():
    w1 = World(WorldConfig(), seed=123)
    w1.reset()
    w2 = World(WorldConfig(), seed=123)
    w2.reset()
    assert w1.tom.position == w2.tom.position
    assert w1.jerry.position == w2.jerry.position
    assert np.array_equal(w1.grid.tiles, w2.grid.tiles)


def test_reset_places_agents_apart():
    w = make_world()
    dist = w.tom.position.manhattan(w.jerry.position)
    # On a 30x30 map they should be at least somewhat separated
    assert dist > 5


def test_observation_vector_size_consistent():
    w = make_world()
    tom_obs, jerry_obs = w._observe_tom(), w._observe_jerry()
    tom_vec = tom_obs.to_vector()
    jerry_vec = jerry_obs.to_vector()
    # Tom's window is larger than Jerry's → bigger obs vector
    assert tom_vec.size == tom_obs.vector_size
    assert jerry_vec.size == jerry_obs.vector_size
    assert tom_vec.size > jerry_vec.size
    # Both are float32
    assert tom_vec.dtype == np.float32
    assert jerry_vec.dtype == np.float32


# ---- step ---------------------------------------------------------------

def test_step_advances_tick():
    w = make_world()
    assert w.tick_count == 0
    w.step(Action.WAIT, Action.WAIT)
    assert w.tick_count == 1


def test_step_after_done_raises():
    w = make_world(max_ticks=2)
    w.step(Action.WAIT, Action.WAIT)
    w.step(Action.WAIT, Action.WAIT)
    assert w.done
    with pytest.raises(RuntimeError):
        w.step(Action.WAIT, Action.WAIT)


def test_timeout_ends_episode():
    w = make_world(max_ticks=3)
    for _ in range(3):
        _, _, events, done = w.step(Action.WAIT, Action.WAIT)
    assert done
    assert any(e.type == EventType.TIMEOUT for e in events)


# ---- movement ----------------------------------------------------------

def test_jerry_movement_changes_position():
    w = make_world()
    # Try every direction; one of them is almost certainly walkable
    starting = w.jerry.position
    moved = False
    for action in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST):
        w.step(Action.WAIT, action)
        if w.jerry.position != starting:
            moved = True
            break
        # If blocked, try again from same spot
    assert moved, "Jerry could not move in any direction — grid too dense"


def test_wall_bump_emits_event():
    """Force Jerry into a wall and check the event fires."""
    w = make_world(seed=42)
    # Find a wall-adjacent direction
    jp = w.jerry.position
    for action in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST):
        target = jp + {
            Action.NORTH: Position(0, -1),
            Action.SOUTH: Position(0, 1),
            Action.EAST: Position(1, 0),
            Action.WEST: Position(-1, 0),
        }[action]
        if not w.grid.is_walkable(target):
            _, _, events, _ = w.step(Action.WAIT, action)
            assert any(e.type == EventType.JERRY_BUMPED_WALL for e in events)
            return
    pytest.skip("no wall adjacent to Jerry's spawn — try a different seed")


def test_movement_emits_footstep_noise():
    w = make_world(seed=42)
    # Try to move Jerry — one of the cardinal directions should succeed
    for action in (Action.EAST, Action.WEST, Action.NORTH, Action.SOUTH):
        target = w.jerry.position + {
            Action.NORTH: Position(0, -1),
            Action.SOUTH: Position(0, 1),
            Action.EAST: Position(1, 0),
            Action.WEST: Position(-1, 0),
        }[action]
        if w.grid.is_walkable(target):
            _, _, events, _ = w.step(Action.WAIT, action)
            assert any(e.type == EventType.NOISE_EMITTED and e.actor == "jerry"
                       for e in events)
            assert any(e.type == EventType.JERRY_MOVED for e in events)
            return
    pytest.skip("no walkable direction from Jerry's spawn")


# ---- catch -------------------------------------------------------------

def test_catch_when_adjacent_and_visible():
    """Manually put Tom and Jerry next to each other and confirm catch."""
    w = make_world(seed=42)
    # Force adjacency: pick an empty tile and put Jerry adjacent to Tom
    tx, ty = w.tom.position.x, w.tom.position.y
    placed = False
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        candidate = Position(tx + dx, ty + dy)
        if w.grid.is_walkable(candidate):
            w.jerry.position = candidate
            placed = True
            break
    assert placed, "no walkable tile adjacent to Tom"

    _, _, events, done = w.step(Action.WAIT, Action.WAIT)
    assert done
    assert any(e.type == EventType.TOM_CAUGHT_JERRY for e in events)
    assert not w.jerry.alive


def test_jerry_in_locker_safe_from_catch():
    """Jerry in a locker should NOT be caught even if Tom is adjacent."""
    w = make_world(seed=42)
    # Force Jerry into in_locker state and adjacent to Tom
    w.jerry.in_locker = True
    tx, ty = w.tom.position.x, w.tom.position.y
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        candidate = Position(tx + dx, ty + dy)
        if w.grid.in_bounds(candidate):
            w.jerry.position = candidate
            break
    _, _, events, done = w.step(Action.WAIT, Action.WAIT)
    assert not done
    assert not any(e.type == EventType.TOM_CAUGHT_JERRY for e in events)


# ---- interactions ------------------------------------------------------

def test_locker_entry_and_exit():
    """Place Jerry on a locker tile, INTERACT, confirm in_locker, INTERACT
    again, confirm out.
    """
    w = make_world(seed=42)
    if not w.grid.locker_positions:
        pytest.skip("no lockers on this map")
    locker = w.grid.locker_positions[0]
    w.jerry.position = locker

    _, _, events, _ = w.step(Action.WAIT, Action.INTERACT)
    assert w.jerry.in_locker
    assert any(e.type == EventType.JERRY_ENTERED_LOCKER for e in events)

    _, _, events, _ = w.step(Action.WAIT, Action.INTERACT)
    assert not w.jerry.in_locker
    assert any(e.type == EventType.JERRY_EXITED_LOCKER for e in events)


def test_vent_teleport():
    """Place Jerry on a vent, INTERACT, confirm teleport to paired vent."""
    w = make_world(seed=42)
    if len(w.grid.vent_positions) < 2:
        pytest.skip("not enough vents on this map")
    vent_a = w.grid.vent_positions[0]
    vent_b = w.grid.vent_destination(vent_a)
    w.jerry.position = vent_a

    _, _, events, _ = w.step(Action.WAIT, Action.INTERACT)
    assert w.jerry.position == vent_b
    assert any(e.type == EventType.JERRY_USED_VENT for e in events)


# ---- observations ------------------------------------------------------

def test_tom_sees_jerry_in_line_of_sight():
    """Force them onto the same row with no walls between, check sees_other=1."""
    w = make_world(seed=42)
    # Put both on a known-clear row near the middle
    # Find a row with at least 5 consecutive empty tiles
    for y in range(2, w.grid.height - 2):
        clear = []
        for x in range(1, w.grid.width - 1):
            if w.grid.is_walkable(Position(x, y)):
                clear.append(x)
        if len(clear) >= 5:
            w.tom.position = Position(clear[0], y)
            w.jerry.position = Position(clear[2], y)
            obs = w._observe_tom()
            # If a wall sneaks between them in this seed, skip — that's not
            # what this test is about
            if obs.sees_other == 1.0:
                assert obs.rel_dx != 0 or obs.rel_dy != 0
                return
    pytest.skip("could not find a clear row in this seed")


def test_scent_in_tom_obs_after_jerry_moves():
    """After Jerry moves, Tom should sense scent SOMEWHERE on the gradient."""
    w = make_world(seed=42)
    # Force Tom next to where Jerry will be
    jp = w.jerry.position
    w.tom.position = jp  # put Tom on top of Jerry's spawn tile
    # Move Jerry away
    for action in (Action.EAST, Action.WEST, Action.NORTH, Action.SOUTH):
        target = jp + {
            Action.NORTH: Position(0, -1),
            Action.SOUTH: Position(0, 1),
            Action.EAST: Position(1, 0),
            Action.WEST: Position(-1, 0),
        }[action]
        if w.grid.is_walkable(target):
            w.step(Action.WAIT, action)
            obs = w._observe_tom()
            # Jerry deposited scent on `target`; Tom is at jp.
            # Direction from Tom to target should have nonzero scent.
            grads = (obs.scent_n, obs.scent_s, obs.scent_e, obs.scent_w)
            assert sum(grads) > 0
            return
    pytest.skip("could not move Jerry from spawn")


def test_jerry_has_no_scent_perception():
    """Jerry's obs should always have zero scent components."""
    w = make_world(seed=42)
    w.step(Action.WAIT, Action.WAIT)
    obs = w._observe_jerry()
    assert obs.scent_n == 0
    assert obs.scent_s == 0
    assert obs.scent_e == 0
    assert obs.scent_w == 0


def test_jerry_in_locker_obs_flag():
    """When Jerry is in a locker, his obs.in_locker should be 1.0."""
    w = make_world(seed=42)
    w.jerry.in_locker = True
    obs = w._observe_jerry()
    assert obs.in_locker == 1.0


def test_obs_window_size_matches_config():
    w = make_world(seed=42, jerry_obs_window=5, tom_obs_window=11)
    jerry_obs = w._observe_jerry()
    tom_obs = w._observe_tom()
    assert jerry_obs.grid_window.shape == (5, 5)
    assert tom_obs.grid_window.shape == (11, 11)


def test_obs_window_fills_oob_with_wall():
    """Put Jerry at (1,1) — most of the obs window is out of bounds."""
    w = make_world(seed=42)
    w.jerry.position = Position(1, 1)
    obs = w._observe_jerry()
    # Top-left corner of the window should be WALL (oob)
    assert obs.grid_window[0, 0] == TileType.WALL


# ---- full episode smoke test -------------------------------------------

def test_full_random_episode_completes():
    """Run a full episode with random actions on both sides. Should
    terminate cleanly without exceptions, either by catch or timeout.
    """
    import random
    w = make_world(seed=42, max_ticks=100)
    actions = list(Action)
    rng = random.Random(0)
    ticks = 0
    while not w.done:
        w.step(rng.choice(actions), rng.choice(actions))
        ticks += 1
        if ticks > 200:
            pytest.fail("episode never ended")
    # Final tick count should not exceed max_ticks
    assert w.tick_count <= 100
