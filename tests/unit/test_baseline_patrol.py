"""Regression tests for the Defect-A patrol WAIT-stall fix.

Bug (found via seed-100 trace, Round 5): the map generator can produce
walkable-but-unreachable tiles (isolated pockets). When _patrol picked one
as a target, BFS could not route to it, _step_toward returned WAIT, and Tom
froze in place — often wedged in a dead-end pocket — until the staleness
timer fired ~40 ticks later. A hunter standing still is never correct.

Fix: (1) patrol picks a BFS-reachable target; (2) if a step is still WAIT,
retarget immediately rather than freezing.
"""
from __future__ import annotations

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.utils.types import Action, Position, TileType


def _find_dead_end_pocket(world: World) -> Position | None:
    """Find a walkable tile with exactly one walkable cardinal neighbor —
    a dead-end pocket of the kind that triggered the stall."""
    from src.hunter.agent.behavior.baseline import ACTION_DELTAS
    for x in range(world.grid.width):
        for y in range(world.grid.height):
            p = Position(x, y)
            if not world.grid.is_walkable(p):
                continue
            open_n = sum(
                1 for a in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)
                if world.grid.is_walkable(p + ACTION_DELTAS[a])
            )
            if open_n == 1:
                return p
    return None


def _carve_dead_end_pocket(world: World) -> Position:
    """Construct a synthetic dead-end pocket for the regression test.

    As of the map-generator dead-end cleanup pass, generated maps contain NO
    dead-end pockets (open_n <= 1 tiles) — the very geometry that caused the
    Defect-A stall is eliminated at the source. But the patrol logic must
    still be robust to a dead-end *should one ever arise* (e.g. a future
    mechanic that closes tiles dynamically), so we wall a tile in by hand to
    recreate the scenario rather than relying on the generator to produce it.
    """
    g = world.grid
    # Find an interior walkable tile with >=2 open neighbors, then wall off
    # all but one to make it a one-exit pocket.
    from src.hunter.agent.behavior.baseline import ACTION_DELTAS
    dirs = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
    for x in range(2, g.width - 2):
        for y in range(2, g.height - 2):
            p = Position(x, y)
            if not g.is_walkable(p):
                continue
            open_dirs = [a for a in dirs if g.is_walkable(p + ACTION_DELTAS[a])]
            if len(open_dirs) >= 2:
                # Keep the first open direction; wall the rest.
                for a in open_dirs[1:]:
                    np_ = p + ACTION_DELTAS[a]
                    g.tiles[np_.y, np_.x] = TileType.WALL
                return p
    raise AssertionError("could not carve a synthetic pocket")


def test_patrol_never_stalls_in_dead_end():
    """From a dead-end pocket, patrol must produce a MOVE (out of the
    pocket), never WAIT, regardless of the current target.

    The generator no longer produces dead-ends (cleanup pass), so we carve a
    synthetic one to exercise the patrol-stall guard directly.
    """
    world = World(WorldConfig(max_ticks=300), seed=100)
    world.reset()
    pocket = _find_dead_end_pocket(world)
    if pocket is None:
        pocket = _carve_dead_end_pocket(world)

    tom = ScriptedTom(seed=100)
    tom.reset()
    world.tom.position = pocket
    # Point the patrol target at the pocket's walled neighbor region — the
    # kind of walkable-but-unreachable tile that caused the stall.
    tom.patrol_target = Position(min(world.grid.width - 1, pocket.x + 1), pocket.y + 1)
    tom.patrol_target_set_tick = 0

    action = Action(int(tom._patrol(world)))
    assert action != Action.WAIT, "patrol must not stall in a dead-end pocket"


def test_patrol_target_is_reachable():
    """_choose_reachable_patrol_target must pick a BFS-reachable target."""
    world = World(WorldConfig(max_ticks=300), seed=100)
    world.reset()
    tom = ScriptedTom(seed=100)
    tom.reset()
    for _ in range(20):
        tom._choose_reachable_patrol_target(world)
        # A reachable target has a defined first BFS step from Tom's position
        # (or is Tom's own tile, which the chooser avoids).
        step = tom._bfs_first_step(world.tom.position, tom.patrol_target, world)
        assert step is not None, (
            f"patrol target {tom.patrol_target} unreachable from "
            f"{world.tom.position}"
        )


def test_no_long_patrol_wait_run_against_silent_camper():
    """End-to-end: against a silent WAIT camper on the seed that exhibited
    the stall, Tom should never rack up a long run of consecutive
    PATROL-state WAIT actions."""
    world = World(WorldConfig(max_ticks=300), seed=100)
    world.reset()
    tom = ScriptedTom(seed=100)
    tom.reset()

    max_wait_run = 0
    cur = 0
    for _ in range(300):
        a = Action(int(tom(world)))
        if a == Action.WAIT and tom.state == tom.state.PATROL:
            cur += 1
            max_wait_run = max(max_wait_run, cur)
        else:
            cur = 0
        world.step(tom_action=int(a), jerry_action=int(Action.WAIT))
        if not world.jerry.alive:
            break

    # Pre-fix this hit ~30+. Allow a tiny margin for legitimate single-tick
    # waits, but nothing resembling the old stall.
    assert max_wait_run <= 3, (
        f"Tom stalled in PATROL-WAIT for {max_wait_run} consecutive ticks"
    )
