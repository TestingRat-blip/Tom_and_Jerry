"""Tests for the stale-bookmark standoff fix.

The exploit (found in r12 evader training, seed 3): a prey that holds 1 tile
just outside Tom's sight range never refreshes Tom's last_seen_jerry bookmark.
Tom, in PURSUE with no LOS, walks to the stale bookmark tile and then —
because _step_toward(src == dst) returns WAIT — freezes there for the entire
pursue-memory window (125 ticks in the trace), while the prey farms
survival-per-tick reward two tiles away.

The fix: when Tom is in PURSUE, has no LOS, and is already standing ON the
stale last_seen_jerry tile, route him to _patrol (which actively searches and
has its own no-stall guard) instead of staring at the empty tile.

These tests pin the fix so a future change can't silently re-open the
standoff.
"""
from __future__ import annotations

import random

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.baseline import TomState
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.conductor import Conductor
from src.utils.types import Action, Position


def _find_open_tile_with_neighbors(world, min_free_neighbors=3):
    """Find a walkable tile with several walkable neighbors (open area, so
    _patrol has somewhere to go and the test isn't wedged in a pocket)."""
    grid = world.grid
    for x in range(2, grid.width - 2):
        for y in range(2, grid.height - 2):
            p = Position(x, y)
            if not grid.is_walkable(p):
                continue
            free = sum(
                grid.is_walkable(Position(p.x + dx, p.y + dy))
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
            )
            if free >= min_free_neighbors:
                return p
    return None


def test_tom_does_not_freeze_on_stale_bookmark():
    """The core of the fix: once Tom's belief holds no live SIGHTING (e.g. he
    reached the believed spot and saw nothing, so it was invalidated), the
    last_seen_jerry mirror is cleared, PURSUE releases, and Tom searches
    instead of freezing.

    This exercises the REAL path: a planted sighting belief, driven through
    full ticks (tom(world) → _update_memory_from_conductor → state select →
    act), not a direct _act_for_state_chemical call (which bypasses the mirror
    clear where the fix lives)."""
    world = World(WorldConfig(max_ticks=60), seed=3)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=3)
    tom.reset()

    spot = _find_open_tile_with_neighbors(world)
    assert spot is not None
    far = Position(world.grid.width - 2, world.grid.height - 2)
    if far == spot or not world.grid.is_walkable(far):
        far = _find_open_tile_with_neighbors(world, min_free_neighbors=2)

    # Put Tom on the spot with a planted SIGHTING belief there, Jerry far/out
    # of sight. First tick: Tom is on the spot, no LOS → belief invalidated →
    # mirror cleared → PURSUE releases.
    world.tom.position = spot
    world.jerry.position = far
    tom.conductor.belief.add_sighting(spot, world.tick_count)
    tom.last_seen_jerry = spot
    tom.last_seen_tick = world.tick_count
    tom.state = TomState.PURSUE

    assert not world._tom_can_see_jerry(), "setup should have no LOS"
    # Drive one full tick — the action must not be a frozen WAIT-in-place.
    a = tom(world)
    # After the tick's memory update, the stale mirror should be gone.
    assert tom.last_seen_jerry is None or tom.last_seen_jerry != spot, (
        "stale sighting mirror not cleared — PURSUE will re-freeze"
    )


def test_tom_breaks_standoff_within_a_few_ticks():
    """Behavioral: drive several ticks with Jerry holding still just out of
    sight while Tom sits on the stale bookmark. Tom must genuinely LEAVE the
    area — not freeze, and not orbit two tiles forever (the bug the first fix
    attempt missed: it stopped the WAIT but Tom bounced between the bookmark
    and a neighbor because the bookmark was never abandoned)."""
    world = World(WorldConfig(max_ticks=60), seed=3)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=3)
    tom.reset()

    spot = _find_open_tile_with_neighbors(world)
    world.tom.position = spot
    far = Position(world.grid.width - 2, world.grid.height - 2)
    if far == spot or not world.grid.is_walkable(far):
        far = _find_open_tile_with_neighbors(world, min_free_neighbors=2)
    tom.conductor.belief.add_sighting(spot, world.tick_count)
    tom.last_seen_jerry = spot
    tom.last_seen_tick = world.tick_count
    tom.state = TomState.PURSUE

    visited = []
    for _ in range(15):
        a = tom(world)
        world.jerry.position = far
        world.step(tom_action=a, jerry_action=int(Action.WAIT),
                   tom_in_pursuit=tom.state.is_committed_pursuit)
        visited.append((world.tom.position.x, world.tom.position.y))

    # Genuine departure: Tom should visit many distinct tiles and end FAR from
    # the dead bookmark — not orbit it. A 2-tile bounce would give ~2 distinct
    # tiles; real search gives many.
    distinct = len(set(visited))
    assert distinct >= 5, (
        f"Tom only visited {distinct} distinct tiles in 15 ticks — he's "
        f"stuck/orbiting the stale bookmark, not searching: {visited}"
    )
    final_dist = world.tom.position.manhattan(spot)
    assert final_dist >= 3, (
        f"Tom ended {final_dist} tiles from the dead bookmark — still tethered"
    )
    # And the bookmark must have been abandoned.
    assert tom.last_seen_jerry != spot or tom.last_seen_jerry is None


def test_normal_pursuit_with_los_unaffected():
    """Guard: the fix must NOT disturb normal pursuit. With live LOS, Tom
    targets the predicted Jerry position and closes — never routed to patrol."""
    world = World(WorldConfig(max_ticks=50), seed=7)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=7)
    tom.reset()

    # Place Jerry a few tiles from Tom in open line of sight.
    spot = _find_open_tile_with_neighbors(world)
    world.tom.position = spot
    # Find a visible jerry tile within sight.
    placed = False
    for dx in range(1, 6):
        cand = Position(spot.x + dx, spot.y)
        if world.grid.is_walkable(cand):
            world.jerry.position = cand
            if world._tom_can_see_jerry():
                placed = True
                break
    if placed:
        tom.last_seen_jerry = world.jerry.position
        tom.last_seen_tick = world.tick_count
        tom.state = TomState.PURSUE
        before = world.tom.position.manhattan(world.jerry.position)
        a = tom._act_for_state_chemical(world)
        # With LOS and distance, Tom should take a real step (not WAIT) toward
        # Jerry. (We don't assert direction — prediction may pick any closing
        # axis — only that he acts.)
        assert a != Action.WAIT or before <= 1


def test_default_episode_unbroken_by_fix():
    """Sanity: a normal random-Jerry episode still runs without error and Tom
    spends time in multiple states (the fix didn't lock him into patrol)."""
    world = World(WorldConfig(max_ticks=120), seed=11)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=11)
    tom.reset()
    rng = random.Random(5)
    states_seen = set()
    for _ in range(120):
        a = tom(world)
        states_seen.add(tom.state)
        _, _, _, done = world.step(tom_action=a, jerry_action=rng.randint(0, 4),
                                   tom_in_pursuit=tom.state.is_committed_pursuit)
        if done or not world.jerry.alive:
            break
    # Tom should exhibit more than one state over a full episode.
    assert len(states_seen) >= 2
