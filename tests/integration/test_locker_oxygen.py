"""Tests for the locker oxygen / cooldown mechanic (cost-of-staying).

See docs/DESIGN_LOCKER_OXYGEN.md. Covers:
  - OFF by default: lockers behave as before (oxygen never depletes, no burst,
    no cooldown) — the regression guard.
  - ON: oxygen depletes per in-locker tick; bursts (forced exit + loud noise)
    at zero; the burst locker goes on cooldown; cooldown blocks re-entry and
    lifts after the window.
"""
from __future__ import annotations

from src.env.world.grid import TileType
from src.env.world.world import Event, EventType, World, WorldConfig
from src.utils.types import Action, Position


def _first_locker(world: World) -> Position:
    for x in range(world.grid.width):
        for y in range(world.grid.height):
            if world.grid.tile_at(Position(x, y)) == TileType.LOCKER:
                return Position(x, y)
    raise AssertionError("no locker on map")


def _enter_locker(world: World, pos: Position) -> None:
    world.jerry.position = pos
    world._resolve_action(world.jerry, Action.INTERACT, "jerry")


# ---- OFF by default: regression guard ---------------------------------

def test_oxygen_disabled_by_default():
    assert WorldConfig().locker_oxygen_enabled is False


def test_disabled_locker_never_depletes_or_bursts():
    """With the mechanic off, a hidden Jerry stays hidden indefinitely and
    oxygen state is never engaged."""
    world = World(WorldConfig(max_ticks=300), seed=42)  # default: disabled
    world.reset()
    pos = _first_locker(world)
    _enter_locker(world, pos)
    assert world.jerry.in_locker is True
    assert world._jerry_oxygen is None  # never set when disabled
    # Tick many times; Jerry must remain hidden (no burst).
    for _ in range(200):
        if world.done:
            break
        world.step(tom_action=Action.WAIT, jerry_action=Action.WAIT)
    assert world.jerry.in_locker is True, "disabled mechanic must not eject Jerry"
    assert world._locker_cooldowns == {}, "no cooldowns when disabled"


# ---- ON: depletion + burst --------------------------------------------

def test_oxygen_depletes_while_hiding():
    cfg = WorldConfig(locker_oxygen_enabled=True, locker_oxygen_capacity=10,
                      max_ticks=300)
    world = World(cfg, seed=42)
    world.reset()
    pos = _first_locker(world)
    _enter_locker(world, pos)
    assert world._jerry_oxygen == 10
    # One manual oxygen tick drops it by one.
    world.tick_count += 1
    world._tick_locker_oxygen()
    assert world._jerry_oxygen == 9


def test_burst_at_zero_forces_exit_and_emits_noise():
    cfg = WorldConfig(locker_oxygen_enabled=True, locker_oxygen_capacity=5,
                      locker_cooldown_ticks=20, locker_burst_intensity=3.0,
                      max_ticks=300)
    world = World(cfg, seed=42)
    world.reset()
    pos = _first_locker(world)
    _enter_locker(world, pos)

    burst = False
    for _ in range(6):
        world._events_this_tick = []
        world.sound.clear()
        world.tick_count += 1
        world._tick_locker_oxygen()
        if not world.jerry.in_locker:
            burst = True
            types = {e.type for e in world._events_this_tick}
            assert EventType.JERRY_EXITED_LOCKER in types, "burst should emit exit event"
            assert EventType.NOISE_EMITTED in types, "burst should emit noise"
            assert world._jerry_oxygen is None
            assert pos in world._locker_cooldowns, "burst locker should be on cooldown"
            break
    assert burst, "oxygen should have hit zero and burst within capacity ticks"


def test_burst_locker_goes_on_cooldown_then_reopens():
    cfg = WorldConfig(locker_oxygen_enabled=True, locker_oxygen_capacity=3,
                      locker_cooldown_ticks=15, max_ticks=300)
    world = World(cfg, seed=42)
    world.reset()
    pos = _first_locker(world)
    _enter_locker(world, pos)
    for _ in range(4):
        world._events_this_tick = []
        world.sound.clear()
        world.tick_count += 1
        world._tick_locker_oxygen()
        if not world.jerry.in_locker:
            break
    cd_until = world._locker_cooldowns[pos]

    # Re-entry while cooling down fails.
    world.jerry.position = pos
    world._resolve_action(world.jerry, Action.INTERACT, "jerry")
    assert world.jerry.in_locker is False, "should not re-enter a cooling-down locker"

    # After the cooldown lifts, re-entry succeeds.
    world.tick_count = cd_until + 1
    world._resolve_action(world.jerry, Action.INTERACT, "jerry")
    assert world.jerry.in_locker is True, "should re-enter after cooldown lifts"


def test_voluntary_exit_starts_cooldown():
    cfg = WorldConfig(locker_oxygen_enabled=True, locker_oxygen_capacity=50,
                      locker_cooldown_ticks=30, max_ticks=300)
    world = World(cfg, seed=42)
    world.reset()
    pos = _first_locker(world)
    _enter_locker(world, pos)
    assert world.jerry.in_locker is True
    # Voluntarily leave.
    world._resolve_action(world.jerry, Action.INTERACT, "jerry")
    assert world.jerry.in_locker is False
    assert pos in world._locker_cooldowns, "voluntary exit should start cooldown"
    assert world._jerry_oxygen is None
