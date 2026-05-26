"""Tests for the pursuit speed-ramp mechanic.

Tom and Jerry are equal speed, which makes pure kiting / closed-loop circling
unbeatable (DIST floors at 2). The ramp lets Tom accelerate during sustained
committed pursuit (PURSUE/ATTACK), capping at speed_cap, decaying only when he
gives up. These tests guard the accumulator math, the gating (off by default =
legacy), and the decay-on-giveup behavior.
"""
from __future__ import annotations

from src.env.world.world import World, WorldConfig
from src.utils.types import Action, Position


def _ramp_world(seed: int = 1, **overrides) -> World:
    cfg = WorldConfig(
        pursuit_speed_ramp_enabled=True,
        pursuit_ramp_per_tick=0.005,
        pursuit_speed_cap=1.15,
        pursuit_ramp_decay=0.05,
        max_ticks=600,
        **overrides,
    )
    w = World(cfg, seed=seed)
    w.reset()
    return w


def test_ramp_off_by_default_no_speed_change():
    """Default config: speed stays 1.0 and no accumulator movement, even if a
    pursuit flag is passed."""
    w = World(WorldConfig(max_ticks=100), seed=1)
    w.reset()
    for _ in range(50):
        w.step(Action.WAIT, Action.WAIT, tom_in_pursuit=True)
    assert w._tom_speed == 1.0
    assert w._tom_step_bank == 0.0


def test_none_pursuit_flag_keeps_ramp_inert():
    """Even with the ramp enabled, a legacy caller passing tom_in_pursuit=None
    gets no acceleration (preserves exact legacy behavior)."""
    w = _ramp_world()
    for _ in range(50):
        w.step(Action.WAIT, Action.WAIT, tom_in_pursuit=None)
    assert w._tom_speed == 1.0


def test_speed_ramps_up_during_pursuit_and_caps():
    """Sustained pursuit ramps speed toward the cap and never exceeds it."""
    w = _ramp_world()
    for _ in range(200):
        w.step(Action.WAIT, Action.WAIT, tom_in_pursuit=True)
    assert abs(w._tom_speed - 1.15) < 1e-6  # reached cap
    # One more tick must not exceed the cap.
    w.step(Action.WAIT, Action.WAIT, tom_in_pursuit=True)
    assert w._tom_speed <= 1.15 + 1e-9


def test_speed_decays_when_tom_gives_up():
    """Once Tom drops out of pursuit, speed decays back toward base."""
    w = _ramp_world()
    for _ in range(200):  # ramp to cap
        w.step(Action.WAIT, Action.WAIT, tom_in_pursuit=True)
    assert w._tom_speed > 1.0
    for _ in range(10):  # give up
        w.step(Action.WAIT, Action.WAIT, tom_in_pursuit=False)
    assert w._tom_speed == 1.0  # fully decayed
    assert w._tom_step_bank == 0.0  # bank cleared on full reset


def test_bonus_step_moves_tom_extra_tile():
    """When the accumulator crosses 1.0, Tom takes a bonus step in his action
    direction — covering 2 tiles in one tick instead of 1."""
    w = _ramp_world()
    # Ramp to cap first (no movement; WAIT doesn't consume bonus steps since
    # the bonus only re-resolves movement actions).
    for _ in range(40):
        w.step(Action.WAIT, Action.WAIT, tom_in_pursuit=True)
    assert w._tom_speed >= 1.1  # near cap, bank building

    # Place Tom in open space facing EAST with room to move two tiles.
    start = None
    for x in range(2, w.grid.width - 3):
        for y in range(2, w.grid.height - 3):
            p = Position(x, y)
            e1 = Position(x + 1, y)
            e2 = Position(x + 2, y)
            if (w.grid.is_walkable(p) and w.grid.is_walkable(e1)
                    and w.grid.is_walkable(e2)):
                start = p
                break
        if start:
            break
    assert start is not None
    w.tom.position = start
    # Drive bank over 1.0 so the next pursuit tick grants a bonus step.
    w._tom_step_bank = 0.99
    before = w.tom.position
    w.step(Action.EAST, Action.WAIT, tom_in_pursuit=True)
    moved = abs(w.tom.position.x - before.x) + abs(w.tom.position.y - before.y)
    assert moved == 2, f"expected a 2-tile bonus-step move, got {moved}"


def test_brief_pursuit_no_bonus_step_early():
    """A short chase shouldn't grant a bonus step immediately — the ramp must
    build first (no free instant lunge at chase start)."""
    w = _ramp_world()
    bonus_seen = False
    last = w.tom.position
    # Put Tom where he can move east freely.
    for _ in range(5):
        before = w.tom.position
        w.step(Action.EAST, Action.WAIT, tom_in_pursuit=True)
        if abs(w.tom.position.x - before.x) + abs(w.tom.position.y - before.y) > 1:
            bonus_seen = True
    assert not bonus_seen, "should not lunge in the first few ticks of pursuit"
