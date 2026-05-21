"""Tests for Component 3 — hold-on-LOS-break / run-down (memory adaptation).

This is the new Conductor behavior that counters the cover-dance: when Tom
loses line of sight to a Jerry he was actively seeing, the Conductor
anchors a high-confidence sighting at the last-seen tile and keeps Tom
pursuing it (instead of letting the suspicion decay and releasing
pressure).

The behavior is flag-gated (hold_on_los_break, default False) because it's
deployed by MEMORY, not always-on. These tests force it on.

Verified:
  - default-off preserves prior Conductor behavior
  - anchor activates on LOS-break and not before
  - while anchored, Tom targets the anchor tile (pursuit flows through the
    handover into last_seen_jerry)
  - a live sighting clears the anchor (real contact dominates)
  - the anchor expires after the window
  - the anchor expires when Tom reaches the tile (runs it down)
  - reset clears anchor state
"""
from __future__ import annotations

import random

import pytest

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.conductor import Conductor, ConductorConfig
from src.utils.types import Action, Position


def _run(world, tom, jerry_rng, max_ticks):
    """Drive an episode with a random-moving Jerry."""
    for _ in range(max_ticks):
        a = tom(world)
        ja = jerry_rng.randint(0, 4)
        _, _, _, done = world.step(tom_action=a, jerry_action=ja)
        if done or not world.jerry.alive:
            break


# ---- default off -------------------------------------------------------

def test_hold_off_by_default():
    """hold_on_los_break defaults to False — the anchor never activates."""
    world = World(WorldConfig(max_ticks=200), seed=3)
    world.reset()
    cond = Conductor()  # default config
    tom = ChemicalTom(conductor=cond, seed=3)
    tom.reset()
    rng = random.Random(11)
    anchored = False
    for _ in range(200):
        a = tom(world)
        if cond.anchor_active:
            anchored = True
        ja = rng.randint(0, 4)
        world.step(tom_action=a, jerry_action=ja)
        if not world.jerry.alive:
            break
    assert anchored is False


def test_hold_off_behaves_like_plain_conductor():
    """With hold off, behavior matches a plain Conductor exactly."""
    def run(cfg):
        world = World(WorldConfig(max_ticks=150), seed=7)
        world.reset()
        cond = Conductor(config=cfg) if cfg else Conductor()
        tom = ChemicalTom(conductor=cond, seed=7)
        tom.reset()
        rng = random.Random(99)
        actions = []
        for _ in range(150):
            a = tom(world)
            actions.append(int(a))
            ja = rng.randint(0, 4)
            world.step(tom_action=a, jerry_action=ja)
            if not world.jerry.alive:
                break
        return actions

    plain = run(None)
    explicit_off = run(ConductorConfig(hold_on_los_break=False))
    assert plain == explicit_off


# ---- anchor activation -------------------------------------------------

def test_anchor_activates_on_los_break():
    """Across several seeds, the anchor should activate in at least some
    episodes (those where Tom loses LOS), and while active Tom should be
    targeting the anchor tile.
    """
    episodes_with_anchor = 0
    pursuit_mismatches = 0
    anchor_ticks = 0
    for seed in range(20):
        cfg = ConductorConfig(hold_on_los_break=True, hold_window_ticks=15)
        world = World(WorldConfig(max_ticks=200), seed=seed)
        world.reset()
        cond = Conductor(config=cfg)
        tom = ChemicalTom(conductor=cond, seed=seed)
        tom.reset()
        rng = random.Random(seed + 11)
        ep_anchored = False
        for _ in range(200):
            a = tom(world)
            if cond.anchor_active:
                ep_anchored = True
                anchor_ticks += 1
                # While anchored, Tom's target should be the anchor tile.
                if tom.last_seen_jerry != cond._anchor_pos:
                    pursuit_mismatches += 1
            ja = rng.randint(0, 4)
            world.step(tom_action=a, jerry_action=ja)
            if not world.jerry.alive:
                break
        if ep_anchored:
            episodes_with_anchor += 1

    assert episodes_with_anchor >= 1, "anchor never activated in 20 episodes"
    assert anchor_ticks > 0
    # Every anchored tick should have Tom pursuing the anchor.
    assert pursuit_mismatches == 0, \
        f"{pursuit_mismatches}/{anchor_ticks} anchored ticks Tom wasn't targeting the anchor"


# ---- anchor lifecycle (controlled) ------------------------------------

def test_live_sighting_clears_anchor():
    """If Tom re-acquires LOS while anchored, the anchor clears (real
    sighting dominates)."""
    cfg = ConductorConfig(hold_on_los_break=True, hold_window_ticks=15)
    cond = Conductor(config=cfg)

    # Build a minimal fake world we can drive precisely.
    from tests.integration.test_conductor_observe import _FakeWorld

    # Tick 0: Tom sees Jerry.
    w = _FakeWorld(tick=0, tom=Position(5, 5), jerry=Position(7, 5), can_see=True)
    cond.observe(w)
    assert cond.anchor_active is False  # seeing → no anchor

    # Tick 1: LOS breaks → anchor should start.
    w = _FakeWorld(tick=1, tom=Position(5, 5), jerry=Position(7, 5), can_see=False)
    cond.observe(w)
    assert cond.anchor_active is True
    assert cond._anchor_pos == Position(7, 5)

    # Tick 2: Tom re-acquires LOS → anchor clears.
    w = _FakeWorld(tick=2, tom=Position(6, 5), jerry=Position(8, 5), can_see=True)
    cond.observe(w)
    assert cond.anchor_active is False
    assert cond._anchor_pos is None


def test_anchor_expires_after_window():
    """The anchor expires once hold_window_ticks pass without resolution."""
    cfg = ConductorConfig(hold_on_los_break=True, hold_window_ticks=5)
    cond = Conductor(config=cfg)
    from tests.integration.test_conductor_observe import _FakeWorld

    # See, then lose at tick 1 → anchor until tick 6.
    cond.observe(_FakeWorld(tick=0, tom=Position(0, 0), jerry=Position(10, 10), can_see=True))
    cond.observe(_FakeWorld(tick=1, tom=Position(0, 0), jerry=Position(10, 10), can_see=False))
    assert cond.anchor_active is True

    # Keep Tom far away so it never "runs down" the anchor; advance past window.
    for t in range(2, 8):
        cond.observe(_FakeWorld(tick=t, tom=Position(0, 0), jerry=Position(10, 10), can_see=False))
    assert cond.anchor_active is False


def test_anchor_occupies_then_clears_when_run_down():
    """When Tom reaches the anchor, he OCCUPIES it for hold_occupy_ticks
    (camping the spot to flush Jerry out) before the anchor clears, rather
    than leaving immediately.
    """
    cfg = ConductorConfig(hold_on_los_break=True, hold_window_ticks=30,
                          hold_occupy_ticks=4)
    cond = Conductor(config=cfg)
    from tests.integration.test_conductor_observe import _FakeWorld

    cond.observe(_FakeWorld(tick=0, tom=Position(0, 0), jerry=Position(3, 0), can_see=True))
    cond.observe(_FakeWorld(tick=1, tom=Position(0, 0), jerry=Position(3, 0), can_see=False))
    assert cond.anchor_active is True
    assert cond._anchor_pos == Position(3, 0)

    # Tom reaches the anchor tile → starts occupying (does NOT clear yet).
    cond.observe(_FakeWorld(tick=2, tom=Position(3, 0), jerry=Position(3, 0), can_see=False))
    assert cond.anchor_active is True, "anchor should persist while occupying"

    # Tom camps the spot; anchor stays through the occupy window...
    cond.observe(_FakeWorld(tick=3, tom=Position(3, 0), jerry=Position(3, 0), can_see=False))
    assert cond.anchor_active is True

    # ...then clears once the occupy dwell (4 ticks from tick 2 => tick 6) passes.
    for t in range(4, 8):
        cond.observe(_FakeWorld(tick=t, tom=Position(3, 0), jerry=Position(3, 0), can_see=False))
    assert cond.anchor_active is False
    assert cond._anchor_pos is None


# ---- reset -------------------------------------------------------------

def test_reset_clears_anchor_state():
    cfg = ConductorConfig(hold_on_los_break=True)
    cond = Conductor(config=cfg)
    from tests.integration.test_conductor_observe import _FakeWorld

    cond.observe(_FakeWorld(tick=0, tom=Position(0, 0), jerry=Position(5, 5), can_see=True))
    cond.observe(_FakeWorld(tick=1, tom=Position(0, 0), jerry=Position(5, 5), can_see=False))
    assert cond.anchor_active is True

    cond.reset()
    assert cond.anchor_active is False
    assert cond._anchor_pos is None
    assert cond._anchor_until_tick == -1
    assert cond._was_seeing_jerry is False
    assert cond._last_seen_pos is None


# ---- inertness check ---------------------------------------------------

def test_hold_does_not_break_normal_catching():
    """With hold on, Tom should still catch a passive Jerry (the behavior
    must not deadlock or sabotage normal hunting)."""
    caught = 0
    for seed in (42, 1, 7, 13, 99):
        cfg = ConductorConfig(hold_on_los_break=True)
        world = World(WorldConfig(max_ticks=300), seed=seed)
        world.reset()
        tom = ChemicalTom(conductor=Conductor(config=cfg), seed=seed)
        tom.reset()
        for _ in range(300):
            a = tom(world)
            world.step(tom_action=a, jerry_action=Action.WAIT)
            if not world.jerry.alive:
                break
        if not world.jerry.alive:
            caught += 1
    assert caught >= 3, f"hold-on Tom only caught {caught}/5 passive Jerrys"


def test_tom_rushes_not_stalks_while_anchored():
    """Component 3 fix: while the run-down anchor is active, Tom must be in
    RUSH mode (close on the vanish point and occupy it), NOT STALK (hold at
    distance). The anchor re-stamps a high-confidence sighting which would
    otherwise suggest STALK — the wrong behavior for a run-down.
    """
    from src.hunter.agent.conductor import HuntMode

    cfg = ConductorConfig(hold_on_los_break=True, hold_window_ticks=15)
    saw_anchor = False
    stalk_while_anchored = 0
    for seed in range(30):
        world = World(WorldConfig(max_ticks=200), seed=seed)
        world.reset()
        cond = Conductor(config=cfg)
        tom = ChemicalTom(conductor=cond, seed=seed)
        tom.reset()
        rng = random.Random(seed + 11)
        for _ in range(200):
            a = tom(world)
            if cond.anchor_active:
                saw_anchor = True
                if tom.current_mode == HuntMode.STALK:
                    stalk_while_anchored += 1
            ja = rng.randint(0, 4)
            world.step(tom_action=a, jerry_action=ja)
            if not world.jerry.alive:
                break
    assert saw_anchor, "no anchor activated across 30 seeds — can't test mode"
    assert stalk_while_anchored == 0, \
        f"Tom was STALKing on {stalk_while_anchored} anchored ticks — should RUSH"
