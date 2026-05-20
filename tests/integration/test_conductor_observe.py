"""Tests for the Conductor scaffolding (Phase 6b, observe-only).

Two layers:
  1. Unit tests of signal sourcing with a hand-built fake world (precise
     control over events / visibility / scent).
  2. Integration tests driving the Conductor alongside a REAL running
     episode, checking the belief tracks reality sensibly.

Phase 6b is observe-only: the Conductor must NOT change Tom's behavior.
A regression test confirms ScriptedTom acts identically with and without
a Conductor watching.
"""
from __future__ import annotations

import pytest

from src.env.world.world import Event, EventType, World, WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.hunter.agent.conductor import (
    Conductor,
    ConductorConfig,
    SuspicionType,
)
from src.utils.types import Action, Position


# ---- fake world for precise unit control ------------------------------

class _FakeScent:
    def __init__(self, grad):
        self._grad = grad

    def gradient_at(self, pos):
        return self._grad


class _FakeTom:
    def __init__(self, pos):
        self.position = pos


class _FakeJerry:
    def __init__(self, pos):
        self.position = pos


class _FakeWorld:
    """Minimal stand-in exposing only what Conductor.observe reads."""
    def __init__(self, *, tick=0, tom=Position(0, 0), jerry=Position(10, 10),
                 can_see=False, events=None, scent_grad=None):
        self.tick_count = tick
        self.tom = _FakeTom(tom)
        self.jerry = _FakeJerry(jerry)
        self._can_see = can_see
        self._events_this_tick = events or []
        self.scent = _FakeScent(scent_grad or {"N": 0, "S": 0, "E": 0, "W": 0})

    def _tom_can_see_jerry(self):
        return self._can_see


# ---- sighting sourcing -------------------------------------------------

def test_sighting_created_only_when_visible():
    c = Conductor()
    # Not visible -> no sighting
    w = _FakeWorld(can_see=False, tick=0)
    c.observe(w)
    assert c.suggested_target(w) is None

    # Visible -> sighting at Jerry's position
    w2 = _FakeWorld(can_see=True, jerry=Position(10, 10), tick=1)
    c.observe(w2)
    target = c.suggested_target(w2)
    assert target == Position(10, 10)
    assert c.last_suggested_type == SuspicionType.SIGHTING


# ---- noise sourcing ----------------------------------------------------

def test_jerry_noise_creates_suspicion():
    c = Conductor()
    ev = Event(type=EventType.NOISE_EMITTED, actor="jerry",
               position=Position(7, 3), payload=1.0)
    w = _FakeWorld(events=[ev], tick=0)
    c.observe(w)
    src = c.suggested_source(w)
    assert src is not None
    s, _ = src
    assert s.type == SuspicionType.NOISE
    assert s.position == Position(7, 3)


def test_tom_own_noise_ignored():
    """Tom's own footstep noise must NOT create a Jerry suspicion."""
    c = Conductor()
    ev = Event(type=EventType.NOISE_EMITTED, actor="tom",
               position=Position(0, 0), payload=1.0)
    w = _FakeWorld(events=[ev], tick=0)
    c.observe(w)
    assert c.suggested_target(w) is None


def test_noise_without_position_ignored():
    c = Conductor()
    ev = Event(type=EventType.NOISE_EMITTED, actor="jerry",
               position=None, payload=1.0)
    w = _FakeWorld(events=[ev], tick=0)
    c.observe(w)
    assert c.suggested_target(w) is None


def test_noise_intensity_from_payload():
    c = Conductor()
    ev_soft = Event(type=EventType.NOISE_EMITTED, actor="jerry",
                    position=Position(5, 5), payload=0.5)
    w = _FakeWorld(events=[ev_soft], tick=0)
    c.observe(w)
    src = c.suggested_source(w)
    assert src is not None
    _, conf = src
    # birth_noise 0.6 * 0.5 = 0.3
    assert conf == pytest.approx(0.3)


# ---- scent sourcing ----------------------------------------------------

def test_scent_projects_suspicion_in_gradient_direction():
    c = Conductor()
    # Strong scent to the East from Tom at (5,5) -> suspicion ~3 tiles East
    w = _FakeWorld(tom=Position(5, 5), tick=0,
                   scent_grad={"N": 0.0, "S": 0.0, "E": 0.8, "W": 0.0})
    c.observe(w)
    src = c.suggested_source(w)
    assert src is not None
    s, _ = src
    assert s.type == SuspicionType.SCENT
    assert s.position == Position(8, 5)  # 5 + 3 East


def test_weak_scent_below_threshold_ignored():
    c = Conductor()
    w = _FakeWorld(tom=Position(5, 5), tick=0,
                   scent_grad={"N": 0.05, "S": 0.0, "E": 0.0, "W": 0.0})
    c.observe(w)
    assert c.suggested_target(w) is None


# ---- priority among signals -------------------------------------------

def test_sighting_beats_noise_and_scent():
    c = Conductor()
    ev = Event(type=EventType.NOISE_EMITTED, actor="jerry",
               position=Position(2, 2), payload=1.0)
    w = _FakeWorld(can_see=True, jerry=Position(20, 20), tick=0,
                   events=[ev],
                   scent_grad={"N": 0.0, "S": 0.0, "E": 0.5, "W": 0.0})
    c.observe(w)
    target = c.suggested_target(w)
    # Sighting (conf 1.0) should win over noise (0.6) and scent (~0.5)
    assert target == Position(20, 20)
    assert c.last_suggested_type == SuspicionType.SIGHTING


# ---- reset -------------------------------------------------------------

def test_reset_clears_belief():
    c = Conductor()
    w = _FakeWorld(can_see=True, jerry=Position(3, 3), tick=0)
    c.observe(w)
    assert c.suggested_target(w) is not None
    c.reset()
    assert c.suggested_target(w) is None
    assert c.last_suggested_target is None


# ---- integration: real episode ----------------------------------------

def test_conductor_builds_belief_during_real_episode():
    """Drive a Conductor alongside a real ScriptedTom-vs-passive-Jerry
    episode. The Conductor should accumulate at least one suspicion over
    the course of the episode (Tom makes noise / sees Jerry at some point).
    """
    world = World(WorldConfig(max_ticks=200), seed=42)
    world.reset()
    tom = ScriptedTom(seed=42)
    conductor = Conductor()
    conductor.reset()

    saw_a_suspicion = False
    for _ in range(200):
        # Conductor observes BEFORE Tom acts (same order the real wiring
        # will use): it sees last tick's events + current visibility.
        conductor.observe(world)
        if conductor.suggested_target(world) is not None:
            saw_a_suspicion = True
        action = tom(world)
        # passive Jerry
        world.step(tom_action=action, jerry_action=int(Action.WAIT))
        if not world.jerry.alive:
            break

    assert saw_a_suspicion, "Conductor never formed any suspicion in 200 ticks"


def test_conductor_does_not_change_tom_behavior():
    """THE Phase 6b regression guard. ScriptedTom must act identically
    whether or not a Conductor is observing. We run two identical episodes
    — one with a Conductor watching, one without — and assert Tom's action
    sequence is byte-for-byte identical.
    """
    def run(with_conductor: bool) -> list[int]:
        world = World(WorldConfig(max_ticks=150), seed=7)
        world.reset()
        tom = ScriptedTom(seed=7)
        conductor = Conductor() if with_conductor else None
        if conductor:
            conductor.reset()
        actions: list[int] = []
        for _ in range(150):
            if conductor:
                conductor.observe(world)
                _ = conductor.suggested_target(world)  # query, ignore
            a = tom(world)
            actions.append(int(a))
            world.step(tom_action=a, jerry_action=int(Action.WAIT))
            if not world.jerry.alive:
                break
        return actions

    without = run(with_conductor=False)
    with_ = run(with_conductor=True)
    assert without == with_, \
        "Conductor observation changed Tom's behavior — 6b must be observe-only"


def test_conductor_suspicion_near_jerry_when_visible():
    """When Tom can see Jerry, the Conductor's suggested target should be
    exactly Jerry's position (a sighting). Find a tick where Tom sees
    Jerry and check.
    """
    world = World(WorldConfig(max_ticks=300), seed=42)
    world.reset()
    tom = ScriptedTom(seed=42)
    conductor = Conductor()
    conductor.reset()

    checked = False
    for _ in range(300):
        conductor.observe(world)
        if world._tom_can_see_jerry():
            target = conductor.suggested_target(world)
            assert target == world.jerry.position
            assert conductor.last_suggested_type == SuspicionType.SIGHTING
            checked = True
            break
        action = tom(world)
        world.step(tom_action=action, jerry_action=int(Action.WAIT))
        if not world.jerry.alive:
            break

    if not checked:
        pytest.skip("Tom never saw Jerry on seed 42 within 300 ticks")
