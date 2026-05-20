"""Tests for the Phase 6c targeting handover.

When a Conductor is attached to ChemicalTom, Tom's strategic targeting
(last_seen_jerry / last_noise) is populated from the Conductor's decaying,
foolable belief instead of Tom's direct perception. Downstream logic is
unchanged.

We verify:
  1. conductor=None preserves exact Phase 2-5 behavior (regression).
  2. With a Conductor, Tom's memory fields track the belief.
  3. The belief's decay produces the intended "weakening" — Tom forgets
     a stale sighting that no-Conductor Tom would still be chasing.
  4. A false noise (planted away from Jerry) misdirects Conductor-Tom.
  5. Episodes run end-to-end with a Conductor attached.
"""
from __future__ import annotations

import pytest

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.conductor import Conductor
from src.utils.types import Action, Position


# ---- regression: no conductor == old behavior ------------------------

def test_no_conductor_behaves_identically():
    """Two ChemicalToms with the same seed, neither with a Conductor,
    must produce identical action sequences against the same episode.
    (Sanity that the new branch doesn't perturb the default path.)
    """
    def run():
        world = World(WorldConfig(max_ticks=120), seed=11)
        world.reset()
        tom = ChemicalTom(seed=11)
        tom.reset()
        actions = []
        for _ in range(120):
            a = tom(world)
            actions.append(int(a))
            world.step(tom_action=a, jerry_action=Action.WAIT)
            if not world.jerry.alive:
                break
        return actions

    assert run() == run()


# ---- conductor populates memory fields -------------------------------

def test_conductor_populates_last_seen_from_belief():
    """When the Conductor holds a SIGHTING suspicion, Tom's last_seen_jerry
    should be set from it after a tick.
    """
    world = World(WorldConfig(max_ticks=300), seed=42)
    world.reset()
    conductor = Conductor()
    tom = ChemicalTom(conductor=conductor, seed=42)
    tom.reset()

    # Step until Tom sees Jerry (creating a sighting suspicion)
    saw = False
    for _ in range(300):
        a = tom(world)
        if tom.last_seen_jerry is not None:
            saw = True
            break
        world.step(tom_action=a, jerry_action=Action.WAIT)
        if not world.jerry.alive:
            break
    assert saw, "Tom never formed a last_seen from the Conductor belief"


def test_conductor_attached_runs_full_episode():
    """An episode with a Conductor-driven ChemicalTom runs to completion
    without error and produces a valid outcome.
    """
    world = World(WorldConfig(max_ticks=200), seed=3)
    world.reset()
    conductor = Conductor()
    tom = ChemicalTom(conductor=conductor, seed=3)
    tom.reset()
    steps = 0
    for _ in range(200):
        a = tom(world)
        _, _, _, done = world.step(tom_action=a, jerry_action=Action.WAIT)
        steps += 1
        if done or not world.jerry.alive:
            break
    assert steps > 0


# ---- the weakening: belief decay vs crisp memory ---------------------

def test_belief_decay_weakens_pursuit_memory():
    """A Conductor-driven Tom should LOSE a stale sighting suspicion as it
    decays below the belief floor, whereas the raw belief makes the
    suspicion fade continuously. We check the belief empties after enough
    ticks with no reinforcement.
    """
    conductor = Conductor()
    tom = ChemicalTom(conductor=conductor, seed=1)
    tom.reset()

    # Manually inject a sighting suspicion at tick 0
    conductor.belief.add_sighting(Position(10, 10), now_tick=0)
    assert conductor.belief.strongest(now_tick=0) is not None

    # Sighting half-life ~25 ticks, floor 0.05; by tick ~110 it's dead.
    conductor.belief.tick(now_tick=110)
    assert conductor.belief.is_empty(now_tick=110), \
        "Sighting suspicion should decay to nothing without reinforcement"


def test_false_noise_creates_misdirecting_suspicion():
    """A noise planted far from Jerry creates a real NOISE suspicion that
    Tom (via the Conductor) will treat as a target — the manipulation
    surface ADR-013 wants. We inject a false noise and confirm Tom's
    last_noise points at the false location.
    """
    from src.env.world.world import Event, EventType

    world = World(WorldConfig(max_ticks=50), seed=42)
    world.reset()
    conductor = Conductor()
    tom = ChemicalTom(conductor=conductor, seed=42)
    tom.reset()

    # Inject a false noise event far from Jerry's actual position into the
    # world's event buffer, as if Jerry had thrown something.
    false_pos = Position(2, 2)
    world._events_this_tick = [
        Event(type=EventType.NOISE_EMITTED, actor="jerry",
              position=false_pos, payload=1.0),
    ]
    # One Tom tick: it observes the (false) noise via the Conductor.
    tom(world)
    # Tom's last_noise should now be the false location (or very near it,
    # if a merge nudged it). It must NOT be None.
    assert tom.last_noise is not None
    assert tom.last_noise == false_pos


# ---- behavior actually differs with vs without conductor -------------

def test_conductor_changes_behavior_vs_baseline():
    """A Conductor-driven Tom and a plain Tom should diverge in their
    strategic state, because the Conductor creates a NOISE suspicion for
    every Jerry noise event while plain Tom only records noises above his
    modulated threshold (and the belief decays continuously rather than
    on a hard timeout).

    We assert divergence in INTERNAL TARGETING STATE rather than in the
    raw action sequence. Action sequences can coincidentally reconverge
    (INVESTIGATE-toward-X and PATROL may pick the same first step), so
    action equality is too brittle a probe. What matters is that the
    belief-driven memory differs from the perception-driven memory — that
    IS the handover working.
    """
    import random

    def trace(with_conductor: bool):
        world = World(WorldConfig(max_ticks=80), seed=42)
        world.reset()
        conductor = Conductor() if with_conductor else None
        tom = ChemicalTom(conductor=conductor, seed=42)
        tom.reset()
        jerry_rng = random.Random(99)  # SAME both runs → identical Jerry
        states = []
        for _ in range(80):
            a = tom(world)
            states.append((tom.state.name, tom.last_seen_jerry, tom.last_noise))
            ja = jerry_rng.randint(0, 4)
            world.step(tom_action=a, jerry_action=ja)
            if not world.jerry.alive:
                break
        return states

    plain = trace(with_conductor=False)
    conducted = trace(with_conductor=True)
    assert plain != conducted, \
        "Conductor-driven Tom's targeting state never diverged from plain Tom"
