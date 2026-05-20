"""Tests for hunt modes + chemistry override (Phase 6e).

Two layers:
  1. Pure mode-logic unit tests (no world): suggestion from belief type,
     the adrenaline→RUSH override, the precedence rule.
  2. Integration: a Conductor-driven Tom computes modes, the override
     fires during real episodes, STALK self-limits (no deadlock), and
     Tom still catches.
"""
from __future__ import annotations

import pytest

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.conductor import (
    Conductor,
    HuntMode,
    ModeConfig,
    SuspicionType,
    apply_chemistry_override,
    conductor_suggested_mode,
    decide_mode,
)
from src.utils.types import Action


# ---- pure: conductor suggestion --------------------------------------

def test_empty_belief_suggests_patrol():
    cfg = ModeConfig()
    assert conductor_suggested_mode(None, 0.0, cfg) == HuntMode.PATROL


def test_high_confidence_sighting_suggests_stalk():
    cfg = ModeConfig()
    m = conductor_suggested_mode(SuspicionType.SIGHTING, 0.9, cfg)
    assert m == HuntMode.STALK


def test_low_confidence_sighting_suggests_investigate():
    cfg = ModeConfig()
    m = conductor_suggested_mode(SuspicionType.SIGHTING, 0.3, cfg)
    assert m == HuntMode.INVESTIGATE


def test_noise_suggests_investigate():
    cfg = ModeConfig()
    assert conductor_suggested_mode(SuspicionType.NOISE, 0.6, cfg) == HuntMode.INVESTIGATE


def test_scent_suggests_investigate():
    cfg = ModeConfig()
    assert conductor_suggested_mode(SuspicionType.SCENT, 0.5, cfg) == HuntMode.INVESTIGATE


# ---- pure: chemistry override ----------------------------------------

def test_low_adrenaline_keeps_suggestion():
    cfg = ModeConfig()
    final, overridden = apply_chemistry_override(
        HuntMode.STALK, adrenaline=0.1, cortisol=0.0, config=cfg)
    assert final == HuntMode.STALK
    assert overridden is False


def test_high_adrenaline_overrides_stalk_to_rush():
    """THE core 6e mechanic: an amped Tom over-commits, turning a patient
    STALK suggestion into a headlong RUSH.
    """
    cfg = ModeConfig()
    final, overridden = apply_chemistry_override(
        HuntMode.STALK, adrenaline=0.9, cortisol=0.0, config=cfg)
    assert final == HuntMode.RUSH
    assert overridden is True


def test_high_adrenaline_overrides_investigate_to_rush():
    cfg = ModeConfig()
    final, overridden = apply_chemistry_override(
        HuntMode.INVESTIGATE, adrenaline=0.9, cortisol=0.0, config=cfg)
    assert final == HuntMode.RUSH
    assert overridden is True


def test_patrol_never_overridden():
    """PATROL has nothing to commit to — adrenaline doesn't change it."""
    cfg = ModeConfig()
    final, overridden = apply_chemistry_override(
        HuntMode.PATROL, adrenaline=1.0, cortisol=0.0, config=cfg)
    assert final == HuntMode.PATROL
    assert overridden is False


def test_already_rush_not_double_overridden():
    """If the suggestion is already RUSH, high adrenaline doesn't flag an
    override (nothing changed).
    """
    cfg = ModeConfig()
    final, overridden = apply_chemistry_override(
        HuntMode.RUSH, adrenaline=0.9, cortisol=0.0, config=cfg)
    assert final == HuntMode.RUSH
    assert overridden is False


def test_override_threshold_boundary():
    """Override fires at >= threshold, not below."""
    cfg = ModeConfig(rush_adrenaline_threshold=0.65)
    below = apply_chemistry_override(HuntMode.STALK, 0.64, 0.0, cfg)
    at = apply_chemistry_override(HuntMode.STALK, 0.65, 0.0, cfg)
    assert below[0] == HuntMode.STALK
    assert at[0] == HuntMode.RUSH


# ---- pure: decide_mode wrapper ---------------------------------------

def test_decide_mode_returns_all_three():
    cfg = ModeConfig()
    final, suggested, overridden = decide_mode(
        SuspicionType.SIGHTING, confidence=0.9,
        adrenaline=0.9, cortisol=0.0, config=cfg)
    assert suggested == HuntMode.STALK
    assert final == HuntMode.RUSH
    assert overridden is True


def test_decide_mode_no_override_path():
    cfg = ModeConfig()
    final, suggested, overridden = decide_mode(
        SuspicionType.SIGHTING, confidence=0.9,
        adrenaline=0.1, cortisol=0.0, config=cfg)
    assert suggested == HuntMode.STALK
    assert final == HuntMode.STALK
    assert overridden is False


# ---- integration: modes in a real episode ----------------------------

def test_conductor_tom_computes_modes():
    """A Conductor-driven Tom should set current_mode each tick and use a
    variety of modes over an episode (at least PATROL plus one active mode).
    """
    world = World(WorldConfig(max_ticks=200), seed=42)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=42)
    tom.reset()
    modes_seen = set()
    for _ in range(200):
        a = tom(world)
        modes_seen.add(tom.current_mode)
        _, _, _, done = world.step(tom_action=a, jerry_action=Action.WAIT)
        if done or not world.jerry.alive:
            break
    # Should see PATROL plus at least one active hunting mode
    assert HuntMode.PATROL in modes_seen
    assert len(modes_seen) >= 2


def test_override_fires_during_episode():
    """Over a real episode, the chemistry override should fire at least
    once — Tom's adrenaline crosses the threshold and upgrades a STALK/
    INVESTIGATE suggestion to RUSH.
    """
    world = World(WorldConfig(max_ticks=300), seed=42)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=42)
    tom.reset()
    overrides = 0
    for _ in range(300):
        a = tom(world)
        if tom.mode_overridden:
            overrides += 1
        _, _, _, done = world.step(tom_action=a, jerry_action=Action.WAIT)
        if done or not world.jerry.alive:
            break
    assert overrides >= 1, "chemistry override never fired in a full episode"


def test_stalk_self_limits_and_tom_still_catches():
    """STALK holds at a distance, but adrenaline rises while Jerry is in
    view, eventually triggering the RUSH override and a commit. So STALK
    must NOT deadlock — Tom should still catch a passive Jerry.
    """
    caught_count = 0
    for seed in (42, 1, 7, 13, 99):
        world = World(WorldConfig(max_ticks=300), seed=seed)
        world.reset()
        tom = ChemicalTom(conductor=Conductor(), seed=seed)
        tom.reset()
        for _ in range(300):
            a = tom(world)
            _, _, _, done = world.step(tom_action=a, jerry_action=Action.WAIT)
            if done or not world.jerry.alive:
                break
        if not world.jerry.alive:
            caught_count += 1
    # STALK must not deadlock — Tom should catch a passive Jerry most of
    # the time (it's a sitting target). Require catching in the majority.
    assert caught_count >= 3, \
        f"Tom only caught {caught_count}/5 passive Jerrys — STALK may be deadlocking"


def test_modes_reset_between_episodes():
    world = World(WorldConfig(max_ticks=50), seed=42)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=42)
    tom.reset()
    for _ in range(20):
        a = tom(world)
        world.step(tom_action=a, jerry_action=Action.WAIT)
        if not world.jerry.alive:
            break
    tom.reset()
    assert tom.current_mode == HuntMode.PATROL
    assert tom.suggested_mode == HuntMode.PATROL
    assert tom.mode_overridden is False
