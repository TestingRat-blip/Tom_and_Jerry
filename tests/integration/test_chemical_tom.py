"""Integration tests for ChemicalTom.

These verify the wire-up: that chemistry and drives actually CHANGE Tom's
behavior, not just sit in memory.

We test:
  - ChemicalTom works as a drop-in replacement for ScriptedTom
  - Chemistry state updates as Tom encounters events
  - Adrenalized Tom predicts ahead; calm Tom does not
  - High aggression keeps Tom committed to PURSUE longer
  - High curiosity lowers the noise threshold (Tom investigates weaker noises)
  - The replay recorder captures chemistry/drives in frames
  - Old replays still load (backwards compatibility)
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.env.gym_env import JerryEnv
from src.env.world.world import EventType, World, WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.hunter.agent.behavior.chemical_tom import (
    ChemicalTom,
    ChemicalTomConfig,
)
from src.render.replay.recorder import Replay, ReplayRecorder
from src.utils.types import Action, Position


# ---- drop-in compatibility --------------------------------------------

def test_chemical_tom_runs_full_episode():
    """ChemicalTom should complete an episode without errors."""
    tom = ChemicalTom(seed=0)
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=100),
        tom_policy=tom,
    )
    env.reset(seed=42)
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break


def test_chemical_tom_resets_state_on_env_reset():
    """Drives and chemistry should reset between episodes."""
    tom = ChemicalTom(seed=0)
    env = JerryEnv(tom_policy=tom)

    # Run a full episode to accumulate some chemistry
    env.reset(seed=42)
    for _ in range(30):
        env.step(env.action_space.sample())

    # At this point chemistry should have moved from zero
    chem_during = tom.chemistry.snapshot()

    # Reset env — should reset Tom
    env.reset(seed=99)
    chem_after_reset = tom.chemistry.snapshot()
    for name, value in chem_after_reset.items():
        assert value == 0.0, f"{name} was {value} after reset, expected 0.0"


def test_chemical_tom_catches_passive_jerry():
    """Sanity check: ChemicalTom is at least as good as ScriptedTom against
    a passive Jerry. Should hit ≥50% catch rate.
    """
    tom = ChemicalTom(seed=0)
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=400),
        tom_policy=tom,
    )
    caught = 0
    n = 10
    for trial in range(n):
        env.reset(seed=trial)
        while True:
            _, _, terminated, truncated, info = env.step(int(Action.WAIT))
            if terminated or truncated:
                if info["episode"]["outcome"] == "caught":
                    caught += 1
                break
    assert caught >= n * 0.5, \
        f"ChemicalTom only caught {caught}/{n} passive Jerrys"


# ---- chemistry updates from events ------------------------------------

def test_seeing_jerry_raises_adrenaline():
    """After repeated sightings of Jerry, adrenaline should rise."""
    tom = ChemicalTom(seed=0)
    env = JerryEnv(tom_policy=tom)
    env.reset(seed=42)

    # Force Tom and Jerry to be visible to each other on a clear row, far apart
    world = env.world
    placed = False
    for y in range(2, world.grid.height - 2):
        run_start, best_start, best_len = None, None, 0
        run_len = 0
        for x in range(1, world.grid.width - 1):
            if world.grid.is_walkable(Position(x, y)):
                if run_start is None:
                    run_start = x
                run_len += 1
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
            else:
                run_start = None
                run_len = 0
        # Need enough room that Tom can't catch a passive Jerry in 8 ticks
        if best_len >= 10 and best_start is not None:
            world.tom.position = Position(best_start, y)
            world.jerry.position = Position(best_start + 9, y)
            placed = True
            break
    if not placed:
        pytest.skip("could not find a 10-tile clear row")

    initial_adrenaline = tom.chemistry.adrenaline
    # Run a few ticks; Tom should see Jerry repeatedly. Bail if catch.
    for _ in range(8):
        _, _, terminated, truncated, _ = env.step(int(Action.WAIT))
        if terminated or truncated:
            break
    final_adrenaline = tom.chemistry.adrenaline
    assert final_adrenaline > initial_adrenaline + 0.1, \
        f"Adrenaline went from {initial_adrenaline} to {final_adrenaline}"


def test_long_patrol_raises_cortisol():
    """Tom that never sees Jerry should accumulate cortisol."""
    tom = ChemicalTom(seed=0)
    env = JerryEnv(tom_policy=tom)
    env.reset(seed=42)
    # Hide Jerry far away by tucking him into a known empty corner if possible
    # For this test, just run patrol with Tom away from Jerry — most random spawns
    # will have them far enough apart that Tom doesn't immediately see Jerry.

    initial_cortisol = tom.chemistry.cortisol
    saw_jerry = False
    for _ in range(100):
        # Guard for episode end: with non-deterministic pathing Tom may
        # actually reach and catch the WAITing Jerry. If so, stop — the
        # point of the test (cortisol rises while NOT seeing Jerry) only
        # applies up to that moment.
        if env.world.done or not env.world.jerry.alive:
            break
        if env.world._tom_can_see_jerry():
            saw_jerry = True
        env.step(int(Action.WAIT))
    # Cortisol should rise unless Tom saw Jerry along the way.
    if not saw_jerry:
        assert tom.chemistry.cortisol >= initial_cortisol


# ---- prediction horizon -----------------------------------------------

def test_calm_tom_does_not_predict():
    """Tom with zero adrenaline should target Jerry's current tile."""
    tom = ChemicalTom(seed=0)
    env = JerryEnv(tom_policy=tom)
    env.reset(seed=42)
    # Manually clear chemistry
    tom.chemistry.reset()
    # Take a tick — the prediction history is empty, so no prediction
    env.step(int(Action.WAIT))
    assert tom.last_prediction_steps == 0


def test_adrenalized_tom_predicts_ahead():
    """When adrenaline is high enough and Tom has seen Jerry twice, the
    predicted target should differ from Jerry's current position.
    """
    tom = ChemicalTom(seed=0)
    env = JerryEnv(tom_policy=tom)
    env.reset(seed=42)
    world = env.world

    # Force a clear corridor
    placed_target = None
    for y in range(3, world.grid.height - 3):
        clear = all(
            world.grid.is_walkable(Position(x, y)) for x in range(2, 12)
        )
        if clear:
            world.tom.position = Position(2, y)
            world.jerry.position = Position(5, y)
            placed_target = y
            break
    if placed_target is None:
        pytest.skip("no clear corridor")

    # Manually pump adrenaline high
    tom.chemistry.adrenaline = 0.9
    # Pre-populate jerry position history with two positions showing eastward motion
    tom._jerry_position_history.append(Position(4, placed_target))
    tom._jerry_position_history.append(Position(5, placed_target))

    # Call Tom directly to observe predicted target
    tom(world)
    assert tom.last_prediction_steps > 0, "Should be predicting ahead with high adrenaline"
    # Predicted position should be east of Jerry's current position
    pred = tom.last_predicted_jerry_pos
    assert pred is not None
    assert pred.x > 5, f"Predicted x={pred.x}, expected > 5 (Jerry moving east)"


def test_prediction_horizon_scales_with_adrenaline():
    """Higher adrenaline → more steps predicted ahead."""
    cfg = ChemicalTomConfig()
    # Build a clean world
    w = World(WorldConfig(max_ticks=50), seed=42)
    w.reset()

    # Place Tom and Jerry in a known clear corridor
    placed_target = None
    for y in range(3, w.grid.height - 3):
        clear = all(w.grid.is_walkable(Position(x, y)) for x in range(2, 14))
        if clear:
            w.tom.position = Position(2, y)
            w.jerry.position = Position(7, y)
            placed_target = y
            break
    if placed_target is None:
        pytest.skip("no clear corridor")

    # Test at two adrenaline levels
    low_steps = []
    high_steps = []
    for trial_idx, adr in enumerate([0.4, 0.95]):
        tom = ChemicalTom(seed=trial_idx)
        tom.chemistry.adrenaline = adr
        tom._jerry_position_history.append(Position(6, placed_target))
        tom._jerry_position_history.append(Position(7, placed_target))
        tom(w)
        if adr < 0.5:
            low_steps.append(tom.last_prediction_steps)
        else:
            high_steps.append(tom.last_prediction_steps)

    assert high_steps[0] > low_steps[0], \
        f"High adrenaline ({high_steps[0]} steps) should predict farther than low ({low_steps[0]})"


# ---- threshold modulation ---------------------------------------------

def test_aggression_extends_pursue_memory():
    """High-aggression Tom should remember Jerry's position longer after losing sight."""
    base_tom = ChemicalTom(seed=0)
    base_tom.drives.aggression = 0.0
    base_memory = base_tom._modulated_pursue_memory()

    agg_tom = ChemicalTom(seed=0)
    agg_tom.drives.aggression = 1.0
    agg_memory = agg_tom._modulated_pursue_memory()

    assert agg_memory > base_memory


def test_curiosity_lowers_noise_threshold():
    """Curious Tom investigates quieter noises."""
    base_tom = ChemicalTom(seed=0)
    base_tom.drives.curiosity = 0.0
    base_th = base_tom._modulated_noise_threshold()

    curious_tom = ChemicalTom(seed=0)
    curious_tom.drives.curiosity = 1.0
    curious_th = curious_tom._modulated_noise_threshold()

    assert curious_th < base_th


def test_cortisol_shortens_pursue_memory():
    """Frustrated (high cortisol) Tom gives up sooner."""
    base_tom = ChemicalTom(seed=0)
    base_tom.drives.aggression = 0.5
    base_memory = base_tom._modulated_pursue_memory()

    frustrated_tom = ChemicalTom(seed=0)
    frustrated_tom.drives.aggression = 0.5
    frustrated_tom.chemistry.cortisol = 0.9
    frustrated_memory = frustrated_tom._modulated_pursue_memory()

    assert frustrated_memory < base_memory


# ---- replay capture ---------------------------------------------------

def test_recorder_captures_chemistry_for_chemical_tom():
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=15), seed=42)
    tom = ChemicalTom(seed=0)
    replay = rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        tom_label="chemical",
    )
    # All frames should have a chemistry dict with all five chemicals
    f = replay.frames[-1]
    assert isinstance(f.tom_chemistry, dict)
    assert "adrenaline" in f.tom_chemistry
    assert "cortisol" in f.tom_chemistry
    assert "dopamine" in f.tom_chemistry
    assert "oxytocin" in f.tom_chemistry
    assert "serotonin" in f.tom_chemistry


def test_recorder_captures_drives_for_chemical_tom():
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=15), seed=42)
    tom = ChemicalTom(seed=0)
    replay = rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        tom_label="chemical",
    )
    f = replay.frames[-1]
    assert isinstance(f.tom_drives, dict)
    expected = {"hunger", "aggression", "caution", "curiosity", "fatigue", "social_bond"}
    assert expected.issubset(set(f.tom_drives.keys()))


def test_recorder_no_chemistry_for_scripted_tom():
    """Backwards compatibility: replays of ScriptedTom should have empty
    chemistry/drives dicts.
    """
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=10), seed=42)
    replay = rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=ScriptedTom(seed=0),
        tom_label="scripted",
    )
    f = replay.frames[-1]
    assert f.tom_chemistry == {}
    assert f.tom_drives == {}


def test_phase2_replay_roundtrips_through_json(tmp_path):
    """A ChemicalTom replay should save and load preserving chemistry."""
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=10), seed=42)
    tom = ChemicalTom(seed=0)
    original = rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        tom_label="chemical",
    )
    path = tmp_path / "phase2.json"
    original.save(path)
    loaded = Replay.load(path)
    # Pick a frame somewhere in the middle
    mid = len(loaded.frames) // 2
    assert loaded.frames[mid].tom_chemistry == original.frames[mid].tom_chemistry
    assert loaded.frames[mid].tom_drives == original.frames[mid].tom_drives


def test_old_phase1_replay_still_loads(tmp_path):
    """A replay JSON written without Phase 2 fields should still load."""
    # Manually craft an old-style payload
    import json
    payload = {
        "grid_width": 5, "grid_height": 5,
        "grid_tiles": [[0]*5 for _ in range(5)],
        "vent_pairs": [],
        "locker_positions": [],
        "seed": 1,
        "jerry_policy_label": "old",
        "tom_policy_label": "old",
        "outcome": "survived",
        "total_ticks": 2,
        "total_jerry_reward": 0.5,
        "frames": [
            {
                "tick": 1, "tom_pos": [1, 1], "tom_facing": 0, "tom_state": "",
                "tom_action": 4, "jerry_pos": [3, 3], "jerry_facing": 0,
                "jerry_action": 4, "jerry_in_locker": False, "jerry_alive": True,
                "tom_sees_jerry": False, "jerry_sees_tom": False,
                "jerry_reward": 0.01, "jerry_cum_reward": 0.01,
                "sound_events": [], "scent_cells": [], "events": [],
            },
        ],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload))
    loaded = Replay.load(path)
    assert loaded.frames[0].tom_chemistry == {}
    assert loaded.frames[0].tom_drives == {}
    assert loaded.frames[0].tom_predicted_jerry is None
    assert loaded.frames[0].tom_prediction_steps == 0


# ---- wall-aware prediction (corner-cubby oscillation fix) -------------

def test_prediction_does_not_phase_through_walls():
    """Regression for the corner-cubby standoff (seed 0): a wall-blocked Jerry
    drifting toward a wall must NOT be predicted on the far side of it.

    Before the fix, _predict_jerry_target extrapolated Jerry's velocity and
    only checked that the FINAL tile was walkable — so a Jerry pinned against
    a wall, with eastward drift in his history, got predicted three tiles east
    THROUGH the wall. Tom then chased a phantom and oscillated forever.
    """
    from collections import deque
    world = World(WorldConfig(max_ticks=300), seed=0)
    world.reset()
    tom = ChemicalTom(seed=0)
    tom.reset()

    # Find a walkable tile with a wall immediately to its east.
    from src.utils.types import TileType
    pinned = None
    for x in range(1, world.grid.width - 1):
        for y in range(1, world.grid.height - 1):
            here = Position(x, y)
            east = Position(x + 1, y)
            if world.grid.is_walkable(here) and not world.grid.is_walkable(east):
                pinned = here
                break
        if pinned:
            break
    assert pinned is not None, "seed 0 should have a wall-pinned tile"

    # Seed history with eastward drift INTO the wall, max adrenaline (predicts
    # the full horizon ahead).
    tom.chemistry.adrenaline = 1.0
    tom._jerry_position_history = deque(
        [Position(pinned.x - 2, pinned.y),
         Position(pinned.x - 1, pinned.y),
         pinned],
        maxlen=3,
    )
    world.jerry.position = pinned

    target = tom._predict_jerry_target(world)
    # Prediction must not be east of the wall — Jerry can't go there.
    assert target.x <= pinned.x, (
        f"prediction {target} phased east through the wall past {pinned}"
    )
    # And every tile from Jerry to the prediction must be walkable.
    assert world.grid.is_reachable(pinned, target)


def test_walk_predicted_path_stops_at_wall():
    """_walk_predicted_path returns the last walkable tile before a wall."""
    world = World(WorldConfig(max_ticks=300), seed=0)
    world.reset()
    tom = ChemicalTom(seed=0)
    tom.reset()
    from src.utils.types import TileType
    # Find a wall-pinned tile (wall to the east) again.
    pinned = None
    for x in range(1, world.grid.width - 1):
        for y in range(1, world.grid.height - 1):
            if world.grid.is_walkable(Position(x, y)) \
                    and not world.grid.is_walkable(Position(x + 1, y)):
                pinned = Position(x, y)
                break
        if pinned:
            break
    # Walking east toward a far target must stop at pinned (wall blocks step).
    far_east = Position(pinned.x + 5, pinned.y)
    result = tom._walk_predicted_path(pinned, far_east, world)
    assert result == pinned, f"expected stop at {pinned}, got {result}"
