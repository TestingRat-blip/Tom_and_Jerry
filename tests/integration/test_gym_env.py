"""Integration tests for the Gymnasium env wrapper and scripted Tom."""
from __future__ import annotations

import random

import numpy as np
import pytest

import gymnasium as gym

from src.env.gym_env import JerryEnv, JerryRewardConfig
from src.env.world.world import EventType, WorldConfig
from src.hunter.agent.behavior.baseline import (
    ScriptedTom,
    ScriptedTomConfig,
    TomState,
)
from src.utils.types import Action


# ---- gymnasium compliance ---------------------------------------------

def test_env_is_gym_env_subclass():
    env = JerryEnv()
    assert isinstance(env, gym.Env)


def test_observation_space_box():
    env = JerryEnv()
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32


def test_action_space_discrete_six():
    env = JerryEnv()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == len(Action)


def test_reset_returns_obs_and_info():
    env = JerryEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_five_tuple():
    env = JerryEnv()
    env.reset(seed=42)
    result = env.step(Action.WAIT.value)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_observation_shape_consistent_across_steps():
    env = JerryEnv()
    obs, _ = env.reset(seed=42)
    expected_shape = obs.shape
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == expected_shape
        if terminated or truncated:
            break


def test_seed_makes_reset_deterministic():
    env1 = JerryEnv()
    env2 = JerryEnv()
    obs1, _ = env1.reset(seed=999)
    obs2, _ = env2.reset(seed=999)
    assert np.array_equal(obs1, obs2)


def test_episode_terminates_on_catch_or_timeout():
    """Random Jerry vs. ScriptedTom should end every episode within bounds."""
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=100),
        tom_policy=ScriptedTom(seed=0),
    )
    rng = random.Random(0)
    for ep in range(10):
        env.reset(seed=ep)
        steps = 0
        while True:
            _, _, terminated, truncated, info = env.step(rng.randint(0, 5))
            steps += 1
            if terminated or truncated:
                assert "episode" in info
                assert info["episode"]["outcome"] in ("caught", "survived")
                break
            assert steps <= 100


# ---- reward behavior ---------------------------------------------------

def test_reward_negative_on_catch():
    """When ScriptedTom catches Jerry, reward should be strongly negative
    on the terminal step.
    """
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=200),
        tom_policy=ScriptedTom(seed=0),
        reward_config=JerryRewardConfig(),
    )
    rng = random.Random(0)
    # Run until catch (or skip if rare)
    for trial in range(20):
        env.reset(seed=trial)
        caught = False
        last_reward = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(rng.randint(0, 5))
            last_reward = reward
            if terminated:
                caught = True
                break
            if truncated:
                break
        if caught:
            assert last_reward < 0
            return
    pytest.skip("No catch occurred in 20 trials — adjust seed")


def test_reward_positive_on_survive():
    """When Jerry runs out the clock (Tom is a no-op), the terminal
    reward should include the survival bonus.
    """
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=20),
        # Default Tom waits → Jerry should easily survive
    )
    env.reset(seed=42)
    final_reward = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(Action.WAIT.value)
        final_reward = reward
        if terminated or truncated:
            break
    # Tom was waiting → Jerry survived → terminal reward should be positive
    assert final_reward > 0


# ---- scripted Tom -----------------------------------------------------

def test_scripted_tom_resets_state():
    tom = ScriptedTom(seed=0)
    tom.last_seen_jerry = Position_helper(5, 5)
    tom.last_seen_tick = 100
    tom.state = TomState.PURSUE
    tom.reset()
    assert tom.last_seen_jerry is None
    assert tom.state == TomState.PATROL


def test_scripted_tom_chooses_valid_action():
    """Over many ticks, every action ScriptedTom returns must be a valid
    Action enum value.
    """
    tom = ScriptedTom(seed=0)
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=50),
        tom_policy=tom,
    )
    env.reset(seed=42)
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        # Tom's last decided action should always be a valid Action
        assert isinstance(tom.last_decided_action, Action)
        if terminated or truncated:
            break


def test_scripted_tom_pursues_when_visible():
    """If Jerry is placed directly visible to Tom on the same row,
    ScriptedTom should choose to move toward him (not away).
    """
    tom = ScriptedTom(seed=0)
    env = JerryEnv(tom_policy=tom)
    env.reset(seed=42)

    # Find a row with a contiguous run of at least 6 clear tiles
    # (so there are no walls in between Tom and Jerry).
    from src.utils.types import Position
    world = env.world
    placed = False
    for y in range(2, world.grid.height - 2):
        run_start = None
        run_len = 0
        best_start, best_len = None, 0
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
        if best_len >= 6 and best_start is not None:
            world.tom.position = Position(best_start, y)
            world.jerry.position = Position(best_start + 3, y)
            placed = True
            break
    if not placed:
        pytest.skip("could not find a contiguous clear row of 6 tiles")

    # Sanity: they must actually see each other
    assert world._tom_can_see_jerry(), \
        "test setup failed: Tom doesn't see Jerry on the chosen row"

    initial_dist = world.tom.position.manhattan(world.jerry.position)
    env.step(Action.WAIT.value)
    new_dist = world.tom.position.manhattan(world.jerry.position)
    assert new_dist <= initial_dist, \
        f"Tom moved away when Jerry was visible (state={tom.state.name})"


def test_scripted_tom_catches_passive_jerry_eventually():
    """The defining benchmark: a Jerry that always WAITs should be caught
    by ScriptedTom within a reasonable number of episodes.

    If this fails, the baseline Tom isn't a real threat and our control
    group is broken.
    """
    tom = ScriptedTom(seed=0)
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=400),
        tom_policy=tom,
    )

    caught_count = 0
    n_trials = 10
    for trial in range(n_trials):
        env.reset(seed=trial)
        while True:
            _, _, terminated, truncated, info = env.step(Action.WAIT.value)
            if terminated or truncated:
                if info["episode"]["outcome"] == "caught":
                    caught_count += 1
                break
    # Against a Jerry that never moves, ScriptedTom should catch on the
    # majority of episodes. He has scent, hearing, sight, and patrol.
    # We allow some misses because some seeds may put them across hard walls.
    assert caught_count >= n_trials * 0.6, \
        f"ScriptedTom only caught {caught_count}/{n_trials} passive Jerrys"


# ---- ADR-009: Tom observation logging ---------------------------------

def test_tom_observation_logging_off_by_default():
    env = JerryEnv(tom_policy=ScriptedTom(seed=0))
    env.reset(seed=42)
    _, _, _, _, info = env.step(Action.WAIT.value)
    assert "tom_log" not in info


def test_tom_observation_logging_on_when_requested():
    """Per ADR-009 — when log_tom_observations=True, info should contain
    Tom's would-be RL observation and the action chosen.
    """
    env = JerryEnv(
        tom_policy=ScriptedTom(seed=0),
        log_tom_observations=True,
    )
    env.reset(seed=42)
    _, _, _, _, info = env.step(Action.WAIT.value)
    assert "tom_log" in info
    tom_log = info["tom_log"]
    assert "tom_obs" in tom_log
    assert "tom_action" in tom_log
    assert isinstance(tom_log["tom_obs"], np.ndarray)
    assert tom_log["tom_obs"].dtype == np.float32
    assert 0 <= tom_log["tom_action"] < len(Action)


# ---- helper -----------------------------------------------------------

def Position_helper(x: int, y: int):
    """Indirection to dodge a top-level circular-import quirk in tests."""
    from src.utils.types import Position
    return Position(x, y)
