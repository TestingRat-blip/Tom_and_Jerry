"""Smoke tests for the PPO policy wrappers.

These do NOT train a model — too slow for unit tests. They confirm:
  - The wrapper class is constructable.
  - The reset() method exists (interface symmetry with ScriptedTom).
  - Loading a saved model gives back a working policy.

The end-to-end pipeline (train + load + eval) is exercised in
scripts/train_phase1.py + scripts/evaluate.py, not here.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.env.gym_env import JerryEnv
from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.players.policies.ppo import PPOJerryPolicy, PPOTomPolicy
from src.utils.types import Action


@pytest.fixture
def tiny_trained_model(tmp_path):
    """Train a near-zero-step PPO model just to produce a saveable
    checkpoint we can load back. The model is essentially random
    initialization with one update.
    """
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=50),
        tom_policy=ScriptedTom(seed=0),
    )
    env = Monitor(env)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    model.learn(total_timesteps=64)
    path = tmp_path / "tiny.zip"
    model.save(str(path))
    return path


def test_ppo_jerry_policy_loads(tiny_trained_model):
    policy = PPOJerryPolicy.load(tiny_trained_model)
    assert policy.model is not None


def test_ppo_jerry_policy_has_reset(tiny_trained_model):
    """Interface symmetry with ScriptedTom — env.reset() will call this
    if present.
    """
    policy = PPOJerryPolicy.load(tiny_trained_model)
    assert hasattr(policy, "reset")
    policy.reset()  # no-op, should not raise


def test_ppo_jerry_policy_returns_valid_action(tiny_trained_model):
    policy = PPOJerryPolicy.load(tiny_trained_model)
    env = JerryEnv(tom_policy=ScriptedTom(seed=0))
    obs, _ = env.reset(seed=0)
    action = policy.from_obs(obs)
    assert isinstance(action, Action)


def test_ppo_jerry_policy_callable_with_world(tiny_trained_model):
    """The policy should be callable with a World, returning an Action
    (symmetric with ScriptedTom)."""
    policy = PPOJerryPolicy.load(tiny_trained_model)
    env = JerryEnv(tom_policy=ScriptedTom(seed=0))
    env.reset(seed=0)
    action = policy(env.world)
    assert isinstance(action, Action)


def test_ppo_tom_policy_exists(tiny_trained_model):
    """PPOTomPolicy is unused in Phase 1 but must exist as a class
    (mirrors PPOJerryPolicy). Phase 4+ uses it.
    """
    policy = PPOTomPolicy.load(tiny_trained_model)
    assert hasattr(policy, "reset")
    assert hasattr(policy, "from_obs")
    assert hasattr(policy, "__call__")
