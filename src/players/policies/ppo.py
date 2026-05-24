"""Thin wrapper around a trained Stable Baselines3 PPO model.

Exposes the same callable interface as ScriptedTom: `policy(world)` returns
an Action. This is what makes tournaments work — any policy
(scripted or learned) can be plugged into the env, and any combination
of policies can fight each other.

Two perspectives are supported:
  - PPOJerryPolicy: consumes Jerry-shaped observations
  - PPOTomPolicy:   consumes Tom-shaped observations (used in Phase 4+)

Both wrap a single PPO model and dispatch to the right observation
extractor based on which agent's vector the model expects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.env.world.world import World
from src.utils.types import Action


class PPOJerryPolicy:
    """Callable wrapper around a trained PPO model trained AS Jerry.

    Usage:
        policy = PPOJerryPolicy.load("data/snapshots/jerry_gen_42.zip")
        env = JerryEnv(...)
        obs, _ = env.reset()
        action = policy.from_obs(obs)
        # OR, for use as a "world policy" (symmetric with ScriptedTom):
        action = policy(world)  # extracts Jerry's obs internally

    The deterministic flag controls whether action sampling uses argmax
    (deterministic=True) or samples from the policy distribution
    (deterministic=False). For evaluation, default True. For training-time
    exploration, use False (but we don't typically call this during PPO's
    own learning — that's handled by SB3 internally).
    """

    def __init__(self, model: PPO, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> "PPOJerryPolicy":
        model = PPO.load(str(path))
        return cls(model, deterministic=deterministic)

    def from_obs(self, obs: np.ndarray) -> Action:
        """Use directly when you already have Jerry's obs vector."""
        action_int, _ = self.model.predict(obs, deterministic=self.deterministic)
        return Action(int(action_int))

    def __call__(self, world: World) -> Action:
        """Use as a world policy. Extracts Jerry's obs from the world."""
        jerry_obs = world._observe_jerry()
        return self.from_obs(jerry_obs.to_vector())

    def reset(self) -> None:
        """No per-episode state to clear. Here for interface symmetry
        with ScriptedTom — env.reset() calls this if present.
        """
        return None


class PPOTomPolicy:
    """Same as PPOJerryPolicy but consumes Tom's obs vector.

    Used in Phase 4+ when a Tom policy is trained via RL. In Phase 1
    this class exists but is unused — included now so the symmetry is
    visible in code from day one.
    """

    def __init__(self, model: PPO, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> "PPOTomPolicy":
        model = PPO.load(str(path))
        return cls(model, deterministic=deterministic)

    def from_obs(self, obs: np.ndarray) -> Action:
        action_int, _ = self.model.predict(obs, deterministic=self.deterministic)
        return Action(int(action_int))

    def __call__(self, world: World) -> Action:
        tom_obs = world._observe_tom()
        return self.from_obs(tom_obs.to_vector())

    def reset(self) -> None:
        return None
