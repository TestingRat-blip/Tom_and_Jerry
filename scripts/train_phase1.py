"""Phase 1 training — PPO Jerry vs scripted baseline Tom.

Usage:
    python -m scripts.train_phase1 --timesteps 500000 --run-name baseline_v1

Outputs (under data/):
    snapshots/<run_name>/ckpt_<n>.zip   — checkpoints every CKPT_FREQ
    snapshots/<run_name>/final.zip      — final model
    logs/<run_name>/                    — TensorBoard event files
    logs/<run_name>/eval_log.jsonl      — per-eval results

This is the Phase 1 exit criterion: a trained Jerry that survives
ScriptedTom meaningfully more often than a random-action Jerry does.

Throughput target on the 3060 Ti box: ~1000 episodes/hour minimum.
With 8 parallel envs at ~3400 eps/hr each = expect 15000-25000 eps/hr
during training, depending on PPO inference cost and SB3 overhead.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.env.gym_env import JerryEnv, JerryRewardConfig
from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom


# ---- env factory --------------------------------------------------------

def make_env(seed: int, world_max_ticks: int = 600):
    """Factory closure for SB3 vector envs. Each call returns a NEW env."""
    def _init():
        env = JerryEnv(
            world_config=WorldConfig(max_ticks=world_max_ticks),
            reward_config=JerryRewardConfig(),
            tom_policy=ScriptedTom(seed=seed),
        )
        env = Monitor(env)  # SB3 wrapper for episode reward / length logging
        env.reset(seed=seed)
        return env
    return _init


# ---- eval callback ------------------------------------------------------

class EvalAgainstBaselineCallback(BaseCallback):
    """Periodically run the current policy against ScriptedTom and log
    survival rate and average reward.

    SB3 has a built-in EvalCallback but ours adds outcome breakdown
    (caught vs survived) and writes JSONL for easy plotting later.
    """

    def __init__(
        self,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed: int,
        log_path: Path,
        world_max_ticks: int = 600,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.log_path = log_path
        self.world_max_ticks = world_max_ticks
        self._n_calls_since_eval = 0

    def _on_step(self) -> bool:
        self._n_calls_since_eval += self.training_env.num_envs
        if self._n_calls_since_eval < self.eval_freq:
            return True
        self._n_calls_since_eval = 0
        self._run_eval()
        return True

    def _run_eval(self) -> None:
        det = self._eval_pass(deterministic=True)
        stoch = self._eval_pass(deterministic=False)

        # TensorBoard logging
        self.logger.record("eval/survival_rate_det", det["survival_rate"])
        self.logger.record("eval/survival_rate_stoch", stoch["survival_rate"])
        self.logger.record("eval/mean_reward_det", det["mean_reward"])
        self.logger.record("eval/mean_reward_stoch", stoch["mean_reward"])
        self.logger.record("eval/mean_episode_length_det", det["mean_length"])
        self.logger.record("eval/det_stoch_gap", stoch["survival_rate"] - det["survival_rate"])

        # JSONL — record both for offline analysis
        record = {
            "timesteps": int(self.num_timesteps),
            "survival_rate_det": det["survival_rate"],
            "survival_rate_stoch": stoch["survival_rate"],
            "mean_reward_det": det["mean_reward"],
            "mean_reward_stoch": stoch["mean_reward"],
            "mean_length_det": det["mean_length"],
            "n_episodes": self.n_eval_episodes,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if self.verbose:
            gap = stoch["survival_rate"] - det["survival_rate"]
            gap_flag = " ⚠" if abs(gap) > 0.10 else ""
            print(
                f"[eval @ {self.num_timesteps:>9} steps] "
                f"det={det['survival_rate']:.0%}  stoch={stoch['survival_rate']:.0%}  "
                f"reward={det['mean_reward']:+.2f}  len={det['mean_length']:.0f}{gap_flag}"
            )

    def _eval_pass(self, deterministic: bool) -> dict:
        rewards: list[float] = []
        lengths: list[int] = []
        outcomes: list[str] = []
        # Rotate eval seeds each cycle so we measure GENERAL survival,
        # not memorization of fixed seeds. Offset by timesteps so the
        # eval set drifts steadily through the seed space.
        seed_offset = (self.num_timesteps // max(self.eval_freq, 1)) * 13
        for i in range(self.n_eval_episodes):
            env_seed = self.eval_seed + seed_offset + i
            env = JerryEnv(
                world_config=WorldConfig(max_ticks=self.world_max_ticks),
                tom_policy=ScriptedTom(seed=env_seed),
            )
            obs, _ = env.reset(seed=env_seed)
            total_r = 0.0
            steps = 0
            while True:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, r, terminated, truncated, info = env.step(int(action))
                total_r += r
                steps += 1
                if terminated or truncated:
                    outcomes.append(info["episode"]["outcome"])
                    break
            rewards.append(total_r)
            lengths.append(steps)
        return {
            "survival_rate": sum(1 for o in outcomes if o == "survived") / len(outcomes),
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
        }


# ---- main ---------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO Jerry vs scripted Tom.")
    p.add_argument("--run-name", required=True,
                   help="Run identifier; outputs go under data/.../<run_name>/")
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="Total PPO env steps to train for.")
    p.add_argument("--n-envs", type=int, default=8,
                   help="Parallel envs. Higher = better GPU utilization.")
    p.add_argument("--subproc", action="store_true",
                   help="Use SubprocVecEnv (one process per env). "
                        "Faster on multi-core boxes but heavier to debug.")
    p.add_argument("--world-max-ticks", type=int, default=300,
                   help="Training episode length cap. Shorter = more terminal "
                        "reward signals per env step. v2 default 300 (was 600).")
    p.add_argument("--eval-max-ticks", type=int, default=None,
                   help="Eval episode length cap. Defaults to world-max-ticks. "
                        "Set this LONGER than training to measure long-survival, "
                        "or shorter to make survival easier to achieve.")
    p.add_argument("--ckpt-freq", type=int, default=50_000,
                   help="Checkpoint every N env steps (across all envs combined).")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024,
                   help="PPO rollout length per env before each update.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.995,
                   help="Discount factor. High because survival is long-horizon.")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="Entropy coefficient — higher = more action diversity. "
                        "Default 0.01 is 10x SB3's default of 0.0 to combat "
                        "the argmax-collapse failure mode we observed in v1.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    snapshots_dir = project_root / "data" / "snapshots" / args.run_name
    logs_dir = project_root / "data" / "logs" / args.run_name
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_log = logs_dir / "eval_log.jsonl"

    # Save config alongside the run for reproducibility
    (logs_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Vector env
    env_fns = [make_env(args.seed + i, args.world_max_ticks)
               for i in range(args.n_envs)]
    if args.subproc:
        venv = SubprocVecEnv(env_fns)
    else:
        venv = DummyVecEnv(env_fns)

    # PPO
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(logs_dir),
        seed=args.seed,
    )

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.ckpt_freq // args.n_envs, 1),
        save_path=str(snapshots_dir),
        name_prefix="ckpt",
    )
    # Default eval ticks to training ticks if not specified
    eval_max_ticks = args.eval_max_ticks if args.eval_max_ticks is not None \
        else args.world_max_ticks

    eval_cb = EvalAgainstBaselineCallback(
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 10_000,
        log_path=eval_log,
        world_max_ticks=eval_max_ticks,
    )

    print(f"Training {args.run_name} for {args.timesteps:,} steps "
          f"across {args.n_envs} envs.")
    print(f"  TensorBoard:  tensorboard --logdir {logs_dir}")
    print(f"  Eval log:     {eval_log}")
    print(f"  Snapshots:    {snapshots_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb],
        tb_log_name="ppo",
        progress_bar=True,
    )

    final_path = snapshots_dir / "final.zip"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
