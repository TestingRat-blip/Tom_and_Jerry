"""Phase 5 training — PPO Jerry archetypes vs ScriptedTom.

One script trains one archetype. Run it six times (or in a loop) to
produce the full archetype population:

    python -m scripts.train_phase5 --archetype generalist
    python -m scripts.train_phase5 --archetype sneaker
    python -m scripts.train_phase5 --archetype sprinter
    python -m scripts.train_phase5 --archetype trickster
    python -m scripts.train_phase5 --archetype camper
    python -m scripts.train_phase5 --archetype explorer

Outputs (under data/):
    snapshots/jerry_<archetype>/ckpt_<n>.zip
    snapshots/jerry_<archetype>/final.zip
    logs/jerry_<archetype>/                    — TensorBoard event files
    logs/jerry_<archetype>/eval_log.jsonl      — per-eval results
    logs/jerry_<archetype>/config.json         — frozen run config

Defaults to 1.5M timesteps to match the Phase 1 baseline budget. Each
archetype trains against the same fixed ScriptedTom (per the Phase 5
design decision — see Phase 5 design notes in DECISIONS.md or chat
history). This gives a clean apples-to-apples comparison.

For a sweep across all six archetypes, see `--all` (sequential) or
just script it in PowerShell yourself for parallelism.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.env.gym_env import (
    ARCHETYPE_NAMES,
    JerryEnv,
    JerryRewardConfig,
)
from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom


# ---- Tom opponent factory ----------------------------------------------

def build_training_tom(tom_spec: str, seed: int):
    """Build the Tom opponent a Jerry trains against.

    Supported:
      - "scripted":  ScriptedTom (BFS pathfinder) — the Phase 1-5 default.
      - "conductor": ChemicalTom + Conductor (the two-brain hunter). Used
                     to train a counter-Jerry against the Phase 6 hunter.

    The opponent is FIXED during a Jerry training run (it does not learn).
    This is single-sided training against a stationary hunter — the
    de-risking step before full alternating co-evolution (Stage 2).
    """
    if tom_spec == "scripted":
        return ScriptedTom(seed=seed)
    if tom_spec == "conductor":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor
        return ChemicalTom(conductor=Conductor(), seed=seed)
    raise SystemExit(f"unknown --tom spec: {tom_spec!r} (use scripted|conductor)")


# ---- env factory --------------------------------------------------------

def make_env(seed: int, world_max_ticks: int, archetype: str, tom_spec: str):
    """Factory closure for SB3 vector envs. Each call returns a NEW env
    configured for the given archetype, training against the given Tom.
    """
    def _init():
        env = JerryEnv(
            world_config=WorldConfig(max_ticks=world_max_ticks),
            reward_config=JerryRewardConfig.for_archetype(archetype),
            tom_policy=build_training_tom(tom_spec, seed),
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ---- eval callback ------------------------------------------------------

class EvalAgainstScriptedCallback(BaseCallback):
    """Periodically evaluate the current policy against ScriptedTom and
    log survival rate + average reward.

    Phase 5 note: we evaluate using the archetype's OWN reward config,
    so reported reward is comparable to the training reward signal. The
    SURVIVAL RATE, however, is reward-config-independent and is the
    primary cross-archetype comparison metric.
    """

    def __init__(
        self,
        archetype: str,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed: int,
        log_path: Path,
        world_max_ticks: int,
        tom_spec: str = "scripted",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.archetype = archetype
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.log_path = log_path
        self.world_max_ticks = world_max_ticks
        self.tom_spec = tom_spec
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
        self.logger.record(
            "eval/det_stoch_gap",
            stoch["survival_rate"] - det["survival_rate"],
        )

        # JSONL — record both for offline analysis
        record = {
            "archetype": self.archetype,
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
                f"[{self.archetype:>10} @ {self.num_timesteps:>9} steps] "
                f"det={det['survival_rate']:.0%}  stoch={stoch['survival_rate']:.0%}  "
                f"reward={det['mean_reward']:+.2f}  len={det['mean_length']:.0f}{gap_flag}"
            )

    def _eval_pass(self, deterministic: bool) -> dict:
        rewards: list[float] = []
        lengths: list[int] = []
        outcomes: list[str] = []
        seed_offset = (self.num_timesteps // max(self.eval_freq, 1)) * 13
        for i in range(self.n_eval_episodes):
            env_seed = self.eval_seed + seed_offset + i
            env = JerryEnv(
                world_config=WorldConfig(max_ticks=self.world_max_ticks),
                reward_config=JerryRewardConfig.for_archetype(self.archetype),
                tom_policy=build_training_tom(self.tom_spec, env_seed),
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
    p = argparse.ArgumentParser(
        description="Train PPO Jerry archetype vs ScriptedTom.",
    )
    p.add_argument(
        "--archetype",
        required=True,
        choices=ARCHETYPE_NAMES,
        help="Which Phase 5 archetype to train. Reward shape comes from "
             "JerryRewardConfig.for_archetype(...).",
    )
    p.add_argument(
        "--run-name",
        help="Override the default run name (default: jerry_<archetype> or "
             "jerry_<archetype>_vs_<tom> when --tom is not scripted). "
             "Useful for ablation runs with different hyperparams.",
    )
    p.add_argument(
        "--tom",
        default="scripted",
        choices=["scripted", "conductor"],
        help="Opponent Tom to train against. 'scripted' (default) is the "
             "Phase 1-5 BFS hunter. 'conductor' is the Phase 6 two-brain "
             "hunter — use this to train a counter-Jerry against the "
             "Conductor (the de-risking step before full co-evolution).",
    )
    p.add_argument("--timesteps", type=int, default=1_500_000,
                   help="Total PPO env steps. Phase 5 default 1.5M matches "
                        "the Phase 1 baseline budget.")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--world-max-ticks", type=int, default=300,
                   help="Training episode length cap.")
    p.add_argument("--eval-max-ticks", type=int, default=None)
    p.add_argument("--ckpt-freq", type=int, default=100_000,
                   help="Checkpoint every N env steps. Phase 5 default 100k "
                        "is sparser than Phase 1's 50k — at 1.5M total we "
                        "still get 15 checkpoints, enough to sweep for the "
                        "best one post-hoc.")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--ent-coef", type=float, default=0.01)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Default run name includes the opponent when it's not the default
    # scripted Tom, so a counter-Jerry doesn't overwrite the baseline
    # jerry_<archetype> snapshots.
    if args.run_name:
        run_name = args.run_name
    elif args.tom == "scripted":
        run_name = f"jerry_{args.archetype}"
    else:
        run_name = f"jerry_{args.archetype}_vs_{args.tom}"

    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    snapshots_dir = project_root / "data" / "snapshots" / run_name
    logs_dir = project_root / "data" / "logs" / run_name
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_log = logs_dir / "eval_log.jsonl"

    # Save config alongside the run for reproducibility
    config_dict = vars(args).copy()
    config_dict["run_name_resolved"] = run_name
    config_dict["started_at"] = time.time()
    (logs_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Vector env — every env uses the same archetype + opponent + a different seed
    env_fns = [
        make_env(args.seed + i, args.world_max_ticks, args.archetype, args.tom)
        for i in range(args.n_envs)
    ]
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
    eval_max_ticks = args.eval_max_ticks if args.eval_max_ticks is not None \
        else args.world_max_ticks

    eval_cb = EvalAgainstScriptedCallback(
        archetype=args.archetype,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 10_000,
        log_path=eval_log,
        world_max_ticks=eval_max_ticks,
        tom_spec=args.tom,
    )

    print(f"Training archetype {args.archetype!r} vs Tom {args.tom!r} "
          f"for {args.timesteps:,} steps across {args.n_envs} envs.")
    print(f"  Run name:     {run_name}")
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
