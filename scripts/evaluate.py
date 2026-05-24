"""Evaluate a trained Jerry against ScriptedTom over N episodes.

Usage:
    python -m scripts.evaluate \\
        --model data/snapshots/baseline_v1/final.zip \\
        --episodes 200

For comparison, run with --random to evaluate a random-action Jerry,
which gives the floor we need to beat. Phase 1 is "done" when a trained
Jerry survives meaningfully more often than the random baseline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from src.env.gym_env import JerryEnv
from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.players.policies.ppo import PPOJerryPolicy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Jerry policy.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str,
                   help="Path to a saved PPO model (.zip).")
    g.add_argument("--random", action="store_true",
                   help="Use a uniform random policy. Baseline floor.")
    g.add_argument("--passive", action="store_true",
                   help="Always WAIT. The 80% catch-rate baseline.")

    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--world-max-ticks", type=int, default=600)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--report-json", type=str, default=None,
                   help="If set, write the result dict as JSON here too.")
    return p.parse_args(argv)


def run_evaluation(
    policy_fn,
    n_episodes: int,
    world_max_ticks: int,
    seed: int,
) -> dict:
    """Run policy against ScriptedTom for n_episodes and return stats.

    policy_fn: callable(obs_vector) -> int (action). Receives the Jerry
               obs vector each step and returns an action int.
    """
    rewards: list[float] = []
    lengths: list[int] = []
    outcomes: list[str] = []
    seen_counts: list[int] = []  # how many times Tom saw Jerry per episode

    for ep in range(n_episodes):
        env = JerryEnv(
            world_config=WorldConfig(max_ticks=world_max_ticks),
            tom_policy=ScriptedTom(seed=seed + ep),
        )
        obs, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        steps = 0
        seen = 0
        while True:
            action = policy_fn(obs)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_r += r
            steps += 1
            # Event type 0 = TOM_SAW_JERRY (per env.EventType)
            seen += sum(1 for e in info["events"] if e == 0)
            if terminated or truncated:
                outcomes.append(info["episode"]["outcome"])
                break
        rewards.append(total_r)
        lengths.append(steps)
        seen_counts.append(seen)

    survival_rate = sum(1 for o in outcomes if o == "survived") / n_episodes
    caught = sum(1 for o in outcomes if o == "caught")
    return {
        "n_episodes": n_episodes,
        "survival_rate": survival_rate,
        "caught": caught,
        "survived": n_episodes - caught,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "mean_seen_count": float(np.mean(seen_counts)),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.model:
        print(f"Loading model from {args.model}")
        policy = PPOJerryPolicy.load(args.model, deterministic=True)
        policy_fn = policy.from_obs
        label = f"PPO ({Path(args.model).name})"
    elif args.random:
        rng = random.Random(args.seed)
        policy_fn = lambda _obs: rng.randint(0, 5)
        label = "random-action floor"
    elif args.passive:
        policy_fn = lambda _obs: 4  # Action.WAIT
        label = "passive (always WAIT)"
    else:
        raise RuntimeError("unreachable")

    print(f"\nEvaluating: {label}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Max ticks: {args.world_max_ticks}")
    print(f"  Seed:      {args.seed}\n")

    result = run_evaluation(
        policy_fn=policy_fn,
        n_episodes=args.episodes,
        world_max_ticks=args.world_max_ticks,
        seed=args.seed,
    )

    print(f"  Survival rate:      {result['survival_rate']:.1%}  "
          f"({result['survived']}/{result['n_episodes']})")
    print(f"  Mean reward:        {result['mean_reward']:+.2f}  "
          f"(std {result['std_reward']:.2f})")
    print(f"  Mean episode len:   {result['mean_length']:.0f} ticks  "
          f"(median {result['median_length']:.0f})")
    print(f"  Avg times seen:     {result['mean_seen_count']:.1f}")

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(result, indent=2))
        print(f"\nReport written to {args.report_json}")


if __name__ == "__main__":
    main(sys.argv[1:])
