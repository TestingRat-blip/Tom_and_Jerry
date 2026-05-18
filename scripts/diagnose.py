"""Diagnose what a Jerry policy is actually doing.

Runs N episodes and reports:
  - Action distribution (is Jerry just spamming WAIT?)
  - Average reward decomposition (where does reward come from / leak?)
  - Per-episode patterns (caught early? mid? late?)
  - Fraction of ticks Tom can see Jerry
  - Fraction of ticks Jerry is in a locker
  - Episode-length histogram
  - Deterministic vs stochastic eval gap (the smoking gun for argmax collapse)

Usage:
    python -m scripts.diagnose --model data/snapshots/jerry_v1_full/final.zip
    python -m scripts.diagnose --random  # baseline floor
    python -m scripts.diagnose --passive # the 80% catch baseline
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.env.gym_env import JerryEnv, JerryRewardConfig
from src.env.world.world import EventType, WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.utils.types import Action


def _action_name(a: int) -> str:
    return Action(a).name


def run_diagnostic(
    policy_fn,
    n_episodes: int,
    world_max_ticks: int,
    seed: int,
    label: str,
) -> dict:
    """policy_fn: callable(obs) -> int (action)."""
    action_counts: Counter = Counter()
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    outcomes: list[str] = []
    ticks_seen: list[int] = []        # per-episode, ticks Tom saw Jerry
    ticks_in_locker: list[int] = []   # per-episode, ticks Jerry hid
    catch_tick: list[int] = []        # only for caught episodes — when?
    noise_events: list[int] = []      # per-episode noise events emitted

    for ep in range(n_episodes):
        env = JerryEnv(
            world_config=WorldConfig(max_ticks=world_max_ticks),
            tom_policy=ScriptedTom(seed=seed + ep),
        )
        obs, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        ep_len = 0
        ep_seen = 0
        ep_locker = 0
        ep_noise = 0

        while True:
            action = int(policy_fn(obs))
            action_counts[action] += 1
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            ep_len += 1
            # Tally events
            for et in info["events"]:
                if et == EventType.TOM_SAW_JERRY:
                    ep_seen += 1
                elif et == EventType.NOISE_EMITTED:
                    # NOISE_EMITTED counts noise from both — filter by actor
                    # Faster path: skip — we approximate with all noise events.
                    # (We could filter by checking the events list with full
                    # Event objects, but ints-only is faster.)
                    ep_noise += 1
            if info["jerry_in_locker"]:
                ep_locker += 1
            if terminated or truncated:
                outcomes.append(info["episode"]["outcome"])
                if terminated:
                    catch_tick.append(ep_len)
                break

        episode_lengths.append(ep_len)
        episode_rewards.append(ep_reward)
        ticks_seen.append(ep_seen)
        ticks_in_locker.append(ep_locker)
        noise_events.append(ep_noise)

    total_actions = sum(action_counts.values())
    action_dist = {
        _action_name(a): action_counts[a] / total_actions
        for a in range(len(Action))
    }

    survival_rate = sum(1 for o in outcomes if o == "survived") / n_episodes

    return {
        "label": label,
        "n_episodes": n_episodes,
        "survival_rate": survival_rate,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "median_length": float(np.median(episode_lengths)),
        "mean_seen_pct": float(np.mean([s/l if l else 0
                                        for s, l in zip(ticks_seen, episode_lengths)])),
        "mean_locker_pct": float(np.mean([l/t if t else 0
                                          for l, t in zip(ticks_in_locker, episode_lengths)])),
        "mean_noise_events": float(np.mean(noise_events)),
        "median_catch_tick": float(np.median(catch_tick)) if catch_tick else float("nan"),
        "action_dist": action_dist,
    }


def print_report(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Diagnostic: {result['label']}")
    print(f"{'='*60}")
    print(f"  Episodes:             {result['n_episodes']}")
    print(f"  Survival rate:        {result['survival_rate']:.1%}")
    print(f"  Mean reward:          {result['mean_reward']:+.2f}  "
          f"(std {result['std_reward']:.2f})")
    print(f"  Mean episode length:  {result['mean_length']:.0f} ticks  "
          f"(median {result['median_length']:.0f})")
    if not np.isnan(result['median_catch_tick']):
        print(f"  Median catch tick:    {result['median_catch_tick']:.0f}")
    print(f"  % ticks Tom saw:      {result['mean_seen_pct']:.1%}")
    print(f"  % ticks in locker:    {result['mean_locker_pct']:.1%}")
    print(f"  Mean noise events:    {result['mean_noise_events']:.1f} per episode")
    print(f"\n  Action distribution:")
    for name, pct in sorted(result["action_dist"].items(),
                            key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(pct * 40)
        print(f"    {name:<10} {pct:>6.1%}  {bar}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose a Jerry policy.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--random", action="store_true")
    g.add_argument("--passive", action="store_true")

    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--world-max-ticks", type=int, default=600)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--compare-deterministic", action="store_true",
                   help="If model is given, run both deterministic and "
                        "stochastic policies and compare. This catches "
                        "argmax-collapse failures.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.random:
        rng = random.Random(args.seed)
        policy_fn = lambda _obs: rng.randint(0, 5)
        result = run_diagnostic(policy_fn, args.episodes,
                                args.world_max_ticks, args.seed,
                                "random-action floor")
        print_report(result)
        return

    if args.passive:
        policy_fn = lambda _obs: 4  # WAIT
        result = run_diagnostic(policy_fn, args.episodes,
                                args.world_max_ticks, args.seed,
                                "passive (always WAIT)")
        print_report(result)
        return

    # PPO model
    from src.players.policies.ppo import PPOJerryPolicy
    name = Path(args.model).name

    print(f"Loading model from {args.model}")
    det_policy = PPOJerryPolicy.load(args.model, deterministic=True)
    det_fn = lambda obs: int(det_policy.from_obs(obs))
    det_result = run_diagnostic(det_fn, args.episodes,
                                args.world_max_ticks, args.seed,
                                f"PPO ({name}) — deterministic")
    print_report(det_result)

    if args.compare_deterministic:
        stoch_policy = PPOJerryPolicy.load(args.model, deterministic=False)
        stoch_fn = lambda obs: int(stoch_policy.from_obs(obs))
        stoch_result = run_diagnostic(stoch_fn, args.episodes,
                                      args.world_max_ticks, args.seed,
                                      f"PPO ({name}) — stochastic")
        print_report(stoch_result)

        # Headline comparison
        print(f"\n{'='*60}")
        print("Deterministic vs Stochastic gap")
        print(f"{'='*60}")
        det_s = det_result["survival_rate"]
        st_s = stoch_result["survival_rate"]
        print(f"  Deterministic survival: {det_s:.1%}")
        print(f"  Stochastic survival:    {st_s:.1%}")
        print(f"  Gap:                    {st_s - det_s:+.1%}")
        if abs(st_s - det_s) > 0.10:
            if st_s > det_s:
                # Stochastic better → argmax collapse: policy concentrated on
                # bad actions, sampling lets it escape sometimes.
                print("  ⚠ Stochastic > Deterministic by >10%.")
                print("    This is ARGMAX COLLAPSE — the policy's mode is bad,")
                print("    sampling rescues it occasionally. Retrain with more")
                print("    entropy or check that the reward function is well-shaped.")
            else:
                # Deterministic better → learned exploit: precise tactic that
                # noise breaks. May be a real strategy, may be a brittle exploit.
                print("  ⚠ Deterministic > Stochastic by >10%.")
                print("    This is EXPLOIT CONVERGENCE — the policy has found")
                print("    a precise tactic that sampling noise breaks. Could be")
                print("    a real strategy (e.g. coordinated movement) or a")
                print("    brittle exploit of the opponent. Inspect the action")
                print("    distribution to decide. NOT an indictment of the")
                print("    policy — just means deterministic eval is the right")
                print("    measurement for this checkpoint.")


if __name__ == "__main__":
    main(sys.argv[1:])
