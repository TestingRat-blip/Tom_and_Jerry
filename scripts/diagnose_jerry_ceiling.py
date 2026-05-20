"""Diagnose WHY a trained Jerry caps at its observed survival rate.

The question: a trained generalist Jerry survives ~30% against ScriptedTom.
Is that 30% a property of JERRY (a beatable policy ceiling) or a property
of the HUNTER/ENVIRONMENT (30% is near-optimal play in a setup that just
doesn't allow better)?

This script holds Jerry FIXED (no retraining) and varies the conditions:

  1. baseline   — ScriptedTom (BFS), normal sensors, 30x30 map.
                  Reproduces the standard eval. Reference point.
  2. greedy     — ScriptedTom forced to GREEDY pathfinding (no BFS).
                  Phase 1 retro found greedy drops catch ~80%->~50% on
                  passive Jerry. If survival JUMPS here, BFS hunter
                  strength was the ceiling.
  3. nearsight  — ScriptedTom with halved sight range (10 -> 5). A hunter
                  that loses track of Jerry constantly. Orthogonal weaken.
  4. bigmap     — Normal ScriptedTom (BFS) on a 45x45 map with more
                  lockers. If survival JUMPS here, the ENVIRONMENT (map
                  size / hiding spots) was the ceiling, not Jerry or Tom.

Reading the results:
  - greedy/nearsight raise survival a lot  -> hunter strength is the
    ceiling. Retraining Jerry vs the strong Tom may find a better policy.
  - bigmap raises survival but greedy/nearsight don't -> the MAP is the
    constraint. "Good base Jerry" means accepting ~30% on 30x30.
  - NOTHING moves survival much -> the current Jerry policy is near its
    ceiling on this setup. 30% is "good"; seed co-evolution as-is.

Usage:
    python -m scripts.diagnose_jerry_ceiling \\
        --jerry data/snapshots/jerry_generalist/final.zip \\
        --episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.env.gym_env import JerryEnv, JerryRewardConfig
from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.utils.types import Action, Position


# ---- greedy variant of ScriptedTom ------------------------------------

class GreedyScriptedTom(ScriptedTom):
    """ScriptedTom with BFS disabled — greedy manhattan stepping only.

    Overrides _step_toward to skip the BFS path search and go straight to
    the greedy fallback. Makes Tom prone to getting stuck behind walls and
    losing efficient pursuit lines — the documented ~80%->~50% catch rate
    drop from the Phase 1 retro.
    """

    def _step_toward(self, src: Position, dst: Position, world: World) -> Action:
        if src == dst:
            return Action.WAIT
        return self._greedy_step_toward(src, dst, world)


# ---- conditions --------------------------------------------------------

def make_conditions(max_ticks: int) -> dict:
    """Return {name: (world_config_factory, tom_factory)} for each condition."""
    return {
        "baseline": (
            lambda: WorldConfig(max_ticks=max_ticks),
            lambda s: ScriptedTom(seed=s),
        ),
        "greedy": (
            lambda: WorldConfig(max_ticks=max_ticks),
            lambda s: GreedyScriptedTom(seed=s),
        ),
        "nearsight": (
            lambda: WorldConfig(max_ticks=max_ticks, tom_sight_range=5),
            lambda s: ScriptedTom(seed=s),
        ),
        "bigmap": (
            lambda: WorldConfig(
                max_ticks=max_ticks,
                grid_width=45, grid_height=45,
                n_lockers=12, n_vent_pairs=5,
            ),
            lambda s: ScriptedTom(seed=s),
        ),
    }


# ---- eval --------------------------------------------------------------

def run_condition(
    name: str,
    world_config_factory,
    tom_factory,
    model,
    n_episodes: int,
    base_seed: int,
    deterministic: bool,
) -> dict:
    rewards: list[float] = []
    lengths: list[int] = []
    outcomes: list[str] = []

    for i in range(n_episodes):
        seed = base_seed + i
        env = JerryEnv(
            world_config=world_config_factory(),
            reward_config=JerryRewardConfig.generalist(),
            tom_policy=tom_factory(seed),
        )
        obs, _ = env.reset(seed=seed)
        total_r = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_r += r
            steps += 1
            if terminated or truncated:
                outcomes.append(info["episode"]["outcome"])
                break
        rewards.append(total_r)
        lengths.append(steps)

    n_survived = sum(1 for o in outcomes if o == "survived")
    return {
        "condition": name,
        "n_episodes": n_episodes,
        "survival_rate": n_survived / len(outcomes),
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


# ---- main --------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose a trained Jerry's survival ceiling.",
    )
    p.add_argument(
        "--jerry",
        default="data/snapshots/jerry_generalist/final.zip",
        help="Path to the Jerry checkpoint to diagnose.",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic inference. Default: stochastic (our primary metric).",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "greedy", "nearsight", "bigmap"],
        choices=["baseline", "greedy", "nearsight", "bigmap"],
        help="Which conditions to run (default: all four).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    jerry_path = Path(args.jerry)
    if not jerry_path.exists():
        raise SystemExit(f"Jerry checkpoint not found: {jerry_path}")

    from stable_baselines3 import PPO
    model = PPO.load(str(jerry_path), device="cpu")

    all_conditions = make_conditions(args.max_ticks)
    conditions = {k: all_conditions[k] for k in args.conditions}

    print(f"Jerry ceiling diagnostic")
    print(f"  Jerry:       {jerry_path}")
    print(f"  Episodes:    {args.episodes} per condition")
    print(f"  Inference:   {'deterministic' if args.deterministic else 'stochastic'}")
    print(f"  Conditions:  {', '.join(conditions.keys())}")
    print()

    results = []
    for name, (wc_factory, tom_factory) in conditions.items():
        res = run_condition(
            name=name,
            world_config_factory=wc_factory,
            tom_factory=tom_factory,
            model=model,
            n_episodes=args.episodes,
            base_seed=args.seed,
            deterministic=args.deterministic,
        )
        results.append(res)
        print(
            f"  {name:<11} survival={res['survival_rate']:>5.0%}  "
            f"reward={res['mean_reward']:>+7.2f}  "
            f"len={res['mean_length']:>6.1f}"
        )

    # Interpretation
    print()
    print("=" * 56)
    baseline = next((r for r in results if r["condition"] == "baseline"), None)
    if baseline is None:
        print("(no baseline condition run — skipping interpretation)")
        return

    base_sr = baseline["survival_rate"]
    print(f"Baseline survival: {base_sr:.0%}")
    print()
    print("Deltas vs baseline (positive = Jerry survives MORE = condition")
    print("weakens the hunt):")
    for r in results:
        if r["condition"] == "baseline":
            continue
        delta = r["survival_rate"] - base_sr
        flag = ""
        if delta >= 0.20:
            flag = "  <-- LARGE: this factor was a major ceiling"
        elif delta >= 0.10:
            flag = "  <-- moderate"
        print(f"  {r['condition']:<11} {delta:>+5.0%}{flag}")

    print()
    print("Interpretation guide:")
    print("  greedy/nearsight large  -> hunter strength caps survival; retraining may help")
    print("  bigmap large (others not)-> map size/hiding caps survival; not a Jerry problem")
    print("  nothing large           -> Jerry is near its ceiling here; 30% is 'good'")


if __name__ == "__main__":
    main(sys.argv[1:])
