"""Evaluate trained Phase 5 archetypes against ScriptedTom (and optionally
ChemicalTom-L2) and produce a comparative table.

Run AFTER training all archetypes:

    python -m scripts.train_phase5 --archetype generalist
    ... (repeat for each)
    python -m scripts.eval_archetypes

By default this looks for `data/snapshots/jerry_<archetype>/final.zip` for
each archetype. Override the checkpoint path per-archetype with
`--ckpt <archetype>=<path>` if you want to evaluate a specific checkpoint
rather than the final model.

Output: a table to stdout, plus a JSON dump to
`data/logs/phase5_eval_<timestamp>.json` with the full per-archetype
per-tom-policy results.

Examples:
    # Eval all six archetypes (final checkpoints) vs ScriptedTom
    python -m scripts.eval_archetypes

    # Eval against BOTH ScriptedTom and ChemicalTom-L2 (frozen)
    python -m scripts.eval_archetypes --tom-policies scripted chemical-l2

    # 50 episodes per (archetype, tom) cell
    python -m scripts.eval_archetypes --episodes 50

    # Eval a SPECIFIC checkpoint for one archetype
    python -m scripts.eval_archetypes --ckpt sneaker=data/snapshots/jerry_sneaker/ckpt_500000_steps.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.env.gym_env import (
    ARCHETYPE_NAMES,
    JerryEnv,
    JerryRewardConfig,
)
from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom


# ---- args ---------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Phase 5 archetypes head-to-head.",
    )
    p.add_argument(
        "--archetypes",
        nargs="+",
        default=list(ARCHETYPE_NAMES),
        choices=ARCHETYPE_NAMES,
        help="Which archetypes to evaluate (default: all six).",
    )
    p.add_argument(
        "--tom-policies",
        nargs="+",
        default=["scripted"],
        choices=["scripted", "chemical", "chemical-l2", "conductor", "conductor-l2", "conductor-holddown"],
        help="Which Tom variants to evaluate against. Multiple => "
             "cross-product of (archetype, tom) cells. Default: scripted only.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Episodes per (archetype, tom) cell.",
    )
    p.add_argument(
        "--max-ticks",
        type=int,
        default=300,
        help="Per-episode tick cap.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for the eval episode loop.",
    )
    p.add_argument(
        "--ckpt",
        action="append",
        default=[],
        help="Override checkpoint per archetype: --ckpt sneaker=path.zip. "
             "Can be repeated. Defaults to data/snapshots/jerry_<archetype>/final.zip.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PPO inference. Default: stochastic (which "
             "is the metric we care about per ADR-011).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Don't write JSON output to data/logs/.",
    )
    p.add_argument(
        "--locker-oxygen",
        action="store_true",
        help="Enable the locker oxygen/cooldown mechanic in the eval env. "
             "Required when evaluating a Jerry trained with --locker-oxygen "
             "(its obs vector includes oxygen; the env must match).",
    )
    return p.parse_args(argv)


# ---- ckpt resolution ----------------------------------------------------

def resolve_checkpoints(
    archetypes: list[str],
    overrides: list[str],
    project_root: Path,
) -> dict[str, Path]:
    """For each requested archetype, return the path to its checkpoint.

    Default: data/snapshots/jerry_<archetype>/final.zip
    Override format: --ckpt archetype=path
    """
    paths: dict[str, Path] = {}
    override_map: dict[str, str] = {}
    for spec in overrides:
        if "=" not in spec:
            raise SystemExit(f"--ckpt requires format archetype=path, got {spec!r}")
        archetype, path_str = spec.split("=", 1)
        override_map[archetype.strip()] = path_str.strip()

    for archetype in archetypes:
        if archetype in override_map:
            p = Path(override_map[archetype])
        else:
            p = project_root / "data" / "snapshots" / f"jerry_{archetype}" / "final.zip"
        if not p.exists():
            raise SystemExit(
                f"checkpoint for {archetype!r} not found at {p}\n"
                f"  → run `python -m scripts.train_phase5 --archetype {archetype}` first, "
                f"or pass --ckpt {archetype}=<path>"
            )
        paths[archetype] = p
    return paths


# ---- Tom factory ---------------------------------------------------------

def make_tom_factory(spec: str, seed: int):
    """Return (factory, label) where factory(seed) -> a fresh Tom callable.

    Phase 5 evaluation supports three Tom variants, all built per-episode
    so they reset cleanly:
      - "scripted":    ScriptedTom
      - "chemical":    ChemicalTom (Phase 2)
      - "chemical-l2": ChemicalTom + L1 + L2 (Phase 4 full memory)

    chemical-l2 evaluates against the FROZEN L2 database at the project's
    default path. If you want the L2 to evolve during eval, use a separate
    script — this one is for head-to-head measurement, which requires a
    stationary opponent.

    Phase 6f adds two more:
      - "conductor":    ChemicalTom + Conductor (the two-brain hunter,
                        no persistent memory)
      - "conductor-l2": ChemicalTom + Conductor + frozen L2 memory
    """
    if spec == "scripted":
        def factory(s: int):
            return ScriptedTom(seed=s)
        return factory, "scripted"

    if spec == "chemical":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom

        def factory(s: int):
            return ChemicalTom(seed=s)
        return factory, "chemical"

    if spec == "conductor":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor

        def factory(s: int):
            return ChemicalTom(conductor=Conductor(), seed=s)
        return factory, "conductor"

    if spec == "conductor-holddown":
        # Component 3: Conductor with hold-on-LOS-break / run-down forced on.
        # The cheap-experiment hunter for testing the cover-dance counter.
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor, ConductorConfig

        def factory(s: int):
            cfg = ConductorConfig(hold_on_los_break=True)
            return ChemicalTom(conductor=Conductor(config=cfg), seed=s)
        return factory, "conductor-holddown"

    if spec == "conductor-l2":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor
        from src.hunter.agent.memory.l1 import L1Memory
        from src.hunter.agent.memory.l2_lookup import L2Lookup
        from src.persistence.redis.client import FakeRedis, RedisClient
        from src.persistence.sqlite.client import SQLiteClient
        from src.persistence.sqlite.l2_store import L2Store

        sqlite_client = SQLiteClient()
        l2_store = L2Store(sqlite_client)
        l2_lookup = L2Lookup(l2_store)
        prior_count = l2_store.count()
        print(f"  [eval setup] conductor-l2 reading L2 with {prior_count} prior summaries")

        def factory(s: int):
            l1 = L1Memory(
                client=RedisClient(client=FakeRedis()),
                episode_id=f"eval_{uuid.uuid4().hex[:8]}",
            )
            return ChemicalTom(
                l1=l1, l2_lookup=l2_lookup, l2_store=None,
                conductor=Conductor(), seed=s,
            )
        return factory, "conductor-l2"

    if spec == "chemical-l2":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.memory.l1 import L1Memory
        from src.hunter.agent.memory.l2_lookup import L2Lookup
        from src.persistence.redis.client import FakeRedis, RedisClient
        from src.persistence.sqlite.client import SQLiteClient
        from src.persistence.sqlite.l2_store import L2Store

        # NOTE: we use FakeRedis (L1 is per-episode anyway) and a SHARED
        # SQLiteClient across all envs in this eval. The L2 is read-only
        # for the eval's purposes since we explicitly do NOT pass l2_store
        # (only l2_lookup) — so each archetype's eval sees the same priors
        # without polluting them.
        sqlite_client = SQLiteClient()
        l2_store = L2Store(sqlite_client)
        l2_lookup = L2Lookup(l2_store)
        prior_count = l2_store.count()
        print(f"  [eval setup] chemical-l2 reading L2 with {prior_count} prior summaries")

        def factory(s: int):
            l1 = L1Memory(
                client=RedisClient(client=FakeRedis()),
                episode_id=f"eval_{uuid.uuid4().hex[:8]}",
            )
            # l2_store deliberately NOT passed → frozen L2, no writes
            return ChemicalTom(
                l1=l1, l2_lookup=l2_lookup, l2_store=None, seed=s,
            )
        return factory, "chemical-l2"

    raise SystemExit(f"unknown tom policy: {spec!r}")


# ---- eval loop ----------------------------------------------------------

def evaluate_cell(
    archetype: str,
    ckpt_path: Path,
    tom_factory,
    tom_label: str,
    n_episodes: int,
    max_ticks: int,
    base_seed: int,
    deterministic: bool,
    locker_oxygen: bool = False,
) -> dict[str, Any]:
    """Run one (archetype, tom_policy) cell. Returns aggregated stats."""
    from stable_baselines3 import PPO
    model = PPO.load(str(ckpt_path), device="cpu")
    reward_cfg = JerryRewardConfig.for_archetype(archetype)

    rewards: list[float] = []
    lengths: list[int] = []
    outcomes: list[str] = []

    for i in range(n_episodes):
        seed = base_seed + i
        tom = tom_factory(seed)
        env = JerryEnv(
            world_config=WorldConfig(max_ticks=max_ticks,
                                     locker_oxygen_enabled=locker_oxygen),
            reward_config=reward_cfg,
            tom_policy=tom,
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
        "archetype": archetype,
        "tom_policy": tom_label,
        "n_episodes": n_episodes,
        "n_survived": n_survived,
        "n_caught": len(outcomes) - n_survived,
        "survival_rate": n_survived / len(outcomes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "ckpt_path": str(ckpt_path),
    }


# ---- table printing -----------------------------------------------------

def print_table(results: list[dict[str, Any]], tom_policies: list[str]) -> None:
    """Pretty-print a per-archetype-per-tom table."""
    archetypes = sorted({r["archetype"] for r in results},
                        key=lambda a: ARCHETYPE_NAMES.index(a))

    # Header
    cells_per_tom = ["survive%", "len", "reward"]
    header_main = f"{'archetype':<12}"
    header_sub = f"{'':12}"
    for tom in tom_policies:
        header_main += f"  {tom:<24}"
        header_sub += f"  {' '.join(f'{c:>7}' for c in cells_per_tom):<24}"
    print()
    print(header_main)
    print(header_sub)
    print("-" * len(header_sub))

    for archetype in archetypes:
        line = f"{archetype:<12}"
        for tom in tom_policies:
            cell = next(
                (r for r in results if r["archetype"] == archetype
                 and r["tom_policy"] == tom), None,
            )
            if cell is None:
                line += f"  {'(missing)':<24}"
                continue
            cell_str = (
                f"{cell['survival_rate']:>6.0%} "
                f"{cell['mean_length']:>6.0f} "
                f"{cell['mean_reward']:>+7.2f}"
            )
            line += f"  {cell_str:<24}"
        print(line)
    print()


# ---- main ---------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]

    # Resolve checkpoints
    ckpts = resolve_checkpoints(args.archetypes, args.ckpt, project_root)

    print(f"Phase 5 evaluation — "
          f"{len(args.archetypes)} archetypes × {len(args.tom_policies)} Toms "
          f"× {args.episodes} episodes")
    print(f"  Total cells:       {len(args.archetypes) * len(args.tom_policies)}")
    print(f"  Inference mode:    "
          f"{'deterministic' if args.deterministic else 'stochastic'}")
    print(f"  Per-episode ticks: {args.max_ticks}")
    print()
    print("Checkpoints:")
    for archetype, p in ckpts.items():
        print(f"  {archetype:<12} {p}")
    print()

    # Build tom factories ONCE (chemical-l2 needs to share its SQLiteClient
    # across all envs in the eval)
    tom_factories: dict[str, tuple] = {}
    for spec in args.tom_policies:
        factory, label = make_tom_factory(spec, seed=args.seed)
        tom_factories[label] = (factory, label)

    results: list[dict[str, Any]] = []
    t_start = time.time()
    for archetype in args.archetypes:
        for tom_label, (factory, _) in tom_factories.items():
            t_cell = time.time()
            cell = evaluate_cell(
                archetype=archetype,
                ckpt_path=ckpts[archetype],
                tom_factory=factory,
                tom_label=tom_label,
                n_episodes=args.episodes,
                max_ticks=args.max_ticks,
                base_seed=args.seed,
                deterministic=args.deterministic,
                locker_oxygen=args.locker_oxygen,
            )
            cell_dt = time.time() - t_cell
            cell["elapsed_seconds"] = cell_dt
            results.append(cell)
            print(
                f"  {archetype:<12} vs {tom_label:<14} "
                f"survival={cell['survival_rate']:.0%}  "
                f"reward={cell['mean_reward']:+.2f}  "
                f"({cell_dt:.1f}s)"
            )

    elapsed = time.time() - t_start
    print()
    print(f"All cells done in {elapsed:.1f}s")

    print_table(results, list(tom_factories.keys()))

    # Save results
    if not args.no_save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = project_root / "data" / "logs" / f"phase5_eval_{timestamp}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "timestamp": timestamp,
            "args": vars(args),
            "results": results,
        }, indent=2))
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
