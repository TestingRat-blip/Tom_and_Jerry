"""Memory-adaptation loop driver — does Tom learn to counter the cover-dance?

This is the decisive experiment for the memory-as-substrate thesis. It runs
a fixed Jerry against a memory-equipped ChemicalTom+Conductor over a series
of episodes, with warm-start BEFORE and distillation AFTER each episode, so
Tom's L2 memory accumulates across the run. As memory fills with the Jerry's
LOS-break signature, the Conductor should DEPLOY the hold-on-LOS-break
run-down (selectively — only because memory says this prey is a cover-dancer)
and Jerry's survival should fall.

Contrast with the always-on experiment (eval_archetypes --tom conductor-
holddown), which gave only a 4% improvement because the run-down fired
indiscriminately. The question here: does SELECTIVE, memory-deployed
run-down do better?

Usage:
    python -m scripts.memory_loop \
        --jerry model:data/snapshots/jerry_generalist_vs_conductor/final.zip \
        --episodes 60 --block 10

Reports survival per block of episodes so you can see the adaptation curve:
early blocks (cold memory) should look like plain conductor; later blocks
(warm memory) should show the run-down deployed and survival dropping — IF
the memory loop works.

A fresh in-memory L2 is used by default (each run starts cold) so the
adaptation curve is clean. Pass --db to persist/accumulate across runs.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.conductor import Conductor
from src.hunter.agent.memory.l1 import L1Memory
from src.hunter.agent.memory.l2_lookup import L2Lookup
from src.persistence.redis.client import FakeRedis, RedisClient
from src.persistence.sqlite.client import SQLiteClient, SQLiteConfig
from src.persistence.sqlite.l2_store import L2Store
from src.utils.types import Action


def make_jerry(spec: str, seed: int):
    if spec == "random":
        import random
        rng = random.Random(seed)
        return (lambda world: rng.randint(0, 5)), "random"
    if spec == "passive":
        return (lambda world: int(Action.WAIT)), "passive"
    if spec.startswith("model:"):
        from src.players.policies.ppo import PPOJerryPolicy
        path = spec[len("model:"):]
        policy = PPOJerryPolicy.load(path, deterministic=False)
        return policy, f"ppo({Path(path).stem})"
    raise SystemExit(f"unknown jerry spec: {spec!r}")


def run_episode(world, tom, jerry_policy, max_ticks):
    """Run one episode. Returns (survived: bool, ticks, deployed: bool)."""
    obs = world.reset()
    tom.reset()
    # Warm-start reads L2 → may deploy the run-down for this Jerry.
    tom.warm_start_for_episode(
        world.grid, jerry_policy=jerry_policy,
        jerry_label=getattr(jerry_policy, "label", None),
    )
    deployed = bool(getattr(tom.conductor, "runtime_hold_on_los_break", False))
    if hasattr(jerry_policy, "reset"):
        jerry_policy.reset()

    for _ in range(max_ticks):
        ta = tom(world)
        ja = jerry_policy(world)
        world.step(tom_action=ta, jerry_action=int(ja))
        if not world.jerry.alive:
            break
    survived = world.jerry.alive
    return survived, world.tick_count, deployed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jerry", required=True,
                   help="Jerry spec: model:PATH | random | passive")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--block", type=int, default=10,
                   help="Report survival per block of this many episodes.")
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--db", default=None,
                   help="L2 DB path. Default: a fresh temp DB (cold start).")
    args = p.parse_args(argv)

    jerry_policy, jerry_label = make_jerry(args.jerry, args.seed)

    db_path = Path(args.db) if args.db else Path(tempfile.mktemp(suffix=".db"))
    sql = SQLiteClient(SQLiteConfig(db_path=db_path))
    store = L2Store(sql)
    lookup = L2Lookup(store)

    print(f"Memory-adaptation loop: {jerry_label} vs ChemicalTom+Conductor")
    print(f"  episodes={args.episodes} block={args.block} max_ticks={args.max_ticks}")
    print(f"  L2 DB: {db_path}{' (fresh)' if not args.db else ''}")
    print(f"  Jerry fingerprint label: {jerry_label}\n")

    block_survived = 0
    block_deployed = 0
    block_n = 0
    overall_survived = 0

    for ep in range(args.episodes):
        # Fresh L1 per episode (per-encounter memory); L2 persists.
        l1 = L1Memory(
            client=RedisClient(client=FakeRedis()),
            episode_id=f"loop_{ep}",
        )
        tom = ChemicalTom(
            l1=l1, l2_lookup=lookup, l2_store=store,
            conductor=Conductor(), seed=args.seed + ep,
        )
        world = World(WorldConfig(max_ticks=args.max_ticks), seed=args.seed + ep)

        survived, ticks, deployed = run_episode(
            world, tom, jerry_policy, args.max_ticks)

        # Distill this episode into L2 (feeds future warm-starts).
        tom.distill_at_episode_end(
            world.grid, jerry_policy=jerry_policy,
            outcome="survived" if survived else "caught",
            total_ticks=ticks,
            total_jerry_reward=0.0,  # reward not needed for this experiment
            jerry_label=jerry_label,
        )

        block_survived += int(survived)
        block_deployed += int(deployed)
        overall_survived += int(survived)
        block_n += 1

        if block_n == args.block:
            sr = block_survived / block_n
            dr = block_deployed / block_n
            lo = ep - block_n + 1
            print(f"  episodes {lo:3d}-{ep:3d}: survival={sr:4.0%}  "
                  f"run-down deployed in {dr:4.0%} of episodes")
            block_survived = block_deployed = block_n = 0

    print(f"\nOverall survival: {overall_survived/args.episodes:.0%} "
          f"({overall_survived}/{args.episodes})")
    print("\nRead the curve: if the memory loop works, later blocks should show")
    print("the run-down deploying (as memory fills) and survival falling vs early")
    print("blocks. Compare overall survival to the always-on 36% and plain 40%.")

    if not args.db:
        db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
