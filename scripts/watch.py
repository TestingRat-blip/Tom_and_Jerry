"""Record a single episode and watch it.

Usage examples:
    # Watch the baseline ScriptedTom vs a random Jerry
    python -m scripts.watch --jerry random --tom scripted

    # Watch your trained Jerry play
    python -m scripts.watch --jerry model:data/snapshots/jerry_v1_baseline.zip --tom scripted

    # Save the replay to disk for later
    python -m scripts.watch --jerry random --tom scripted --save data/replays/run1.json

    # Replay a saved file (skip recording entirely)
    python -m scripts.watch --replay data/replays/run1.json

    # Record without rendering (for batch capture)
    python -m scripts.watch --jerry random --tom scripted --no-render --save data/replays/run1.json

    # Force deterministic OR stochastic PPO inference (default: deterministic).
    python -m scripts.watch --jerry model:path.zip --tom scripted --stochastic
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.baseline import ScriptedTom
from src.render.replay.recorder import Replay, ReplayRecorder
from src.utils.types import Action


# ---- policy factories ---------------------------------------------------

def _make_jerry_policy(spec: str, deterministic: bool, seed: int):
    """Resolve a policy spec string into (callable, label).

    Supported specs:
      "random"          uniform random action
      "passive"         always WAIT
      "model:PATH"      a saved PPO Jerry checkpoint
    """
    if spec == "random":
        rng = random.Random(seed)
        return (lambda obs, world: rng.randint(0, 5), "random")
    if spec == "passive":
        return (lambda obs, world: int(Action.WAIT), "passive")
    if spec.startswith("model:"):
        from src.players.policies.ppo import PPOJerryPolicy
        path = spec[len("model:"):]
        policy = PPOJerryPolicy.load(path, deterministic=deterministic)
        label = f"ppo({Path(path).stem})"
        return (policy, label)
    raise SystemExit(f"unknown jerry policy spec: {spec!r}")


def _make_tom_policy(spec: str, seed: int):
    """Resolve a Tom policy spec.

    Supported:
      "scripted"        the Phase 1 baseline (ScriptedTom)
      "chemical"        Phase 2 ChemicalTom (drives + chemistry + prediction)
      "chemical-l1"     ChemicalTom + L1 per-encounter memory (Phase 3)
                        requires Redis at the default config
                        (localhost:6380, db=1, namespace tj:)
      "chemical-l2"     ChemicalTom + L1 + L2 persistent memory (Phase 4)
                        requires Redis (as chemical-l1) AND writes to
                        data/persistence/tj_l2.db. Each run of watch.py
                        warm-starts from past sessions and writes a new
                        summary at the end — so successive runs against
                        the same map+jerry build up persistent memory.
      "wait"            no-op Tom (useful for watching Jerry behavior alone)
      "model:PATH"      a saved PPO Tom checkpoint (Phase 4+; here for symmetry)
    """
    if spec == "scripted":
        tom = ScriptedTom(seed=seed)
        return (tom, "scripted")
    if spec == "chemical":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        tom = ChemicalTom(seed=seed)
        return (tom, "chemical")
    if spec == "conductor":
        # Phase 6: ChemicalTom + Conductor (two-brain hunter). No Redis
        # needed — the Conductor's belief is per-episode in-memory.
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor
        tom = ChemicalTom(conductor=Conductor(), seed=seed)
        return (tom, "conductor")
    if spec == "conductor-holddown":
        # Component 3: Conductor with hold-on-LOS-break / run-down FORCED
        # ON. This is the cheap-experiment hunter — used to test whether the
        # run-down behavior counters the cover-dance Jerry. Normally this
        # behavior is deployed by memory, not always-on.
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor, ConductorConfig
        cfg = ConductorConfig(hold_on_los_break=True)
        tom = ChemicalTom(conductor=Conductor(config=cfg), seed=seed)
        return (tom, "conductor-holddown")
    if spec == "chemical-l1":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.memory.l1 import L1Memory
        from src.persistence.redis.client import RedisClient
        import uuid
        # Each watch session gets its own L1 episode_id so multiple
        # sessions don't share keyspace.
        client = RedisClient()
        try:
            client.ping()
        except Exception as e:
            raise SystemExit(
                f"chemical-l1 requires Redis, but ping failed: {e}\n"
                f"Start it with: docker compose up -d"
            )
        l1 = L1Memory(client, episode_id=f"watch_{uuid.uuid4().hex[:8]}")
        tom = ChemicalTom(l1=l1, seed=seed)
        return (tom, "chemical-l1")
    if spec == "chemical-l2":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.memory.l1 import L1Memory
        from src.hunter.agent.memory.l2_lookup import L2Lookup
        from src.persistence.redis.client import RedisClient
        from src.persistence.sqlite.client import SQLiteClient
        from src.persistence.sqlite.l2_store import L2Store
        import uuid
        # Redis check (same as chemical-l1)
        redis_client = RedisClient()
        try:
            redis_client.ping()
        except Exception as e:
            raise SystemExit(
                f"chemical-l2 requires Redis, but ping failed: {e}\n"
                f"Start it with: docker compose up -d"
            )
        # SQLite — uses the default path data/persistence/tj_l2.db.
        # The SQLiteClient creates parent dirs and migrates schema on first use.
        sqlite_client = SQLiteClient()
        l2_store = L2Store(sqlite_client)
        l2_lookup = L2Lookup(l2_store)
        l1 = L1Memory(redis_client,
                      episode_id=f"watch_l2_{uuid.uuid4().hex[:8]}")
        tom = ChemicalTom(
            l1=l1, l2_lookup=l2_lookup, l2_store=l2_store, seed=seed,
        )
        print(f"  [chemical-l2] L2 has {l2_store.count()} prior episode summaries")
        return (tom, "chemical-l2")
    if spec == "wait":
        return (lambda world: int(Action.WAIT), "wait")
    if spec.startswith("model:"):
        from src.players.policies.ppo import PPOTomPolicy
        path = spec[len("model:"):]
        policy = PPOTomPolicy.load(path, deterministic=True)
        label = f"ppo({Path(path).stem})"
        return (policy, label)
    raise SystemExit(f"unknown tom policy spec: {spec!r}")


# ---- main ---------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record and watch a single episode.")
    p.add_argument("--replay", type=str, default=None,
                   help="Watch an existing replay file. Skips recording.")
    p.add_argument("--jerry", type=str, default="random",
                   help="Jerry policy: 'random', 'passive', or 'model:PATH'.")
    p.add_argument("--tom", type=str, default="scripted",
                   help="Tom policy: 'scripted', 'wait', or 'model:PATH'.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ticks", type=int, default=600)
    p.add_argument("--stochastic", action="store_true",
                   help="If Jerry is a PPO model, use stochastic sampling.")
    p.add_argument("--save", type=str, default=None,
                   help="Save the recorded replay to this JSON path.")
    p.add_argument("--no-render", action="store_true",
                   help="Don't open the renderer window. Useful with --save.")
    p.add_argument("--tile-px", type=int, default=32)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --replay path: just open it
    if args.replay:
        from src.render.pygame_renderer.renderer import RenderConfig, watch_replay
        watch_replay(args.replay, config=RenderConfig(tile_px=args.tile_px))
        return

    # Otherwise: record then watch
    jerry, jerry_label = _make_jerry_policy(
        args.jerry, deterministic=not args.stochastic, seed=args.seed
    )
    tom, tom_label = _make_tom_policy(args.tom, seed=args.seed)

    print(f"Recording episode: jerry={jerry_label}  tom={tom_label}  seed={args.seed}")
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=args.max_ticks),
                         seed=args.seed)
    replay = rec.record_episode(
        jerry_policy=jerry, tom_policy=tom,
        jerry_label=jerry_label, tom_label=tom_label,
    )
    print(f"  Ticks:   {replay.total_ticks}")
    print(f"  Outcome: {replay.outcome}")
    print(f"  Reward:  {replay.total_jerry_reward:+.2f}")

    if args.save:
        replay.save(args.save)
        print(f"Saved replay to {args.save}")

    if not args.no_render:
        from src.render.pygame_renderer.renderer import RenderConfig, watch_replay
        watch_replay(replay, config=RenderConfig(tile_px=args.tile_px))


if __name__ == "__main__":
    main(sys.argv[1:])
