"""Phase 4 demonstration: show L2 persistent memory across episodes.

Runs N consecutive episodes on the same map+jerry combo, with a single
ChemicalTom that has L1 (FakeRedis) + L2 (SQLite) wired up. After each
episode, prints what got distilled and (for ep 2+) what got warm-started.

Usage:
    # Default: 5 episodes against passive Jerry on seed 42
    python -m scripts.show_l2_effect

    # Custom episode count + map seed
    python -m scripts.show_l2_effect --episodes 10 --seed 17

    # Use a real PPO Jerry checkpoint
    python -m scripts.show_l2_effect --jerry-model data/snapshots/jerry_v1_baseline.zip

    # Use real Redis instead of FakeRedis (matches the chemical-l1 watch.py path)
    python -m scripts.show_l2_effect --real-redis

    # Pick a custom SQLite database (default: data/persistence/tj_l2_demo.db)
    python -m scripts.show_l2_effect --db data/persistence/my_run.db

    # Wipe the database before starting (default: append to existing)
    python -m scripts.show_l2_effect --fresh

The script prints a structured per-episode log and a final summary. It
does NOT render replays — for visual inspection, use scripts/watch.py
with --tom chemical-l2.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.memory.l1 import L1Memory
from src.hunter.agent.memory.l2_lookup import L2Lookup
from src.persistence.redis.client import FakeRedis, RedisClient
from src.persistence.sqlite.client import SQLiteClient, SQLiteConfig
from src.persistence.sqlite.l2_store import L2Store
from src.render.replay.recorder import ReplayRecorder
from src.utils.types import Action


DEFAULT_DB_PATH = Path("data/persistence/tj_l2_demo.db")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 L2 memory demonstration.")
    p.add_argument("--episodes", type=int, default=5,
                   help="how many consecutive episodes to run (default 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="map seed — same for every episode (default 42)")
    p.add_argument("--max-ticks", type=int, default=400,
                   help="episode timeout (default 400)")
    p.add_argument("--jerry-label", default="demo_passive_jerry",
                   help="label used for jerry fingerprint (default 'demo_passive_jerry')")
    p.add_argument("--jerry-model",
                   help="path to a PPO Jerry checkpoint (overrides passive Jerry)")
    p.add_argument("--real-redis", action="store_true",
                   help="use real Redis at localhost:6380 (default: FakeRedis)")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH),
                   help=f"SQLite DB path (default {DEFAULT_DB_PATH})")
    p.add_argument("--fresh", action="store_true",
                   help="wipe L2 before starting (default: append)")
    return p.parse_args(argv)


# ---- helpers ----------------------------------------------------------

def _build_jerry(args):
    """Return (jerry_callable, jerry_label) for the episode loop."""
    if args.jerry_model:
        from src.players.policies.ppo import PPOJerryPolicy
        policy = PPOJerryPolicy.load(args.jerry_model, deterministic=True)
        label = Path(args.jerry_model).stem
        return policy, label
    # Passive Jerry — simplest baseline for the demo
    return (lambda obs, world: int(Action.WAIT)), args.jerry_label


def _build_redis_client(use_real: bool) -> RedisClient:
    if use_real:
        client = RedisClient()
        try:
            client.ping()
        except Exception as e:
            print(f"ERROR: --real-redis but Redis not reachable: {e}")
            print("  Start it with: docker compose up -d")
            sys.exit(1)
        return client
    return RedisClient(client=FakeRedis())


def _format_top(entries, n: int = 3, label_kind: str = "tiles") -> str:
    """Render a list of (Position, weight) or {(x,y): weight} as a short string."""
    if isinstance(entries, dict):
        items = sorted(entries.items(), key=lambda kv: kv[1], reverse=True)
        return ", ".join(f"({x},{y})={w:.2f}" for (x, y), w in items[:n]) or "—"
    # list of (Position, weight)
    if not entries:
        return "—"
    return ", ".join(f"({p.x},{p.y})={w:.2f}" for p, w in entries[:n])


# ---- main loop --------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # SQLite setup
    db_path = Path(args.db)
    client = SQLiteClient(SQLiteConfig(db_path=db_path))
    store = L2Store(client)
    lookup = L2Lookup(store)
    if args.fresh:
        n_deleted = store.delete_all()
        print(f"Wiped {n_deleted} existing L2 entries from {db_path}")

    # Redis setup (FakeRedis by default — L1 is per-episode anyway, so the
    # only effect of using real Redis is to make the demo match production)
    redis_client = _build_redis_client(args.real_redis)
    redis_kind = "real Redis" if args.real_redis else "FakeRedis (in-memory)"

    # One L1 for the whole run — we reset it each episode, so it's
    # effectively per-episode but the connection is reused.
    l1 = L1Memory(
        client=redis_client,
        episode_id=f"demo_{uuid.uuid4().hex[:8]}",
    )

    tom = ChemicalTom(
        l1=l1,
        l2_lookup=lookup,
        l2_store=store,
        seed=0,
    )

    jerry, jerry_label = _build_jerry(args)

    # Header
    print()
    print("=" * 70)
    print(f"Phase 4 demonstration — {args.episodes} episodes on seed {args.seed}")
    print("=" * 70)
    print(f"  SQLite DB:     {db_path}")
    print(f"  Redis backend: {redis_kind}")
    print(f"  Jerry:         {jerry_label}")
    print(f"  Tom:           ChemicalTom + L1 + L2 (full Phase 4)")
    print(f"  Max ticks:     {args.max_ticks}")
    print(f"  L2 size at start: {store.count()}")
    print()

    # Track aggregates for the final summary
    outcomes: list[str] = []
    first_sight_ticks: list[int] = []

    for ep in range(1, args.episodes + 1):
        rec = ReplayRecorder(
            world_config=WorldConfig(max_ticks=args.max_ticks),
            seed=args.seed,
        )

        # Snapshot warm-start state RIGHT BEFORE the episode (recorder
        # will trigger warm_start internally; we observe it BEFORE that
        # by running the same code path manually here)
        from src.env.world.world import World
        from src.hunter.agent.memory.fingerprint import (
            fingerprint_jerry,
            fingerprint_map,
        )
        peek_world = World(WorldConfig(max_ticks=args.max_ticks), seed=args.seed)
        peek_world.reset()
        fine_fp, coarse_fp = fingerprint_map(peek_world.grid)
        jerry_fp = fingerprint_jerry(jerry, label=jerry_label)
        peek_warm = lookup.build_warm_start(fine_fp, coarse_fp, jerry_fp)

        # Run the episode (recorder triggers warm-start + distillation)
        replay = rec.record_episode(
            jerry_policy=jerry,
            tom_policy=tom,
            jerry_label=jerry_label,
            tom_label="phase4_demo_tom",
        )

        # Read back the freshly-distilled summary
        latest = store.query_fine(fine_fp, jerry_fp, limit=1)
        s = latest[0] if latest else None

        # Per-episode log
        print(f"Episode {ep}/{args.episodes}:")
        print(f"  Outcome:               {replay.outcome}")
        print(f"  Ticks:                 {replay.total_ticks}")
        print(f"  Ticks to first sight:  "
              f"{s.ticks_to_first_sight if s and s.ticks_to_first_sight is not None else '(never)'}")
        print(f"  Episode reward:        {replay.total_jerry_reward:+.2f}")
        # Warm-start at the START of this episode
        if peek_warm.is_empty:
            print(f"  Warm-start (this ep):  empty (no prior episodes)")
        else:
            print(
                f"  Warm-start (this ep):  "
                f"{peek_warm.total_episodes} prior episodes "
                f"({peek_warm.fine_episode_count} fine + "
                f"{peek_warm.coarse_episode_count} coarse)"
            )
            print(f"    Top heatmap priors:    {_format_top(peek_warm.heatmap)}")
            print(f"    Top locker priors:     {_format_top(peek_warm.lockers)}")
            print(f"    Top false-noise priors:{_format_top(peek_warm.false_noise)}")
        # What this episode added to L2
        if s:
            print(
                f"  Distilled this episode:"
                f" {len(s.heatmap_top)} heatmap, {len(s.lockers)} lockers, "
                f"{len(s.false_noise_top)} false-noise, "
                f"{s.total_noise_events} total noise events"
            )
        print()

        # Aggregate
        outcomes.append(replay.outcome)
        if s and s.ticks_to_first_sight is not None:
            first_sight_ticks.append(s.ticks_to_first_sight)

    # Final summary
    print("=" * 70)
    print("Final summary")
    print("=" * 70)
    print(f"  L2 size at end:    {store.count()}")
    print(f"  Outcomes:          {', '.join(outcomes)}")
    caught = outcomes.count("caught")
    survived = outcomes.count("survived")
    print(f"  Caught:            {caught}/{len(outcomes)}")
    print(f"  Survived:          {survived}/{len(outcomes)}")
    if first_sight_ticks:
        avg = sum(first_sight_ticks) / len(first_sight_ticks)
        print(f"  Avg ticks to first sight: {avg:.1f} (over {len(first_sight_ticks)} episodes)")
    else:
        print(f"  Tom never sighted Jerry in any episode "
              f"(seed {args.seed} may not produce sightings; try a different seed)")

    # Cross-episode heatmap insight
    all_summaries = store.query_fine(fine_fp, jerry_fp, limit=100)
    if len(all_summaries) >= 2:
        from collections import Counter
        tile_seen_in: Counter = Counter()
        for s in all_summaries:
            for (x, y, _c) in s.heatmap_top:
                tile_seen_in[(x, y)] += 1
        if tile_seen_in:
            most_common = tile_seen_in.most_common(3)
            print(f"  Most consistent sighting tiles across {len(all_summaries)} episodes:")
            for (x, y), count in most_common:
                print(f"    ({x}, {y}) — appeared in {count}/{len(all_summaries)} episode summaries")

    print()
    client.close()


if __name__ == "__main__":
    main(sys.argv[1:])
