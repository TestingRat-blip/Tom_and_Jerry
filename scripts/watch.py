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
      "wait"            no-op Tom (useful for watching Jerry behavior alone)
      "model:PATH"      a saved PPO Tom checkpoint (Phase 4+; here for symmetry)
    """
    if spec == "scripted":
        tom = ScriptedTom(seed=seed)
        return (tom, "scripted")
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
