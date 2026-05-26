"""Characterize HOW each mover-survivor survives: genuine kiting vs a closed
loop ("circle"), and if a loop, its geometry — size, wall-adjacency, and how
long Jerry kited freely before locking into it.

This exists to answer a specific question raised by seed 20: is the circle an
EXPLOIT (a bug in Tom's pursuit/prediction the prey games) or the inevitable
GEOMETRY of equal-speed pursuit (a same-speed hunter chasing a looping prey's
current position can never close, regardless of how good prediction is)? And a
sub-hypothesis: do the loops hug walls (suggesting wall-aware prediction
misbehaves on a wall-bounded curve) or sit in open space (pure speed parity)?

For each seed it reports:
  - free_kite_ticks : ticks of non-repeating movement before a stable cycle
  - loop_len        : period of the terminal cycle (0 = no clean loop / genuine)
  - loop_tiles      : the cycle Jerry repeats
  - bbox            : bounding box of the loop (3x3 ring = 'circle')
  - wall_adjacent   : how many loop tiles touch a wall (tests wall hypothesis)
  - caught_tick     : if Tom eventually catches, when

Usage:
    python -m scripts.analyze_loops \
        --jerry model:data/snapshots/jerry_generalist_vs_patched_tom_r9/final.zip \
        --tom conductor --seeds 3,11,16,17,20,31,37,42,49 \
        --locker-oxygen --deterministic --max-ticks 600
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.env.world.world import World, WorldConfig
from src.env.world.grid import TileType
from src.utils.types import Action, Position


def make_jerry(spec: str, seed: int, deterministic: bool):
    if spec.startswith("model:"):
        from src.players.policies.ppo import PPOJerryPolicy
        path = spec[len("model:"):]
        return PPOJerryPolicy.load(path, deterministic=deterministic)
    raise SystemExit(f"analyze_loops needs a model: jerry, got {spec!r}")


def make_tom(spec: str, seed: int):
    from src.hunter.agent.behavior.chemical_tom import ChemicalTom
    from src.hunter.agent.conductor import Conductor
    if spec == "conductor":
        return ChemicalTom(conductor=Conductor(), seed=seed)
    if spec == "chemical":
        return ChemicalTom(seed=seed)
    raise SystemExit(f"unknown tom: {spec!r}")


def detect_terminal_loop(positions: list[tuple[int, int]],
                         max_period: int = 16):
    """Find the smallest period P such that the tail of `positions` repeats
    with period P. Returns (free_prefix_len, period, loop_tiles) or
    (len, 0, []) if no clean terminal cycle.

    Scans from the end: for each candidate period P, find the longest suffix
    that is P-periodic, then take the earliest tick where that periodicity
    began.
    """
    n = len(positions)
    best = (n, 0, [])
    for period in range(1, max_period + 1):
        if n < 2 * period:
            continue
        # How far back from the end does period-P repetition hold?
        start = n - 1
        while start - period >= 0 and positions[start] == positions[start - period]:
            start -= 1
        # positions[start+1 .. end] is P-periodic; need at least 2 full cycles
        periodic_len = (n - 1) - start
        if periodic_len >= 2 * period:
            free_prefix = start + 1 - period  # first tick of the first full cycle
            free_prefix = max(0, free_prefix)
            loop_tiles = positions[free_prefix:free_prefix + period]
            # Prefer the shortest period that explains a long tail.
            if period < best[1] or best[1] == 0:
                best = (free_prefix, period, loop_tiles)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jerry", required=True)
    ap.add_argument("--tom", default="conductor")
    ap.add_argument("--seeds", required=True,
                    help="comma-separated seeds, e.g. 3,11,20")
    ap.add_argument("--locker-oxygen", action="store_true")
    ap.add_argument("--pursuit-speed-ramp", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=600)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Loop analysis — {args.jerry}")
    print(f"  mode: {'deterministic' if args.deterministic else 'stochastic'}, "
          f"max_ticks={args.max_ticks}")
    print(f"  {'seed':>4} {'free_kite':>9} {'loop_len':>8} {'bbox':>11} "
          f"{'wall_adj':>8} {'caught':>7}  verdict")
    print("  " + "-" * 78)

    for seed in seeds:
        jerry = make_jerry(args.jerry, seed, args.deterministic)
        tom = make_tom(args.tom, seed)
        if hasattr(jerry, "reset"):
            jerry.reset()
        tom.reset()
        w = World(WorldConfig(locker_oxygen_enabled=args.locker_oxygen,
                              pursuit_speed_ramp_enabled=args.pursuit_speed_ramp,
                              max_ticks=args.max_ticks), seed=seed)
        w.reset()

        jerry_pos: list[tuple[int, int]] = []
        caught_tick = None
        for t in range(args.max_ticks):
            ja = int(jerry(w)) if callable(jerry) else int(jerry.act(w))
            ta = int(tom(w))
            _st = getattr(tom, "state", None)
            _ip = _st.is_committed_pursuit if _st is not None else None
            jerry_pos.append((w.jerry.position.x, w.jerry.position.y))
            w.step(tom_action=ta, jerry_action=ja, tom_in_pursuit=_ip)
            if not w.jerry.alive:
                caught_tick = t
                break

        free, period, loop = detect_terminal_loop(jerry_pos)
        if period == 0:
            verdict = "NO CLEAN LOOP (genuine kite or chaotic)"
            bbox = "-"
            wall_adj = "-"
        else:
            xs = [p[0] for p in loop]; ys = [p[1] for p in loop]
            bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
            bbox = f"{bw}x{bh}"
            # Count loop tiles adjacent to a wall (tests the wall hypothesis).
            wall_adj = 0
            for (x, y) in loop:
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    p = Position(x + dx, y + dy)
                    if not w.grid.in_bounds(p) or w.grid.tile_at(p) == TileType.WALL:
                        wall_adj += 1
                        break
            if bw <= 3 and bh <= 3:
                verdict = "CIRCLE (tight ring)"
            elif period <= 6:
                verdict = "small loop"
            else:
                verdict = "large loop / patrol-shaped"
        caught_str = str(caught_tick) if caught_tick is not None else "SURVIVED"
        print(f"  {seed:>4} {free:>9} {period:>8} {bbox:>11} "
              f"{str(wall_adj):>8} {caught_str:>7}  {verdict}")
        if period:
            print(f"       loop: {loop}")


if __name__ == "__main__":
    main()
