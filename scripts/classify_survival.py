"""Classify HOW a Jerry survives — locker-camping vs open evasion.

The eval reports aggregate survival% but not WHY each survivor lived. This
sweeps a range of seeds, finds the ones where Jerry survives, and for each
survivor reports diagnostic signals so we can tell whether the survival is:

  - LOCKER-CAMP: Jerry spent most of the episode in a locker (in_locker
    fraction high), little movement.
  - OPEN EVASION: Jerry survived while mostly OUT of lockers (kiting,
    cornering, distance management).
  - MIXED: some of each.

This confirms (or refutes) the hypothesis that the post-fix Round 5 survival
is predominantly locker-camping BEFORE we design the oxygen/cooldown fix.

Usage:
    python -m scripts.classify_survival \
        --jerry model:data/snapshots/jerry_generalist_vs_conductor_postfix/final.zip \
        --tom conductor --seeds 0-49
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.env.world.world import World, WorldConfig
from src.utils.types import Action


def make_jerry(spec: str, seed: int, deterministic: bool):
    if spec == "random":
        import random
        rng = random.Random(seed)
        return (lambda world: rng.randint(0, 5)), "random"
    if spec == "passive":
        return (lambda world: int(Action.WAIT)), "passive"
    if spec.startswith("model:"):
        from src.players.policies.ppo import PPOJerryPolicy
        path = spec[len("model:"):]
        return PPOJerryPolicy.load(path, deterministic=deterministic), f"ppo({Path(path).stem})"
    raise SystemExit(f"unknown jerry: {spec!r}")


def make_tom(spec: str, seed: int):
    from src.hunter.agent.behavior.chemical_tom import ChemicalTom
    from src.hunter.agent.conductor import Conductor
    if spec == "conductor":
        return ChemicalTom(conductor=Conductor(), seed=seed)
    if spec == "chemical":
        return ChemicalTom(seed=seed)
    if spec == "scripted":
        from src.hunter.agent.behavior.baseline import ScriptedTom
        return ScriptedTom(seed=seed)
    raise SystemExit(f"unknown tom: {spec!r}")


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def _spawns_connected(world) -> bool:
    """True iff Tom can reach Jerry's spawn via a walkable path.

    Standalone 4-neighbor BFS over the grid (doesn't depend on Tom internals),
    so the check is valid for any Tom type. Used to flag map-generation
    artifacts: _spawn_agents maximizes manhattan distance with no reachability
    guarantee, so a Jerry can spawn in a pocket Tom physically cannot enter —
    that Jerry 'survives' trivially and must NOT be counted as evasion.
    """
    from collections import deque
    grid = world.grid
    src = world.tom.position
    dst = world.jerry.position
    if src == dst:
        return True
    Pos = type(src)
    seen = {(src.x, src.y)}
    q = deque([(src.x, src.y)])
    while q:
        x, y = q.popleft()
        if (x, y) == (dst.x, dst.y):
            return True
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            if 0 <= nx < grid.width and 0 <= ny < grid.height \
                    and grid.is_walkable(Pos(nx, ny)):
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jerry", required=True)
    p.add_argument("--tom", default="conductor")
    p.add_argument("--seeds", default="0-49")
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--locker-oxygen", action="store_true",
                   help="Enable the locker oxygen/cooldown mechanic.")
    p.add_argument("--locker-frac-threshold", type=float, default=0.5,
                   help="in_locker fraction above which a survivor is "
                        "classified LOCKER-CAMP")
    args = p.parse_args(argv)

    jerry, jlabel = make_jerry(args.jerry, 0, args.deterministic)
    seeds = parse_seeds(args.seeds)

    survivors = []
    n_caught = 0
    for seed in seeds:
        world = World(
            WorldConfig(max_ticks=args.max_ticks,
                        locker_oxygen_enabled=args.locker_oxygen),
            seed=seed,
        )
        world.reset()
        tom = make_tom(args.tom, seed)
        if hasattr(tom, "reset"):
            tom.reset()
        if hasattr(jerry, "reset"):
            jerry.reset()

        connected = _spawns_connected(world)

        locker_ticks = 0
        moved_ticks = 0
        last_pos = world.jerry.position
        total = 0
        for _ in range(args.max_ticks):
            ta = tom(world)
            ja = jerry(world)
            world.step(tom_action=ta, jerry_action=int(ja))
            total += 1
            if world.jerry.in_locker:
                locker_ticks += 1
            if world.jerry.position != last_pos:
                moved_ticks += 1
            last_pos = world.jerry.position
            if not world.jerry.alive:
                break

        if world.jerry.alive:
            lf = locker_ticks / max(1, total)
            mf = moved_ticks / max(1, total)
            if not connected:
                # Map artifact: Tom could never reach Jerry. Not evasion.
                kind = "UNREACHABLE-SPAWN"
            elif lf >= args.locker_frac_threshold:
                kind = "LOCKER-CAMP"
            else:
                kind = "OPEN-EVASION"
            survivors.append((seed, lf, mf, kind))
        else:
            n_caught += 1

    unreachable = [s for s in survivors if s[3] == "UNREACHABLE-SPAWN"]
    real = [s for s in survivors if s[3] != "UNREACHABLE-SPAWN"]

    print(f"Survival classification — {jlabel} vs {args.tom}")
    print(f"  seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes), "
          f"{'deterministic' if args.deterministic else 'stochastic'}")
    print(f"  caught: {n_caught}   survived: {len(survivors)}  "
          f"({len(survivors)/len(seeds):.0%} raw)")
    if unreachable:
        print(f"  !! {len(unreachable)} of those are UNREACHABLE-SPAWN map "
              f"artifacts (Tom could never reach Jerry) — NOT evasion.")
        print(f"  => TRUE survival (reachable maps only): {len(real)}"
              f"/{len(seeds)} ({len(real)/len(seeds):.0%})")
    print()
    if not survivors:
        print("  No survivors in this seed range.")
        return
    print(f"  {'seed':>5} {'locker_frac':>11} {'move_frac':>9}  classification")
    print(f"  {'-'*5} {'-'*11} {'-'*9}  {'-'*17}")
    for seed, lf, mf, kind in survivors:
        print(f"  {seed:>5} {lf:>11.2f} {mf:>9.2f}  {kind}")

    n_locker = sum(1 for s in real if s[3] == "LOCKER-CAMP")
    n_open = sum(1 for s in real if s[3] == "OPEN-EVASION")
    print(f"\n  Of {len(real)} REAL (reachable) survivors: "
          f"{n_locker} LOCKER-CAMP, {n_open} OPEN-EVASION."
          + (f"  ({len(unreachable)} unreachable artifact(s) excluded.)"
             if unreachable else ""))
    if not real:
        print("  => No real survivors — all survival was map artifacts.")
    elif n_locker == len(real):
        print("  => All real survival is locker-camping.")
    elif n_locker == 0:
        print("  => No locker-camping among real survivors; survival is "
              "open evasion.")
    else:
        print("  => MIXED real survival: some locker-camping, some open "
              "evasion.")


if __name__ == "__main__":
    main()
