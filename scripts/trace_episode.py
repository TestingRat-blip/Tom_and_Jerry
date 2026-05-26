"""Tick-by-tick trace of a single episode — diagnose HOW Jerry evades.

The aggregate numbers ruled out both LOS-denial and LOS-breaking: Tom sees
Jerry for ~half the episode in one long unbroken streak, yet often fails to
catch. This script prints the per-tick dynamics so we can see what's actually
happening: distance, visibility, Tom's mode/state/action, and catch-range
proximity.

Usage:
    python -m scripts.trace_episode \
        --jerry model:data/snapshots/jerry_generalist_vs_conductor/final.zip \
        --tom conductor --seed 42

Read the DIST column: catch needs manhattan <= 1 + line of sight. If DIST
hovers at 2-3 for the whole episode while SEE=Y, Jerry is KITING — staying
just outside catch range while fully visible. If DIST spikes whenever Tom
gets close, Jerry is JUKING at the last moment. Watch the MODE column for
whether Tom is STALK-holding (not closing) vs RUSH (closing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.env.world.world import World, WorldConfig
from src.utils.types import Action


def make_jerry(spec: str, seed: int, deterministic: bool = False):
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
    if spec == "scripted":
        from src.hunter.agent.behavior.baseline import ScriptedTom
        return ScriptedTom(seed=seed)
    if spec == "chemical":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        return ChemicalTom(seed=seed)
    if spec == "conductor":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor
        return ChemicalTom(conductor=Conductor(), seed=seed)
    if spec == "conductor-holddown":
        from src.hunter.agent.behavior.chemical_tom import ChemicalTom
        from src.hunter.agent.conductor import Conductor, ConductorConfig
        return ChemicalTom(conductor=Conductor(config=ConductorConfig(
            hold_on_los_break=True)), seed=seed)
    raise SystemExit(f"unknown tom: {spec!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jerry", required=True)
    p.add_argument("--tom", default="conductor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--every", type=int, default=1,
                   help="Print every Nth tick (1 = every tick).")
    p.add_argument("--deterministic", action="store_true",
                   help="Run Jerry deterministically (matches watch.py's "
                        "default). Use this to line up the trace with the "
                        "rendered replay tick-for-tick.")
    p.add_argument("--locker-oxygen", action="store_true",
                   help="Enable the locker oxygen/cooldown mechanic. Use to "
                        "blind-trace an oxygen-unaware Jerry against the wall.")
    p.add_argument("--pursuit-speed-ramp", action="store_true",
                   help="Enable the pursuit speed-ramp mechanic (Tom "
                        "accelerates during sustained pursuit).")
    args = p.parse_args(argv)

    jerry, jlabel = make_jerry(args.jerry, args.seed, deterministic=args.deterministic)
    tom = make_tom(args.tom, args.seed)

    world = World(
        WorldConfig(max_ticks=args.max_ticks,
                    locker_oxygen_enabled=args.locker_oxygen,
                    pursuit_speed_ramp_enabled=args.pursuit_speed_ramp),
        seed=args.seed,
    )
    world.reset()
    if hasattr(tom, "reset"):
        tom.reset()
    if hasattr(jerry, "reset"):
        jerry.reset()

    print(f"Trace: {jlabel} vs {args.tom}  seed={args.seed}"
          f"{'  [locker-oxygen ON]' if args.locker_oxygen else ''}")
    print(f"catch needs DIST<=1 + line-of-sight. catch_distance="
          f"{world.config.catch_distance}, tom_sight={world.config.tom_sight_range}")
    print(f"{'tick':>4} {'DIST':>4} {'SEE':>3} {'LOCK':>4} {'OXY':>4} "
          f"{'STATE':>10} {'MODE':>11} {'ACT':>5} {'tom_pos':>9} {'jerry_pos':>9}")

    # distance histogram while visible
    dist_hist = {}
    min_dist_seen = 99

    for t in range(args.max_ticks):
        ta = tom(world)
        ja = jerry(world)

        dist = world.tom.position.manhattan(world.jerry.position)
        see = world._tom_can_see_jerry()
        lock = world.jerry.in_locker
        oxy = world._jerry_oxygen
        oxy_str = str(oxy) if oxy is not None else "-"
        state = getattr(tom, "state", None)
        state_name = state.name if state is not None else "-"
        mode = getattr(tom, "current_mode", None)
        mode_name = mode.name if mode is not None else "-"

        if see:
            dist_hist[dist] = dist_hist.get(dist, 0) + 1
            min_dist_seen = min(min_dist_seen, dist)

        if t % args.every == 0:
            print(f"{t:>4} {dist:>4} {'Y' if see else 'n':>3} "
                  f"{'Y' if lock else '-':>4} {oxy_str:>4} "
                  f"{state_name:>10} {mode_name:>11} "
                  f"{Action(int(ta)).name[:5]:>5} "
                  f"{f'{world.tom.position.x},{world.tom.position.y}':>9} "
                  f"{f'{world.jerry.position.x},{world.jerry.position.y}':>9}")

        in_pursuit = state.is_committed_pursuit if state is not None else None
        world.step(tom_action=ta, jerry_action=int(ja),
                   tom_in_pursuit=in_pursuit)
        if not world.jerry.alive:
            print(f"\n*** CAUGHT at tick {world.tick_count} ***")
            break
    else:
        print(f"\n*** Jerry SURVIVED {args.max_ticks} ticks ***")

    print(f"\nDistance histogram while Tom could SEE Jerry "
          f"(dist: count):")
    for d in sorted(dist_hist):
        bar = "#" * min(60, dist_hist[d])
        print(f"  dist {d:2d}: {dist_hist[d]:4d} {bar}")
    print(f"\nClosest Tom ever got while seeing Jerry: {min_dist_seen} "
          f"(catch needs <= {world.config.catch_distance})")
    if min_dist_seen > world.config.catch_distance:
        print("  => Tom NEVER got within catch range while seeing Jerry.")
        print("     Jerry KITES: stays visible but always >1 tile away.")
        print("     The exploit is close-range evasion, not LOS denial/breaking.")


if __name__ == "__main__":
    main()
