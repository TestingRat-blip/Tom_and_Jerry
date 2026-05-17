# Tom_and_Jerry

A research project building a hunter AI intended to surpass the Alien from *Alien: Isolation*, through three pillars Isolation cannot reach:

1. **Persistent learning across player deaths and sessions**
2. **Coordinated multi-agent pack hunting** (deferred — single species first)
3. **Emergent behavior from drives and a chemical analog layer**

The hunter co-evolves against a population of diverse RL test-players (the "Jerrys"), each with a different play style. The hunter is "Tom." The yin/yang dynamic — neither side allowed to dominate — drives both toward genuine generality.

## Quick start

> Not yet runnable. See `docs/DESIGN.md` for the full architecture and `docs/ROADMAP.md` for build order.

## Layout

- `src/env/` — headless top-down grid environment, sensors, sound propagation, line of sight
- `src/hunter/` — Tom. Conductor, pack coordinator (stubbed), individual hunter agent
- `src/players/` — Jerrys. Six archetype-conditioned PPO agents
- `src/coevo/` — co-evolution scheduler, hall of fame, metrics
- `src/persistence/` — Redis / Postgres / Chroma backends for L1/L2/L3 memory
- `src/render/` — Pygame replay renderer (training runs headless)
- `configs/` — YAML configs for archetypes, hunter parameters, training runs
- `tests/` — unit, integration, and scenario tests
- `docs/` — design doc, roadmap, decision log
- `data/` — snapshots, replays, training logs (gitignored)

## Status

Phase 0: scaffold and design doc.

See `docs/DESIGN.md`.
