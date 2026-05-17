# Tom_and_Jerry — Glossary

Shared vocabulary. When in doubt, check here first.

- **Tom** — the hunter AI. The thing we're trying to make terrifying.
- **Jerry / Jerrys** — the population of RL test-players. Six archetypes in v1.
- **Conductor** — Layer 1. Omniscient director. Never directly controls the hunter. Holds menace gauge and per-player threat models.
- **Pack Coordinator** — Layer 2. Stubbed in v1, active in v2. Routes signals between hunters without giving them telepathy.
- **Hunter Agent** — Layer 3. The actual creature: drives, chemistry, behavior tree, memory, perception.
- **Drives** — six-axis vector: hunger, aggression, caution, curiosity, fatigue, social-bond. Slow-changing motivational state.
- **Chemistry** — five chemicals: adrenaline, cortisol, dopamine, oxytocin, serotonin. Fast-changing modulators on top of drives.
- **L1 memory** — per-encounter. Redis. This life only.
- **L2 memory** — per-player. Postgres. Persistent profile of how a specific player tends to behave.
- **L3 memory** — hunter identity. Postgres. The hunter's own evolved priors — tactics that have historically worked.
- **Menace gauge** — scalar tension value managed by the Conductor. Modulates how aggressively Tom is allowed to pressure the player.
- **Search zone** — the area the Conductor nudges the hunter toward. Never reveals player position; only narrows the search.
- **Archetype** — a reward-function shape that defines a Jerry's play style. Six in v1.
- **Hall of fame** — the archive of historical Tom snapshots used to spawn fresh-blood Jerrys, preventing co-evolutionary collapse.
- **Cross-generation tournament** — evaluation harness pitting gen-N Tom against gen-1 Jerrys (and vice versa) to measure real improvement.
- **Legibility** — the property that Tom's behavior is readable to a human observer. Non-negotiable constraint on architectural choices.
- **Co-evolution** — joint training of Tom and Jerrys, neither side dominating. Distinct from GAN-style adversarial training in that diversity (not just minimax) drives both sides.
