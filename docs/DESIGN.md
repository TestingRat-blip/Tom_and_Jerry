# Tom_and_Jerry — Design Document

**Author:** Grove
**Status:** v0.1, Phase 0 (scaffold)
**Last updated:** 2026-05-17

---

## 1. Vision

Build a hunter AI that surpasses the Alien from *Alien: Isolation* on three axes:

1. **Persistent learning across player deaths and sessions** — the hunter remembers *this player* across the saved-game boundary, which Isolation explicitly cannot do.
2. **Coordinated pack hunting** (single species at v1; pack mechanics deferred to v2) — Isolation supports only one Alien.
3. **Emergent behavior from drives and chemical analog state** — Isolation uses a fixed behavior tree with unlockable nodes; Tom uses the same skeleton modulated by a drive/chemistry layer that produces non-scripted behavior under pressure.

The project name reflects the core dynamic: Tom (hunter) and Jerry (the population of player-bots) co-evolve. Neither side is allowed to dominate. The yin/yang balance is enforced by population diversity on the player side and by hall-of-fame regularization on the hunter side.

## 2. Non-goals

- **Not a shipping game.** This is an AI research project. The environment is deliberately minimal.
- **Not LLM-driven.** LLMs are too slow and too unreliable for tight predator loops. Tom's intelligence is structural, not generative. (LLMs may layer in later as flavor — vocalizations, narrative — but never on the hot path.)
- **Not connected to Vera or APEX.** Strictly self-contained. Architectural lessons cross over; code does not.
- **Not photoreal.** Top-down 2D grid for v1. 3D port is a future axis, not a v1 concern.

## 3. Reference: how Isolation's Alien actually works

Two-AI design:

- **Director AI** — omniscient about the player, never commands the Alien directly. Maintains a menace gauge and feeds the Alien hints by setting a search zone.
- **Alien AI** — behavior tree starting at ~30 unlocked nodes (of ~100 total). Nodes unlock as the player demonstrates behaviors in front of the Alien (locker-hiding three times unlocks locker-checking, etc.).

Strengths Isolation exploits well:
- The Director hides information so the Alien feels emergent rather than scripted.
- Per-playthrough unlock system makes the Alien feel personal to *this run*.
- Single agent is tightly tuned and dramatically pace-controlled.

Known limits (Tom's openings):
- Single enemy only — the system breaks with two.
- No cross-session memory — reload erases everything.
- Bounded behavior tree — only does what designers pre-authored.
- Director is rule-based — pacing comes from hand-tuned heuristics.
- No real social/communication layer.

## 4. Architecture

Three-layer hierarchy, mirroring Isolation's split but extended.

### 4.1 Layer 1 — The Conductor

Omniscient, never directly controls hunters.

Responsibilities:
- Maintains a per-player **threat model** (learned profile of how this player behaves).
- Manages the menace gauge: tension, dread, relief cycles, modulated by chemical analog signals.
- Persists across deaths. This is the layer that *remembers* the player.
- Nudges the pack coordinator via search-zone hints — never gives positions directly.

Backing store: Postgres for persistent threat models, Redis for live menace gauge state.

### 4.2 Layer 2 — Pack Coordinator (single hunter in v1, stub in code)

In v1: a trivial passthrough. The architecture is in place for v2.

When activated (v2+):
- Knows hunter positions but enforces **fog-of-war between hunters** — they share information via in-world signals (vocalizations, scent trails, line-of-sight handoffs), not telepathy.
- Assigns roles dynamically from drive state: flanker, pursuer, ambusher, watcher.
- Roles emerge from chemistry + drive vectors, not hard assignments.

Backing store: Redis pub/sub (same pattern as Billy.exe / Rat Gang familiar comms).

### 4.3 Layer 3 — Individual Hunter Agent

The hunter itself. Structurally scripted, parametrically learned.

Components:

- **Drives** (6-axis vector): hunger, aggression, caution, curiosity, fatigue, social-bond. Drives shift behavior priorities.
- **Chemistry** (5 chemicals): adrenaline, cortisol, dopamine, oxytocin, serotonin. Chemicals modulate drive expression on short timescales.
- **Behavior tree**: fixed skeleton (search → investigate → pursue → attack → patrol → retreat). Node activation thresholds are learnable parameters.
- **Memory** (3 tiers):
  - L1 per-encounter (Redis, this life only)
  - L2 per-player (Postgres, persistent profile of how the player tends to behave)
  - L3 hunter-identity (Postgres, the hunter's own evolved priors — which tactics have historically worked for it)
- **Perception**: line of sight, sound propagation, scent grid. Critically, the hunter's perception is *not* the same as the Conductor's — the hunter only knows what it senses.

Design rule: **the hunter's behavior must remain legible.** Players need to feel hunted by a creature, not buffeted by a neural network. The skeleton stays scripted; only parameters drift.

## 5. The Jerrys — player-bot population

Six archetypes, all PPO via Stable Baselines3 (consistent with prior OSRS pipeline experience). Each archetype is the same base agent, differentiated by **reward function shape**, not architecture.

| Archetype     | Core reward signal                              | Style                       |
|---------------|--------------------------------------------------|-----------------------------|
| Sneaker       | survival + distance from hunter, sound punished | Stealth-maximizing          |
| Sprinter      | objective progress, low survival weight         | Speed-running, risk-taking  |
| Trickster     | item use, hunter-distracted bonus               | Noisemaker spam, baiting    |
| Camper        | hiding time, movement punished                  | Locker / vent abuser        |
| Explorer      | novel tiles visited, unique paths               | Curious, weird routes       |
| Generalist    | learned reward weights (population-based)       | Self-tuning, often surprising |

Diversity is what saves co-evolution from collapse. If Tom is trained against one player, both overfit and become brittle. Six adversarial styles force Tom to remain generally capable.

## 6. Co-evolution loop

```
For each generation:
  1. Each player-bot plays N episodes against the current Tom.
  2. Tom updates: per-archetype parameter adjustments
     (e.g. "against trickster, weight skepticism higher").
  3. Player-bots take a PPO update step on their own reward.
  4. Every K generations: archive a snapshot of Tom.
  5. Every M generations: spawn a "fresh blood" player-bot
     trained against an OLD Tom snapshot — prevents collapse
     into narrow counter-strategies that only work on current Tom.
```

The **hall of fame** mechanism is the most important regularizer. Without it, both populations chase each other into a degenerate corner of strategy space and stop generalizing. With it, Tom must remain effective against historical players too, which forces robustness.

## 7. The hunter is not pure RL — and why

This is the most important architectural decision in the project.

A pure-RL hunter is unreadable. It jitters, makes non-intuitive moves, and does not feel like a creature. The Alien works precisely because its behavior is **legible** — players can build a mental model of what it's doing and form intent against it. That intent is the source of all the tension.

So: **structure stays scripted, parameters get learned.**

- Behavior tree skeleton: scripted, hand-authored.
- Node unlock thresholds: learned (when does Tom start checking lockers?).
- Drive baselines: learned per encountered archetype.
- Chemical reaction curves: learned (adrenaline spike rate, cortisol decay).
- Conductor menace-gauge weights: learned per archetype.

The result is a creature with a fixed "soul" but adaptive instincts. Legibility of Isolation; adaptability of pure RL.

## 8. Metrics — how we know Tom is actually getting better

Win rate is a trap. If both sides improve, win rate hovers near 50% indefinitely. Real metrics:

1. **Cross-generation tournament**: gen-N Tom vs gen-1 Jerrys. If gen-N wins faster than gen-1 Tom did, Tom improved. Run backwards for Jerrys.
2. **Behavioral diversity score**: count distinct tactics across 1000 episodes. Should rise, not collapse.
3. **Surprise rate**: how often does Tom do something it didn't do last generation? Non-zero is healthy.
4. **Hold-out archetype**: keep one Jerry archetype hidden from training, evaluate against it every 100 generations. Tells us if Tom is overfitting.
5. **Human-readability score**: weekly, render replays and Likert-rate "does this look like a creature hunting?" The one metric that cannot be automated.

Hit 3 of 5 success criteria below and Tom matches Isolation. Hit 5 of 5 and Tom surpasses it:

- Cross-session terror: a playtester says "it remembered" unprompted after death-and-reload.
- Emergent tactics: Tom does something not directly programmed (flanking, baiting, fake retreats).
- Distinct identity: playtester can describe Tom's character after a session.
- Pack dynamics readable (v2): player can tell when the pack coordinates vs. when one breaks off.
- No hivemind feel (v2): player exploits information gaps between hunters at least once per session.

## 9. Environment design (v1)

Headless top-down grid. Deliberately minimal — the AI is the product, not the environment.

- 30×30 tile grid, walls, vents (one-tile teleports), lockers (hide spots).
- Sound propagation: noise events emit on a tile, propagate by inverse-square through open space, blocked by walls.
- Line of sight: raycast through the grid.
- Scent grid: a decaying scalar field the player leaves behind. Tom can sample it but only at its current tile.
- Optional Pygame renderer for replay; training runs headless via a pure numpy/list-based world for speed.

Gymnasium-compatible env wrapper. Reuses interface conventions from the prior OSRS RL pipeline.

## 10. Persistence

All three memory tiers are real backing stores from day one — no in-memory hacks that we have to rip out later.

- **Redis**: live menace gauge, L1 per-encounter memory, pack-coordinator pub/sub (stub in v1).
- **Postgres**: persistent threat models per player ID, L2 player profiles, L3 hunter identity priors, generation snapshots metadata.
- **ChromaDB**: episodic memory embeddings — when Tom encounters a similar situation, retrieves nearest neighbors from past lives.

Snapshot files (model weights, behavior tree configs) live on disk under `data/snapshots/`.

## 11. Training infrastructure

Target hardware: the 3060 Ti box (asuna machine). Headless training, overnight runs. Tom_and_Jerry never shares process space with Vera, APEX, or any other running system — strict separation enforced by running in its own Docker compose or venv.

Estimated throughput target: 1000+ player-deaths per hour during co-evolution. This is the unfair advantage over AAA studios — they cannot iterate this fast on hunter AI because they rely on human playtesters.

## 12. Build phases

See `ROADMAP.md` for detail. High-level:

- **Phase 0** (this commit): scaffold + design doc.
- **Phase 1**: headless grid env, Gymnasium wrapper, one PPO bot, one scripted Tom. Prove the loop runs.
- **Phase 2**: drives + chemistry layer. Same Tom, different emotional states → different behavior.
- **Phase 3**: L1 memory. Tom learns within a life.
- **Phase 4**: L2/L3 memory. Cross-session terror.
- **Phase 5**: six-archetype Jerry population, archetype-conditioned rewards.
- **Phase 6**: co-evolution scheduler, hall of fame, cross-generation tournaments.
- **Phase 7**: tune, playtest, tune.
- **Phase 8+**: pack mechanics (v2), 3D port, multi-species.

## 13. Open questions

- **How fast should L2 player models be allowed to shift per death?** Too fast and the game is unplayable after a few deaths. Too slow and the persistent-learning differentiator disappears. Initial guess: learning rate 0.1–0.2 per death, with hard caps on certain priors. Needs empirical tuning.
- **How readable is "readable"?** The legibility constraint is qualitative. Define a periodic human-eval protocol early.
- **Should Tom have a single fixed personality, or should the Conductor draw from a small pool of pre-baked personalities for variety across playthroughs?** Probably the latter, but defer the decision until Phase 4.
- **When does pack mechanics get reintroduced?** Probably v2, after Tom-as-individual is genuinely terrifying. Don't try to co-evolve along too many axes at once.

## 14. Decision log

Tracked separately in `docs/DECISIONS.md` as architectural decisions land.
