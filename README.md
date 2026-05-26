# Tom_and_Jerry

A research project building a hunter AI intended to surpass the Alien from
*Alien: Isolation*, through three pillars Isolation cannot reach:

1. **Persistent learning across player deaths and sessions** — the hunter
   remembers you, and gets better at hunting *you* specifically over time.
2. **Emergent behavior from drives and a chemical analog layer** — hunting that
   arises from internal state (adrenaline, cortisol, and other chemical
   analogs) feeding a belief system, not from a hardcoded difficulty script.
3. **Coordinated multi-agent pack hunting** *(deferred — single predator
   first)*.

The hunter — **Tom** — co-evolves against a population of reinforcement-learning
prey — the **Jerrys** — each able to learn its own play style. Neither side is
allowed to win permanently: every time one gets better, the other must adapt.
The arms race *is* the engine. The design constraint throughout is that Tom is
**scripted-over-rich-belief** — his structure is hand-authored and fully
inspectable, while his parameters and memory are learned. Emergence comes from
rich inputs colliding inside a legible controller, never from an opaque policy.

> **Why this is interesting to read:** this isn't just "I built a predator AI."
> Over the development arc, the RL prey repeatedly found *exploits* in the
> hunter that its creator didn't know were there — and watching those get
> discovered, diagnosed, and closed is the actual story. See
> [The arms race so far](#the-arms-race-so-far).

---

## Status

**Phase 8 — co-evolution active, the senses-and-speed era.** The core
simulation, the two-brain hunter ("the Conductor"), the persistent memory
system, and the PPO prey population are all built and tested (**485 tests
passing**). The project is in the co-evolution loop: train prey against the
hunter, find what they exploit, close it — round by round.

As of the latest round, **a prey trained specifically to beat the current
hunter survives 0 out of 50 nights.** Every static exploit is closed, and the
most recent fix shut down the last *mobile* survival strategy too (see below).
The open frontier is sensory: a perfectly silent, motionless prey hidden where
the hunter has no line of sight can still survive — which is precisely what the
next layer (scent / presence sensing) exists to solve.

Not a plug-and-play game yet; it's a research sandbox. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for build order and
[`docs/DESIGN.md`](docs/DESIGN.md) for the full architecture.

---

## The premise (and the long game)

The framing that ties the system together: **Tom is a hallucination, and the
player's goal is to survive the night.** Each night, Tom uses his memories more.
That last sentence is not flavor — it is the literal rendering of the persistent
learning system as narrative. Night 1, Tom is generic. Night 5, Tom knows you
bolt for the same corner, knows you go quiet then sprint when cornered, knows
your tells. The escalating dread *is* the memory system filling up. The horror
premise and the technical centerpiece are the same thing.

---

## The arms race so far

The heart of the project. Each round, the prey is trained against the current
hunter, the dominant survival strategy is found, and it is closed — which forces
the next adaptation. A recurring lesson: **any fixed rule is a resonance target.**
A reinforcement learner doesn't need to *understand* a system to exploit it — it
just needs more attempts than the designer has foresight.

| Round | Prey learned… | Hunter's answer | Result |
|------:|---------------|-----------------|--------|
| 1 | An oscillation "dance" that beat the early pathfinder | — | first exploit |
| 2 | *(the Conductor replaced the pathfinder; noise-sensitive belief)* | — | open-dance dies |
| 3 | A wall-pinned column-bob that froze the hunter 2 tiles away, **forever** | — | 40% survival |
| 4 | *(exploit closed)* | **randomized the movement tie-break** | 40% → 6% |
| 5 | **Locker-camping** — hide in the one tile with no cost-of-staying | — | 14%, all camping |
| 6 | *(exploit closed)* | **locker oxygen + cooldown** (the refuge betrays you) | camping 14% → 2% |
| 7 | A **corner-cubby**: wedge in a wall pocket where the hunter's prediction phased *through* the wall and it chased a ghost forever | **wall-aware prediction** | cubby dies → 0/50 |
| 8 | First **genuine kiter** — real map-wide evasion; plus motionless **statues** in unswept corners | **finer patrol coverage; dead-end map cleanup** | open-corner statues die |
| 9 | The most capable prey yet — broad mobile evasion **and a "circle"** (run a closed loop the equal-speed hunter can't cut) | *(diagnosed: equal-speed pursuit, not a bug)* | 9 surviving strategies |
| 10 | *(exploit closed)* | **pursuit speed ramp** — the hunter winds up during a sustained chase | all 9 caught; retrain in progress |

Two of these are the project's favorite stories:

**The column-bob (Round 3→4).** The prey learned to pin itself against a wall
and bob up and down in a single column, and the hunter *mirrored the bobbing
forever* — standing two tiles away, unable to close. The root cause was a fixed
`N/S`-before-`E/W` tie-break in the hunter's pathfinding (recognized by analogy
to **Old School RuneScape movement priority**). Randomizing the tie-break
dropped that prey's survival from 40% to 6%, verified three independent ways.
The "clever stealth tactic" was never stealth — it was one resonance, ridden
hard.

**The circle (Round 9→10).** Once every *static* exploit was closed, a prey
trained 2.5M steps finally learned real map-wide kiting — genuinely beautiful to
watch, using the whole stage. But at the edge of that skill it found a new
resonance: run a tight closed loop, and an **equal-speed** hunter chasing your
current position can never cut the corner to close the last two tiles. It orbits
behind you forever. The fix wasn't a bug patch — the loop is genuine geometry.
It was a *design* answer: the hunter now **accelerates during a sustained
chase** (a slow wind-up toward 1.15×, decaying only once it loses you), so
running in the open too long means the predator closes — and the only
counterplay left is to **break line of sight and use the environment**. Pure
Alien: Isolation: you cannot out-run it, you can only get out of sight.

Full write-ups live in `docs/` — the retrospectives and findings document the
failed bets as honestly as the wins, including five build cycles spent
countering an exploit that turned out to be a misread.

---

## Architecture

**The hunter (Tom)** is a three-layer design: drives + a chemical analog layer
modulate a **belief system** (typed, decaying suspicion from sightings, noise,
and scent), which feeds a scripted state machine and "the Conductor" — a
two-brain strategic layer that tracks *where it believes the prey is* separately
from *how to hunt*. Tom never reads the prey's true position except through the
visibility gate; he acts on belief, which can be wrong.

**The prey (Jerrys)** are PPO agents (Stable-Baselines3) trained against Tom.
They are the adversary that keeps the hunter honest — and the source of every
exploit the project has had to close.

**Memory** is layered: per-encounter (L1, Redis-backed) and persistent
cross-session (L2, SQLite), with behavioral distillation so the hunter can learn
*how a given prey behaves* across encounters — the substrate for the
"remembers you across nights" goal.

**The environment** is a headless top-down grid with line-of-sight, sound
propagation, a diffusing scent field, lockers (timed hiding with an oxygen cost,
so the refuge betrays you), vents, and a map generator that guarantees
connected, dead-end-free layouts — survival has to come from skill, not from a
geometry quirk.

---

## Layout

- `src/env/` — headless top-down grid environment, sensors, sound propagation, line of sight, scent
- `src/hunter/` — Tom: drives, chemistry, the belief system, the Conductor, memory
- `src/players/` — Jerrys: archetype-conditioned PPO prey
- `src/persistence/` — Redis (L1) and SQLite (L2) memory backends
- `src/render/` — replay renderer (training runs headless)
- `scripts/` — training, evaluation, and diagnostic tools (trace, classify-survival, analyze-loops, memory-loop)
- `tests/` — unit, integration, and scenario tests
- `docs/` — design docs, roadmap, decision log (ADRs), retrospectives, findings
- `data/` — snapshots, replays, training logs *(gitignored)*

---

## Tech

Python · [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO)
· Gymnasium · Pygame (rendering) · Redis + SQLite (memory) · pytest. Training
runs headless; the renderer is for inspecting replays after the fact.

---

## A note on the docs

Most of the real thinking lives in `docs/` — not just the design and roadmap,
but the **retrospectives and findings**, which document the failed bets as
honestly as the wins. If you're here to understand *how the project actually
went* rather than just what it is, start there.

---

*Built solo, for the love of it. The whole ecosystem traces back to Mega Man
Battle Network.*
