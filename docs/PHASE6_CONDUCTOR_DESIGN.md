# Phase 6 — The Conductor + Co-evolution

**Status:** Design phase. Living document.
**Depends on:** Phases 1-5 (env, ChemicalTom, memory, base Jerry). ADR-013.
**Started:** 2026-05-20

---

## Tracking this document

Sections marked `[DECIDED]` are locked with reasoning. `[TBD]` are open.
The doc is structured as: why → architecture → build sequence → open
questions. Phase 6 is a **two-stage phase** per ADR-013 and Order A:
build and verify the scripted Conductor first (Stage 1), then make it
learnable and co-evolve (Stage 2).

---

## 1. Why Phase 6 changed shape

Originally Phase 6 was "co-evolution": add a training scheduler and hall
of fame on top of the existing hunter. The Phase 5 ceiling diagnostic
forced a rethink.

The diagnostic showed BFS pathfinding is Tom's dominant, non-adaptive
weapon — generalist Jerry survives 28% against BFS-Tom but 74% with BFS
disabled. BFS is a hand-coded strategic oracle: it computes the shortest
path to Jerry's *true* position. That caps any co-evolution arms race,
because Tom always knows the optimal route to the real Jerry, leaving
Jerry only thin margins to exploit.

Per ADR-013, we replace BFS-as-targeting with a **Conductor** — the
director half of an Alien: Isolation-style two-brain system. This was
always latent in the design (ADR-003 named "conductor weights" as
learnable from the start). Phase 6 activates it.

The payoff: Tom's competence now flows from how well the Conductor
synthesizes *imperfect* information (sounds, sightings, scent) into a
belief about where Jerry is — a tunable, learnable thing — rather than
from a perfect-pathfinding oracle. This opens the entire information
pipeline as a behavioral arms-race surface. Jerry can plant false
suspicions; the Conductor learns to see through them.

---

## 2. The architecture `[DECIDED]`

Model 2 (per design discussion): the Conductor supplies a directive;
Tom's existing chemistry / drives / behavior tree still run and color
execution. The Conductor *replaces BFS targeting specifically*;
everything else from Phases 2-5 stays live.

### 2.1 The two brains

**Tom (local brain).** Perception, movement, the five-state behavior
tree, chemistry, drives, memory. Tom pathfinds locally toward a target
the Conductor supplies — NOT toward Jerry's true position. Tom never
reads Jerry's ground-truth location (ADR-013 hard invariant).

**The Conductor (director brain).** Sits above Tom, runs once per tick
before Tom decides. Consumes observable world events and Tom's memory,
maintains a belief about where Jerry is, and issues Tom a directive.

### 2.2 The belief: typed, decaying suspicion sources `[DECIDED]`

The Conductor does not track Jerry's position. It tracks *reasons to
suspect Jerry is somewhere* — a small set of **suspicion sources**:

```
SuspicionSource:
    location:   tile (or sector centroid)
    type:       NOISE | SIGHTING | SCENT
    confidence: float in [0,1]
    born_tick:  when it was created
```

Confidence decays at a **type-specific rate**:
- SIGHTING — decays slowly; a direct sighting is trustworthy for a while
- SCENT — decays medium; diffuse and directional
- NOISE — decays fast; fresh for only a few ticks

When Tom needs direction, the Conductor selects the highest live-
confidence source and issues a directive toward it. When no source is
live, the Conductor falls back to a sector patrol sweep (2.4).

**Why typed sources, not a heatmap or a single point:** the *type*
carries tactical meaning. A NOISE source in a sector means "check the
area" (not "open every locker"). A SIGHTING means "I actually saw it
here." This typing is what lets the Conductor pick the right *mode*
(2.5) and what gives Jerry a manipulation target (a false noise creates
a real-but-wrong suspicion source).

`[TBD]` Exact decay half-lives per type. Tune in Stage 1.
`[TBD]` Whether multiple sources of the same type merge or stack.
`[TBD]` Max number of live sources (cap to keep behavior legible).

### 2.3 Sectors `[DECIDED]`

The grid is decomposed into sectors (e.g. a 3×3 or 4×4 zoning of the
30×30 grid). Sectors are the Conductor's *coarse reasoning vocabulary*:
suspicion is tracked per sector, refined to specific tiles only when a
source is high-confidence.

**Important boundary:** sectors are a Conductor abstraction. Tom still
moves tile-by-tile and pathfinds locally within/across sectors. Sectors
do NOT mean "Tom teleports between sector centers" or "Tom only moves
between zones" — that would look robotic. Sectors structure the
Conductor's belief and high-level routing; tile-level navigation stays
Tom's job.

`[TBD]` Sector grid resolution (3×3 vs 4×4 vs adaptive).

### 2.4 Patrol when no suspicion is live `[TBD]`

When no suspicion source is live, the Conductor directs a sector sweep.
Open question: what order? Options — round-robin, prioritize sectors not
visited recently, prioritize sectors near the last-known Jerry area,
weight by L2 historical heatmap (memory-informed patrol). The
memory-informed option is the most interesting and ties Phase 4 memory
into the Conductor directly.

### 2.5 The directive: (target, suggested_mode) `[DECIDED]`

The Conductor issues Tom a directive each tick:

```
Directive:
    target:         tile to move toward
    suggested_mode: INVESTIGATE | STALK | RUSH | BAIT | PATROL
```

The suggested mode is chosen from the belief *type and confidence*:
- Fresh SIGHTING, high confidence → STALK (close patiently) or RUSH
- NOISE in a sector → INVESTIGATE (check the area)
- High-confidence locker-type suspicion → BAIT available (fake walkaway,
  then ambush)
- No live source → PATROL

### 2.6 Chemistry can override the mode `[DECIDED]`

This is the load-bearing interaction. The Conductor *suggests* a mode;
Tom's chemistry can override it and sets execution speed. The rule:

> **The Conductor suggests what's tactically available; the body decides
> whether it has the patience for it.**

- High **adrenaline** → upgrade toward RUSH (commit now), regardless of
  what the Conductor suggested. An adrenalized predator over-commits.
- High **cortisol** (frustration from a stalling hunt) → unlock BAIT
  (try deception).
- **Calm** (low arousal) → leave the Conductor's suggestion intact.

**Chemistry can override CORRECT guidance, and that is a feature
(`[DECIDED]`).** A Tom with a high-confidence SIGHTING + STALK suggestion
can be pushed into RUSH by adrenaline, blow the ambush, and let Jerry
escape. This is Tom's flaw — and Jerry's manipulation target. A clever
Jerry (or human) can deliberately spike Tom's adrenaline (sudden close
pass, burst of noise) to provoke a premature commit. This mirrors the
Billy/APEX chemical-exploitation pattern from the wider ecosystem:
chemical illegibility is what makes the agent feel real AND what makes
it exploitable.

**The knob:** how strong must arousal be before it overrides the
Conductor? Set conservatively in Stage 1 (override only at extremes, so
the Conductor's guidance usually stands but a strongly-provoked Tom errs).
Co-evolution tunes where the line sits in Stage 2. This is a Tom-side
override weight — a "conductor weight" in the ADR-003 sense.

`[TBD]` Exact arousal thresholds for each override.
`[TBD]` Does mode also affect movement SPEED, or only behavior? (Design
intent from discussion: speed scales with arousal too.)

### 2.7 What the Conductor reads `[DECIDED]`

- **World events** (observable only): noise emissions + locations,
  Tom's own sightings, scent field. NEVER Jerry's true position.
- **Tom's memory tiers**: L1 sighting heatmap, locker suspicion, false-
  noise counts; L2 historical priors. These are exactly the "hints" a
  director uses, and wiring them into the Conductor finally makes Phase
  3-4 memory behaviorally load-bearing.

The Conductor does NOT need to read Tom's chemistry — chemistry stays
Tom's private execution-layer business (that's what makes 2.6 a Tom-side
override rather than a Conductor decision).

---

## 3. Build sequence

### Stage 1 — Scripted Conductor (Order A: verify before learning)

The Conductor and all its weights are hand-tuned. Goal: a static system
that produces sensible hunting — investigates sounds, follows hints,
corners prey, occasionally baits, occasionally over-commits when
adrenalized — WITHOUT any learning. This becomes the "ScriptedTom-
equivalent baseline" for the Conductor era.

- **6a — Suspicion-source belief.** The data structure, decay, creation
  from events. Unit-tested in isolation.
- **6b — Conductor scaffolding.** The class, the tick hook between
  world.step() and tom.__call__(), directive issuance. Tom's BFS
  targeting replaced by "pathfind to directive.target". Verify Tom's
  existing behavior still works with a no-op/passthrough Conductor first.
- **6c — Sectors + patrol.** Sector decomposition, patrol sweep when no
  live suspicion.
- **6d — Modes + chemistry override.** The five modes, the suggested-
  mode logic, the chemistry override rule.
- **6e — Verify the static system.** Eval scripted-Conductor-Tom vs the
  base generalist Jerry. Tune until hunting looks right (replays!) and
  catch rate is sane (expected: BELOW BFS-Tom's, because we removed the
  pathfinding crutch — that's correct, see §4).

### Stage 2 — Learnable Conductor + co-evolution

Only after Stage 1 produces a sound static system.

- **6f — Identify learnable parameters.** Conductor weights (decay
  rates, mode-selection thresholds, patrol weighting) and Tom-side
  override weights become a parameter vector.
- **6g — Co-evolution scheduler.** Alternating or simultaneous Jerry/Tom
  updates, generation snapshots.
- **6h — Hall of fame** (ADR-005). Archive Toms; spawn fresh Jerrys
  against old Toms to force generalization and prevent strategy cycling.
- **6i — Metrics + tournament harness.** Progress measured by improvement-
  over-generations and generalization, NOT raw catch rate (§4).

---

## 4. Measurement caution `[DECIDED]`

Removing BFS makes early Conductor-Tom WEAKER than ScriptedTom —
generalist already beats greedy-Tom 74%. So:

- Raw catch rate is NOT the progress signal. Early co-evolution catch
  rates will look bad and that's expected.
- Progress = (a) improvement of Tom-gen-N over Tom-gen-1 at the gen-1
  task, and (b) generalization to held-out Jerrys/conditions.
- The Phase 5 ceiling diagnostic conditions (greedy/nearsight/bigmap)
  become a generalization test suite: "how does co-evolved Tom do
  against conditions it never trained against?"

---

## 5. Open questions

- `[TBD]` Suspicion decay half-lives per type.
- `[TBD]` Sector resolution.
- `[TBD]` Patrol ordering (memory-informed is the interesting option).
- `[TBD]` Chemistry override thresholds.
- `[TBD]` Does mode affect movement speed, or only behavior?
- `[TBD]` Stage 2: Tom's learnable substrate — Conductor weights only?
  ChemicalTom config too? A PPO Tom? Hybrid per ADR-003?
- `[TBD]` Stage 2: co-evolution schedule (alternating vs simultaneous,
  generations vs continuous).
- `[TBD]` Hall-of-fame composition: generalist-only seed was chosen, but
  does the HoF accumulate co-evolved Jerry/Tom checkpoints over time?
- `[TBD]` How the Phase 8 stalker layers on: the Conductor gains a fear
  signal and a stalk-vs-commit directive. Confirm this is purely
  additive to the Conductor built here.

---

## 6. How this sets up Phase 8

The Conductor built here IS the director the Phase 8 stalker needs. The
stalker is then a natural extension, not a rebuild:
- The Conductor gains a **fear signal** (estimated from observable Jerry
  behavior, per the Phase 8 doc's behavioral-signal layer).
- It gains a **kill-threshold / stalk-vs-commit** directive: below the
  fear threshold, prefer STALK/BAIT (maintain pressure, don't end it);
  at/above, allow RUSH/commit.
- The chemistry-override mechanism (2.6) already produces the "Tom
  over-commits and blows it" drama that makes stalking tense.

In other words, Phase 6's Conductor + Phase 2's chemistry already
contain most of the stalker's machinery. Phase 8 mostly adds the fear
estimator and re-weights the modes around it.

---

## Document changelog

- 2026-05-20 — Initial draft. Architecture decided (Model 2; typed
  decaying suspicion sources; sectors; (target, mode) directive;
  chemistry-can-override-mode as a feature; Conductor reads world events
  + memory but not chemistry). Two-stage build sequence (scripted first
  per Order A, then learnable + co-evolution). Measurement caution and
  Phase 8 setup documented.
