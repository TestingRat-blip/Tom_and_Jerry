# Phase 6 Stage 1 — Retrospective (the scripted Conductor)

**Status:** Stage 1 complete. Stage 2 (learnable + co-evolution) not started.
**Wrapped:** 2026-05-20
**Scope:** ADR-013's two-brain architecture, built and verified as a
*scripted* system (Order A: verify before learning).

**Artifacts (batches 11a–11f):**
- `src/hunter/agent/conductor/belief.py` — typed decaying suspicion belief (6a)
- `src/hunter/agent/conductor/conductor.py` — director brain, observe + suggest (6b)
- `src/hunter/agent/conductor/sectors.py` — sector decomposition + LRV patrol (6d)
- `src/hunter/agent/conductor/modes.py` — hunt modes + chemistry override (6e)
- `ChemicalTom` — optional `conductor=` integration (6c, 6d, 6e)
- `--conductor` / `conductor-l2` eval + watch tooling (6f)
- 65 new tests across the conductor suite

---

## What Stage 1 set out to do

Per ADR-013: replace BFS-as-targeting with a Conductor (the director half
of an Alien: Isolation two-brain system), built scripted first. Tom's
local brain keeps running (chemistry, drives, behavior tree); the
Conductor replaces only the strategic "where is Jerry" decision, working
from a lossy, foolable belief instead of a perfect-pathfinding oracle.

## Exit criteria check

| Criterion | Result |
|---|---|
| Belief from observable signals only (ADR-013 invariant) | ✓ |
| Conductor builds sensible belief during real episodes | ✓ |
| Targeting routed through belief, not Tom's private memory | ✓ |
| Directed patrol covers the map (vs random wander) | ✓ (9/9 sectors) |
| Hunt modes with chemistry override | ✓ |
| `conductor=None` preserves exact Phase 2-5 behavior | ✓ (regression-guarded) |
| Scripted system produces legible hunting | ✓ (see findings) |
| Tooling to eval/watch Conductor on real checkpoints | ✓ |

Stage 1 complete. Whether the static system produces the *right* hunting
is partly verified (sandbox) and partly handed to Grove (trained-Jerry
eval — see open items).

---

## The architecture as built

Five components, each added to a working base (the project's house style):

1. **Belief (6a).** A set of typed, decaying suspicion sources —
   SIGHTING / NOISE / SCENT — each with its own half-life (~25 / ~12 / ~5
   ticks). Pure data structure, unit-tested in isolation. Merges nearby
   same-type signals, caps total sources. This is what Jerry manipulates:
   a false noise creates a real suspicion.

2. **Conductor scaffolding (6b).** The director brain. Ingests observable
   signals (sightings via the visibility gate, non-Tom noise events,
   projected scent) into the belief, decays it, suggests a target. Built
   observe-only first — a regression test proved Tom's behavior was
   byte-for-byte unchanged while the Conductor merely watched.

3. **Targeting handover (6c).** When a Conductor is attached, Tom's memory
   fields (`last_seen_jerry` / `last_noise`) are populated FROM the belief
   instead of direct perception. Downstream logic unchanged — only the
   *source* of the target changed. Flag-gated; `conductor=None` is exact
   Phase 2-5 behavior.

4. **Sectors + patrol (6d).** A 3×3 sector decomposition gives the
   Conductor a coarse vocabulary. When the belief is empty, it directs
   Tom toward the least-recently-visited sector — legible coverage sweeps
   instead of random wandering. Verified 9/9 sector coverage with real
   movement.

5. **Modes + chemistry override (6e).** The directive gains a mode
   (PATROL / INVESTIGATE / STALK / RUSH; BAIT deferred). The Conductor
   suggests a mode from belief type; Tom's chemistry can override it —
   high adrenaline upgrades to RUSH (over-commit). STALK holds at a
   distance rather than closing.

---

## Findings

### The STALK→RUSH "wind-up to pounce" emerged, unscripted

The single nicest result of Stage 1. STALK holds Tom at a distance
watching prey; while Jerry is in view, adrenaline climbs; it crosses the
RUSH threshold; the override fires; Tom commits. We did not script this
sequence — it falls out of the chemistry responding to visible prey while
STALK holds position. Every test episode showed it: a few STALK ticks,
adrenaline → 1.0, RUSH, catch. The override fired 3–7 times per episode —
each one a moment Tom's temperament overrode the Conductor's patience.

This is exactly the "Tom has a temperament that can sabotage his
strategy" thesis, working as designed, and it's the seed of the Phase 8
stalker: STALK is the stalking, RUSH is the pounce, and the timing is
emergent rather than coded.

STALK also self-limits — no deadlock. A naive worry was that "hold at
distance" would mean Tom never commits and episodes never resolve. The
adrenaline escalation prevents that automatically.

### The Conductor changes Tom's FAILURE MODE, not just his strength

ADR-013 predicted the Conductor would *weaken* Tom (because trained
generalist beat greedy-Tom 74%). The sandbox eval told a more
interesting story:

| Opponent | ScriptedTom | ChemicalTom+Conductor |
|---|---|---|
| passive Jerry | 75% catch / 157t | 70% catch / 143t |
| random Jerry  | 100% catch / 110t | 100% catch / **46t** |

Against quiet/passive prey, the Conductor is slightly *weaker* (the
foolable belief loses targets crisp memory would hold) — as predicted.
But against noisy/moving prey, it's dramatically *faster* (46 vs 110
ticks), because every Jerry move generates a noise suspicion the
Conductor RUSHes toward.

**The Conductor doesn't uniformly weaken Tom — it makes him noise-
sensitive.** This is BETTER than uniform weakening for co-evolution: it
hands Jerry a concrete strategic target. *Be quiet. Noise is what feeds
the Conductor.* The sneaker archetype suddenly has a real reason to
exist, and the arms race has a genuine axis (stealth vs noise-tracking)
rather than just "evade the pathfinder."

### The handover changes internal state more than visible actions (on passive Jerry)

A subtlety from 6c: with a passive Jerry, the belief-driven targeting
diverges from perception-driven targeting in INTERNAL STATE immediately
(Tom enters INVESTIGATE on noise suspicions plain Tom ignores), but the
ACTIONS sometimes reconverge (INVESTIGATE-toward-X and PATROL can pick
the same step). The behavioral impact grows with how much Jerry moves and
makes noise — i.e. it's largest exactly where it matters (active prey),
smallest where it doesn't (a sitting target).

---

## What didn't work / what was harder than expected

### Tests that measured the wrong thing (twice)

Two test-design mistakes, both caught and both instructive:

1. **The 6c divergence test** checked action-sequence equality, which is
   too brittle — actions coincidentally reconverge. Fixed to assert on
   internal targeting state (the real change).

2. **The 6d coverage test** compared directed-vs-random patrol over a
   passive-Jerry episode, but Tom barely patrols against a passive Jerry
   (he's mostly chasing), so the comparison was noise (29 vs 30 sectors).
   A hand-rolled greedy walker in a follow-up test then got stuck on walls
   and falsely suggested patrol covered only 2 sectors. The fix: test
   patrol correctness directly, and confirm coverage with Tom's REAL BFS
   movement (which gave a clean 9/9).

The lesson, twice over: **test against the real machinery and the right
observable, not a simplified reimplementation or a brittle proxy.** Same
discipline as the Phase 5 "trust replays over the scoreboard" finding.

### BAIT deferred

BAIT (approach-then-retreat deception) is the richest mode and the most
characterful — it's the literal "play with food." It's a stateful multi-
tick maneuver (approach, retreat, wait, re-approach) that deserves
dedicated design and tests, so it was deferred rather than rushed into
6e. The enum includes it; it currently falls back to INVESTIGATE. The
cortisol→BAIT-unlock path is deferred with it.

---

## Honest take

Stage 1 is the most architecturally ambitious work in the project so far
and it landed cleanly across five batches without ever breaking the
existing system (the `conductor=None` path stayed exact throughout). The
two-brain architecture exists end-to-end as a scripted system: it
perceives, targets, searches, and modulates approach, with Tom's
chemistry able to override its strategy.

The headline finding — that the Conductor makes Tom noise-sensitive
rather than uniformly weaker — is arguably a *better* outcome than the
ADR-013 prediction, because it gives co-evolution a real axis to work.

The honest gap: the decisive measurement (Conductor-Tom vs the TRAINED
generalist, not sandbox passive/random Jerrys) hasn't been run. The
tooling now exists (`--tom-policies conductor`); the run is Grove's.

---

## What Stage 1 hands to Stage 2

1. **A working scripted Conductor** with all weights hand-set and
   centralized in config dataclasses (BeliefConfig, ConductorConfig,
   SectorConfig, ModeConfig). These ARE the learnable parameter vector
   for co-evolution — decay half-lives, merge radius, scent threshold,
   sector resolution, stalk distance, the rush-adrenaline threshold.

2. **A clear arms-race axis.** Stealth-vs-noise-tracking is the surface.
   Co-evolution should produce quieter, more deceptive Jerrys and a
   Conductor that learns to weight noise less naively (e.g. discount
   noise that pattern-matches past misdirection — which is exactly what
   L1's false-noise machinery already does, now newly relevant).

3. **The eval tooling** (`--tom-policies conductor conductor-l2`) for
   measuring co-evolved generations and generalization.

4. **The measurement caution from ADR-013 stands.** Don't measure
   co-evolution progress by raw catch rate; the Conductor changes failure
   mode, so catch rate alone is misleading. Use improvement-over-
   generations and generalization across opponent types (quiet vs noisy).

---

## Open items (for Grove / Stage 2)

- `[RUN]` The decisive eval: trained generalist vs `chemical` vs
  `conductor` (and `conductor-l2`). Does the Conductor weaken hunting
  against a real trained Jerry, and how does it change with memory?
  Command: `python -m scripts.eval_archetypes --tom-policies chemical conductor --episodes 50`
- `[WATCH]` Eyeball the Conductor hunting: `python -m scripts.watch
  --jerry model:<generalist> --tom conductor --seed 42`. Look for the
  STALK→RUSH wind-up and any visibly dumb behavior.
- `[TUNE]` Mode thresholds (rush_adrenaline_threshold, stalk_hold_distance)
  and belief decay rates — tune against what the replays show before
  Stage 2 makes them learnable.
- `[BUILD]` BAIT mode — the deferred deception maneuver.
- `[BUILD]` Stage 2: identify the learnable parameter vector, the
  co-evolution scheduler, the hall of fame (ADR-005).

---

## Discipline calls preserved

1. **Verify before learning (Order A).** The scripted Conductor works
   before any of it becomes learnable. Same as ScriptedTom-before-
   ChemicalTom, L1-before-L2.
2. **Every component optional, regression-guarded.** `conductor=None` is
   exact prior behavior; the no-conductor test suite never broke across
   five batches.
3. **Observe before driving.** The Conductor watched (6b) before it
   steered (6c) — proven non-interfering first.
4. **Test against real machinery.** Both test-design mistakes were caught
   by re-checking against actual movement/state rather than proxies.
5. **Defer rather than rush.** BAIT deferred to its own batch instead of
   half-built into 6e.

---

## Document changelog

- 2026-05-20 — Initial Stage 1 retrospective. Documents the five-component
  scripted Conductor (belief / scaffolding / handover / patrol / modes),
  the emergent STALK→RUSH wind-up, the "changes failure mode not strength"
  eval finding, the two test-design mistakes, BAIT deferral, and the
  handoff to Stage 2.
