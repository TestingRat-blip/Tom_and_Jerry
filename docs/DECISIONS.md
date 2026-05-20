# Tom_and_Jerry — Decision Log

Architectural decisions, in order, with rationale. Append-only. Each entry is dated and immutable; supersede by adding a new entry that references the old one.

---

## ADR-001 — Top-down 2D for v1, 3D deferred
**Date:** 2026-05-17
**Status:** Accepted

The AI architecture is perspective-agnostic. Top-down grid iterates 10x faster than any 3D environment, supports fully headless training, and isolates the AI as the variable under study. 3D port becomes a future axis once Tom is genuinely terrifying in 2D.

---

## ADR-002 — Single hunter species in v1, pack mechanics in v2
**Date:** 2026-05-17
**Status:** Accepted

Co-evolving along too many axes at once muddies the signal. Single species means we can measure persistent-learning and emergence-from-chemistry effects cleanly. Pack coordinator architecture is in place from day one but stubbed; v2 activates it.

---

## ADR-003 — Hunter is structurally scripted, parametrically learned
**Date:** 2026-05-17
**Status:** Accepted

Pure-RL hunters are unreadable. Players need to form mental models of the predator for tension to exist. Behavior tree skeleton stays hand-authored; node thresholds, drive baselines, chemical curves, and conductor weights are learned. This gives us Isolation's legibility with adaptability Isolation cannot reach.

---

## ADR-004 — PPO via Stable Baselines3 for Jerrys
**Date:** 2026-05-17
**Status:** Accepted

Consistent with prior OSRS RL pipeline experience. Mature, well-documented, supports population-based variants natively. Archetypes differentiated by reward shape, not policy architecture.

---

## ADR-005 — Hall of fame for co-evolution stability
**Date:** 2026-05-17
**Status:** Accepted

Pure adversarial co-evolution collapses into degenerate strategy corners. Periodic spawning of fresh-blood Jerrys against archived Tom snapshots forces Tom to remain effective against historical players, which forces generalization. Non-negotiable.

---

## ADR-006 — Project is strictly isolated from Vera and APEX
**Date:** 2026-05-17
**Status:** Accepted

Architectural lessons cross over freely. Code does not. Cross-contamination would muddy both projects. Tom_and_Jerry runs in its own process / venv / Docker compose. The chemical analog layer is *re-implemented* here from the same patterns, not imported.

---

## ADR-013 — Conductor (two-brain) architecture replaces BFS as Tom's strategic layer
**Date:** 2026-05-20
**Status:** Accepted (supersedes the implicit "BFS is Tom's targeting" assumption from Phase 1)

**Context.**
The Phase 5 ceiling diagnostic proved that BFS pathfinding is Tom's
dominant weapon: trained generalist Jerry survives 28% against BFS-Tom
but 74% against the same Tom with BFS disabled (greedy). BFS is a
hand-coded, non-adaptive strategic layer — it computes the shortest path
to Jerry's *true* position and walks it. This caps the co-evolution
arms race: a Jerry evolving against perfect pathfinding can only exploit
thin margins, because Tom always knows the optimal route to the real
Jerry.

This contradicts the design intent. ADR-003 already named "conductor
weights" as a learnable component, and the project's north star is the
Alien: Isolation two-brain system — a local creature brain plus a
director brain that knows more than the creature and feeds it hints.
BFS short-circuits that vision by giving Tom perfect strategic knowledge
for free.

**Decision.**
Replace BFS-as-targeting with a **Conductor** layer — a director brain
that sits above Tom (the local brain) and directs his attention.

- **Tom (local brain):** perception + movement + the existing five-state
  behavior tree, chemistry, drives, and memory. Tom pathfinds toward a
  *target the Conductor supplies*, NOT toward Jerry's true position.
  Tom never reads Jerry's ground-truth location directly.
- **The Conductor (director brain):** consumes world events (sounds and
  their locations, sightings, scent gradients) and Tom's memory tiers,
  maintains a *belief* about where Jerry is, and feeds Tom directives
  ("investigate here", "patrol this region", "Jerry was last seen
  there"). The Conductor's belief is necessarily imperfect — it works
  from hints, not omniscience.

Tom's competence now flows from how well the Conductor synthesizes
imperfect information, which is a tunable (and later learnable) thing —
not from a hand-coded shortest-path oracle.

This pulls the director architecture forward from Phase 8 (where it was
originally scoped as the stalker layer) into Phase 6 (as the core
hunting architecture). The Phase 8 stalker becomes a natural extension:
the Conductor gains a fear signal and a stalk-vs-commit directive,
rather than requiring a from-scratch director build.

**Sequencing (per Order A discipline):** build and verify the Conductor
as a *scripted/hand-tuned* system first — it must produce sensible
hunting (investigate sounds, follow hints, corner prey) as a static
system before any part of it becomes learnable. Only then does
co-evolution begin. This preserves the project's one-hard-thing-at-a-time
discipline: ScriptedTom worked before ChemicalTom; L1 worked before L2;
the scripted Conductor must work before the learnable Conductor.

**Consequences.**
- Phase 6 expands from "co-evolution" to "build Conductor architecture,
  then co-evolve." Likely a two-stage phase. Downstream phase numbers
  (pack mechanics, stalker) shift accordingly; numbering is not load-
  bearing, sequencing discipline is.
- The new scripted-Conductor + weakened-Tom system becomes the
  "ScriptedTom-equivalent" baseline for the Conductor era. Existing
  ScriptedTom is preserved as the Phase 1-5 reference.
- Co-evolution progress can NOT be measured by raw catch rate — early
  Conductor-Tom will be weaker than BFS-Tom (generalist already beats
  greedy 74%). Progress = improvement-over-generations + generalization
  to held-out Jerrys/conditions.
- Tom's interface gains a hard rule: the local brain never accesses
  Jerry's ground-truth position. All targeting flows through the
  Conductor's belief. This is the architectural invariant that makes
  the system honest (and that makes human-Jerry support possible later —
  the Conductor reads observable signals, which exist whether Jerry is
  PPO or human).
- ADR-005 (hall of fame) still applies and may matter more — a
  Conductor that can be confused has more degenerate-strategy corners to
  fall into.

---

## Template for future entries

```
## ADR-NNN — Short title
**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by ADR-XXX

Context.
Decision.
Consequences.
```
