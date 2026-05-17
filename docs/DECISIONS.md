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

## Template for future entries

```
## ADR-NNN — Short title
**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by ADR-XXX

Context.
Decision.
Consequences.
```
