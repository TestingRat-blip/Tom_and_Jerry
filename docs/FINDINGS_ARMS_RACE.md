# Findings — The Arms Race Is Real (counter-Jerry experiment)

**Date:** 2026-05-20
**Context:** Phase 6 Stage 1 complete (scripted Conductor). Before
committing to Stage 2 (co-evolution), we ran a single counter-Jerry
training experiment (Option A) to de-risk: can a Jerry learn to beat the
Conductor, and what does that reveal?

This doc captures what we found and the design decision it forces.

---

## The experiment

Trained one PPO Jerry (generalist reward shape) for 1.5M steps against a
FIXED Conductor-Tom (`--tom conductor`), producing
`jerry_generalist_vs_conductor`. Then evaluated it against both hunters
and across training vs held-out seeds.

## The numbers

**The arms-race matrix** (50-episode evals, stochastic unless noted):

| Jerry | vs BFS-Tom | vs Conductor |
|---|---|---|
| original generalist (open-field dance) | ~28-34% | **6%** |
| counter-Jerry (cover dance / LOS-break) | 44% | 40-58% |

**Brittleness check** (counter-Jerry vs Conductor):

| Condition | Survival |
|---|---|
| training seeds (42+), stochastic | 40% |
| held-out seeds (9999+), stochastic | 58% |
| held-out seeds (9999+), deterministic | 76% |

---

## Finding 1 — The counter-Jerry's strategy is ROBUST, not brittle

Held-out survival went UP (40% → 58%), not down, when tested against
Conductors seeded differently from training. A brittle exploit overfit to
one Conductor's timing would survive WORSE on unseen Conductors. It
survived better. Therefore the strategy exploits something STRUCTURAL
about the Conductor, not memorized timing — it transfers.

The structural thing it exploits: **belief decay on line-of-sight break.**
The counter-Jerry learned to dance behind cover, constantly breaking LOS.
Each time it vanishes, the Conductor's SIGHTING suspicion decays, Tom's
adrenaline falls, the RUSH commit pressure releases, and Tom has to
re-search. The Jerry weaponized the exact belief-decay mechanic we built.

This is genuinely intelligent evasion — reading the two-brain system's
real structure — not the dumb open-field oscillation that beat BFS-Tom.

## Finding 2 — There IS a precision component, but the skill floor is high

The det/stoch gap persists on held-out seeds (76% det vs 58% stoch, 18
points). So the deterministic policy executes the cover-dance more
precisely than the noised version — a precision bonus. But the STOCHASTIC
floor is 58%: even heavily perturbed, the strategy survives well. A pure
memorized exploit would have a stochastic floor near zero (noise shatters
the precise sequence). 58% means robust-with-precision, not
precision-or-nothing.

This refines our exploit-detection rule for Stage 2:

| det/stoch pattern | meaning |
|---|---|
| wide gap + LOW stochastic floor | brittle exploit (ALARM) |
| wide gap + HIGH stochastic floor | robust skill + precision bonus (fine) |
| narrow gap | robust skill (ideal) |

The counter-Jerry is case 2. The original open-dance generalist (90% det
/ 60% stoch in training but 6% vs Conductor) was the cautionary case —
high training numbers masking a strategy that was catastrophic against
the wrong hunter.

## Finding 3 — No single strategy dominates both hunters

The matrix shows it plainly: the open-dance beats BFS-Tom decently but
DIES to the Conductor (6%). The cover-dance beats the Conductor AND is
better against BFS-Tom (44% vs ~30%). The counter-Jerry isn't a pure
specialist — it found genuinely better general evasion — but the open-
dance's catastrophic 6% shows how brutally a strategy can fail against
the hunter it wasn't built for.

This is the empirical justification for a HALL OF FAME, demonstrated
rather than asserted: if Jerry only ever faces one hunter, it collapses
into a strategy that exploits that hunter and may fail catastrophically
against others. Forcing Jerry to face BOTH hunter types (and forcing Tom
to catch BOTH dance types) is what prevents single-opponent collapse.

## Finding 4 — The same exploit pattern has now appeared THREE times

1. Phase 1: generalist learned open-field oscillation vs BFS-Tom.
2. Phase 5: archetypes reward-hacked instead of learning strategy.
3. Now: counter-Jerry learned the cover-dance vs the Conductor.

The lesson is consistent: **a single fixed opponent, given a dedicated
optimizer with unlimited tries, always gets exploited.** This is not an
architecture failure — it's a fundamental property of optimization
against a stationary target, and it's the entire reason co-evolution +
hall of fame exists as a method.

---

## The decision this forces: WHAT IS TOM'S LEARNABLE SUBSTRATE?

Co-evolution assumes BOTH sides learn. Jerry learns via PPO. But **Tom
currently cannot "learn" in the same sense** — Tom is scripted-with-config
(ScriptedTom + ChemicalTom + Conductor, all hand-tuned parameters). There
is no gradient, no policy network on Tom's side.

So before Stage 2 can be real co-evolution, we must answer the open
`[TBD]` from the Phase 6 design doc: what is Tom's learnable substrate?

The cover-dance result makes this concrete. To beat the cover-dance, the
Conductor needs to stop releasing pressure the instant LOS breaks — e.g.
hold position near where Jerry vanished, predict re-emergence, or treat
"repeated deliberate LOS-breaking near cover" as a signal it's being
played. Can that be achieved by TUNING CONFIG, or does it need LEARNED
BEHAVIOR?

Three options (from the Phase 6 design doc, now urgent):

**Option 1 — Conductor config as the learnable substrate.**
Tom "learns" by optimizing its config vector (belief decay rates, merge
radius, scent/noise thresholds, sector resolution, stalk distance,
rush-adrenaline threshold, and NEW params like "hold-near-last-seen on
LOS-break"). Optimization via evolution strategies / CMA-ES / random
search rather than PPO (the config is low-dimensional and non-
differentiable).
- Pro: keeps Tom legible (ADR-003 — players form mental models), reuses
  the whole scripted architecture, low-dimensional so fast to optimize.
- Con: limited to behaviors expressible by the existing parameter set;
  beating the cover-dance might need a behavior the config can't express.

**Option 2 — PPO Tom.**
Replace/augment the scripted brain with a learned policy network.
- Pro: maximally expressive; can in principle learn any counter.
- Con: unreadable (violates ADR-003's legibility intent); throws away the
  scripted architecture; the project's whole thesis is structurally-
  scripted-parametrically-learned (ADR-003), and a PPO Tom abandons that.

**Option 3 — Hybrid (ADR-003's stated design).**
Behavior-tree skeleton + Conductor stay hand-authored; the PARAMETERS
(config vector + "conductor weights") are learned. This is literally what
ADR-003 already commits to: "node thresholds, drive baselines, chemical
curves, and conductor weights are learned." Option 1 is essentially this,
with the learnable set scoped to config params.

**Leaning (to discuss):** Option 3/1 — learn the config vector, keep the
structure scripted. It's what ADR-003 already committed to, it preserves
legibility, and the cover-dance counter (hold-near-last-seen on LOS-break)
is plausibly expressible as new config params + a small behavior-tree
addition. If we find a counter the config genuinely can't express, that's
the moment to revisit Option 2 — but not before.

A key sub-question if we go Option 1/3: **how do we add the "hold on
LOS-break" capability?** It's probably a new Conductor behavior (don't
let the SIGHTING suspicion decay so fast when it was lost near cover; or
add a "last-seen anchor" the patrol biases toward) plus a config param
controlling its strength. That's a scripted addition whose strength
becomes learnable — exactly the ADR-003 pattern.

---

## Recommended next steps (for discussion)

1. **Decide Tom's learnable substrate** (Option 1/3 lean). This is the
   gating decision for all of Stage 2.
2. **Manual experiment first (cheap):** before building any optimization
   loop, hand-tune the Conductor config (and/or add a hold-on-LOS-break
   behavior) and see if it beats the cover-dance. If a config tweak beats
   it, Tom's substrate can be "just config" and Stage 2 is lighter. If no
   config tweak touches it, we need richer learnable behavior.
3. **Then build Stage 2** with the substrate decision made: the
   co-evolution scheduler (alternate Jerry-PPO updates with Tom-config
   optimization), the hall of fame (both Jerry checkpoints and Tom
   configs), and the det/stoch-gap + stochastic-floor exploit monitor.

---

## What's good about where we are

The counter-Jerry experiment did exactly its job. It proved:
- A Jerry CAN robustly beat the scripted Conductor (so the Conductor
  isn't a dead end — it has a real, learnable weakness).
- The way it does so (LOS-break / belief-decay exploit) reveals a
  CONCRETE frontier for the next Tom generation.
- The hall of fame is empirically necessary (no strategy dominates both
  hunters).
- The det/stoch gap + stochastic floor is a working exploit detector.

And the cover-dance itself is a lovely result: a PPO agent reading the
real mechanics of a hand-built belief system and exploiting them
intelligently. That's the kind of emergent behavior the whole project
exists to produce.

---

## Document changelog

- 2026-05-20 — Initial findings doc. Arms-race matrix, robustness
  confirmation (held-out survival rose), the three-case exploit-detection
  rule, the three-time exploit pattern, and the forced decision on Tom's
  learnable substrate (leaning Option 1/3 per ADR-003).
