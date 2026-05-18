# Phase 2 — Retrospective

**Status:** Complete
**Wrapped:** 2026-05-18
**Artifact:** `src/hunter/agent/behavior/chemical_tom.py` — `ChemicalTom` class

---

## Exit criteria check

From the roadmap:

> Same Tom visibly behaves differently in different chemical states.
> Stalking vs frenzied vs cautious are distinguishable on replay.

| Criterion | Target | Achieved |
|---|---|---|
| 6-axis drive vector | ✓ | hunger, aggression, caution, curiosity, fatigue, social_bond |
| 5-chemical analog layer | ✓ | adrenaline, cortisol, dopamine, oxytocin, serotonin |
| Buffer-then-level transfer | ✓ | Vera v2 pattern; no event saturation |
| Cross-chemical interactions | ✓ | 3 couplings (adrenaline⊣cortisol, serotonin⊣adrenaline, dopamine→serotonin) |
| Drive/chemistry → behavior tree modulation | ✓ | 5 modulated thresholds |
| Prediction horizon scales with adrenaline | ✓ | 0–3 step linear extrapolation on average velocity |
| Renderer shows chemistry/drives state | ✓ | 11 bar charts + Tom adrenaline tint + prediction marker |
| Backwards-compatible replay format | ✓ | Old Phase 1 replays still load |

**Phase 2 is functionally complete.**

---

## Headline numbers

ChemicalTom vs ScriptedTom over 30 trials per Jerry pattern, max 400 ticks:

| Jerry pattern | ScriptedTom catch rate | ChemicalTom catch rate |
|---|---|---|
| Passive (always WAIT) | 70% | 70% |
| Pure oscillator (N/S/N/S) | 30% | 27% |
| Drift + oscillate | 80% | 77% |

After softening the default modulation coefficients, ChemicalTom matches
ScriptedTom within 1–2 catches over 30 trials on every pattern tested.
The architecture is wired in; the *behavior* is similar by design — see
"On the coefficients" below.

---

## What worked

- **Two-stage buffer→level chemistry** (borrowed from Vera v2) prevents
  single events from saturating chemicals. Sustained stimulus produces
  smooth ramps; isolated spikes decay cleanly.
- **Three modest cross-chemical interactions** are enough to produce
  qualitatively distinct behavioral postures across encounters. Adrenaline
  suppresses cortisol → frantic Tom doesn't *feel* tired. Serotonin caps
  adrenaline → confident Tom doesn't get rattled. Dopamine raises
  serotonin → success slowly builds confidence.
- **Event filtering by actor** keeps Tom's chemistry clean of Jerry-internal
  events (e.g., JERRY_ENTERED_LOCKER doesn't move Tom's drives).
  Test `test_jerry_internal_events_dont_affect_tom` pins this.
- **Frame extension was backwards-compatible.** Old Phase 1 replays still
  load with empty chemistry/drives dicts. The renderer renders both
  fluidly — no chemistry section appears for non-chemical Toms.
- **Test-driven discovery of coefficient pathology.** The catch-rate
  comparison surfaced that initial default modulation coefficients were
  too aggressive, hurting Tom's scent-tracking. Softening them roughly
  in half restored parity without touching the architecture.

---

## What didn't work — the flash-prediction issue

Watching ChemicalTom play against the trained Jerry made the central
limitation obvious: **adrenaline-coupled prediction operates on the wrong
timescale.**

The intended design:
1. Tom sees Jerry → adrenaline spikes
2. Adrenaline ≥ 0.3 → Tom predicts Jerry's future position
3. Tom paths to predicted tile, intercepts the dance

The reality:
1. Tom sees Jerry → adrenaline spikes
2. Tom predicts for 1–2 ticks (visible as the orange X marker)
3. Tom loses sight (Jerry's dance moves him out of LOS)
4. Adrenaline decays at 0.85/tick → below threshold in ~3–4 ticks
5. Tom reverts to chasing current tile
6. Dance resumes

Prediction is the right *mechanism* for breaking the dance, but coupling
it to a transient chemical was a category error. **Prediction isn't an
emotion, it's a capability.** Biological predators don't *decide* to
predict during high arousal — they continuously model their prey
whenever the prey is visible.

We considered several fixes (always-predict-on-sight; sustained
adrenaline during PURSUE; oscillation-pattern recognition) and elected
to **NOT** apply any of them in Phase 2. Reasoning:

- The Phase 2 exit criterion is "Tom visibly behaves differently in
  different chemical states." That's met. The dance is a Phase 6
  problem.
- Endless hand-tuning of the predator-vs-prey loop is exactly what
  co-evolution is supposed to escape. If we keep stacking patches
  until ChemicalTom beats today's Jerry, the Jerrys evolved in
  Phase 5+ will end up specialized against this hand-tuned thing
  rather than against a clean substrate.
- The architecture is correct. The defaults are conservative on purpose.
  Phase 5+ should *learn* better coefficients via co-evolution.

A real fix for the flash-prediction issue is documented for Phase 5+
in the "Open notes" section below.

---

## On the coefficients

My first-pass coefficients made ChemicalTom **worse** than ScriptedTom
on oscillating Jerrys (13% vs 30% catch rate). After diagnosing, the
cause was clear:

- `pursue_memory_cortisol_mult = -0.6` shortened pursuit memory too
  aggressively when cortisol built up during long patrols
- `investigate_dwell_curiosity_mult = +0.8` made Tom over-commit to
  weak noise sources

Softening each by ~50% restored parity. The lesson generalizes:

> Modulation magnitudes in this kind of architecture are not free
> parameters that can be guessed. They need either (a) careful
> hand-tuning against held-out scenarios, or (b) population-based
> learning. Hand-tuning is brittle; the architectural choice is to
> use (b) in Phase 5+.

Current defaults represent a deliberate compromise: large enough that
behavior visibly differs across chemical states (catches the
architectural intent), small enough that worst-case configurations
don't catastrophically degrade Tom's tracking (preserves the
substrate's usefulness as a baseline).

---

## Honest take on what Phase 2 actually delivers

**Architecturally** — Phase 2 is the most distinctive piece of the
project so far. No other published predator AI has this structure:

- Asymmetric perception (Tom has scent, Jerry doesn't)
- Slow-changing motivational state (drives)
- Fast-changing emotional modulators (chemistry)
- Behavior tree with parametric modulation, structurally scripted
- Prediction horizon coupled to internal state

This is the substrate that Phase 5's six archetypes and Phase 6's
co-evolution can sit on. Without it, those phases would have nothing
distinct to evolve.

**Behaviorally** — ChemicalTom is roughly indistinguishable from
ScriptedTom in catch rate against the prey patterns we tested. The
behavioral differences are visible on replay (drives/chemistry bars,
prediction marker, body color shifts) but don't yet translate to
clear *performance* differences.

This is the right Phase 2 outcome, but it's worth being honest that
"chemistry will make Tom feel like a creature" was somewhat oversold
in earlier project framing. What chemistry *visibly* does in v1:

- Tom's body color tints red when adrenalized
- Pursuit duration varies with aggression/cortisol balance
- Investigation duration varies with curiosity
- A few cross-tick chemical dynamics produce subtle postural shifts

What chemistry *doesn't visibly do* in v1:

- Make Tom obviously frenzied vs cautious in a single screenshot
- Produce dramatic behavioral pivots within an episode
- Solve the dancing exploit

The architecture supports all of these. The default coefficients don't
demonstrate them. Phase 5+ has the levers to crank them up via
evolution.

---

## What Phase 2 unlocks for later phases

- **Phase 5 archetypes** can each have a distinct `ChemicalTomConfig`
  baseline. A "frenzied" archetype starts with high aggression baseline;
  a "patient" archetype starts with high caution baseline. The same
  underlying ChemicalTom code produces all six.
- **Phase 6 co-evolution** can vary the modulation coefficients
  themselves via population-based training. The current 12-ish
  hand-tuned constants in `ChemicalTomConfig` become a 12-axis
  evolvable parameter space.
- **Phase 4 persistent memory** can be added as an additional layer
  ON TOP of chemistry. L1/L2/L3 memory tiers will influence which
  drives/chemicals get pre-loaded at episode start. A Tom that has
  died to oscillation 100 times in past episodes can start with
  higher caution and slightly higher cortisol — biasing toward
  *expecting* tricks.

---

## Open notes for Phase 5+

These are findings that should inform later phases:

1. **Decouple prediction from adrenaline.** Make prediction always-on
   while Jerry is visible. Keep adrenaline coupled to *threshold*
   modulation (pursue memory, noise threshold, attack commitment) but
   NOT to the prediction itself. Capability vs. emotion.

2. **Average-velocity prediction is too simple.** Recognize oscillation
   as a distinct motion pattern. A history of `[(5,5), (5,6), (5,5),
   (5,6)]` should trigger "oscillator detected" and produce a different
   intercept strategy — maybe ATTACK the midpoint, or break off and
   return from a different angle.

3. **Chemistry timescales need a rethink for the dance specifically.**
   Currently adrenaline decays in ~4–8 ticks. Pursuit episodes can
   easily run 30+ ticks. Consider per-state decay overrides: during
   PURSUE, slow adrenaline decay; during PATROL, current fast decay.

4. **Coefficient defaults document v1 behavior — they are not law.**
   The currently-shipped defaults make ChemicalTom roughly equivalent
   to ScriptedTom. Phase 5/6 should explicitly explore *higher*
   modulation magnitudes, accepting that some configurations will be
   strictly worse than ScriptedTom. The point isn't to win — it's to
   explore the space.

5. **The dancing exploit is robust because Jerry's policy is
   deterministic.** Once Phase 6 co-evolution mixes in stochastic
   Jerrys (from the hall of fame), the dance will frequently fail to
   start cleanly, and Toms with even modest prediction will start to
   win consistently. Don't fix the dance in Phase 2/3/4 — let
   Phase 6 fix it organically.

---

## What I'd tell a future contributor

If you're looking at ChemicalTom and wondering "why doesn't this beat
the dance?":

- The architecture is correct. Trust the tests.
- The defaults are deliberately conservative.
- This is the *substrate* for Phase 5/6, not a finished hunter.
- If you find yourself hand-tuning coefficients to win against a
  specific Jerry pattern, stop. You're optimizing against a target
  that Phase 6 should evolve against.
- Cool things this substrate enables: heterogeneous archetypes with
  distinct baseline drives/chemistry, learned modulation weights,
  prediction-mode pluggability, chemistry-coupled memory bias.

---

## Artifacts pinned

- `src/hunter/agent/drives/` — drives system (Batch 7a)
- `src/hunter/agent/chemistry/` — chemistry system (Batch 7b)
- `src/hunter/agent/behavior/chemical_tom.py` — wired-up ChemicalTom (Batch 7c)
- Frame extensions in `src/render/replay/recorder.py`
- Bar-chart panel + prediction marker in `src/render/pygame_renderer/renderer.py`
- 53 new tests (16 drives + 21 chemistry + 16 ChemicalTom integration)

Total project state: **146 tests passing in ~30s.**
