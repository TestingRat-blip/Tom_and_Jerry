# Phase 5 — Retrospective

**Status:** Complete (with a deliberate scope change mid-phase)
**Wrapped:** 2026-05-20
**Artifacts:**
- `JerryRewardConfig` archetype factory methods + new reward terms (`src/env/gym_env.py`)
- `scripts/train_phase5.py` — archetype training
- `scripts/eval_archetypes.py` — head-to-head evaluation
- `scripts/diagnose_jerry_ceiling.py` — survival-ceiling diagnostic
- Six trained Jerry checkpoints (3 to 1.5M, 3 partial to ~1M)

---

## What Phase 5 set out to do

Build six behaviorally-distinct Jerry archetypes via reward shaping —
sneaker, sprinter, trickster, camper, explorer, generalist — to serve
as a diverse prey population for Phase 6 co-evolution. Same neural
architecture, different reward configs (the clean-experiment choice).
Train each against a fixed ScriptedTom (the stable-opponent choice).

The bet: hand-shaped reward functions would produce a useful, diverse
prey population to co-evolve against.

## What actually happened

The bet partly failed, and the failure was instructive.

### Exit criteria check

| Criterion | Result |
|---|---|
| Six archetype reward shapes implemented + tested | ✓ (23 tests) |
| Training infrastructure (per-archetype, checkpointed) | ✓ |
| Eval harness (head-to-head + vs multiple Toms) | ✓ |
| Six archetypes trained | Partial — 3 to 1.5M, 3 killed early at ~1M |
| Archetypes behaviorally distinct | ✗ — see below |
| Diverse prey population ready for co-evo | ✗ — pivoted to single seed |

### The numbers

50 episodes per archetype vs ScriptedTom, stochastic inference:

| Archetype | Survival | Mean len | Mean reward |
|---|---|---|---|
| sneaker | 32% | 162 | +9.80 |
| generalist | 28% | 170 | +9.51 |
| sprinter | 24% | 150 | +22.23 |
| trickster | 24% | 160 | +8.48 |
| camper | 24% | 157 | +2.10 |
| explorer | 20% | 136 | +4.77 |

Survival compressed to a 12-point band (20-32%). And the replays
revealed the compression hid a worse problem.

### The reward-hacking finding

Reward numbers suggested the archetypes were distinct (sprinter's +22
reward vs camper's +2 is a 10x spread). But watching replays showed the
reward numbers were measuring **reward-hacking, not strategy**:

- **Sprinter** (highest reward, +22) ran down 2 tiles and died in 42
  ticks on seed 42. It learned to bank distance-from-Tom reward in the
  opening moves, then stopped mattering. Reward-rich, behavior-poor —
  the textbook proxy-optimization failure. It optimized "distance" (the
  proxy) instead of "survival" (the goal).
- **Explorer** (worst survival, 20%) found a corner and sat in it until
  death — the *opposite* of its intended novelty-seeking. The new-tile
  bonus apparently saturated so fast that sitting still became locally
  optimal.
- **Camper** (lowest reward, +2) never committed to camping. Its
  locker-dwell bonus and open-tile penalty roughly cancelled, leaving a
  confused policy that survived only when it accidentally camped.

So at least three of the five new archetypes did not learn their
intended behavior. The reward shaping produced reward-hackers, not
strategists. **This is exactly why we check replays instead of trusting
the scoreboard.**

### The memory finding (chemical-l2 eval)

Ran the population against ChemicalTom-L2 (full Phase 4 memory) as well
as ScriptedTom. Result: the memory hunter was **not meaningfully better**
than the dumb pathfinder — slightly worse on 3 archetypes, same on 2,
better on only 1.

Cause: L2 had only 3 prior summaries to draw on. The persistent-memory
advantage is built on accumulating hundreds of episodes; 3 priors is
noise, not memory. We were effectively evaluating Phase 2 ChemicalTom
with the memory machinery attached but empty. **This confirms (rather
than contradicts) the architecture: memory is latent until something
fills it.** It also confirms the Phase 2 retro's prediction that
ChemicalTom ≈ ScriptedTom in raw catch rate by design.

### The ceiling diagnostic

Before settling on a base Jerry, we asked: why does generalist cap at
~28% survival? Is that a weak Jerry policy, or a strong hunter?

Ran the generalist checkpoint (no retraining) against four conditions:

| Condition | Survival | Δ vs baseline |
|---|---|---|
| baseline (BFS, sight 10, 30×30) | 28% | — |
| greedy (BFS disabled) | 74% | **+46** |
| nearsight (sight range halved) | 52% | +24 |
| bigmap (45×45, more lockers) | 62% | +34 |

**Every weakening produced a large jump.** The headline: removing BFS
pathfinding alone takes generalist from 28% to 74% survival. That
proves generalist *already contains a strong evasion policy* — it's
being suppressed by BFS specifically, not by any weakness in Jerry.

**Conclusion: 28% is a hunter-strength ceiling, not a Jerry-policy
ceiling.** Generalist is a good Jerry held down by a near-optimal
hunter. There is no meaningfully-better Jerry to train against this
hunter on this map; retraining would grind for an improvement the
environment physically caps.

---

## The scope change: from six archetypes to one base Jerry

Mid-phase, given the reward-hacking finding and the ceiling diagnostic,
we changed the plan:

- **Stop optimizing the archetype population.** Hand-shaped reward
  diversity produced reward-hackers. The fix isn't better hand-shaping;
  it's *co-evolution* — let diversity emerge from an arms race rather
  than from our guesses about reward.
- **Generalist is the base Jerry.** It's the one archetype that both
  trained cleanly and behaves sensibly, and the diagnostic proved it's
  near its policy ceiling. No retraining needed.
- **Seed co-evolution with generalist alone.** Single clean seed, not a
  population of mixed-quality hand-shaped policies.

This is not Phase 5 failing. It's Phase 5 producing a clear empirical
result — *static reward-shaping is a weak way to get diverse prey* —
that directly motivates the Phase 6 design.

---

## What worked

- **The clean-experiment design.** Same architecture + different rewards
  meant we could attribute outcomes to reward shapes, which is exactly
  what let us SEE the reward-hacking. A messier design would have hidden
  it.
- **Replay verification caught what metrics missed.** The reward
  scoreboard said "distinct archetypes." The replays said "reward-
  hackers." We trusted the replays. This discipline is the single most
  important methodological win of the phase.
- **The ceiling diagnostic.** A 5-minute, no-training experiment that
  converted a vague feeling ("generalist should be better") into a hard
  result ("28% is a hunter ceiling"). Cheap diagnostics that distinguish
  hypotheses are worth their weight.
- **Training infrastructure.** The per-archetype training + checkpointing
  + eval-callback pattern worked smoothly. The killed-early runs still
  produced usable checkpoints because checkpointing was frequent.
- **The reward-misspecification finding is genuinely valuable.** Sprinter
  getting max reward while dying instantly is a clean, teachable example
  of proxy-vs-goal divergence. Worth keeping as a reference.

## What didn't work

- **Hand-shaped reward diversity.** The core bet of the phase. Three of
  five new archetypes reward-hacked. Hand-designing "what good sneaker
  behavior looks like" and encoding it as reward is harder than it
  looks, and easy to get subtly wrong in ways that only show up in
  replays.
- **Training budget guess.** We committed to 1.5M steps for all six for
  experimental cleanliness, then killed three early due to stagnation
  anyway. The stagnation was real (converged-then-flat), so the early
  kills were correct, but it means the "all six at equal budget" plan
  was abandoned in practice.
- **The archetypes as a Phase 6 population.** Originally meant to be the
  co-evolution seed population. Now demoted — generalist seeds co-evo
  alone; the others are at most hall-of-fame regularization fodder
  (and Grove chose generalist-only, so even that is deferred).

---

## Honest take

Phase 5 is the first phase that didn't fully achieve its stated goal,
and that's fine — arguably good. The goal (diverse hand-shaped prey)
turned out to be the wrong goal, and the phase produced the evidence for
why. We end Phase 5 with:

- One genuinely good base Jerry (generalist, ceiling-verified)
- A proven finding that static reward-shaping produces reward-hackers
- A proven finding that BFS pathfinding is the hunter's dominant,
  non-adaptive weapon
- A clear, evidence-backed motivation for co-evolution

That last point matters most. Phase 6 was always next on the roadmap,
but now it's not just "the next phase" — it's *the answer to a problem
Phase 5 demonstrated*. Co-evolution exists precisely because hand-shaped
diversity failed. That's a much stronger justification than "the roadmap
said so."

---

## What Phase 5 hands to Phase 6

1. **The base Jerry.** `data/snapshots/jerry_generalist/final.zip`. A
   ceiling-verified competent evader. Single co-evolution seed.

2. **The key strategic finding: BFS is the hunter's crutch.** Generalist
   survives 74% the moment BFS is removed. This means a *static* hunter
   wins by pathfinding, not by cleverness — and co-evolution's job is to
   make Tom win by cleverness instead. Grove's call: **weaken Tom's
   pathfinding for co-evolution** to open a larger behavioral arms race.

3. **A measurement caution.** If Phase 6 weakens BFS, early co-evolved
   Tom will be WORSE than ScriptedTom (generalist already beats greedy
   Tom 74%). Progress can't be measured by raw catch rate; it must be
   measured by improvement-over-generations and generalization to
   held-out conditions.

4. **A design constraint surfaced.** "Weaken pathfinding" must become a
   *specific, tunable* weakening, not binary BFS-off. Pure greedy may be
   too weak (Tom can't apply pressure → arms race never starts). Likely
   candidates: BFS-with-noise, or limited-horizon BFS. The sweet spot is
   "weakened but still able to corner prey." This is a Phase 6 design
   decision.

5. **Reusable infrastructure.** The training script, eval harness, and
   ceiling diagnostic all carry forward. The eval harness's multi-Tom
   support is directly useful for "co-evolved Tom vs held-out Toms"
   generalization testing.

---

## Open questions for Phase 6

- `[TBD]` Exact pathfinding weakening: BFS-with-noise (what noise rate?)
  vs limited-horizon BFS (what horizon?) vs something else?
- `[TBD]` Co-evolution schedule: alternating (train Jerry N steps, then
  Tom N steps, repeat) vs simultaneous? Generations vs continuous?
- `[TBD]` Hall of fame: Grove chose generalist-only seed, but does the
  HoF accumulate past Jerry/Tom checkpoints to prevent cycling? (The
  original ADR-005 says yes — revisit.)
- `[TBD]` What's Tom's learnable substrate in co-evolution? ChemicalTom's
  config parameters? A PPO Tom? A hybrid (scripted structure, learned
  parameters per ADR-003)?
- `[TBD]` Progress metrics: since catch rate is unreliable under weakened
  BFS, what's the headline "co-evolution is working" signal?

---

## Discipline calls preserved

1. **Trust replays over the scoreboard.** The reward-hacking finding
   only surfaced because we watched the agents play.
2. **Cheap diagnostics before expensive retraining.** The ceiling
   diagnostic (5 min, no training) prevented a wasteful "retrain
   generalist harder" effort that the environment would have capped.
3. **Stop optimizing when evidence says you're at the ceiling.** We
   stopped tuning generalist the moment the diagnostic showed 28% was a
   hunter-strength artifact, not a Jerry weakness.
4. **A failed bet documented honestly is worth more than a success
   spun.** Phase 5 didn't achieve its stated goal; the retro says so
   plainly, because the *reason* it failed is the most valuable thing
   the phase produced.

---

## Document changelog

- 2026-05-20 — Initial retrospective. Documents archetype training
  outcomes, reward-hacking finding (sprinter/explorer/camper),
  chemical-l2 memory finding (latent until L2 fills), ceiling
  diagnostic (28% is hunter-strength ceiling, generalist is good),
  and the pivot to single-seed co-evolution with weakened-BFS Tom.
