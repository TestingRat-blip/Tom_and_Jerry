# Phase 8 — Tom-as-Stalker

**Status:** Design phase. No code yet. Doc evolves with discussion.
**Branched:** Will live in its own project repository when implementation begins.
**Depends on:** Phase 1-7 substrate (trained ChemicalTom with full memory + co-evolved population)
**Started:** 2026-05-18

---

## Tracking this document

This doc is **alive**, not final. Sections marked `[TBD]` are open questions
that get resolved through discussion as the design solidifies. Sections
marked `[DECIDED]` are locked-in calls with reasoning.

The doc has four parts in deliberate order:
1. **Why** — the intent and the experience we're building toward
2. **What** — concrete behavioral specifications (success criteria)
3. **How** — architectural design (the build plan)
4. **Failure modes** — what could go wrong and discipline calls

Part 1-2 is conversational and Grove-led. Part 3-4 is technical and
build-led. Edits welcome at any level.

---

## 1. Why this exists

### The architectural gap in the existing project

Phases 1-7 build a hunter that optimizes for **catch**. Tom's drives and
chemistry modulate his moment-to-moment behavior, his memory tiers store
what he's learned about Jerry, but the terminal reward is always "did
the chase end with a kill?" By every measurable axis, a Tom that catches
on tick 30 is a *better* Tom than one that catches on tick 200.

This is the standard predator-AI structure. It's also, on reflection,
not what most memorable predators in fiction actually do. The xenomorph
in *Alien: Isolation* doesn't end the chase the moment it could. Real
orcas play with seals before consuming them. Domestic cats torture mice.
The pattern "predator engages prey for longer than strict efficiency
requires" is robust across nature and fiction, and producing it from a
purely-efficient architecture is impossible by construction.

Phase 8 is the rebuild of the objective: Tom doesn't want to catch fast.
He wants to **terrorize until the food is afraid enough**, *then* catch.

### Where this fits in the project ecosystem

Vera is the guardian. APEX is the red team. Tom-as-Stalker is the third
corner — what cognition looks like when its drives align with cruelty
rather than care. Not in a moralistic sense, but in the architectural
one: same memory tiers, same chemistry, same persistence, but pointed at
a fundamentally different optimization target. The triad implicitly asks
"what is the shape of mind that emerges from different intent?" and
Phase 8 is one of the three answers.

### What we mean by "Tom at his peak"

Grove's framing. Phase 8 is the *culmination* of the project, not
another rung. Phases 1-7 produce a competent hunter; Phase 8 produces a
hunter that *uses* its competence toward an end the substrate alone
doesn't optimize for. The competence has to exist first — you can't be
patient with prey when you can't catch them at all.

This is also why it's its own project. The substrate work is a
contained, comparatively-conventional ML/RL project. The stalker work
is a layer on top that addresses a different kind of question. Mixing
them in one repo would obscure both.

---

## 2. What we're building (success criteria)

This is the **observational** specification. If the system doesn't
produce these scenarios on the regular, we haven't built the thing.

### Target behaviors

**Patient approach.** Tom does not close the distance immediately on
sighting Jerry. He maintains a tactical proximity — close enough to
maintain pressure, far enough that Jerry has room to feel hunted rather
than caught. The duration of this phase scales with Jerry's apparent
fear level (more fear → faster commit to attack; less fear → longer
stalking).

**Visible patience.** When Jerry hides in a locker, Tom doesn't
necessarily check the locker. Sometimes he walks past, lingers nearby,
then leaves the area — only to circle back. The "I know where you are
and I'm choosing not to act on it" signal is itself terrifying.

**Tactical retreat.** When Jerry's fear plateaus or declines (e.g.
Jerry seems to have settled into a hiding spot and isn't generating
panic signals), Tom retreats. Not to abandon the hunt — to allow Jerry
to think it's over, so the next encounter spikes fear higher.

**Asymmetric engagement.** Tom corners Jerry, makes eye contact (so to
speak), and *doesn't attack*. Then Tom leaves. This is the canonical
"plays with food" moment. The lack of catch isn't a failure mode; it's
the system working as designed.

**Eventual catch.** The episode does end with Jerry caught. Tom is a
predator, not just a stressor. When the fear-state reaches the kill
threshold AND positional conditions are right, Tom commits with full
intent and the chase ends. The catch should feel earned — the prey was
shaped by the chase, not just located.

### Anti-goals (what we are NOT building)

- A Tom that maximizes Jerry's suffering without bound. The system has
  to *resolve*; episodes that drag forever without resolution are bugs.
- A Tom that's just *slow*. Patience isn't a global slowdown — it's
  state-conditional. Tom should still be capable of fast, lethal closes
  when fear conditions are met.
- A Tom that explicitly reads "real" fear from Jerry's hidden state.
  All fear signals come through *observable behavior* (see Part 3).
  This is a hard architectural rule — Tom and the human player Jerry
  must operate through the same interface.
- A Tom that's unfair to a human Jerry. The system should produce
  *experiential* terror, not *mechanical* unfairness. A human Jerry
  who plays well should be able to escape; the terror is the journey
  even when the destination is survival.

### Success metrics (preliminary)

Existing project metrics (catch rate, ticks-to-catch) are *inadequate*
for Phase 8. We need new ones. Some candidates:

- **Mean fear-time integral** per episode — area under the fear curve
- **Fear at moment of catch** — high is good, indicates earned catches
- **Stalking phase duration** — wall-clock time Tom spends in STALK state
- **Reversal count** — how many times Tom approached, retreated, returned
- **Drama score** — composite metric weighted by all of the above

`[TBD]` Settle on a small set of canonical metrics. Probably 3-5
numbers we report per episode, with the integral one as the headline.

---

## 3. How we build it

### The four new components

Each of these is a TODO at this stage — written as a sketch, ready to
be elaborated when implementation begins.

#### 3.1 The Director Layer

A meta-agent that sits **above Tom**, runs once per tick *between*
`world.step()` and `tom.__call__()`, and modulates Tom's drives and
chemistry. Tom doesn't see the director, doesn't query it, doesn't even
know it exists. From Tom's POV, his internal state shifts in ways
slightly outside his own control — like a real animal whose nervous
system is doing things its conscious mind isn't tracking.

**Inspiration:** The Alien: Isolation (2014) two-brain architecture.
The xenomorph has two parallel decision layers:
1. **The alien's local brain** — runs the sensory and motor systems,
   makes moment-to-moment movement and hunt decisions based on what
   it can directly perceive.
2. **The director brain** — a separate process that has access to
   information the alien itself doesn't, including the player's
   location. The director leaks hints to the alien sparingly,
   producing the famous "the alien just *knows* where you are
   sometimes" experience without ever just teleporting it to the
   player.

Two agents, two information sets, one emergent behavior. We extend
this pattern: our director doesn't just leak position, it shapes
*intent*. The director estimates fear from observable Jerry behavior,
decides the appropriate engagement level, and pushes that decision
into Tom's drives and chemistry rather than directly controlling
Tom's actions. Tom's local brain (the ChemicalTom we've built across
Phases 2-7) still runs the show; the director just adjusts the
chemical milieu in which that brain operates.

**Inputs the director sees:**
- World state (Tom position, Jerry position, recent events)
- Behavioral signals derived from Jerry's actions (Part 3.2)
- Tom's current state (which behavior-tree node, what chemistry levels)
- Optional: biometric input if connected to a human Jerry (heartbeat
  via wearable, microphone for breath/voice, etc.) — we're not building
  this in v1 but designing for the option

**Outputs the director produces:**
- Estimated fear level for Jerry, `fear ∈ [0, 1]`
- Drive nudges: small additive shifts to Tom's drive values
- Chemistry pushes: deposits into Tom's chemistry buffers
- Optional: direct state-preference signals ("STALK is appropriate now")

**`[TBD]`** Director implementation. Options range from a simple
hand-coded heuristic (early prototype) to a learned policy
(later iteration). Start hand-coded so we can iterate on the
*behavior* without entangling with training.

#### 3.1.1 Wiring chemistry and memory to prioritize fear `[DECIDED]`

A discipline call worth locking in now: the director's job isn't to
*replace* Tom's existing internal state — it's to *reweight* it so the
substrate we've built across Phases 2-7 starts caring about Jerry's
fear as a first-class signal.

Three concrete wiring intents (final coefficients are `[TBD]`):

**Chemistry priorities:**
- **Dopamine binds to fear-delivery rather than catch.** Currently
  dopamine spikes on `TOM_CAUGHT_JERRY`. In Phase 8, dopamine also
  spikes on fear *rising* (a positive fear-delta event from the
  director). The catch becomes terminal release; per-tick "reward
  feeling" comes from successfully scaring her. This produces the
  "playing with food" emotional register at the substrate level.
- **Adrenaline gates on fear-gradient, not absolute fear.** A Tom in
  the presence of steady-state high-fear Jerry is not as adrenalized
  as a Tom watching fear rise toward the kill threshold. The hunt is
  felt at the inflection points — when fear approaches the threshold
  (commit window) or when fear is dropping (re-pressure needed).
- **Cortisol responds to fear stalling.** When Jerry's fear plateaus
  below the threshold and refuses to rise, Tom's cortisol creeps up
  (frustration). This naturally pushes Tom toward a tactical retreat
  state — "back off, let her think it's safe, come back."

**Memory priorities:**
- **L1 sighting heatmap weights by fear delivered.** Currently every
  sighting increments the heatmap equally. In Phase 8, a sighting
  that *also* produced a fear-rise event adds more weight. Tom's
  memory of "where Jerry tends to be" becomes biased toward "where
  Jerry tends to be *terrified*."
- **L2 distillation includes fear-summary statistics.** Each episode's
  L2 row gains fields for mean-fear, peak-fear, fear-at-catch.
  Warm-start picks up "this map produces high-fear hunts" priors,
  which preconditions Tom toward STALK in those contexts.
- **L1 locker suspicion weights by fear-while-near.** When Tom passes
  near a locker and Jerry's fear is high, that locker's suspicion
  score climbs faster than from a low-fear pass. Lockers where
  terrified Jerrys hide become *more* psychologically loaded for
  future Toms, regardless of catch outcome.

**Why this works architecturally:** the existing
behavior tree (PATROL → SEARCH → INVESTIGATE → PURSUE → STALK →
ATTACK) reads from drives, chemistry, and memory. If those three
substrates start *prioritizing fear-related signals*, the existing
state machine produces stalker behavior without needing to be
rewritten. The architecture stays intact; the substrate just learns
to care about new things.

#### 3.2 The Behavioral Signal Layer

The director can't read Jerry's hidden state. It has to *infer* fear
from observable behavior. This is the layer that does the inferring.

**Signals to track per tick:**
- Movement frequency in last N ticks (stillness → either calm or
  frozen-in-fear; context-dependent)
- Action diversity (varied actions → exploring; repetitive → panicking
  or strategically holding)
- Locker entry/exit patterns (entering locker when Tom is far →
  pre-emptive hiding; entering when Tom is near → reactive)
- Pathing reversals (turning around mid-route)
- Speed of response after a Tom sighting (immediate flight → high
  alert; delayed reaction → either confidence or frozen)
- Distance-from-Tom trajectory (increasing → fleeing; oscillating →
  uncertain; decreasing → bold or trapped)

**Aggregation:** these per-tick signals feed an estimator that produces
a smoothed fear estimate. The estimator could be:
- A simple weighted sum (v1 prototype)
- A small recurrent network trained on labeled trajectories (later)
- A classifier trained on PPO Jerry behavior matched with explicit fear
  labels we set during training (advanced)

**Critical property:** the same signal layer must work for PPO Jerry
*and* a human Jerry. PPO Jerry's signals come from its action stream;
human Jerry's signals come from their input stream. The layer must not
care about the source.

**`[TBD]`** Concrete signal weights and the fear-estimator function.
Best discovered empirically once we have trained Toms to observe.

#### 3.3 Tom-side STALK State and Kill Threshold

A new state in Tom's behavior tree alongside PATROL / SEARCH /
INVESTIGATE / PURSUE / ATTACK. STALK activates when:

- Tom has line-of-sight to Jerry (or recent memory of position)
- AND director's fear estimate is *below* Tom's kill threshold
- AND positional conditions favor maintaining pressure (not too far,
  not too close)

STALK behavior:
- Maintain a target distance from Jerry (configurable; default ~3-5
  tiles)
- Move *parallel* to Jerry's flight direction rather than directly
  toward
- Occasionally retreat — back off a tile or two — to allow fear to
  build or stabilize
- Read director cues about retreat-vs-press

Transition out of STALK:
- → ATTACK when fear ≥ kill_threshold AND Tom has positional advantage
- → PURSUE when Jerry breaks LOS and Tom needs to relocate
- → PATROL when Tom decides to disengage (director-suggested)

**Kill threshold** becomes a *trait* parameter on each Tom, in
[0, 1]. Low-threshold Toms commit early (less patient). High-threshold
Toms wait for true panic. Connects to the drive system: high-aggression
Toms tend toward low thresholds; high-caution Toms toward high.

#### 3.4 Sensor Extensions

`[TBD]` What does Tom *see* about Jerry's fear, vs what does the
director see? Two architectural options:

**Option A:** Tom sees nothing about fear directly. The director reads
Jerry behavior, then nudges Tom's chemistry (especially adrenaline,
dopamine, cortisol). Tom acts based on his nudged internal state. This
is cleanest — Tom's interface stays unchanged.

**Option B:** Tom gets a new "stress sense" — a vector observation with
range that lets him directly perceive Jerry's behavioral indicators.
Like predators detecting prey distress chemicals biologically. This
gives Tom richer information but complicates the human-Jerry path.

My lean is **Option A** for v1, with Option B reserved as a future
enhancement. Option A preserves the "Tom is the same agent, the
*context* is what changes" architectural cleanliness.

---

### Episode resolution rules `[DECIDED]`

Per discipline call #3 in Section 4, episodes MUST resolve. No
infinite stalking. The concrete rules:

**Episode ends when ANY of:**
1. **Catch occurs.** Tom successfully closes on Jerry. (Existing
   project termination.)
2. **Tick budget exhausted.** Episode tick count exceeds 550.
   Default in Phase 1-7 is 300 for training, 600 for some eval.
   Phase 8 uses 550 — long enough for meaningful stalking
   arcs to play out, short enough that humans don't disengage
   from sheer fatigue.
3. **Fear threshold reached without commit.** If Jerry's
   director-estimated fear stays at or above the kill threshold
   for a sustained window (`[TBD]` exact tick count; ~30-50
   ticks) and Tom hasn't committed to ATTACK, force-trigger
   the catch. Architecturally prevents the "Tom learned to
   sustain max-fear indefinitely without ever ending it"
   degenerate strategy.
4. **Jerry permanently safe.** If Jerry has been hiding in a
   locker for an extreme duration (`[TBD]` exact; ~100 ticks)
   AND Tom has been out of sensor range for that whole period,
   the encounter resolves as "Jerry survived." Reflects the
   reality that some hunts end with the predator giving up.

**Why these specific rules:**
- Rule 1 preserves the original catch resolution unchanged.
- Rule 2 (550 ticks) is a hard cap. We picked this number
  consciously — it's ~80% of the Phase 1 eval budget of 600,
  which gives Tom *meaningfully more* time than a catch-optimized
  Tom would need, encoding the patience intent into the time
  budget itself.
- Rule 3 is the "Tom hesitated too long" resolution. The fail-safe
  catch isn't a clean win; we should probably tag the resolution
  with a `forced_catch=True` marker in episode metadata so the
  evaluation harness can distinguish "earned" catches from forced
  ones.
- Rule 4 is the "Jerry won by hiding" resolution — important for
  human players. A human who finds a safe spot and waits should be
  able to survive. This rule prevents Tom from indefinitely
  patrolling a map looking for a Jerry who's effectively gone to
  ground.

**`[TBD]`** Exact tick counts for sustained-fear and locker-hiding
windows. Will tune through playtesting once the system runs.

### Training and evaluation

`[TBD]` This section will be substantial when we get to implementation.
Key open questions:

- Do we train Tom anew for stalker behavior, or do we take a Phase 6
  co-evolved Tom and just wrap it in the director?
- If we train, what's the reward signal? The director-nudged chemistry
  *is* a kind of reward shaping. Is that enough, or do we need explicit
  reward for fear-time?
- How do we evaluate against PPO Jerry vs human Jerry? Different metric
  suites? Different baselines?
- Do we need a calibration phase where the director's fear estimator is
  tuned against ground-truth labels (which only exist for PPO Jerry)?

---

### Implementation phasing (within Phase 8)

`[TBD]` But sketch:

1. **8a — Director scaffolding.** Empty director class, plumb it into
   the tick loop. Verify Tom's existing behavior is unaffected when
   director is a no-op.
2. **8b — Behavioral signal extraction.** Implement signal trackers,
   verify they produce sensible values on existing PPO Jerry replays.
3. **8c — Hand-coded fear estimator.** Simple weighted sum, tunable.
   Verify it produces sensible fear curves on existing replays.
4. **8d — STALK state on Tom.** New state, transition logic, kill
   threshold trait. Verify Tom-with-director-no-op still works.
5. **8e — Director → Tom chemistry pipeline.** Director's fear estimate
   modulates Tom's drives/chemistry. STALK starts firing.
6. **8f — First evaluation.** Run trained Tom-with-stalker against PPO
   Jerry, see what happens. Adjust.
7. **8g — Human-Jerry interface.** Wrap the env so a human can play
   Jerry. Signal layer reads from input stream rather than action
   stream. First human playtest.
8. **8h — Iteration.** This will be most of the work.

---

## 4. Failure modes and discipline calls

### What "wrong" looks like

The Phase 2 retro established the discipline pattern: don't hand-tune
agents to beat specific opponents. Phase 8 has its own versions:

**Tom that never catches.** Director pushes Tom away from ATTACK
indefinitely; episodes time out. Easy diagnostic: catch rate = 0%.
Fix: cap the patience, force commit at some maximum episode length.

**Tom that catches as fast as ever.** Director's nudges don't actually
change behavior; Tom's "default" optimization wins. Diagnostic: STALK
state never fires, or fires for one tick and immediately transitions.
Fix: stronger director influence, larger drive nudges.

**Tom that's just slow.** STALK fires correctly but the behavior is
"PURSUE but at half speed." Not what we want. Diagnostic: STALK state
duration correlates with kill threshold but doesn't produce
qualitatively different action patterns.
Fix: distinguish STALK from PURSUE by *what action it produces*, not
just timing.

**Director that exploits hidden Jerry state.** During development with
PPO Jerry, it'll be tempting to give the director access to Jerry's
PPO hidden state or actual action probabilities — "to make the fear
estimate more accurate." This is a bug. It breaks the human-Jerry
path. Diagnostic: any code in the director that reads
`jerry._policy_state` or `world.jerry.policy.action_dist` etc.
Discipline: hard-fail at code review.

**Fear estimator that doesn't generalize.** Trained on PPO Jerry,
fails completely on human Jerry. Probable cause: the estimator latched
onto PPO-specific signals (e.g. specific action distributions) rather
than universal behavioral features. Fix: develop the estimator with
*both* sources from early on, even if human Jerry data is sparse.

### Discipline calls we're committing to now

These are the rules we're locking in *before* implementation, so we
can't post-hoc rationalize their violation.

1. **The director never reads Jerry's internal state.** Only
   observable actions, positions, and world state.
2. **The signal layer is policy-agnostic.** Same code path for PPO
   Jerry, scripted Jerry, and human Jerry.
3. **Episode resolution is required.** No infinite stalking. Eventual
   commitment or eventual disengagement is part of the design.
4. **Existing project metrics stay valid.** Catch rate, survival ticks,
   etc. still apply. Phase 8 adds new metrics; it doesn't invalidate
   old ones.
5. **The substrate Toms (Phase 7 outputs) remain usable.** Wrapping
   them in the director is additive. A Tom that works in Phase 7
   should still work without the director — degraded, perhaps, but
   functional.
6. **Hand-tuning to beat a specific Jerry is forbidden.** Same rule as
   Phase 2.

---

## 5. Framing note

A short, deliberate framing. Not because the work needs defense, but
because this doc will outlive its author's intent and someone reading
in 2030 should know what we were and weren't doing.

Phase 8 is a research and game-design exploration of predator-prey
dynamics in artificial agents. The goal is to produce a hunter whose
behavior is structurally different from "minimize time to kill" —
specifically, a hunter that exhibits patience, retreat, and engagement
patterns analogous to those documented in real predators and in the
horror-game genre.

When we talk about "terror" or "fear," we mean the architectural
mechanic we are building. We are not making claims about whether the
PPO Jerry agent experiences anything. We are explicitly building toward
human Jerry players who *will* experience something — that's the point
of horror games and they've existed for fifty years. The novelty here
is doing it through an agent architecture rather than through scripted
events.

Nothing in this project should be deployed as a tool to cause distress
outside the consensual gameplay frame. The project is a research
exploration and (eventually) a game; it is not a deployment platform
for adversarial AI behavior toward unwitting humans.

---

## Open conversation thread

Things to discuss as the design evolves. Add to this list freely.

- `[TBD]` Exact signal set for the behavioral signal layer
- `[TBD]` Director-fear-to-Tom-chemistry mapping
- `[TBD]` Whether to use Option A or Option B for sensor extension
- `[TBD]` Training approach: re-train Tom, or wrap a Phase 6 Tom?
- `[TBD]` Human-Jerry input mode: keyboard? gamepad? VR? something
  weirder like webcam-based stress detection?
- `[TBD]` Map design considerations for stalker gameplay (do 30×30
  grids still make sense, or do we need different geometries?)
- `[TBD]` Multi-Jerry scenarios — does a stalker pursue one specific
  victim while others are present?
- `[TBD]` Whether Phase 8 should include a "playable demo" deliverable
  or stay purely on the research side.

---

## Document changelog

- 2026-05-18 — Initial draft. Why and What sections elaborated based
  on Grove's design intent (Jerry-as-scaffolding, AI:I-inspired
  director, terrorize-then-catch). How and Failure-Modes sketched as
  scaffolding for implementation phase. Framing note added.
- 2026-05-18 — Revision 1. Corrected reference from "AI:I 2" (rumored
  sequel) to Alien: Isolation (2014)'s documented two-brain
  architecture — local alien brain + director brain with separate
  information sets. Added section 3.1.1 "Wiring chemistry and memory
  to prioritize fear" with concrete intent for dopamine, adrenaline,
  cortisol, L1 heatmap weighting, L2 distillation fields, and locker
  suspicion weighting. Added concrete Episode Resolution Rules
  section: catch / tick budget 550 / sustained-fear forced catch /
  permanent-hiding survival.
