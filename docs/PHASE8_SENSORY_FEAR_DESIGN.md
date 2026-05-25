# Design — Phase 8: Sensory Fusion, Fear, and the Night (Tom's perception & the win-con)

**Status:** Design. No code yet. All numeric constants are TODO-tune.
**Date:** 2026-05-22
**Context:** Phase 8 is the human-facing endgame — the stalker that hunts a
real player across nights. Current Tom is "an attack animal": binary
line-of-sight, position-perfect tracking, no facing, no graded perception.
This doc designs the conversion of that step-function predator into a
*continuous* one — graded senses, induced fear, a real win-condition — while
keeping Tom **scripted-over-rich-belief** (ADR-003): no black-box policy, all
behavior traceable, emergence coming from the richness of inputs colliding,
not from an opaque learned controller.

---

## The unifying premise (this is load-bearing, not flavor)

**Tom is a hallucination. The player's win-con is surviving the night. Each
night, Tom uses his memories more.**

This framing is doing structural work, not decoration:

1. **It explains the 1-prey-vs-1-relentless-hunter setup.** You are alone with
   the thing because you are *meant* to be alone with it. No allies, no help,
   no plot hole.
2. **It is the diegetic justification for the persistent-learning system** —
   the single feature that makes Tom surpass Isolation's Alien (which cannot
   remember the player across the save boundary). "Each night Tom uses his
   memories more" is the *literal rendering of L2 memory accumulating across
   sessions, as narrative.* Night 1 Tom is generic. Night 5 Tom knows you bolt
   for the same corner, knows you go quiet-then-sprint when cornered, knows
   your tells. The difficulty curve is not scripted — it is **Tom's memory
   filling up**, shown as escalating dread. The horror premise and the
   technical centerpiece are the same thing.

**Design consequence:** fear (Atom 1) and memory interact across the night
boundary. Tom never reads Jerry's fear stat, but he remembers the *behavioral
tells* fear produces — "this prey vents fear near the east lockers," "goes
silent then breaks into a sprint at high fear." That feeds **behavioral
signatures into L2** — i.e. exactly the distillation pipeline built in batch
12c (`StrategicStance`, behavioral fields, distill→warm-start). The signal
that was moot for the column-bob exploit is the *right machinery* for
remembering how a prey behaves under fear across nights. Repoint it here when
the time comes; do not rebuild it.

---

## The core reframe — perception is sensory FUSION, not vision-with-holes

Everything in Phase 8 is a modulator on a **three-channel sensory model** that
fuses into the belief system Tom already has (typed, decaying, confidence-
weighted suspicion sources — sighting/noise/scent today; this enriches their
shape). Tom does not get "vision." Tom gets three senses with different shapes
that combine into one belief:

| Channel | Shape | Detects | Driven by |
|---|---|---|---|
| **Sight** | Directional (cones along facing) | Jerry's position, high confidence | Facing (Atom 2), light (later) |
| **Hearing** | Omnidirectional radius | Jerry's *sound* emission | Movement mode + fear (Atom 1) |
| **Smell** | Diffuse gradient (existing scent system) | Where Jerry *was* (laggy) | Movement + fear; existing diffusion |

**The "donut" is not a built thing.** It is the emergent region where
hearing+smell are strong enough to guarantee detection *regardless of facing*.
It is not a fixed radius — it is "wherever the non-visual channels currently
cross the detection threshold." A loud/scared Jerry has a *bigger* effective
donut (he is giving himself away); a calm silent sneaker shrinks it. This
means:

- The genuine blind spot is only where **all three channels fail at once**:
  outside the sight cones AND beyond hearing AND scent has not reached. That
  pocket *moves* as Tom turns and as scent drifts — so it cannot be memorized
  as a fixed angle. **Resonance-defense falls out of honest sensory modeling**
  (cf. the whole batch 11–12 arc: fixed rules are exploitable; this rule
  isn't fixed).
- The lateral mid-range gap (off Tom's sides, past hearing, outside cones) is
  a real flanking lane *by design*. Verify it with the perception instrument
  (below) rather than discovering it as an exploit.

---

## ATOM 1 — The Fear Stat

### What it is
A scalar on **Jerry**, 0→1. It is Jerry's internal state. **Tom never reads
it.** It exists to (a) modulate Jerry's emission on the hearing+smell
channels, and (b) drive theatrical escalation against human players. Tom acts
only on his *perceived* fear — inferred from how much Jerry is giving away —
which can be wrong.

### Sources (raise fear) — fast attack
- **Proximity-while-seen** — Tom visible and close; core driver, scales with
  closeness and duration.
- **Near-miss** — Tom entered catch range and Jerry escaped; a spike.
- **Tom's theatrics** — menacing stare, hit-and-run burst; *deliberate* fear
  injection (the loop where Tom chooses to terrorize).
- **Sustained pursuit** — actively chased many ticks, even at range.

### Decay (lower fear) — slow release
- **Time without Tom perceived** — core decay; produces the appear/vanish
  *wave* instead of a flatline.
- **Safety** — in locker / cover / no Tom for N ticks → faster decay.
- **Fear-item** (the win-con enabler; see below) — a discrete, scarce vent.

### The rise/decay shape (THE critical tuning ratio)
**Fast to spook, slow to fully calm.** This asymmetry is the entire feel.
- Too-fast-rise + slow-decay → pins at max (the saturation failure mode we hit
  repeatedly; cf. adrenaline-stuck-at-1.0 in the prediction-window exploit).
- Slow-rise + fast-decay → never accumulates; no dread possible.
- **Target shape:** several ticks of close-pursuit to approach high; decay over
  a *comparable but slightly longer* timescale, so a brief sighting leaves
  lingering unease a single back-off does not erase.
- Constants: **TODO-tune empirically.** Trace before trusting (lesson of the
  arc). The *shape* is the design decision; the numbers are a measured pass.

### Hysteresis (Tom's pressure rhythm — prevents runaway AND reads as menace)
Tom's back-off instinct is keyed to **fear level, NOT a timer** (a timer has a
fixed period PPO will dance — cf. the column-bob). Two thresholds with a gap:
- Perceived-fear crosses **HIGH** (TODO ~0.75) → Tom eases off, lets Jerry stew.
- Decays past **LOW** (TODO ~0.4) → Tom re-engages.
- The gap (LOW→HIGH) is the "let him stew" zone. Hysteresis prevents on/off
  chatter at a single line; the gap is *where the dread lives.*
- These are thresholds on Tom's *perceived* fear, so Tom can misjudge — back
  off on a Jerry who was already calm, or push one who is about to break. Good.

A predator that eases off when its prey is maximally panicked **reads as
savoring it.** Same damping term, two readings: mechanically it caps the
feedback loop; experientially it is menace. This is how scripted Tom feels
"thinking."

### Emission mapping (fear → senses) — the elegant coupling
Fear does **not** add a channel. It *scales existing emission*:
- `scent_emitted = base_scent * (1 + fear * SCENT_FEAR_MULT)`  (TODO mult)
- Sound gains a **floor that rises with fear** — panicked breathing. You are
  never fully silent when terrified, *even sneaking.*

**Consequence — high fear partially defeats sneak.** A panicked Jerry trying to
sneak still breathes hard, so sneak's quietness degrades as fear rises. The
time to use sneak is *before* you panic → a smart Jerry **manages its own
fear** as a resource. A strategic layer falls out of one multiplier.

---

## The win-con and the fear-item (the watershed)

**Survival is active, not passive.** Not "avoid until the timer," but "manage
fear well enough, and use scarce tools at the right moment, to make it through
the night." Fear is *the resource the player plays against*; Tom's pressure
rhythm drains it; the player's choices (which movement mode, when to vent)
manage it.

**Fear-item:** discrete, **1 use per item**, multiple may exist per map.
Instantly (or quickly) lowers fear — an emergency vent. Scarcity is the design
constraint: you cannot spam calm, so vents must be *timed* against Tom's
pressure waves. This is the win-condition enabler — skilled fear management +
well-timed vents = survive the night.

This single decision turns the whole system into a *game*: sneak/walk/sprint
becomes fear-vs-speed-vs-noise resource management; the item is the emergency
release valve; Tom's rhythm is the drain. Everything specced here now serves
one legible objective.

**Layer separation to respect:**
- **Mechanical fear layer** (scent/sound emission, the stat, the item) —
  trainable, works against bots, is the foundation.
- **Theatrical fear layer** (jumpscare distance-sightings → hit-and-run →
  shoving Jerry off his feet) — only has an audience when a *human's* real
  heart rate is climbing. Rides on top; matters at the human-facing endgame.
  Do not over-build the theater before the mechanics are solid.

---

## ATOM 2 — Facing & the Three-Sense Perception

### Facing
- Tom has an orientation: **4 cardinals** (diagonal facing complicates cone
  math for little gain).
- Default: facing follows movement direction.
- **Turn-in-place is a costed action** — Tom can scan/check corners without
  moving, but it *takes the tick* (opportunity cost). Free scanning would mean
  Tom spins constantly; costed scanning reads as deliberate and careful. This
  is the cost-of-staying principle (below) applied to perception.

### The three channels (shapes)
- **Sight (directional):**
  - **Long cone** — low confidence; sees Jerry → **menace** (stare / slow
    advance). The dread-builder and a primary *fear source*. Tom announces
    presence rather than attacking.
  - **Inner/focal cone** (narrower, "bright") — high confidence; Jerry caught
    here → **hit-and-run** initiates (quick aggressive burst, then release per
    the back-off instinct).
  - Range modulated later by light.
- **Hearing (omnidirectional radius):** picks up Jerry's *sound* emission.
  Effective range = how loud Jerry currently is (mode + fear). Sprinting
  panicked Jerry audible from far; calm sneaker near-silent.
- **Smell (diffuse gradient):** uses existing scent diffusion. Drifts, lingers,
  points toward where Jerry *was*. Laggy and gradient-directional — realistic.

### The donut (emergent, not built)
The region where hearing+smell guarantee detection regardless of sight. Scales
with Jerry's emission (fear + mode), so it is dynamic. Rule: if Jerry is close
enough that smell+sound make him certain-detected, and he is trying to *close*
(sneak up), Tom strikes — **sneaking into the donut is death.**

This is the **built-in counter to the blind-spot exploit.** Facing creates
blind spots behind Tom (good — the sneak-behind-the-predator fantasy). But the
donut bounds the abuse: blind spots exist *at range*, but get close enough to
exploit one and you die. And a panicked Jerry **cannot shrink the donut** —
fear forces emission — so fear literally makes you easier to ambush at close
range. The loop closes: the exploit and its defense were designed
simultaneously.

### Fusion
All three channels produce typed, confidence-weighted suspicion → the existing
belief merges them → Tom's scripted state machine reads the merged belief. We
are **not rebuilding the Conductor.** Its typed suspicion sources
(sighting/noise/scent) were almost designed for this; Atom 2 enriches the
*shape* of channels that already exist.

---

## The structural defense rule (write this down, reuse everywhere)

**Every committed Tom state must have a cost-of-staying, so no state is a safe
place for Jerry to stall him.**

The column-bob exploit worked because ATTACK-state had no cost-of-staying —
Tom mirrored forever for free. Applied to Phase 8:
- **Ambush:** if it does not trigger within a window, **soft-reset** — Tom
  abandons the back stage and returns to stalk/patrol. Removes the "Jerry sits
  at fear 0.59 forever to deny the ambush" exploit: refusing the bait costs
  Jerry an *actively stalking* Tom. The abandon condition should be
  **state-based where possible** (Jerry left the zone / fear dropped below
  re-engage) rather than a fixed tick count (a fixed count gets waited out by
  one tick — jitter it if it must be a timer).
- **Stare/menace, hit-and-run, run-down:** each needs its own exit so none can
  be camped.

---

## Dependency order (each independently testable & shippable)

1. **Jerry movement modes** (discrete: sneak / walk / sprint).
   Speed/sound tradeoff, strictly ordered so none dominates (tune for a *mixed*
   equilibrium). Smallest change; extends existing noise emission. Foundation —
   feeds the hearing channel and couples to fear.
2. **Fear stat + emission mapping + win-con item** (Atom 1, mechanical layer).
   Can be added to *today's* Tom for an immediately richer hunt, before
   facing/cones exist. The item makes survival active.
3. **Facing + three-channel fusion + the donut** (Atom 2). The structural
   keystone. Introduces orientation; enriches the belief's channels.
4. **Theatrical escalation** (fear-gated attack styles, human-facing). Rides on
   2+3.
5. **Light levels** — modulates sight-cone range/confidence; dark corners as
   refuge. Top of the dependency stack; near-complete-build.

---

## Build discipline (earned the hard way in batches 11–12)

- **Instrument before behavior.** Before Tom *acts* on the three-channel
  perception, build a **perception trace/visualizer**: place Tom, walk Jerry
  in a circle at varying ranges and movement modes and fear levels, and render
  the seen/unseen/donut map. Confirm the geometry matches intent — *then* wire
  behavior. (We theorized from the eyeball twice last arc and were wrong twice.
  Trace first, theorize second, build third.)
- **No fixed thresholds without a stalling cost or jitter.** Fear gates,
  ambush timers, mode-switch points — each is a resonance target. Soft/
  probabilistic/hysteretic/jittered from day one.
- **Tune constants empirically, traced.** The fear rise/decay ratio especially
  — the *shape* is decided (fast attack / slow release); the numbers are a
  measured pass against real episodes, not guessed.
- **Tom stays scripted-over-rich-belief.** Every addition is a new belief input
  or a new gated branch — never a black-box policy. Legibility is the point.

---

## Open questions (for when we build, not now)
- Exact channel ranges *relative to each other* (sight cone length vs hearing
  radius vs scent reach) — this ratio defines the blind-spot geometry.
- Fear rise/decay constants and the HIGH/LOW hysteresis thresholds.
- Movement-mode speed/sound numbers for a mixed equilibrium.
- How behavioral-tell distillation (the 12c pipeline) encodes fear-driven
  patterns for cross-night memory — what exactly gets stored.
- Whether turn-in-place is one action or "turn toward a chosen heading."

## Changelog
- 2026-05-22 — Initial Phase 8 sensory/fear design. Established the
  three-channel fusion frame (sight/hearing/smell → existing belief), fear as
  emission-modulator (Tom reads symptoms not the stat), the hysteresis pressure
  rhythm, the win-con + scarce fear-item, facing + costed turn-in-place, the
  emergent donut as built-in blind-spot counter, the cost-of-staying defense
  rule, and the hallucination/escalating-memory premise as the diegetic frame
  for the persistent-learning centerpiece. All constants TODO-tune.

---

## Motivating case: the occluded-pocket statue (found Round 8, batch 19)

A concrete, reproducible failure that ONLY the sensory model fixes — recorded
here so the Phase 8 build has a real target instead of an abstract goal.

**Setup:** seed 26, the kiter_600 Jerry. Jerry spawns at (1,28), a pocket in
the bottom-left corner sealed by walls along row 27 with a single entrance gap
at (3,27). If Jerry simply stands still (move_frac 0.00), it survives the full
600-tick night.

**Why every Phase-6 fix fails on it:**
- It is NOT a reachability bug — `_spawn_agents` (batch 16) guarantees Tom can
  path there.
- It is NOT a coverage bug — the 5x5 sector patrol (batch 19) brings Tom within
  ~6 tiles, well inside his sight *range* of 10.
- It survives anyway because the row-27 walls **block line of sight** into the
  pocket. Tom is near, but cannot SEE a motionless prey, so belief never fires,
  PURSUE never triggers, and he wanders off. A silent statue emits no noise and
  negligible scent, so sight is Tom's only channel — and sight is occluded.

**Why this is a Phase 8 case, not a Phase 6 bug:** the only general fixes are
(a) rework the map geometry (whack-a-mole — the generator will always produce
some occluded pocket), or (b) give Tom non-visual senses. (b) is the Phase 8
sensory model. A hidden, motionless, silent prey is exactly what hearing/scent
exist to flush out: even a statue leaves a faint scent gradient, and the three-
channel fusion would let Tom sense "something is in that pocket" without seeing
it. The donut/blind-spot reasoning already in this doc is the same machinery.

**Thematic note (Alien: Isolation north star):** a predator that finds a
perfectly-still, hidden prey using *sight alone* would be less interesting, not
more. The ~2% "statue in a sealed pocket survives" outcome is arguably the
correct sliver of hope — Phase 8 scent is what *shrinks* that sliver toward
zero, turning "hold still and pray" from a reliable strategy into a desperate,
usually-failing gamble.

**Acceptance test for the Phase 8 build:** seed 26, stationary Jerry, must be
found and caught once Tom has scent. If it still survives, the scent channel
isn't reaching occluded pockets and needs tuning.
