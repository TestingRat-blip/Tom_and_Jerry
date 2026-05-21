# Design — Memory-Driven Adaptation (Tom's learning substrate)

**Status:** Design. No code yet.
**Date:** 2026-05-20
**Context:** The counter-Jerry experiment proved a PPO Jerry can robustly
beat the scripted Conductor via a cover-dance (break LOS near cover →
exploit belief decay). That forced the question: how does Tom learn to
counter? The answer, intended from the project's start: **Tom learns from
his memory.** This doc designs that loop.

---

## The thesis

**Memory is Tom's learning substrate.** Tom does not learn by gradient
descent (no PPO Tom) or by config search (no CMA-ES). Tom learns the way
a real predator does — within its own lifetime, from its own
experience, stored as memory and recalled on the next encounter.

Concretely: Tom notices how Jerry played, distills that into persistent
memory (L2), and on the next encounter recalls it and adjusts his hunt.
"Jerry kept breaking line of sight near cover last time → this time, when
he breaks LOS, don't release pressure; run him down to that spot and deny
him the cover."

**The door stays open for scripted capability additions.** Memory governs
*when and how strongly* Tom deploys behaviors. But Tom can only deploy
behaviors he HAS. When memory says "counter the cover-dance" but Tom's
repertoire can't express that counter, we ADD the behavior (hand-authored)
and let memory govern its deployment. This is exactly ADR-003: structure
is scripted, parameters/deployment are learned. Memory is the parameter
layer; new behaviors expand the structure.

This asymmetry with Jerry is intentional and biologically real: the
predator adapts within its lifetime (memory, fast, reactive); the prey
species adapts across generations (PPO training, slow, anticipatory).

---

## Why memory, not gradient

1. **Legibility (ADR-003).** "Tom remembers Jerry hid in the NE corner and
   checks there first" is readable — you can inspect L2 and understand why
   Tom acts as he does. A PPO weight vector or a CMA-ES config blob is a
   black box. Legibility is what makes Tom feel like a *who*.

2. **Right timescale.** Gradient methods adapt over thousands of training
   episodes. Memory adapts over a HANDFUL of encounters — even within one
   long hunt. That's the timescale that produces the target experience: a
   (future human) Jerry does the cover-dance twice, and by the third
   encounter Tom has already adjusted. Live, in the deployed agent, no
   retraining.

3. **Already 80% built.** L1 (per-encounter), L2 (persistent), distillation,
   warm-start, fingerprinting — all exist (Phases 3-4). We are WIRING
   existing memory into the Conductor's strategy, not building a learner
   from scratch.

4. **It's the same machinery Phase 8 needs.** Distilling Jerry's
   *behavioral patterns* (not just locations) is exactly what the stalker's
   fear-reading requires. Build it here, reuse it there.

---

## The honest limitation (named, not hidden)

Memory adaptation is **reactive**: Tom counters a pattern only AFTER he's
seen and distilled it. The first time a new Jerry strategy appears, it
works; Tom adapts on subsequent encounters. For the deployed hunter facing
a human, this is exactly right ("next round he remembers").

For co-evolution as a *method*, it means Tom's adaptation is bounded by
(a) what distillation captures and (b) what the Conductor's behaviors +
parameters can express. If a counter needs a behavior the repertoire lacks,
memory alone won't crack it — we add the behavior (scripted), and memory
triggers it. That's the "leave the door open" clause, made concrete.

---

## The loop (on paper)

```
EPISODE N
  Tom hunts Jerry with Conductor warm-started from L2 (see below).
  During the episode, L1 records per-encounter events (existing) PLUS
    behavioral signals (NEW): LOS-breaks, where they happened, hide
    durations, oscillation, etc.
  At episode end, distillation writes an EpisodeSummary to L2 — existing
    spatial stats PLUS new behavioral-signature stats.

EPISODE N+1 (same map + same/similar Jerry fingerprint)
  Conductor warm-start queries L2: "how did this prey behave here before?"
  Gets back behavioral signatures: e.g. "high LOS-break rate near sector X
    cover; tends to hide NE."
  Conductor ADJUSTS strategy from that:
    - parameter adjustments (slow SIGHTING decay, bias patrol to cover)
    - behavior deployment (enable "hold-on-LOS-break / run-down")
  Jerry's old strategy works less well.

EPISODE N+2
  Jerry (if also adapting via training) must find a new strategy, or Tom
  keeps winning. Arms race continues — Tom via memory, Jerry via PPO.
```

---

## Component 1 — Behavioral distillation (the missing data)

**Problem:** today's `EpisodeSummary` is purely SPATIAL (heatmap_top,
lockers, false_noise_top — all `[x,y,count]`) plus outcome stats. Nothing
captures HOW Jerry behaved. To counter the cover-dance, Tom must remember
the *pattern*, not just the *places*.

**Add behavioral-signature fields to `EpisodeSummary`.** Candidates (start
small, expand as needed):

- `los_break_count: int` — how many times Jerry broke Tom's line of sight
- `los_break_hotspots: list[TileCount]` — WHERE LOS-breaks clustered
  (these are the cover spots Jerry exploits)
- `mean_hide_duration: float` — how long Jerry stays hidden once he breaks
  contact (long = patient camper; short = active dancer)
- `oscillation_score: float` — how much Jerry reverses direction
  (the dance signature)
- `time_in_cover_fraction: float` — fraction of ticks adjacent to walls/
  cover vs in the open

**Where these come from:** L1 already observes per-tick events and the
Conductor already tracks sightings/LOS. We add lightweight per-episode
counters in L1 (or the Conductor) that tally these signals, then
distillation summarizes them into the EpisodeSummary. The signals must be
OBSERVABLE (ADR-013) — LOS-break is observable (Tom had sight, then
didn't), hide duration is observable (time since last sight), etc. None
require reading Jerry's hidden state. This keeps the door open for the
human-Jerry path (Phase 8): the same signals exist for a human.

**`[TBD]`** Final signal set. Start with `los_break_count` +
`los_break_hotspots` because they directly target the cover-dance — the
concrete problem in front of us — then add others as new Jerry strategies
appear.

---

## Component 2 — Conductor warm-start from behavioral memory

**Today:** the Phase 4 warm-start pre-seeds L1 with SPATIAL priors
(heatmap, locker suspicion, false-noise) so Tom's noise threshold and
locker checks start episode N+1 informed by the past. The Conductor reads
L1's current belief but does NOT yet read L2's cross-episode behavioral
memory for STRATEGY.

**Add:** at episode start, the Conductor queries L2 (by map + Jerry
fingerprint, fine→coarse cascade — existing mechanism) and reads the
behavioral signatures. It uses them to set its STRATEGIC stance for the
episode:

- High `los_break_count` near `los_break_hotspots` →
    - slow SIGHTING-suspicion decay (don't forget so fast on LOS-break)
    - enable the "hold-on-LOS-break / run-down" behavior (Component 3)
    - bias patrol toward the LOS-break hotspots (the cover Jerry uses)
- High `mean_hide_duration` (patient camper) →
    - longer INVESTIGATE dwell; check lockers more thoroughly
- High `oscillation_score` (dancer) →
    - the existing prediction system already half-handles this; warm-start
      could raise prediction weight

**Mechanism:** the Conductor gets a small `StrategicStance` object at
episode start, derived from the L2 behavioral query, that adjusts its
config-level parameters for the episode. This is the "memory as parameter
layer" made concrete: the Conductor's structure is fixed; memory tunes its
parameters per-encounter based on who it's hunting.

**`[TBD]`** The exact mapping from behavioral signatures → stance
adjustments. This is where hand-tuning lives in Stage 1; it could become
the learnable surface later (but via memory-derived rules, not gradient).

---

## Component 3 — New Conductor behavior: hold-on-LOS-break / run-down

**The specific counter to the cover-dance.** When Tom loses LOS to a Jerry
he was actively pursuing, the current Conductor lets the SIGHTING suspicion
decay and eventually disengages (releases RUSH pressure) — which is exactly
what the cover-dance exploits.

**New behavior:** when warm-start (Component 2) says "this prey breaks LOS
to escape," the Conductor instead:
- anchors a high-priority suspicion at the last-seen tile that decays MUCH
  slower (or not at all for a window)
- directs Tom to advance TO that tile and search it / adjacent cover,
  rather than giving up and re-patrolling
- "runs him down to his square" — denies the cover by occupying it

**Capability vs deployment:** the behavior is hand-authored (scripted
capability). WHETHER it's active is governed by memory (Component 2). A
naive Tom with no memory of this Jerry doesn't use it; a Tom who's been
burned by the cover-dance before deploys it. Memory governs deployment —
the ADR-003 pattern, exactly.

**`[TBD]`** Exact mechanics: how long to hold the anchor, how aggressively
to occupy cover, when to give up (episodes must still resolve — the
Phase 8 resolution rules apply: tick budget, etc).

---

## The cheap experiment that should come FIRST (before full loop)

Per the project's discipline (cheap diagnostic before expensive build):
**before wiring the whole memory loop, hardcode Component 3's behavior ON
and test it against the cover-dance Jerry we already have on disk.**

- If forced-on "hold-on-LOS-break / run-down" beats the cover-dance →
  the counter is EXPRESSIBLE, and the loop's job is just to deploy it via
  memory. Proceed to build Components 1+2.
- If forced-on doesn't beat it → the counter needs to be richer; iterate
  on Component 3's mechanics BEFORE building the memory pipeline that would
  trigger a behavior that doesn't work.

This de-risks the whole loop the same way the ceiling diagnostic and the
counter-Jerry experiment did: prove the expensive thing is worth building
with a cheap test first.

---

## Build order (once the cheap experiment validates Component 3)

1. **Component 3 first (validated by the cheap experiment).** Add the
   hold-on-LOS-break / run-down behavior to the Conductor, behind a flag
   (forced on for the experiment).
2. **Component 1.** Behavioral distillation — add `los_break_count` +
   `los_break_hotspots` to L1 tracking + EpisodeSummary + the schema
   migration.
3. **Component 2.** Conductor warm-start reads the behavioral signatures
   and produces a StrategicStance that (among other things) flips
   Component 3's behavior on when warranted.
4. **Verify the loop end-to-end:** episode N (cover-dance Jerry beats
   Tom, behavior distilled) → episode N+1 (Tom warm-starts, deploys the
   counter, does better). The proof is the SAME shape as the Phase 4
   end-to-end test: episode N+1's behavior differs because of episode N's
   memory.

---

## How this reframes "Tom's learnable substrate"

The findings doc (`FINDINGS_ARMS_RACE.md`) listed three options: config-
as-substrate, PPO Tom, hybrid. This doc is the option that was the plan all
along and got left off that list:

**Tom's learnable substrate is his MEMORY. The Conductor reads L2 to set
its per-encounter strategic stance. Where memory needs a behavior the
Conductor can't express, we add it as scripted capability and memory
governs its deployment.**

It's a memory-as-parameters realization of ADR-003's hybrid, distinct from
gradient/search optimization. It adapts at encounter timescale, stays
legible, reuses Phase 3-4, and doubles as the Phase 8 stalker's behavioral-
reading machinery.

---

## Open questions

- `[TBD]` Final behavioral-signature set (start: los_break_count +
  hotspots).
- `[TBD]` Behavioral-signature → StrategicStance mapping (hand-tuned
  Stage 1).
- `[TBD]` Component 3 mechanics (anchor decay, cover occupation, give-up
  rule).
- `[TBD]` Does the StrategicStance mapping itself eventually become
  learnable — and if so, learned how (memory-derived rules, not gradient)?
- `[TBD]` How memory adaptation interacts with Jerry's PPO adaptation in a
  full co-evolution run (different timescales — does Tom's memory reset
  per generation, or persist across Jerry generations, accumulating a
  "book" on the whole Jerry lineage?). This last one is deep and
  interesting — a Tom that remembers the ENTIRE history of Jerry strategies
  is a formidable thing.

---

## Changelog

- 2026-05-20 — Initial design. Memory as Tom's learning substrate (not
  gradient/search). Three components: behavioral distillation (L2 gains
  pattern stats), Conductor warm-start from behavioral memory (per-
  encounter StrategicStance), and a new hold-on-LOS-break/run-down behavior
  governed by memory. Cheap-experiment-first discipline: validate the
  counter behavior forced-on before building the memory pipeline. Reframes
  the findings-doc substrate question: the answer is memory, the ADR-003-
  consistent option left off the original list.
