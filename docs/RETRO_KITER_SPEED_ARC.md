# Retrospective: The Kiter → Circle → Speed-Ramp Arc (Rounds 8–10)

*The arc where the prey stopped exploiting and started running — then found the
one thing equal speed can't beat, and the fix turned out to be design, not code.*

This continues the honest-history tradition of `RETRO_MEMORY_ADAPTATION_ARC.md`
and `FINDINGS_ARMS_RACE.md`. It documents the mis-scopes as plainly as the wins,
because the mis-scopes are where the real lessons are.

---

## Where this arc started

Coming in, every *static* exploit was closed: the column-bob tie-break (R4), the
locker camp via oxygen (R6), and the corner-cubby via wall-aware prediction (R7).
A prey trained against that hunter survived 0/50. The only thing still working
was the **motionless statue** — a perfectly still, silent prey in a spot the
hunter couldn't see into. We had established, by exhaustion, that the statue is a
*sensory* problem (sight-only hunters can't find a zero-signal target) and
deferred it to a future scent/presence layer.

The plan for this arc: don't pre-build the sensory fix against a hypothetical.
**Retrain first** and see what the prey actually does against the fully-patched
hunter, then build against observed reality.

---

## Round 8 — the first genuine kiter, and a correction

A fresh prey produced three survivors. The classifier labeled all three
"open-evasion," but the `move_frac` column split them: two statues (motionless)
and **one mover** — seed 21, `move_frac 0.88`, which ran a real map-wide chase.

The lesson that repeats every arc landed again here, on *me* (the assistant):
from a **300-tick trace** I called seed 21 "exploit #4 — a fixed-distance kite,"
because DIST sat pinned at 3. Grove corrected it: the replay (`watch.py`) runs
**600 ticks**, and at ~tick 336 the chase resolved in a **catch**. The "pinned
DIST" was a long-but-winnable chase, not a locked exploit. Tracing a 300-window
on a 600-tick night had clipped the catch and made genuine evasion look like a
standoff.

**Fixes banked:** eval/classify windows 300 → 600 (so slow catches stop being
miscounted as survivals); patrol sectors 3×3 → 5×5 (kills open-corner statues).

**Durable lesson:** match the measurement window to the phenomenon. A truncated
trace doesn't just lose data — it can *invert* the conclusion (evasion ↔ exploit).

---

## The map-generator detour (and a second bug the sanity-check caught)

The remaining statue lived in a dead-end pocket. We tried, in order:

1. **Scent diffusion** — physically correct, but the pocket's dogleg geometry
   kept the escaped scent ~80× below detection threshold; forcing it stronger
   smeared scent globally and broke directional tracking. Caught **zero**
   additional statues. Kept as gated-off Phase-8 infrastructure, documented
   honestly as dormant.
2. **Dead-end map cleanup** — eliminated all 573 dead-end tiles across 50 seeds.
   But the statue simply **relocated** to an LOS-occluded *non*-dead-end corner.
   Whack-a-mole, exactly as predicted.

The conclusion held: the statue is sensory, not geometric — three different
geometry/patrol fixes each only *moved* it. Deferred to the sensory layer with a
revised acceptance test.

**The sanity-check payoff:** before retraining on the cleaned maps, Grove
insisted on a map sanity pass. It caught a bug the cleanup had *introduced* —
seeds 5 and 29 had sealed 2×2 rooms (every tile has 2 neighbors, so "no
dead-ends" passed, but the room was globally disconnected). "No dead-ends" is a
*local* property; "connected" is *global*; they are not the same. Added
`_enforce_connectivity` and 60 parametrized invariant tests. **We would have
trained a baseline on broken maps without that check.**

**Durable lesson:** verify the environment before spending GPU on it. A local
invariant passing is not a global invariant holding.

---

## Round 9 — the prey learns to run, and finds the circle

Retrained 2.5M steps against the fully-patched, cleaned-map hunter. The result
was the most capable prey the project had produced, and it came with a tell in
the training logs: **deterministic eval 20–25%, stochastic eval 0%.** A large
det/stoch gap means a *precise, fragile* policy — one that perfect (argmax)
execution sustains but sampling noise breaks. That pattern usually means an
exploit that needs exact positioning.

Tracing the deterministic survivors (9 of them) split them cleanly:

- **5 pure kiters** — ran the full 600 ticks, no repeating loop. Genuine
  map-wide evasion. The thing the whole arms race was driving toward.
- **3 kite-then-loop** — kited for hundreds of ticks, then settled into a closed
  ring (3×3, 4×4, 4×5 — *different sizes*, which is the tell that it's geometry,
  not one gamed ring).
- **1 kite-then-stall** — kited, then parked where the hunter oscillated 2 tiles
  short (a brief reprise of the cubby flavor, only in the final few ticks).

Grove's read, before any analysis: *"I doubt it's an exploit. Tom and Jerry are
same-speed creatures."* That was exactly right. A loop-detector across all 9
survivors confirmed it — the loops were varied sizes and wall-shaped, the
signature of equal-speed pursuit geometry rather than a single exploitable bug.
DIST floored at **2** for every survivor, never 1: the universal fingerprint of
a same-speed hunter chasing the prey's *current* position. It can match every
step the prey takes but never gain the one it needs.

**Durable lesson (reinforced):** characterize the *population* before designing
the fix. One traced survivor (seed 20's 3×3 ring) would have led to a
"break-3×3-rings" patch that missed the 4×4 and 4×5 cases. Nine traces showed it
was geometry, which is a different fix entirely.

### Two assistant mis-scopes this arc, for the record

1. Called seed 21 "exploit #4" from a 300-tick trace (corrected by the 600-tick
   replay).
2. Earlier in the broader effort: edited the wrong patrol class, and assumed a
   coverage fix would catch all statues. Each time, *watching the behavior* beat
   *inferring from aggregates*. The project's discipline — trace before
   theorize — exists because inference keeps being wrong and observation keeps
   being right.

---

## Round 10 — the speed ramp

Because the circle is geometry, not a bug, the answer is a **design** decision:
should an equal-speed hunter be able to catch a perfect kiter? In Alien:
Isolation the answer is no — *you can't out-run the Alien*; the whole game is
breaking line of sight. So the hunter shouldn't be equal speed during a chase.

Grove's mechanic (better than the assistant's first proposals): **a pursuit
speed ramp.** Per tick of sustained committed pursuit (PURSUE/ATTACK), the
hunter gains +0.005 speed, capping at 1.15×. The fractional speed banks into an
accumulator; when it crosses 1.0 the hunter takes a **bonus step** (~1 extra
step per 7 ticks at cap). The ramp **decays only when the hunter gives up** the
chase (drops to SEARCH/PATROL) — brief line-of-sight flickers during a chase do
*not* reset it, because "the thrill of the chase doesn't vanish because the
hunter blinked."

Why this is the right shape, not a blunt speed buff:

- The hunter is **normal speed at the start** of every engagement — it has to
  *wind up*. A short chase to cover costs nothing.
- It only wins by **sustained** pursuit — so the counterplay is to make the
  chase *not* sustained: break line of sight, force the hunter back to SEARCH,
  reset the ramp. The only way to break LOS is to **use the stage** (corners,
  lockers, vents). The ramp doesn't just close the circle — it *creates the
  pressure* that should finally push the prey into the environment.
- It is config-gated and off by default (byte-identical legacy).

**Verification.** Math: cap at tick 29, first bonus step at tick 19 (no free
early lunge), 3-tick decay on give-up. Against the R9 circle-runner with the ramp
on: **all 9 survivors caught** (ticks 60–234). Seed 20 — the perfect 3×3 circle
that survived 600 ticks — was caught at **tick 63**, before it could even settle
into the ring. The hunter winds up during the open chase and closes the DIST-2
floor that defined the entire equal-speed era.

The footer heuristic in `trace_episode` still prints "never caught while seeing
Jerry" on a closing bonus-step catch — a cosmetic staleness to fix; the
`*** CAUGHT ***` line is authoritative.

---

## State at the end of this arc

- Every static exploit closed; the circle / kiting / equal-speed era closed by
  the ramp. A prey trained against the *old* hunter is 0/9 against the ramp.
- Scent diffusion + locker scent-puff: built, gated off, dormant infrastructure
  for the sensory layer.
- Map generator: connected and dead-end-free, with invariant tests.
- **485 tests passing.**

## What's open

1. **The ramp retrain (R10).** The real question this arc sets up: when the prey
   trains *against* the ramp, does it learn to break line of sight and use the
   stage (vents/lockers/corners) — the win condition — or does it just die? If
   survival stays at 0%, the ramp may be too strong and wants softer params
   (lower cap / slower ramp) to leave a learnable gap.
2. **The sensory layer.** The motionless-statue-in-a-blind-spot is still open and
   still sensory. Scent/presence sensing is its acceptance test (seed-26 stationary
   prey must be findable without smearing directional tracking).

## Durable lessons from this arc

1. Match the measurement window to the phenomenon — a truncated trace can invert
   the conclusion.
2. Verify the environment before spending GPU on it; a local invariant is not a
   global one.
3. Characterize the population before designing the fix; one trace can mislead.
4. Some "exploits" are geometry, not bugs — and the right answer is a design
   change, not a patch.
5. The designer's intuition about *mechanism* ("same-speed creatures") beat the
   assistant's pattern-match ("DIST pinned = exploit") again. Trace before
   theorize; ask the person who's watched it run.
