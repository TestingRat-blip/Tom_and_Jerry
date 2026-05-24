## ADR-010 — Rebalanced Jerry reward weights (Phase 1 v2)
**Date:** 2026-05-17
**Status:** Accepted (supersedes implicit values in initial JerryRewardConfig)

After a 1M-step training run produced a Jerry with 0% survival rate
(worse than random and worse than passive-WAIT), the diagnostic
revealed two failure modes:

1. **Reward poisoning.** Original weights made surviving 200 ticks
   worth `+7.0` while losing in 150 ticks was worth `-8.5`. The gap
   existed but small per-tick penalties (`-0.05 seen`, `-0.02 noise`)
   meant that "stop moving, accept fate" had a less-bad expected return
   than "actively try to survive but probably fail."

2. **Argmax collapse.** The deterministic policy concentrated 47%+ of
   actions on WAIT and 0% on EAST/SOUTH. Stochastic eval survived ~2x
   more often than deterministic eval.

New weights make survival decisively rewarded:
  survival_per_tick:  0.01 → 0.05   (5x)
  penalty_seen:      -0.05 → -0.02  (less punishing of visibility)
  penalty_noise:     -0.02 → -0.005 (much less punishing of sound)
  penalty_bump_wall: -0.01 → -0.002
  penalty_caught:   -10.0 → -5.0    (LESS catastrophic, see below)
  bonus_survived:    +5.0 → +20.0   (4x)

Expected reward math:
  300-tick survive:  300*0.05 + 20.0 = +35.0
  150-tick catch:    150*0.05 -  5.0 =  +2.5
  decisive 32.5-point gap, clearly preferring survival.

Counterintuitively, the catch penalty was REDUCED. A massive catch
penalty causes reward poisoning — every action looks bad, gradients
smear toward WAIT. The correct shape is "catch is roughly equal to
the expected reward of surviving," so the gradient says "survive,
don't shut down."

---

## ADR-011 — Stochastic eval as primary metric; entropy coefficient 0.01
**Date:** 2026-05-17
**Status:** Accepted

Phase 1 v1 used deterministic eval (`deterministic=True`), which masked
argmax-collapse failures. Going forward:

- Training reports BOTH deterministic and stochastic survival rates.
- When the gap exceeds 10%, the training script flags it.
- The default entropy coefficient is bumped from SB3's 0.0 to 0.01 to
  discourage policies from collapsing onto a tiny set of high-mode
  actions.
- The diagnostic script (`scripts/diagnose.py`) is the canonical tool
  for inspecting any trained policy's action distribution and
  deterministic/stochastic gap.

This will matter even more in Phase 5 when six archetypes train side
by side — each must have a healthy action distribution or its
"personality" is fake.

---

## ADR-012 — Training episode length vs eval episode length
**Date:** 2026-05-17
**Status:** Accepted

Original Phase 1 used `max_ticks=600` for both training and eval.
With an 80%-catch baseline Tom, surviving 600 ticks is nearly
unachievable — Jerry sees terminal "you survived" signals essentially
never during early training, so the survival pathway never gets
gradient.

New convention:
  --world-max-ticks: training cap, default 300
  --eval-max-ticks:  eval cap, default = world-max-ticks

Shorter training episodes mean more terminal signals per env step,
giving the survival pathway gradient sooner. Eval can independently
test longer-horizon survival once a policy starts working.

This is a curriculum knob more than a true hyperparameter. Phase 5+
will likely add explicit difficulty curricula (deaf-Tom, blind-Tom)
for the same reason.
