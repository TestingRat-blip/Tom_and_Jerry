# Phase 1 — Retrospective

**Status:** Complete
**Wrapped:** 2026-05-17
**Baseline checkpoint:** `data/snapshots/jerry_v1_baseline.zip` (1.5M steps, run `jerry_v4_overnight`)

---

## Exit criteria check

From the original roadmap:

> Jerry plays Tom headlessly at ~1000 episodes/hour, you can load any episode log and watch it back in pygame.

| Criterion | Target | Achieved |
|---|---|---|
| Headless training throughput | 1000+ eps/hr | ~6,000 random / ~3,400 active |
| PPO Jerry learns vs ScriptedTom | "meaningfully better than random" | **30%** survival vs **0%** random |
| Replay renderer | Yes | Deferred to Phase 1.5 |
| Gymnasium-compliant env | Yes | Yes |
| Full test coverage | Yes | 80 tests passing |

**Phase 1 is functionally complete.** The replay renderer is the one item deferred; it's small and will land as Phase 1.5 before Phase 2 begins.

---

## Headline results

Evaluation of the 1.5M-step Jerry over 50 episodes, 600 max ticks, vs ScriptedTom:

| Policy | Survival rate | Mean reward | Mean episode length |
|---|---|---|---|
| Random | 0% | -11.25 | 104 |
| Passive (always WAIT) | 0% | -9.11 | 128 |
| **Trained PPO Jerry @ 1500k** | **30%** | **+12.12** | **270** |

ScriptedTom catches passive prey ~80% of the time; trained Jerry has cut that catch rate by ~10 percentage points and roughly doubled survival duration on the episodes he eventually loses.

---

## What worked

- **Tooling-first investment paid off.** When the v1 run failed (1M steps, 0% survival), the issue was reward shape — not architecture, not bugs. The diagnostic script we wrote in response found the problem immediately on the next run, and will catch the same class of bug in Phase 5 when six archetypes train in parallel.
- **Deterministic AND stochastic eval.** Reporting both caught the argmax-collapse failure mode that had been hiding in v1. This is now permanent project hygiene.
- **Rotating eval seeds.** Eliminated the seed-memorization risk that fixed seeds carry. Without this, the per-checkpoint eval results would have looked stable when they shouldn't have.
- **Asymmetric observations for Tom and Jerry (ADR-008).** Tom gets scent, Jerry doesn't. Tom has 360° sound, Jerry has cone-of-vision sound. The asymmetry made Tom feel like a *creature* rather than an opponent and gave the env real strategic depth.
- **Behavior-tree-with-learned-parameters (ADR-003).** ScriptedTom's BFS pathfinding and priority-ordered states (ATTACK > PURSUE > INVESTIGATE > SEARCH > PATROL) produced a control-group opponent strong enough to actually test Jerry. 80% catch rate on passive prey means he's a real threat.

---

## What failed, and why

### Run v1: 1M steps, 0% survival

Caused by **reward poisoning + argmax collapse**.

- The original reward weights had `survival_per_tick=0.01` and `bonus_survived=+5.0` vs `penalty_caught=-10.0`. Over a 200-tick episode, "survive" was worth +7 and "die" was worth -8.5. The gap existed but the per-tick noise/visibility penalties pushed Jerry into a "give up and minimize per-tick punishment" local optimum.
- Deterministic eval was being used as the sole metric, hiding argmax collapse: Jerry's policy concentrated 47%+ of actions on WAIT and ~0% on EAST/SOUTH. Eval went 0% even though stochastic survival was 13%.

**Fix:** rewards rebalanced to make survival decisively positive (+35 expected) vs catching decisively but not catastrophically negative (+2.5 expected) — see ADR-010. Eval extended to report both modes — ADR-011.

### Misread of the v4 run as a plateau

Mid-run I called the v4 run a plateau and recommended stopping. I was wrong — looking at the full per-checkpoint eval sweep afterward showed a clear climb from 0% (at 700k) to 30% (at 1500k). The mistake was using per-50k eval snapshots as my signal; they're too noisy at 20-30 episodes each. **Lesson:** trust the per-checkpoint post-hoc sweep over live eval snapshots when judging convergence.

### Counterintuitive: Jerry discovered an oscillation exploit, not stealth

The 1500k diagnostic (100 episodes, deterministic):

| Action | Frequency |
|---|---|
| NORTH | 31.9% |
| SOUTH | 31.3% |
| WAIT | 26.1% |
| WEST | 5.0% |
| EAST | 4.1% |
| INTERACT | 1.7% |

Jerry's deterministic policy is **vertical oscillation with strategic pauses**. He flips North/South rapidly with WAITs interleaved, almost never going East/West, almost never using lockers or vents. This produces:

- 31% deterministic survival
- 480 noise events per episode (he's not stealthing — he's loudly oscillating)
- Tom sees him only 29.4% of ticks despite the noise
- Median episode length: 180 ticks

The exploit is against ScriptedTom's pathfinding: Tom does BFS to Jerry's current tile every tick. When Jerry oscillates N/S, Tom commits to a path that's stale by the time he takes a step. The BFS oscillates with Jerry and never quite closes the distance.

This is fascinating for several reasons:

- **Exploit, not strategy.** Jerry found a quirk in ScriptedTom's lookahead, not a general evasion principle. A learned Tom in Phase 4+ would not be fooled by this.
- **Brittle.** Stochastic eval collapses to 10% survival because the trick *requires* near-deterministic oscillation. Sampling noise breaks the pattern at random moments and gets Jerry killed.
- **Failure mode of the project's premise.** The whole point of co-evolution is to escape exploit-based policies. ScriptedTom is the wrong opponent to co-evolve against (which we already knew — see ADR-003 — but the exploit makes the reason concrete).
- **Useful data for Phase 5.** When the six archetypes train, one of them will likely discover this same exploit. The hall-of-fame mechanism in Phase 6 is what stops it from sticking.

### Implications for Phase 1.5 + Phase 2

- The deterministic/stochastic gap warning in `scripts/diagnose.py` needs nuancing. A large gap can mean either *argmax collapse* (untrained / over-collapsed policy) or *exploit convergence* (precise learned tactic). Currently the script can't tell them apart. Worth distinguishing by checking entropy_loss during training: if entropy is healthy (-1.0 to -1.8) AND gap is large, that's an exploit; if entropy is tiny AND gap is large, that's collapse.
- The 1500k checkpoint is **pinned for deterministic use only**. `scripts/evaluate.py` defaults to deterministic so this is fine, but document the brittleness.
- This is exactly the kind of policy that Phase 6's hall-of-fame regularizer is designed to break. Fresh-blood Jerrys trained against archived Toms can't exploit the same way because the archived Tom's flaws are different.

---

## What this tells us about Phase 2 and beyond

- **The PPO + 7×7 grid window + scalar sensors policy substrate hits a ceiling around 30%.** Pushing it further with the current architecture would require either curriculum learning (deaf Tom → blind Tom → full Tom) or richer observations (tick counter, memory).
- **The plateau is the right signal that we should change architecture, not throw more compute at the current one.** Phase 2's chemistry layer (for Tom) and Phase 4's persistent memory (for both) are exactly the substrate upgrades the project plan anticipated.
- **The strong baseline ScriptedTom may need to be excluded from Phase 6 co-evolution.** He uses BFS pathfinding — perfect information about the grid — which a learned Tom wouldn't have. Co-evolving Toms against this baseline would be unfair. Use him only as a *control group* (ADR-003), never as a co-evolution opponent.

---

## Artifacts pinned

- `data/snapshots/jerry_v1_baseline.zip` — the 1.5M-step Jerry, 30% survival vs ScriptedTom
- `data/logs/jerry_v4_overnight/eval_log.jsonl` — per-eval-cycle metrics
- `data/logs/jerry_v4_overnight/config.json` — full training config for reproducibility
- ADRs 010, 011, 012 in `docs/DECISIONS.md` — the reward and eval fixes

The v4 snapshots between checkpoints (`ckpt_100000_steps.zip` through `ckpt_1500000_steps.zip`) are kept for the per-checkpoint sweep we ran. Consider keeping `ckpt_700000_steps.zip` specifically — it's the interesting failure-mode "ultra-stealth Jerry" that died fast trying to be invisible. Useful for Phase 5 comparisons.

---

## Next steps

- **Phase 1.5:** pygame replay renderer. Watch the 1500k Jerry hunt-and-evade. Useful infrastructure for Phase 2 when chemistry effects need to be *seen*.
- **Phase 2:** drives + chemistry layer. Originally scoped for Tom; we may want to reconsider whether Jerry also benefits from chemistry (panic, fatigue, etc.) given how visible-Jerry's evasion behavior is.
- **Phase 4 preview:** the "Jerry learned to survive *contact* rather than avoid it" emergent behavior is a strong argument that Phase 4's memory tiers should include the equivalent for Jerry — knowledge of "where I tend to die" could push survival meaningfully higher.
