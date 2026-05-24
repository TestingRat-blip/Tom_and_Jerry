# Archive Manifest — Pre-Oxygen Obs-Space Change

**Archived:** 2026-05-22 (your local timeline may differ)
**Reason:** The locker oxygen/cooldown mechanic (see
`docs/DESIGN_LOCKER_OXYGEN.md`) adds Jerry's **oxygen level to the observation
vector.** This changes the observation-space shape, which makes every
checkpoint trained on the *old* obs vector **incompatible with the new env** —
they cannot be loaded against post-change code without a shape mismatch.

These checkpoints are the evidence for the entire arms-race arc (the tie-break
saga and the Round 5 locker-camping finding). They are preserved here as
historical artifacts. To reproduce any result below, check out the repo at a
commit BEFORE the oxygen obs change (or run against a build with the old obs
vector).

---

## Obs-space version
All checkpoints in this archive expect the **pre-oxygen observation vector**
(the obs space as of the Batch 13 patrol-stall fix; no oxygen field in Jerry's
observation). The post-change obs vector adds at least Jerry's oxygen level and
is NOT backward compatible.

---

## Checkpoints

### `jerry_generalist/`
- **What it is:** The baseline generalist — the original open-dancer.
- **Trained against:** the scripted/BFS-era Tom (pre-Conductor lineage).
- **Headline numbers:** vs scripted Tom ~32% (28-34% historical range);
  vs Conductor 6%.
- **What it demonstrated:** the honest survival *floor* against the Conductor —
  a Jerry with no exploit. Became the reference point for "what beats sound
  Tom" comparisons. Its open-dance feeds the Conductor's belief and gets it
  caught (6%).

### `jerry_generalist_vs_conductor/`  (the counter-Jerry)
- **What it is:** Generalist trained specifically against the **pre-fix**
  Conductor (the one with the fixed N/S-first BFS movement-priority tie-break).
- **Trained against:** pre-fix Conductor. ~1.5M steps.
- **Headline numbers:**
  - vs pre-fix Conductor: **40%** survival (the anomaly that started the saga).
  - vs scripted Tom: ~44% / held-out seeds robust.
  - vs **post-fix** Conductor: **6%** (collapses to floor once the tie-break
    groove is closed).
- **What it demonstrated:** the load-bearing exploit of the project. Its 40%
  was almost entirely the fixed movement-priority tie-break (the wall-pin
  column-bob standoff). Recognized by analogy to OSRS movement priority. When
  the tie-break was randomized (Batch 12d), this Jerry dropped 40%→6%, proving
  the groove *was* the exploit. **Do not lose this one** — it is the proof
  artifact for the tie-break fix.

### `jerry_generalist_vs_conductor_postfix/`  (Round 5)
- **What it is:** Fresh generalist trained from scratch against the **post-fix**
  (sound) Conductor.
- **Trained against:** post-fix Conductor (randomized tie-break). ~1.5M steps.
- **Headline numbers:** vs post-fix Conductor **14%** (24% on seeds 0-49);
  reward +2.10; mean episode length ~80 ticks.
- **What it demonstrated:** with the movement exploit gone, the dominant
  surviving strategy is **locker-camping.** `classify_survival` showed
  **12 of 12 survivors were locker-campers** (locker_frac 0.80-1.00, near-zero
  movement). This finding motivated the locker oxygen/cooldown design — the
  next balance change and the reason for this archive.

---

## Lineage (the arms race, in order)
1. Generalist learns open-dance → beats BFS-era Tom.
2. Conductor beats open-dance (noise-sensitive) → generalist falls to 6%.
3. Counter-Jerry finds the fixed-tie-break groove → 40% vs pre-fix Conductor.
4. Tie-break randomized (Batch 12d) → counter-Jerry collapses to 6%; sound Tom.
5. Fresh generalist vs sound Tom → 14%, **all of it locker-camping** (Round 5).
6. (next) Locker oxygen/cooldown → forces this archive's obs change; retrain.

## Reproduction notes
- These expect the pre-oxygen obs vector. Loading them post-change WILL fail
  on observation shape.
- Eval/trace tooling used: `scripts/eval_archetypes.py`,
  `scripts/trace_episode.py`, `scripts/classify_survival.py`.
- Tom variant for the post-fix results: `conductor` (ChemicalTom + Conductor,
  randomized BFS tie-break).
