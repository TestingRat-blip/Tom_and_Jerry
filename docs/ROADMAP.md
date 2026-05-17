# Tom_and_Jerry — Roadmap

Build order is deliberate. Earlier phases prove infrastructure works before later phases bet on it.

## Phase 0 — Scaffold (current)
- [x] Project root and folder structure
- [x] Design doc (`DESIGN.md`)
- [x] Roadmap (this file)
- [ ] Decision log skeleton (`DECISIONS.md`)
- [ ] `requirements.txt`, `.gitignore`, README

**Exit criteria:** repo cloneable, structure clear, design doc reviewed.

## Phase 1 — Minimum viable loop
- [ ] Headless grid env (30×30, walls, vents, lockers)
- [ ] Sound propagation + line of sight
- [ ] Gymnasium env wrapper
- [ ] One PPO Jerry, single reward (survival)
- [ ] One scripted Tom — Alien-baseline behavior tree, no learning
- [ ] Win-rate logging to TensorBoard
- [ ] Pygame replay renderer (load saved episode, play back)

**Exit criteria:** Jerry plays against Tom headless, ~1000 episodes/hour, you can watch any replay.

## Phase 2 — Drives + chemistry
- [ ] 6-axis drive vector implementation
- [ ] 5-chemical analog layer (port pattern from Vera)
- [ ] Drive/chemistry → behavior tree priority modulation
- [ ] Visualization: replay renderer overlays Tom's drive/chemistry state

**Exit criteria:** same Tom visibly behaves differently in different chemical states. Stalking vs frenzied vs cautious are distinguishable on replay.

## Phase 3 — L1 memory (per-encounter)
- [ ] Redis-backed encounter memory
- [ ] Hide-spot tracking, throw-direction tracking, path tracking
- [ ] Behavior tree unlock thresholds wired to L1
- [ ] Replay overlay: "Tom learned X this life"

**Exit criteria:** Tom that gets fooled by a noisemaker stops getting fooled after the third throw within a single episode. Visible on replay.

## Phase 4 — L2 / L3 memory (persistent)
- [ ] Postgres schema for player profiles + hunter identity
- [ ] ChromaDB episodic-memory embeddings
- [ ] Distillation pipeline: L1 → L2 on episode end
- [ ] Cross-session loading: gen-N+1 Tom starts with gen-N's L3 priors
- [ ] Bounded learning rate on L2 updates

**Exit criteria:** a Tom that lost to a particular Jerry 100 times wins more frequently in subsequent runs. Effect visible in tournament metrics.

## Phase 5 — Jerry population
- [ ] Archetype config system (`configs/archetypes/*.yaml`)
- [ ] Reward-shaping framework
- [ ] Six archetypes implemented: sneaker, sprinter, trickster, camper, explorer, generalist
- [ ] Population manager — rotate archetypes per training batch
- [ ] Generalist with population-based reward weight learning

**Exit criteria:** all six Jerrys trainable, distinct behavior visible on replay, can hold one out for evaluation.

## Phase 6 — Co-evolution
- [ ] Co-evolution scheduler (alternating updates, generation snapshots)
- [ ] Hall of fame: archive Toms, periodically spawn fresh-blood Jerrys against old Toms
- [ ] Cross-generation tournament harness
- [ ] Behavioral diversity metric
- [ ] Surprise rate metric
- [ ] Hold-out archetype eval loop

**Exit criteria:** Tom-N beats Tom-1 at the gen-1 task; diversity score does not collapse over 50 generations.

## Phase 7 — Tuning + human eval
- [ ] Weekly human-readability Likert rating protocol
- [ ] Hyperparameter sweep harness
- [ ] Ablation studies: chemistry on/off, hall of fame on/off, persistent memory on/off
- [ ] Documented results

**Exit criteria:** at least 3 of 5 success criteria from `DESIGN.md` §8 met.

## Phase 8+ — Future axes (not committed)
- Pack mechanics (v2): pack coordinator activated, fog-of-war between hunters, signal-based comms
- 3D port: same AI, different env layer
- Multi-species: heterogeneous packs with distinct drive baselines
- LLM flavor layer: vocalizations, environment storytelling (never on hot path)

## Time estimates

Rough, based on prior project velocity:

- Phase 1: ~1 week
- Phase 2: ~1 week
- Phase 3: ~3–5 days
- Phase 4: ~1–2 weeks (this is the novel one, expect tuning pain)
- Phase 5: ~3–5 days
- Phase 6: ~1–2 weeks
- Phase 7: ongoing

First genuinely terrifying Tom: target ~6–8 weeks from Phase 1 start.
