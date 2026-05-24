## ADR-007 — Discrete action space, swappable
**Date:** 2026-05-17
**Status:** Accepted

Phase 1 uses a discrete action space for both Tom and Jerry: `{NORTH, SOUTH, EAST, WEST, WAIT, INTERACT}`. Grids are inherently discrete and PPO trains substantially faster on small action spaces. Action interface is abstracted behind an `ActionSpace` protocol so a continuous variant can be swapped in later (3D port, fine motor control) without rewriting agents.

---

## ADR-008 — Stacked observation space: egocentric sensors + local grid window
**Date:** 2026-05-17
**Status:** Accepted

Observations combine:
- Egocentric sensor readings (sound levels by direction, LOS hits, scent gradient, own drive/chemistry state if applicable)
- A small local grid window (e.g. 7×7 centered on the agent) flattened into the vector

This dual representation gives the agent both *what it senses as a creature* and *spatial context for navigation*. The sensor channel is the one that ports cleanly to 3D later; the grid window is a Phase 1–4 expedient that will be replaced or supplemented when perspective shifts.

Critically: Tom and Jerry get *different* observation vectors. Jerry sees what a survivor sees. Tom sees what a predator senses (longer sound range, scent gradient, no minimap-style awareness of the full layout). This asymmetry is intentional and core to the design.

---

## ADR-009 — Hybrid Tom interface: scripted execution, RL-shaped logging
**Date:** 2026-05-17
**Status:** Accepted

Tom uses the same Gymnasium action interface as Jerry from day one, even though Phase 1 Tom is scripted. The scripted behavior tree emits actions through the env's `step()` API, identical to how a future learned Tom will. Additionally, Tom logs:
- The observation it would have received as an RL agent
- The action it actually took (chosen by behavior tree)
- The reward it would have received under candidate reward functions

This means: when Phase 4+ wants to train Tom (or train *parts* of Tom — node-unlock thresholds, drive baselines, chemistry curves), the data is already there. No refactor, no second integration pass. The scripted Tom is implicitly producing imitation-learning data the whole time.
