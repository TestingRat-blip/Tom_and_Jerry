"""ChemicalTom — ScriptedTom + drives + chemistry.

The behavior tree skeleton is identical to ScriptedTom (same five states,
same priority order). What changes:

  1. Prediction horizon: when adrenaline is high, Tom extrapolates Jerry's
     last two positions and paths to the PREDICTED future tile instead
     of Jerry's current tile. This is the dancing-exploit killer.

  2. Threshold modulation: noise_investigate_threshold, scent_search_threshold,
     pursue_memory, and investigate_dwell are all computed each tick from
     the combination of drives + chemistry, instead of being fixed constants.

  3. State selection modulation: caution and aggression drives bias the
     ATTACK vs PURSUE decision boundary.

The agent feels different in different chemical states without learning
anything — the same neural-circuit-style modulation that makes the
biological version of this work.

Per ADR-003, the STRUCTURE stays scripted. PARAMETERS are computed.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.env.world.world import Event, World
from src.hunter.agent.behavior.baseline import (
    ScriptedTom,
    ScriptedTomConfig,
    TomState,
)
from src.hunter.agent.chemistry.chemistry import Chemistry, ChemistrySystem
from src.hunter.agent.chemistry.config import ChemistryConfig
from src.hunter.agent.drives.config import DrivesConfig
from src.hunter.agent.drives.drives import DriveSystem, Drives
from src.utils.types import ACTION_DELTAS, Action, Position

if TYPE_CHECKING:
    from src.env.world.world import Grid
    from src.hunter.agent.conductor.conductor import Conductor
    from src.hunter.agent.memory.l1 import L1Memory
    from src.hunter.agent.memory.l2_lookup import L2Lookup
    from src.persistence.sqlite.l2_store import L2Store


@dataclass(frozen=True, slots=True)
class ChemicalTomConfig:
    """Tunable knobs for how chemistry/drives map to behavior.

    These are the WEIGHTS connecting internal state to behavioral params.
    The base ScriptedTomConfig parameters become the *baseline* values
    that get modulated.

    Default values are deliberately MODEST in v1 — large enough that
    behavior visibly differs across chemical states, small enough that
    the worst-case configuration doesn't catastrophically degrade
    Tom's tracking ability. Phase 5+ should learn better coefficients
    via co-evolution.
    """
    # Prediction horizon: adrenaline ∈ [0, 1] → predict 0 to 3 steps ahead
    max_prediction_steps: int = 3
    prediction_adrenaline_floor: float = 0.3  # below this, no prediction

    # Threshold modulation magnitudes (how much chemistry can shift each)
    # Final value = baseline * (1.0 + drive/chem contributions, clamped)
    noise_threshold_curiosity_mult: float = -0.25   # curious Tom hears more
    scent_threshold_caution_mult: float = +0.2      # cautious Tom needs more scent
    pursue_memory_aggression_mult: float = +0.5     # aggressive Tom remembers longer
    pursue_memory_cortisol_mult: float = -0.3       # frustrated Tom gives up faster
    investigate_dwell_curiosity_mult: float = +0.4  # curious Tom investigates longer

    # L1 contribution: weight applied to L1's false-noise factor when
    # computing the modulated noise threshold. The factor is already
    # bounded ∈ [0, L1Config.max_false_noise_factor], so this scales it
    # further when integrating into the threshold expression.
    l1_false_noise_weight: float = 1.0

    # State selection biases
    # When adjacent to Jerry, ATTACK vs PURSUE blends by aggression
    attack_aggression_threshold: float = 0.25       # below this, no committed attack at d=1


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ChemicalTom(ScriptedTom):
    """Stateful scripted hunter modulated by drives + chemistry.

    Same callable interface as ScriptedTom; can be dropped into any
    place ScriptedTom is used. The difference is felt, not announced.
    """

    def __init__(
        self,
        config: ScriptedTomConfig | None = None,
        chemical_config: ChemicalTomConfig | None = None,
        drives_config: DrivesConfig | None = None,
        chemistry_config: ChemistryConfig | None = None,
        l1: "L1Memory | None" = None,
        l2_lookup: "L2Lookup | None" = None,
        l2_store: "L2Store | None" = None,
        conductor: "Conductor | None" = None,
        seed: int | None = None,
    ):
        super().__init__(config=config, seed=seed)
        self.chemical_config = chemical_config or ChemicalTomConfig()
        self.drives = Drives()
        self.chemistry = Chemistry()
        self.drive_system = DriveSystem(drives_config)
        self.chemistry_system = ChemistrySystem(chemistry_config)

        # L1 per-encounter memory. Optional — None means "no L1," which
        # restores the original Phase 2 ChemicalTom behavior exactly.
        # When set, Tom records noise events / sightings each tick and
        # reads false-noise factor when computing the noise threshold.
        self.l1: "L1Memory | None" = l1

        # Phase 4 L2 persistent memory. Both optional and INDEPENDENT:
        #   - l2_lookup: read side. Used by warm_start_for_episode() to
        #     pre-seed L1 with priors from past episodes.
        #   - l2_store:  write side. Used by distill_at_episode_end() to
        #     write this episode's summary.
        # You can have one without the other (e.g. warm-start from a
        # frozen L2 database in eval mode, no writes). Both require L1
        # to be attached — without L1 there's nothing to warm-start or
        # to distill.
        self.l2_lookup: "L2Lookup | None" = l2_lookup
        self.l2_store: "L2Store | None" = l2_store

        # Phase 6c — the Conductor (director brain). Optional. When None,
        # ChemicalTom behaves EXACTLY as Phase 2-5: targeting uses Tom's
        # own per-encounter memory (last_seen_jerry / last_noise). When
        # attached, the strategic targeting decision routes through the
        # Conductor's belief instead — a unified, decaying, manipulable
        # belief that REPLACES Tom's private memory for the purpose of
        # answering "where do I think Jerry is?".
        #
        # Per ADR-013 this is how BFS-as-targeting gets replaced: Tom no
        # longer pathfinds toward his own freshly-remembered Jerry sightings;
        # he pathfinds toward the Conductor's (lossier, foolable) belief.
        # Chemistry modulation, prediction, and catch logic all stay — only
        # the SOURCE of the target changes.
        self.conductor: "Conductor | None" = conductor

        # Tracking for prediction: last two Jerry positions Tom has seen
        self._jerry_position_history: deque[Position] = deque(maxlen=3)

        # For external inspection
        self.last_predicted_jerry_pos: Position | None = None
        self.last_prediction_steps: int = 0

        # Phase 4 tracking: when did Tom first sight Jerry this episode?
        # Used in distillation as a "how findable is Jerry on this map" stat.
        # None means Tom never sighted Jerry this episode.
        self._ticks_to_first_sight: int | None = None

    def reset(self) -> None:
        super().reset()
        self.drive_system.reset(self.drives)
        self.chemistry_system.reset(self.chemistry)
        self._jerry_position_history.clear()
        self.last_predicted_jerry_pos = None
        self.last_prediction_steps = 0
        self._ticks_to_first_sight = None
        # If L1 is attached, clear its episode state too.
        if self.l1 is not None:
            self.l1.reset()
        # If a Conductor is attached, clear its belief for the new episode.
        if self.conductor is not None:
            self.conductor.reset()

    def warm_start_for_episode(
        self,
        grid: "Grid",
        jerry_policy: object,
        jerry_label: str | None = None,
    ) -> bool:
        """Phase 4: pull priors from L2 and pre-seed L1 before episode begins.

        Call AFTER reset() and BEFORE the first tick. The caller (typically
        ReplayRecorder, env wrapper, or training loop) supplies the grid
        and Jerry policy so we can compute fingerprints.

        Returns True if warm-start was applied (L1 + L2 lookup both present
        and at least one past episode found), False otherwise.

        No-op (returns False) when L1 or L2 lookup isn't attached — Tom
        behaves like Phase 3.
        """
        if self.l1 is None or self.l2_lookup is None:
            return False
        # Compute fingerprints
        from src.hunter.agent.memory.fingerprint import (
            fingerprint_jerry,
            fingerprint_map,
        )
        fine_fp, coarse_fp = fingerprint_map(grid)
        jerry_fp = fingerprint_jerry(jerry_policy, label=jerry_label)
        # Query L2 + build the warm-start
        warm = self.l2_lookup.build_warm_start(
            map_fp_fine=fine_fp,
            map_fp_coarse=coarse_fp,
            jerry_fp=jerry_fp,
        )
        if warm.is_empty:
            return False
        self.l1.apply_warm_start(warm)
        return True

    def distill_at_episode_end(
        self,
        grid: "Grid",
        jerry_policy: object,
        outcome: str,
        total_ticks: int,
        total_jerry_reward: float,
        tom_label: str = "",
        jerry_label: str | None = None,
        notes: dict | None = None,
    ) -> bool:
        """Phase 4: summarize this episode's L1 into L2 at episode end.

        Call BEFORE the next reset(). reset() clears L1's in-episode
        counters, which distillation needs to read.

        Returns True if distillation happened (L1 + L2 store both present),
        False otherwise. No-op when L1 or L2 store isn't attached.
        """
        if self.l1 is None or self.l2_store is None:
            return False
        from src.hunter.agent.memory.distillation import distill_l1_to_summary
        summary = distill_l1_to_summary(
            l1=self.l1,
            grid=grid,
            jerry_policy=jerry_policy,
            outcome=outcome,
            total_ticks=total_ticks,
            total_jerry_reward=total_jerry_reward,
            ticks_to_first_sight=self._ticks_to_first_sight,
            tom_label=tom_label,
            jerry_label=jerry_label,
            notes=notes,
        )
        self.l2_store.insert(summary)
        return True

    def __call__(self, world: World) -> Action:
        """Per-tick decision.

        When this is called, world._events_this_tick contains the events
        from the MOST RECENT completed step (the env calls our policy
        before step()). So we update chemistry/drives from those events,
        then decide this tick's action.
        """
        # Track first sight time for the episode summary
        if self._ticks_to_first_sight is None and world._tom_can_see_jerry():
            self._ticks_to_first_sight = world.tick_count

        # 1. Update drives/chemistry from the previous step's events.
        events = list(getattr(world, "_events_this_tick", []))
        self.drive_system.tick(
            self.drives,
            events=events,
            agent_moved=self.last_decided_action not in (Action.WAIT, Action.INTERACT),
        )
        self.chemistry_system.tick(
            self.chemistry,
            events=events,
            jerry_visible=world._tom_can_see_jerry(),
        )

        # 1b. Update L1 with what just happened. L1 will record noises,
        # try to verify pending noises against the current sighting, and
        # bump locker/heatmap counters as appropriate.
        if self.l1 is not None:
            self.l1.observe_events(
                events,
                tom_pos=world.tom.position,
                jerry_pos=world.jerry.position,
                jerry_visible=world._tom_can_see_jerry(),
                tick=world.tick_count,
            )

        # 2. Update Jerry-position history (used for prediction)
        # Only push when Jerry's position has actually changed, so the deque
        # tracks DISTINCT recent positions for velocity estimation.
        if world._tom_can_see_jerry():
            jp = world.jerry.position
            if not self._jerry_position_history or self._jerry_position_history[-1] != jp:
                self._jerry_position_history.append(jp)

        # 3. Update memory from current perceptions.
        #    Without a Conductor: Tom uses his own direct perception
        #    (the Phase 2-5 path). With a Conductor: the Conductor observes
        #    the world and Tom's memory fields are populated FROM the
        #    Conductor's belief instead — a unified, decaying, foolable
        #    belief replacing Tom's private binary memory. Downstream logic
        #    (state selection, prediction, action) is identical either way;
        #    only the SOURCE of last_seen_jerry / last_noise changes.
        if self.conductor is not None:
            self.conductor.observe(world)
            self._update_memory_from_conductor(world)
        else:
            self._update_memory(world)

        # 4. Select state with chemistry-modulated thresholds
        self.state = self._select_state_chemical(world)

        # 5. Choose action with prediction horizon
        action = self._act_for_state_chemical(world)

        # 6. Wall avoidance (parent behavior)
        if self.config.wall_bump_avoidance:
            action = self._avoid_walls(world, action)

        self.last_decided_action = action
        return action

    # ---- modulated thresholds ------------------------------------------

    def _modulated_pursue_memory(self) -> int:
        """How many ticks Tom remembers Jerry's position after losing sight."""
        base = self.config.pursue_memory
        cfg = self.chemical_config
        # Aggression lengthens, cortisol shortens
        agg_term = self.drives.aggression * cfg.pursue_memory_aggression_mult
        cort_term = self.chemistry.cortisol * cfg.pursue_memory_cortisol_mult
        multiplier = 1.0 + agg_term + cort_term
        return max(1, int(base * _clamp(multiplier, 0.2, 2.5)))

    def _modulated_investigate_dwell(self) -> int:
        """How many ticks Tom keeps chasing a noise after it goes quiet."""
        base = self.config.investigate_dwell
        cur_term = self.drives.curiosity * self.chemical_config.investigate_dwell_curiosity_mult
        return max(1, int(base * (1.0 + cur_term)))

    def _modulated_noise_threshold(self, tom_pos: Position | None = None) -> float:
        """Noise level needed to trigger INVESTIGATE state.

        If `tom_pos` is given AND L1 is attached, the L1 false-noise factor
        near Tom's current position is folded in. The factor is bounded
        ∈ [0, max_false_noise_factor] so the composition stays sane:

            threshold = base * (1 + curiosity_term + l1_factor * l1_weight)

        Curiosity LOWERS the threshold; L1 false-noise factor RAISES it.
        They cancel out near zero, compose smoothly elsewhere.
        """
        base = self.config.noise_investigate_threshold
        cur_term = self.drives.curiosity * self.chemical_config.noise_threshold_curiosity_mult

        l1_term = 0.0
        if self.l1 is not None and tom_pos is not None:
            l1_factor = self.l1.false_noise_factor_near(tom_pos)
            l1_term = l1_factor * self.chemical_config.l1_false_noise_weight

        # Negative cur_term → curious Tom has a LOWER threshold (hears more)
        # Positive l1_term → fooled Tom has a HIGHER threshold (skeptical)
        return max(0.05, base * (1.0 + cur_term + l1_term))

    def _modulated_scent_threshold(self) -> float:
        base = self.config.scent_search_threshold
        caut_term = self.drives.caution * self.chemical_config.scent_threshold_caution_mult
        return max(0.05, base * (1.0 + caut_term))

    # ---- state selection -----------------------------------------------

    def _select_state_chemical(self, world: World) -> TomState:
        """Like ScriptedTom._select_state but with modulated thresholds.

        Plus: aggression biases ATTACK vs PURSUE at close range.
        """
        tick = world.tick_count

        # ATTACK / PURSUE branch
        if world._tom_can_see_jerry():
            d = world.tom.position.manhattan(world.jerry.position)
            if d <= world.config.catch_distance + 1:
                # Aggressive Tom commits to ATTACK; cautious Tom may PURSUE
                # to wait for a better angle
                if self.drives.aggression >= self.chemical_config.attack_aggression_threshold:
                    return TomState.ATTACK
                # Still close but not committing — treat as PURSUE
                return TomState.PURSUE
            return TomState.PURSUE

        # PURSUE: lost sight but recently saw, with modulated memory
        if (
            self.last_seen_jerry is not None
            and tick - self.last_seen_tick <= self._modulated_pursue_memory()
        ):
            return TomState.PURSUE

        # INVESTIGATE: heard something recently, with modulated dwell
        if (
            self.last_noise is not None
            and tick - self.last_noise_tick <= self._modulated_investigate_dwell()
        ):
            return TomState.INVESTIGATE

        # SEARCH: scent gradient strong, with modulated threshold
        obs = world._observe_tom()
        max_scent = max(obs.scent_n, obs.scent_s, obs.scent_e, obs.scent_w)
        if max_scent >= self._modulated_scent_threshold():
            return TomState.SEARCH

        return TomState.PATROL

    # ---- override _update_memory to use modulated noise threshold ----

    def _update_memory_from_conductor(self, world: World) -> None:
        """Phase 6c: populate Tom's memory fields from the Conductor's belief
        instead of from Tom's direct perception.

        The Conductor has already observed the world this tick (sightings,
        noises, scent → belief, with decay). Here we translate the belief's
        live suspicions into the SAME memory fields the downstream logic
        reads (last_seen_jerry / last_noise), so state selection, prediction,
        and action choice work unchanged.

        Mapping:
          - strongest SIGHTING suspicion → last_seen_jerry / last_seen_tick
          - strongest NOISE   suspicion → last_noise      / last_noise_tick
          - SCENT suspicions are NOT mapped here; scent is read live by
            _select_state_chemical via world._observe_tom(), preserving the
            existing SEARCH-state path.

        The crucial difference from _update_memory: these fields now reflect
        a DECAYING, FOOLABLE belief. A false noise creates a real NOISE
        suspicion Tom will chase; a sighting fades continuously rather than
        being remembered crisply until a hard timeout. This is the
        "weakening" ADR-013 calls for — Tom hunts from belief, not from
        perfect private memory.

        We still update _jerry_position_history and first-sight tracking
        from genuine live visibility (handled in __call__), because those
        feed prediction and episode stats, not strategic targeting.
        """
        from src.hunter.agent.conductor.belief import SuspicionType

        tick = world.tick_count
        now_live = self.belief_live_sources(world)

        # Find the strongest live suspicion of each relevant type.
        best_sighting = None
        best_noise = None
        for src, conf in now_live:
            if src.type == SuspicionType.SIGHTING and best_sighting is None:
                best_sighting = (src, conf)
            elif src.type == SuspicionType.NOISE and best_noise is None:
                best_noise = (src, conf)
            if best_sighting is not None and best_noise is not None:
                break

        # SIGHTING suspicion → Tom's "last seen Jerry" memory.
        if best_sighting is not None:
            self.last_seen_jerry = best_sighting[0].position
            self.last_seen_tick = tick

        # NOISE suspicion → Tom's "last noise" memory.
        if best_noise is not None:
            self.last_noise = best_noise[0].position
            self.last_noise_tick = tick

    def belief_live_sources(self, world: World):
        """Convenience accessor: the Conductor's live suspicion sources this
        tick, strongest first. Empty list if no Conductor attached.
        """
        if self.conductor is None:
            return []
        return self.conductor.belief.live_sources(world.tick_count)

    def _patrol(self, world: World) -> Action:
        """Phase 6d: Conductor-directed patrol when a Conductor is attached.

        When no Conductor: fall back to the base ScriptedTom random-target
        patrol (Phase 1-5 behavior).

        With a Conductor: walk toward the Conductor's directed patrol target
        (the least-recently-visited sector), so patrol becomes a legible
        coverage sweep instead of random wandering. We reuse Tom's own
        patrol-retarget cadence so movement stays smooth — we only swap the
        SOURCE of the target, mirroring the 6c handover philosophy.
        """
        if self.conductor is None:
            return super()._patrol(world)

        # Re-target on the same cadence as the base patrol: when we have no
        # target, when we've reached it, or when it's gone stale.
        need_new = (
            self.patrol_target is None
            or world.tom.position.manhattan(self.patrol_target)
                <= self.config.patrol_retarget_distance
            or world.tick_count - self.patrol_target_set_tick
                > self.config.patrol_retarget_after
        )
        if need_new:
            self.patrol_target = self.conductor.patrol_target(world)
            self.patrol_target_set_tick = world.tick_count
        return self._step_toward(world.tom.position, self.patrol_target, world)

    # ---- override _update_memory to use modulated noise threshold ----

    def _update_memory(self, world: World) -> None:
        """Same as parent, but uses the modulated noise threshold."""
        tick = world.tick_count
        if world._tom_can_see_jerry():
            self.last_seen_jerry = world.jerry.position
            self.last_seen_tick = tick

        obs = world._observe_tom()
        dirs = {"N": obs.sound_n, "S": obs.sound_s,
                "E": obs.sound_e, "W": obs.sound_w}
        loudest = max(dirs.items(), key=lambda kv: kv[1])
        if loudest[1] >= self._modulated_noise_threshold(tom_pos=world.tom.position):
            step_map = {
                "N": Position(0, -3), "S": Position(0, 3),
                "E": Position(3, 0), "W": Position(-3, 0),
            }
            self.last_noise = world.tom.position + step_map[loudest[0]]
            self.last_noise_tick = tick

    # ---- action selection with prediction -----------------------------

    def _act_for_state_chemical(self, world: World) -> Action:
        """Like ScriptedTom._act_for_state, but the target for PURSUE/ATTACK
        is a PREDICTED future Jerry position when adrenaline is high enough.
        """
        if self.state == TomState.ATTACK:
            target = self._predict_jerry_target(world)
            return self._step_toward(world.tom.position, target, world)

        if self.state == TomState.PURSUE:
            if world._tom_can_see_jerry():
                target = self._predict_jerry_target(world)
            elif self.last_seen_jerry is not None:
                target = self.last_seen_jerry
            else:
                return self._patrol(world)
            return self._step_toward(world.tom.position, target, world)

        if self.state == TomState.INVESTIGATE:
            if self.last_noise is None:
                return self._patrol(world)
            return self._step_toward(world.tom.position, self.last_noise, world)

        if self.state == TomState.SEARCH:
            return self._follow_scent(world)

        return self._patrol(world)

    def _predict_jerry_target(self, world: World) -> Position:
        """Predict Jerry's future position using AVERAGE velocity over recent
        history, scaled by adrenaline.

        Averaging matters: a pure oscillator (N/S/N/S) has zero average
        velocity, so we correctly predict its current position rather than
        chasing the last single-tick step. Linear drift still extrapolates
        normally because consistent direction produces a non-zero average.
        """
        adr = self.chemistry.adrenaline
        cfg = self.chemical_config

        if adr < cfg.prediction_adrenaline_floor or len(self._jerry_position_history) < 2:
            self.last_predicted_jerry_pos = world.jerry.position
            self.last_prediction_steps = 0
            return world.jerry.position

        # Average velocity across all recent position deltas (up to 3 in deque)
        hist = list(self._jerry_position_history)
        deltas: list[tuple[int, int]] = []
        for i in range(1, len(hist)):
            deltas.append((hist[i].x - hist[i-1].x, hist[i].y - hist[i-1].y))
        if not deltas:
            self.last_predicted_jerry_pos = world.jerry.position
            self.last_prediction_steps = 0
            return world.jerry.position

        avg_dx = sum(d[0] for d in deltas) / len(deltas)
        avg_dy = sum(d[1] for d in deltas) / len(deltas)

        # adrenaline 0.3 → 0.5 step ahead; 1.0 → max_prediction_steps ahead
        adr_normalized = (adr - cfg.prediction_adrenaline_floor) / \
                         max(1e-6, 1.0 - cfg.prediction_adrenaline_floor)
        steps_ahead = max(1, int(round(adr_normalized * cfg.max_prediction_steps)))

        curr = hist[-1]
        # Predict using the AVERAGE velocity, rounded to nearest tile
        predicted = Position(
            curr.x + int(round(avg_dx * steps_ahead)),
            curr.y + int(round(avg_dy * steps_ahead)),
        )

        # If the average velocity is essentially zero (oscillator), the
        # prediction equals current position — record steps=0 to be honest
        # that we're not really predicting.
        if predicted == curr:
            self.last_predicted_jerry_pos = curr
            self.last_prediction_steps = 0
            return curr

        # Snap to nearest walkable tile in case prediction lands in a wall
        if not world.grid.in_bounds(predicted) or not world.grid.is_walkable(predicted):
            for s in range(steps_ahead - 1, 0, -1):
                candidate = Position(
                    curr.x + int(round(avg_dx * s)),
                    curr.y + int(round(avg_dy * s)),
                )
                if world.grid.in_bounds(candidate) and world.grid.is_walkable(candidate):
                    predicted = candidate
                    break
            else:
                predicted = curr

        self.last_predicted_jerry_pos = predicted
        self.last_prediction_steps = steps_ahead
        return predicted
