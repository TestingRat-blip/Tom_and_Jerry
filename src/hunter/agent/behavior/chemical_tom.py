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
    from src.hunter.agent.memory.l1 import L1Memory


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

        # Tracking for prediction: last two Jerry positions Tom has seen
        self._jerry_position_history: deque[Position] = deque(maxlen=3)

        # For external inspection
        self.last_predicted_jerry_pos: Position | None = None
        self.last_prediction_steps: int = 0

    def reset(self) -> None:
        super().reset()
        self.drive_system.reset(self.drives)
        self.chemistry_system.reset(self.chemistry)
        self._jerry_position_history.clear()
        self.last_predicted_jerry_pos = None
        self.last_prediction_steps = 0
        # If L1 is attached, clear its episode state too.
        if self.l1 is not None:
            self.l1.reset()

    def __call__(self, world: World) -> Action:
        """Per-tick decision.

        When this is called, world._events_this_tick contains the events
        from the MOST RECENT completed step (the env calls our policy
        before step()). So we update chemistry/drives from those events,
        then decide this tick's action.
        """
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

        # 3. Update memory from current perceptions
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
