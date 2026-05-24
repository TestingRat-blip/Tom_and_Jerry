"""The scripted baseline Tom — Alien-style hunter, no learning.

This is the CONTROL GROUP. Every learned Tom in Phase 4+ gets
benchmarked against this scripted version. If a learned Tom isn't
better than this on hold-out Jerrys, we haven't actually improved
anything.

Five behavior states, priority-ordered:
  ATTACK     — adjacent to Jerry, visible → step toward and catch
  PURSUE     — Jerry visible at range → close distance
  INVESTIGATE— heard loud noise → move toward it
  SEARCH     — strong scent gradient → follow it
  PATROL     — none of the above → wander toward a patrol target

Memory is per-encounter only (cleared every reset). For Phase 1 this
is fine; persistent memory (L1/L2/L3) arrives in Phase 3+.

Per ADR-003, the *structure* of this behavior is scripted; parameters
(thresholds, ranges, dwell times) are the thing that will become learnable
later. We expose them on a config dataclass for that reason.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import IntEnum

from src.env.sensors.los import visible_from
from src.env.world.world import World
from src.utils.types import ACTION_DELTAS, Action, Position, TileType


class TomState(IntEnum):
    """Top-level behavior states. Higher value = higher priority."""
    PATROL = 0
    SEARCH = 1
    INVESTIGATE = 2
    PURSUE = 3
    ATTACK = 4


@dataclass(frozen=True, slots=True)
class ScriptedTomConfig:
    """Tunable parameters for the scripted Tom.

    These are the values that will eventually become learnable per-archetype
    in Phase 4+. Keep them named and documented so the learning code
    knows what to vary.
    """
    # Sensory thresholds
    noise_investigate_threshold: float = 0.3   # noise level that triggers INVESTIGATE
    scent_search_threshold: float = 0.15       # scent level that triggers SEARCH

    # Persistence
    investigate_dwell: int = 12  # ticks to keep moving toward last noise after it goes quiet
    pursue_memory: int = 18      # ticks to remember Jerry's last-seen position after losing sight

    # Patrol
    patrol_retarget_distance: int = 3  # how close to a patrol point before picking a new one
    patrol_retarget_after: int = 40    # max ticks on one patrol target before giving up
    patrol_target_max_tries: int = 8   # attempts to draw a REACHABLE patrol target (Defect-A fix)

    # Movement
    wall_bump_avoidance: bool = True   # if a chosen action would bump a wall, try alternatives


class ScriptedTom:
    """Stateful scripted hunter. Instantiate once, call as a policy
    (it's a `__call__`-able), it tracks its own internal state.

    Usage:
        tom = ScriptedTom(seed=42)
        env = JerryEnv(tom_policy=tom)
    """

    def __init__(self, config: ScriptedTomConfig | None = None, seed: int | None = None):
        self.config = config or ScriptedTomConfig()
        self._rng = random.Random(seed)

        # Per-encounter memory — reset implicitly when caller calls reset()
        self.state: TomState = TomState.PATROL
        self.last_seen_jerry: Position | None = None
        self.last_seen_tick: int = -10**9
        self.last_noise: Position | None = None
        self.last_noise_tick: int = -10**9
        self.patrol_target: Position | None = None
        self.patrol_target_set_tick: int = 0

        # For external inspection (replay rendering, debug)
        self.last_decided_action: Action = Action.WAIT

    def reset(self) -> None:
        """Clear per-encounter memory. Called at the start of each episode."""
        self.state = TomState.PATROL
        self.last_seen_jerry = None
        self.last_seen_tick = -10**9
        self.last_noise = None
        self.last_noise_tick = -10**9
        self.patrol_target = None
        self.patrol_target_set_tick = 0
        self.last_decided_action = Action.WAIT

    def __call__(self, world: World) -> Action:
        """Decide an action given the current world state."""
        # 1. Update memory from current perceptions
        self._update_memory(world)
        # 2. Decide top-level state from updated memory
        self.state = self._select_state(world)
        # 3. Choose a concrete action implementing that state
        action = self._act_for_state(world)
        # 4. Optional: wall-bump avoidance
        if self.config.wall_bump_avoidance:
            action = self._avoid_walls(world, action)
        self.last_decided_action = action
        return action

    # ---- memory --------------------------------------------------------

    def _update_memory(self, world: World) -> None:
        """Read current sensors, update last-seen / last-noise records."""
        tick = world.tick_count

        # Visibility check
        if world._tom_can_see_jerry():
            self.last_seen_jerry = world.jerry.position
            self.last_seen_tick = tick

        # Noise check — pick the loudest event Tom can hear this tick
        # We use the sound field's directional hearing to find a dominant
        # direction, then estimate the source position by stepping that way.
        obs = world._observe_tom()
        dirs = {"N": obs.sound_n, "S": obs.sound_s,
                "E": obs.sound_e, "W": obs.sound_w}
        loudest = max(dirs.items(), key=lambda kv: kv[1])
        if loudest[1] >= self.config.noise_investigate_threshold:
            # Project a rough noise origin: step a few tiles in that direction
            step_map = {
                "N": Position(0, -3), "S": Position(0, 3),
                "E": Position(3, 0), "W": Position(-3, 0),
            }
            self.last_noise = world.tom.position + step_map[loudest[0]]
            self.last_noise_tick = tick

    # ---- state selection -----------------------------------------------

    def _select_state(self, world: World) -> TomState:
        """Pick the highest-priority state whose preconditions are met."""
        tick = world.tick_count

        # ATTACK: adjacent to Jerry, visible (catch range = 1)
        if world._tom_can_see_jerry():
            d = world.tom.position.manhattan(world.jerry.position)
            if d <= world.config.catch_distance + 1:
                return TomState.ATTACK
            return TomState.PURSUE

        # PURSUE: lost sight but recently saw — keep chasing toward last-seen
        if (
            self.last_seen_jerry is not None
            and tick - self.last_seen_tick <= self.config.pursue_memory
        ):
            return TomState.PURSUE

        # INVESTIGATE: heard something recently
        if (
            self.last_noise is not None
            and tick - self.last_noise_tick <= self.config.investigate_dwell
        ):
            return TomState.INVESTIGATE

        # SEARCH: scent gradient is strong
        obs = world._observe_tom()
        max_scent = max(obs.scent_n, obs.scent_s, obs.scent_e, obs.scent_w)
        if max_scent >= self.config.scent_search_threshold:
            return TomState.SEARCH

        return TomState.PATROL

    # ---- action selection per state -----------------------------------

    def _act_for_state(self, world: World) -> Action:
        if self.state == TomState.ATTACK:
            return self._step_toward(world.tom.position, world.jerry.position, world)

        if self.state == TomState.PURSUE:
            target = world.jerry.position if world._tom_can_see_jerry() \
                else self.last_seen_jerry
            if target is None:
                return self._patrol(world)
            return self._step_toward(world.tom.position, target, world)

        if self.state == TomState.INVESTIGATE:
            if self.last_noise is None:
                return self._patrol(world)
            return self._step_toward(world.tom.position, self.last_noise, world)

        if self.state == TomState.SEARCH:
            return self._follow_scent(world)

        return self._patrol(world)

    # ---- behaviors -----------------------------------------------------

    def _step_toward(self, src: Position, dst: Position, world: World) -> Action:
        """Choose a cardinal action that follows the shortest walkable path
        from src to dst.

        Uses BFS to find the path, then returns the first step of it.
        Falls back to a greedy manhattan choice if no path exists (which
        shouldn't happen on connected maps, but handle gracefully).
        """
        if src == dst:
            return Action.WAIT
        first_step = self._bfs_first_step(src, dst, world)
        if first_step is not None:
            return first_step
        # Fallback: greedy by axis
        return self._greedy_step_toward(src, dst, world)

    def _bfs_first_step(
        self, src: Position, dst: Position, world: World,
    ) -> Action | None:
        """BFS from src toward dst. Returns the action for the first step
        of the shortest path, or None if dst is unreachable.

        Treats LOCKER and VENT tiles as walkable (Tom can step onto them
        even though he doesn't use them).
        """
        from collections import deque
        if not world.grid.is_walkable(src):
            return None
        # BFS frontier; parent map tracks (cell -> (prev_cell, action_taken))
        frontier: "deque[Position]" = deque([src])
        parents: dict[Position, tuple[Position, Action]] = {src: (src, Action.WAIT)}
        # Cap exploration to keep it cheap on big maps
        max_explore = world.grid.width * world.grid.height
        explored = 0
        while frontier and explored < max_explore:
            cell = frontier.popleft()
            explored += 1
            if cell == dst:
                break
            # NOTE: neighbor expansion order determines which of several
            # EQUAL-LENGTH shortest paths BFS returns. A FIXED order (the
            # old N,S,E,W) creates an exploitable movement-priority groove:
            # a target reachable by multiple equal paths always resolves to
            # the same axis first (vertical, since N/S were enqueued first),
            # so a column-bobbing Jerry could hold Tom at distance 2 forever
            # (Tom mirrors the bob, never spends a tick closing horizontally).
            # Shuffling the expansion order per-cell removes the stable
            # groove — there's no fixed frequency for a learner to lock onto.
            # (Verification patch: confirms the tie-break is the exploit
            # before we commit to a specific priority philosophy.)
            neighbor_order = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
            self._rng.shuffle(neighbor_order)
            for a in neighbor_order:
                nxt = cell + ACTION_DELTAS[a]
                if nxt in parents:
                    continue
                if not world.grid.is_walkable(nxt):
                    continue
                parents[nxt] = (cell, a)
                frontier.append(nxt)
        # If we never reached dst, look for the explored cell closest to dst
        # and path there instead — this handles "Jerry inside a locker" cases
        # where dst itself isn't walkable from Tom's side.
        if dst not in parents:
            if not parents:
                return None
            dst = min(parents.keys(), key=lambda p: p.manhattan(dst))
            if dst == src:
                return None
        # Walk parents backward from dst to src; the action taken from
        # the cell just after src is what we want.
        path: list[Action] = []
        cur = dst
        while cur != src:
            prev, act = parents[cur]
            path.append(act)
            cur = prev
        if not path:
            return None
        return path[-1]  # the action taken from src

    def _greedy_step_toward(self, src: Position, dst: Position, world: World) -> Action:
        """Greedy manhattan fallback used only if BFS yields no path."""
        dx = dst.x - src.x
        dy = dst.y - src.y
        candidates: list[Action] = []
        if abs(dx) >= abs(dy):
            if dx > 0: candidates.append(Action.EAST)
            if dx < 0: candidates.append(Action.WEST)
            if dy > 0: candidates.append(Action.SOUTH)
            if dy < 0: candidates.append(Action.NORTH)
        else:
            if dy > 0: candidates.append(Action.SOUTH)
            if dy < 0: candidates.append(Action.NORTH)
            if dx > 0: candidates.append(Action.EAST)
            if dx < 0: candidates.append(Action.WEST)
        for a in candidates:
            target = src + ACTION_DELTAS[a]
            if world.grid.is_walkable(target):
                return a
        return Action.WAIT

    def _follow_scent(self, world: World) -> Action:
        """Move toward the strongest neighboring scent."""
        obs = world._observe_tom()
        gradient = {
            Action.NORTH: obs.scent_n, Action.SOUTH: obs.scent_s,
            Action.EAST: obs.scent_e,  Action.WEST: obs.scent_w,
        }
        # Filter to walkable neighbors only
        valid = {
            a: v for a, v in gradient.items()
            if world.grid.is_walkable(world.tom.position + ACTION_DELTAS[a])
        }
        if not valid or max(valid.values()) <= 0:
            return self._patrol(world)
        return max(valid.items(), key=lambda kv: kv[1])[0]

    def _patrol(self, world: World) -> Action:
        """Wander toward a randomly chosen patrol target, retargeting
        when reached or stale.

        Defect-A fix (the WAIT-stall): the map generator can produce
        walkable-but-unreachable tiles (isolated pockets). If patrol picks
        one, BFS can't route to it and _step_toward returns WAIT — parking
        Tom in place (often wedged in a dead-end pocket) until the staleness
        timer finally fires ~40 ticks later. A hunter standing still is never
        correct. Two guards:
          1. Pick a target Tom can actually BFS-reach from here (retry a few
             times), so unreachable pockets are never selected.
          2. If the step toward the current target is still WAIT (target
             went unreachable, or Tom is wedged), retarget IMMEDIATELY rather
             than waiting out the staleness timer.
        """
        need_new = (
            self.patrol_target is None
            or world.tom.position.manhattan(self.patrol_target)
                <= self.config.patrol_retarget_distance
            or world.tick_count - self.patrol_target_set_tick
                > self.config.patrol_retarget_after
        )
        if need_new:
            self._choose_reachable_patrol_target(world)

        action = self._step_toward(world.tom.position, self.patrol_target, world)

        # No-stall guard: a WAIT here means the target is unreachable from
        # Tom's current tile (or src==dst on a tile that didn't retarget).
        # Don't freeze — pick a fresh reachable target and move now.
        if action == Action.WAIT:
            self._choose_reachable_patrol_target(world)
            action = self._step_toward(world.tom.position, self.patrol_target, world)

        return action

    def _choose_reachable_patrol_target(self, world: World) -> None:
        """Pick a patrol target Tom can actually path to from his current
        position. Retries a bounded number of times; falls back to any empty
        position if every attempt is unreachable (degenerate maps), so this
        never loops unboundedly."""
        src = world.tom.position
        for _ in range(self.config.patrol_target_max_tries):
            candidate = world.grid.random_empty_position(self._rng)
            if candidate == src:
                continue
            # Reachable iff BFS finds a first step toward it.
            if self._bfs_first_step(src, candidate, world) is not None:
                self.patrol_target = candidate
                self.patrol_target_set_tick = world.tick_count
                return
        # Fallback: accept whatever we last drew (rare; keeps behavior defined).
        self.patrol_target = world.grid.random_empty_position(self._rng)
        self.patrol_target_set_tick = world.tick_count

    # ---- wall avoidance ------------------------------------------------

    def _avoid_walls(self, world: World, action: Action) -> Action:
        """If the chosen movement action would bump a wall, try a fallback.

        WAIT and INTERACT are returned unchanged.
        """
        if action in (Action.WAIT, Action.INTERACT):
            return action
        target = world.tom.position + ACTION_DELTAS[action]
        if world.grid.is_walkable(target):
            return action
        # Try the other cardinal directions in a random but stable order
        fallback_order = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
        self._rng.shuffle(fallback_order)
        for a in fallback_order:
            if a == action:
                continue
            t = world.tom.position + ACTION_DELTAS[a]
            if world.grid.is_walkable(t):
                return a
        return Action.WAIT
