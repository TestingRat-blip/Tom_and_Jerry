"""Replay recording — capture episodes as data, play them back later.

A Replay is a list of Frames. Each Frame is a snapshot of everything
needed to render and reason about one tick of the simulation. Frames
are compact (no PyTorch tensors, no env objects) and JSON-serializable
after a single conversion — so replays can be saved to disk for later
analysis, shared between machines, and inspected programmatically.

Usage:
    rec = ReplayRecorder(world_config=WorldConfig(), seed=42)
    replay = rec.record_episode(jerry_policy=trained_jerry,
                                tom_policy=ScriptedTom(seed=0))
    # replay is now a Replay object you can render or save

Phase 2 will extend Frame with chemistry/drive state. Designed so that
addition is backwards-compatible: optional fields with defaults.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.env.world.world import EventType, World, WorldConfig
from src.utils.types import Action, Position, TileType


# Type alias: any callable that takes (obs_vector, world) → Action.
# This is the most general policy signature; both ScriptedTom and
# PPOJerryPolicy fit it (one ignores obs, the other ignores world).
PolicyFn = Callable[[np.ndarray, World], int]


@dataclass(frozen=True, slots=True)
class Frame:
    """Snapshot of one tick.

    All fields are JSON-serializable (ints, floats, lists, tuples) so
    a Replay can be saved with json.dump() after a single conversion
    step. No numpy arrays in Frame itself — the grid is captured ONCE
    in Replay.grid_tiles and frame deltas reference it by position.
    """
    tick: int
    tom_pos: tuple[int, int]
    tom_facing: int
    tom_state: str          # "PATROL" | "SEARCH" | "INVESTIGATE" | "PURSUE" | "ATTACK" | ""
    tom_action: int         # Action int chosen this tick
    jerry_pos: tuple[int, int]
    jerry_facing: int
    jerry_action: int
    jerry_in_locker: bool
    jerry_alive: bool
    tom_sees_jerry: bool
    jerry_sees_tom: bool
    # Reward this tick (Jerry's), for HUD display
    jerry_reward: float = 0.0
    # Cumulative reward up through this tick
    jerry_cum_reward: float = 0.0
    # Sound events emitted this tick: (x, y, intensity)
    sound_events: tuple[tuple[int, int, float], ...] = ()
    # Scent field snapshot: only cells with scent > threshold,
    # as (x, y, value) tuples. Sparse representation keeps frames small.
    scent_cells: tuple[tuple[int, int, float], ...] = ()
    # Event types fired this tick (for highlighting in renderer)
    events: tuple[int, ...] = ()

    # ---- Phase 2 extensions (optional, defaulted) ----------------------
    # Tom's chemistry levels this tick — empty dict for non-chemical Toms
    tom_chemistry: dict = field(default_factory=dict)
    # Tom's drive state this tick — empty dict for non-chemical Toms
    tom_drives: dict = field(default_factory=dict)
    # Predicted Jerry position if Tom is predicting ahead (else == jerry_pos)
    tom_predicted_jerry: tuple[int, int] | None = None
    # How many steps ahead Tom is currently predicting (0 = no prediction)
    tom_prediction_steps: int = 0


@dataclass
class Replay:
    """A complete recorded episode."""
    # Static info — captured once
    grid_width: int
    grid_height: int
    grid_tiles: list[list[int]]                       # row-major, TileType ints
    vent_pairs: list[tuple[tuple[int, int], tuple[int, int]]]  # for renderer
    locker_positions: list[tuple[int, int]]
    seed: int | None
    # Run info
    jerry_policy_label: str
    tom_policy_label: str
    # Frame data
    frames: list[Frame] = field(default_factory=list)
    # Outcome
    outcome: str = "in_progress"   # "caught" | "survived" | "in_progress"
    total_ticks: int = 0
    total_jerry_reward: float = 0.0

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, path: str | Path) -> None:
        """Save the replay as JSON. Compact-ish (no pretty-printing)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "grid_tiles": self.grid_tiles,
            "vent_pairs": [
                [list(a), list(b)] for a, b in self.vent_pairs
            ],
            "locker_positions": [list(p) for p in self.locker_positions],
            "seed": self.seed,
            "jerry_policy_label": self.jerry_policy_label,
            "tom_policy_label": self.tom_policy_label,
            "outcome": self.outcome,
            "total_ticks": self.total_ticks,
            "total_jerry_reward": self.total_jerry_reward,
            "frames": [asdict(f) for f in self.frames],
        }
        with p.open("w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "Replay":
        """Load a replay JSON."""
        with Path(path).open("r") as f:
            data = json.load(f)
        frames = []
        for fd in data["frames"]:
            # Convert nested lists back to tuples where Frame expects them
            fd = dict(fd)
            fd["tom_pos"] = tuple(fd["tom_pos"])
            fd["jerry_pos"] = tuple(fd["jerry_pos"])
            fd["sound_events"] = tuple(tuple(s) for s in fd.get("sound_events", []))
            fd["scent_cells"] = tuple(tuple(s) for s in fd.get("scent_cells", []))
            fd["events"] = tuple(fd.get("events", []))
            # Phase 2 fields — handle old replays that don't have them
            fd.setdefault("tom_chemistry", {})
            fd.setdefault("tom_drives", {})
            pred = fd.get("tom_predicted_jerry")
            fd["tom_predicted_jerry"] = tuple(pred) if pred is not None else None
            fd.setdefault("tom_prediction_steps", 0)
            frames.append(Frame(**fd))
        return cls(
            grid_width=data["grid_width"],
            grid_height=data["grid_height"],
            grid_tiles=data["grid_tiles"],
            vent_pairs=[(tuple(a), tuple(b)) for a, b in data["vent_pairs"]],
            locker_positions=[tuple(p) for p in data["locker_positions"]],
            seed=data["seed"],
            jerry_policy_label=data["jerry_policy_label"],
            tom_policy_label=data["tom_policy_label"],
            frames=frames,
            outcome=data["outcome"],
            total_ticks=data["total_ticks"],
            total_jerry_reward=data["total_jerry_reward"],
        )


class ReplayRecorder:
    """Wraps a World and captures Frames during an episode.

    Usage:
        rec = ReplayRecorder(world_config=WorldConfig(max_ticks=600), seed=42)
        replay = rec.record_episode(
            jerry_policy=my_jerry,
            tom_policy=ScriptedTom(seed=0),
            jerry_label="ppo_v4_final",
            tom_label="scripted_baseline",
        )
        replay.save("data/replays/run1.json")
    """

    SCENT_THRESHOLD: float = 0.05  # cells below this are not recorded

    def __init__(self, world_config: WorldConfig | None = None, seed: int | None = None):
        self.world_config = world_config or WorldConfig()
        self.seed = seed

    def record_episode(
        self,
        jerry_policy: PolicyFn | Any,
        tom_policy: PolicyFn | Any,
        jerry_label: str = "unknown",
        tom_label: str = "unknown",
        episode_id: str | None = None,
    ) -> Replay:
        """Run one full episode and return the captured Replay.

        Both policies should be callable. They can:
          - Be ScriptedTom-style: policy(world) → Action
          - Be PPO-style with .from_obs(): we'll call from_obs(obs)
          - Be raw callables: callable(obs, world) → Action int

        We try each protocol in order.

        If `tom_policy` has an attached L1 memory (ChemicalTom with
        `self.l1`), this method automatically sets the L1's locker
        positions from the freshly-generated map. The L1's episode_id
        is whatever was already on the L1 (typically set externally).
        Callers that want a per-episode-id-isolated L1 should construct
        a new L1Memory(client, episode_id=...) before each call and
        attach it to the Tom.
        """
        world = World(self.world_config, seed=self.seed)
        _, jerry_obs = world.reset()

        # Reset policies that support it
        for pol in (jerry_policy, tom_policy):
            if hasattr(pol, "reset") and callable(pol.reset):
                pol.reset()

        # If Tom has L1 memory attached, give it the map's locker positions.
        # This must happen AFTER policy reset (which clears L1 state) but
        # BEFORE any tick runs.
        if hasattr(tom_policy, "l1") and tom_policy.l1 is not None:
            tom_policy.l1.set_locker_positions(list(world.grid.locker_positions))

        # Capture static map info
        vent_pairs = []
        seen_vents: set = set()
        for a, b in world.grid.vent_links.items():
            key = frozenset([a, b])
            if key in seen_vents:
                continue
            seen_vents.add(key)
            vent_pairs.append(((a.x, a.y), (b.x, b.y)))

        replay = Replay(
            grid_width=world.grid.width,
            grid_height=world.grid.height,
            grid_tiles=world.grid.tiles.tolist(),
            vent_pairs=vent_pairs,
            locker_positions=[(p.x, p.y) for p in world.grid.locker_positions],
            seed=self.seed,
            jerry_policy_label=jerry_label,
            tom_policy_label=tom_label,
        )

        # Reward config — for HUD reward computation, mirrors JerryEnv default
        from src.env.gym_env import JerryRewardConfig
        reward_cfg = JerryRewardConfig()

        cum_reward = 0.0

        while not world.done:
            # Step the world
            tom_action = self._call_policy(tom_policy, world, agent="tom")
            jerry_action = self._call_policy(jerry_policy, world, agent="jerry",
                                             obs=jerry_obs)
            tom_action_int = int(tom_action)
            jerry_action_int = int(jerry_action)
            _, jerry_obs, events, _ = world.step(
                Action(tom_action_int), Action(jerry_action_int)
            )

            # Compute Jerry's per-tick reward to mirror JerryEnv exactly
            r = reward_cfg.survival_per_tick
            for ev in events:
                if ev.type == EventType.TOM_SAW_JERRY:
                    r += reward_cfg.penalty_seen
                elif ev.type == EventType.NOISE_EMITTED and ev.actor == "jerry":
                    r += reward_cfg.penalty_noise
                elif ev.type == EventType.JERRY_BUMPED_WALL:
                    r += reward_cfg.penalty_bump_wall
                elif ev.type == EventType.TOM_CAUGHT_JERRY:
                    r += reward_cfg.penalty_caught
                elif ev.type == EventType.TIMEOUT:
                    if world.jerry.alive:
                        r += reward_cfg.bonus_survived
            cum_reward += r

            # Sound events this tick — read from world.sound._events
            # (pre-clear, since the env clears next tick)
            sound_events = tuple(
                (e.position.x, e.position.y, float(e.intensity))
                for e in world.sound._events
            )

            # Scent: sparse snapshot of cells above threshold
            scent_cells: list[tuple[int, int, float]] = []
            scent_arr = world.scent.field
            ys, xs = np.where(scent_arr > self.SCENT_THRESHOLD)
            for x, y in zip(xs.tolist(), ys.tolist()):
                scent_cells.append((int(x), int(y), float(scent_arr[y, x])))

            # Tom state — only ScriptedTom exposes .state; default to ""
            tom_state = ""
            if hasattr(tom_policy, "state"):
                tom_state = getattr(tom_policy.state, "name", "")

            # Phase 2: capture chemistry, drives, prediction if Tom is ChemicalTom
            tom_chemistry: dict = {}
            tom_drives: dict = {}
            tom_predicted_jerry: tuple[int, int] | None = None
            tom_prediction_steps: int = 0
            if hasattr(tom_policy, "chemistry"):
                try:
                    tom_chemistry = tom_policy.chemistry.snapshot()
                except Exception:
                    tom_chemistry = {}
            if hasattr(tom_policy, "drives"):
                try:
                    tom_drives = tom_policy.drives.snapshot()
                except Exception:
                    tom_drives = {}
            pred = getattr(tom_policy, "last_predicted_jerry_pos", None)
            if pred is not None:
                tom_predicted_jerry = (pred.x, pred.y)
            tom_prediction_steps = int(getattr(tom_policy, "last_prediction_steps", 0))

            frame = Frame(
                tick=world.tick_count,
                tom_pos=(world.tom.position.x, world.tom.position.y),
                tom_facing=int(world.tom.facing),
                tom_state=tom_state,
                tom_action=tom_action_int,
                jerry_pos=(world.jerry.position.x, world.jerry.position.y),
                jerry_facing=int(world.jerry.facing),
                jerry_action=jerry_action_int,
                jerry_in_locker=world.jerry.in_locker,
                jerry_alive=world.jerry.alive,
                tom_sees_jerry=world._tom_can_see_jerry(),
                jerry_sees_tom=world._jerry_can_see_tom(),
                jerry_reward=float(r),
                jerry_cum_reward=float(cum_reward),
                sound_events=sound_events,
                scent_cells=tuple(scent_cells),
                events=tuple(int(e.type) for e in events),
                tom_chemistry=tom_chemistry,
                tom_drives=tom_drives,
                tom_predicted_jerry=tom_predicted_jerry,
                tom_prediction_steps=tom_prediction_steps,
            )
            replay.frames.append(frame)

            # Detect outcome
            if any(e.type == EventType.TOM_CAUGHT_JERRY for e in events):
                replay.outcome = "caught"
            elif any(e.type == EventType.TIMEOUT for e in events):
                replay.outcome = "survived" if world.jerry.alive else "caught"

        replay.total_ticks = world.tick_count
        replay.total_jerry_reward = cum_reward
        return replay

    # ---- private -------------------------------------------------------

    def _call_policy(
        self,
        policy: Any,
        world: World,
        agent: str,
        obs: Any | None = None,
    ) -> int:
        """Resolve any policy shape to an action int.

        Order of preference:
          1. policy.from_obs(obs_vector) if obs is provided and policy supports it
          2. policy(world) for ScriptedTom-style world callables
          3. policy(obs, world) for raw two-arg callables
        """
        # PPO-style: has .from_obs and obs is provided
        if obs is not None and hasattr(policy, "from_obs"):
            return int(policy.from_obs(obs.to_vector() if hasattr(obs, "to_vector") else obs))
        # ScriptedTom-style: callable with world
        try:
            result = policy(world)
            return int(result)
        except TypeError:
            pass
        # Raw two-arg callable
        return int(policy(obs, world))
