"""The Conductor — director brain (Phase 6b, observe-only stage).

This batch builds the Conductor as an OBSERVER that runs alongside Tom
but does NOT yet drive him. It watches the same world Tom sees, ingests
observable signals into its SuspicionBelief, and can report what target
it WOULD suggest. Tom's decision path is unchanged this batch, so all
existing behavior (and tests) are preserved.

The handover — Tom actually targeting the Conductor's suggestion instead
of his own per-encounter memory — happens in a later batch (6c+), once
we've verified the Conductor builds sensible beliefs from real gameplay.
This mirrors the project's house style: L1 memory was observed before it
modulated behavior; the Conductor builds belief before it targets.

Per ADR-013, the Conductor ingests ONLY observable signals:
  - SIGHTING : created when Tom can actually see Jerry (a real sensor hit)
  - NOISE    : created from NOISE_EMITTED events that are NOT Tom's own
  - SCENT    : projected from the scent gradient Tom can smell at his tile

It never reads world.jerry.position except through the visibility gate
(seeing Jerry is perception, not cheating — same gate ScriptedTom uses).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.env.world.world import Event, EventType, World
from src.hunter.agent.conductor.belief import (
    BeliefConfig,
    SuspicionBelief,
    SuspicionType,
)
from src.hunter.agent.conductor.sectors import SectorConfig, SectorMap
from src.utils.types import Position


@dataclass(frozen=True)
class ConductorConfig:
    """Tunable knobs for the Conductor's perception layer.

    The belief's own knobs live in BeliefConfig; these govern how the
    Conductor turns world signals into belief updates.
    """
    # Scent: only project a SCENT suspicion if the strongest directional
    # gradient at Tom's tile exceeds this. Mirrors ScriptedTom's
    # scent_search_threshold so behavior is comparable.
    scent_threshold: float = 0.15
    # How many tiles to project a scent suspicion in the gradient direction.
    scent_projection_tiles: int = 3
    # How many tiles to project a noise suspicion (when a heard-noise event
    # carries a position we use it directly; this is only a fallback for
    # directional-only noise — see observe()).
    noise_projection_tiles: int = 3
    # Scale factor applied to scent gradient strength -> birth confidence
    # input. Keeps scent suspicions appropriately weak vs sightings.
    scent_strength_scale: float = 1.0
    # Sector decomposition for patrol (Phase 6d).
    sectors: SectorConfig = SectorConfig()
    # Max manhattan search radius when snapping a sector centroid to the
    # nearest walkable tile for a patrol target.
    patrol_snap_radius: int = 6

    # --- Component 3: hold-on-LOS-break / run-down (memory-adaptation) ---
    # When enabled, if Tom LOSES line of sight to a Jerry he was actively
    # seeing, the Conductor ANCHORS a high-confidence sighting suspicion at
    # the last-seen tile for a window of ticks, re-stamping it each tick so
    # it does not decay. Tom keeps pursuing toward where Jerry vanished
    # ("runs him down to his square") instead of releasing pressure and
    # re-patrolling — the counter to the cover-dance exploit.
    #
    # Disabled by default: this behavior is DEPLOYED by memory
    # (a StrategicStance from L2), not always-on. For the cheap validation
    # experiment we force it on via hold_on_los_break=True.
    hold_on_los_break: bool = False
    # How many ticks to keep the anchor alive after LOS breaks.
    hold_window_ticks: int = 15
    # Confidence the anchor is re-stamped to each tick (high = strong pull).
    hold_anchor_confidence: float = 1.0
    # Occupy-dwell: once Tom REACHES the anchor tile, hold the anchor (keep
    # him on the spot) for this many ticks before clearing — denying the
    # cover and forcing Jerry to either stay trapped or break into the open.
    # "Runs him down to his square AND camps it." 0 = leave immediately
    # (run-through). >0 = occupy and flush.
    hold_occupy_ticks: int = 6


class Conductor:
    """Director brain. Observe-only in Phase 6b.

    Lifecycle per tick (driven by the caller, e.g. the env/recorder or,
    later, ChemicalTom):
        conductor.observe(world)          # ingest signals, tick belief
        target = conductor.suggested_target(world)   # query (may be None)

    reset() clears the belief at episode start.
    """

    def __init__(
        self,
        config: ConductorConfig | None = None,
        belief_config: BeliefConfig | None = None,
    ):
        self.config = config or ConductorConfig()
        self.belief = SuspicionBelief(belief_config)
        # For external inspection / debugging / replay overlay later.
        self.last_suggested_target: Position | None = None
        self.last_suggested_type: SuspicionType | None = None
        # Sector map for patrol (Phase 6d). Lazily built on first observe,
        # because the Conductor doesn't know grid dimensions at construction.
        self._sectors: SectorMap | None = None

        # Component 3: hold-on-LOS-break tracking.
        # _was_seeing_jerry: did Tom have LOS last tick?
        # _last_seen_pos: where Jerry was when last seen.
        # _anchor_pos / _anchor_until_tick: active anchor (None = inactive).
        self._was_seeing_jerry: bool = False
        self._last_seen_pos: Position | None = None
        self._anchor_pos: Position | None = None
        self._anchor_until_tick: int = -1
        # Occupy-dwell: once Tom reaches the anchor, hold it until this tick.
        self._anchor_occupy_until: int = -1
        # Exposed for inspection / replay overlay.
        self.anchor_active: bool = False
        # Runtime override of config.hold_on_los_break, set by memory
        # (warm-start) per-episode. None = use config value; True/False =
        # override. Lets memory deploy/stand-down the run-down without
        # mutating the frozen config.
        self.runtime_hold_on_los_break: bool | None = None

    def _ensure_sectors(self, world: World) -> SectorMap:
        """Build the sector map on first use (we need grid dimensions)."""
        if self._sectors is None:
            self._sectors = SectorMap(
                grid_width=world.grid.width,
                grid_height=world.grid.height,
                config=self.config.sectors,
            )
        return self._sectors

    def reset(self) -> None:
        """Clear belief and sector visit history at episode start."""
        self.belief.clear()
        self.last_suggested_target = None
        self.last_suggested_type = None
        if self._sectors is not None:
            self._sectors.reset()
        # Component 3 state
        self._was_seeing_jerry = False
        self._last_seen_pos = None
        self._anchor_pos = None
        self._anchor_until_tick = -1
        self._anchor_occupy_until = -1
        self.anchor_active = False
        # NOTE: runtime_hold_on_los_break is NOT reset here — it's set by
        # warm-start AFTER reset(), and reset shouldn't clobber a value the
        # caller may set immediately after. Callers that want a clean slate
        # set it explicitly (or it persists from the prior warm-start, which
        # is fine since warm-start re-sets it each episode).

    # ---- perception ---------------------------------------------------

    def observe(self, world: World) -> None:
        """Ingest this tick's observable signals into the belief, then
        decay the belief. Call once per tick, before querying.

        Signal sourcing:
          - SIGHTING from Tom actually seeing Jerry (visibility gate)
          - NOISE from NOISE_EMITTED events not produced by Tom himself
          - SCENT projected from the gradient Tom smells at his own tile
        """
        now = world.tick_count

        # --- SIGHTING: only when Tom genuinely sees Jerry ---
        # This is the one place we read Jerry's position, and it's gated
        # exactly like ScriptedTom's perception — seeing is a sensor, not
        # ground-truth omniscience.
        seeing = world._tom_can_see_jerry()
        if seeing:
            self.belief.add_sighting(world.jerry.position, now)
            self._last_seen_pos = world.jerry.position

        # --- Component 3: hold-on-LOS-break / run-down ---
        # When enabled, and Tom JUST lost LOS to a Jerry he was seeing,
        # plant an anchor at the last-seen tile and keep re-stamping a
        # high-confidence sighting there for hold_window_ticks. This makes
        # Tom keep pursuing where Jerry vanished instead of forgetting —
        # the counter to the cover-dance. Memory deploys this (via the
        # hold_on_los_break flag); for the validation experiment it's
        # forced on.
        if self._hold_on_los_break_active():
            self._update_los_break_anchor(world, now, seeing)
        self._was_seeing_jerry = seeing

        # --- SIGHTING invalidation: checked and empty ---
        # If Tom is NOT seeing Jerry but is standing on a SIGHTING suspicion's
        # tile, he's physically verified that spot is empty — drop the stale
        # sighting so it stops re-feeding last_seen_jerry and re-triggering
        # PURSUE (the standoff exploit, where a prey held just outside sight
        # range tethers Tom to a dead tile for the whole slow-decay window).
        # Guard: never invalidate while the run-down anchor is active — that's
        # the deliberate "keep believing the vanish point" cover-dance counter,
        # and it owns sighting belief at the anchor tile on purpose.
        anchor_active = bool(getattr(self, "anchor_active", False))
        if not seeing and not anchor_active:
            self.belief.invalidate_sighting_at(world.tom.position, now, radius=1)

        # --- NOISE: from events Tom did not himself cause ---
        events = getattr(world, "_events_this_tick", [])
        for ev in events:
            if ev.type != EventType.NOISE_EMITTED:
                continue
            if ev.actor == "tom":
                continue  # Tom's own footsteps are not a Jerry signal
            if ev.position is None:
                continue
            intensity = float(ev.payload) if isinstance(ev.payload, (int, float)) else 1.0
            self.belief.add_noise(ev.position, now, intensity=intensity)

        # --- SCENT: projected from the gradient at Tom's tile ---
        self._observe_scent(world, now)

        # --- sector visit tracking (Phase 6d patrol) ---
        # Record that Tom is physically in this sector now, so the LRV
        # patrol policy knows which zones are freshly covered.
        sectors = self._ensure_sectors(world)
        sectors.mark_visited(world.tom.position, now)

        # --- decay / prune ---
        self.belief.tick(now)

    def _hold_on_los_break_active(self) -> bool:
        """Whether the run-down behavior is active: runtime override (set by
        memory) wins over the config default."""
        if self.runtime_hold_on_los_break is not None:
            return self.runtime_hold_on_los_break
        return self.config.hold_on_los_break

    def _update_los_break_anchor(self, world: World, now: int, seeing: bool) -> None:
        """Component 3: maintain a 'run-down' anchor when Tom loses LOS.

        Logic:
          - If Tom currently SEES Jerry, no anchor needed (real sighting
            dominates). Clear any active anchor.
          - If Tom just LOST LOS this tick (saw last tick, not now), start
            an anchor at the last-seen tile, alive for hold_window_ticks.
          - While an anchor is active and unexpired, re-stamp a high-
            confidence sighting at the anchor tile each tick so Tom keeps
            pursuing it (it would otherwise decay and release pressure).
          - Expire the anchor when its window passes OR when Tom reaches
            the anchor tile (he's run it down — nothing there, resume normal
            hunting / patrol).
        """
        if seeing:
            # Real sighting active — drop any anchor.
            self._anchor_pos = None
            self._anchor_until_tick = -1
            self.anchor_active = False
            return

        # Detect the moment LOS breaks: saw last tick, not this tick.
        just_lost = self._was_seeing_jerry and not seeing
        if just_lost and self._last_seen_pos is not None:
            self._anchor_pos = self._last_seen_pos
            self._anchor_until_tick = now + self.config.hold_window_ticks
            self._anchor_occupy_until = -1  # not yet reached

        # Maintain an active anchor.
        if self._anchor_pos is not None:
            reached = world.tom.position.manhattan(self._anchor_pos) <= 1

            # If Tom has reached the anchor, start (or continue) the occupy
            # dwell — camp the spot to flush Jerry out instead of leaving.
            if reached and self._anchor_occupy_until < 0:
                self._anchor_occupy_until = now + self.config.hold_occupy_ticks

            occupying = self._anchor_occupy_until >= 0
            window_ok = now <= self._anchor_until_tick
            occupy_ok = now <= self._anchor_occupy_until

            # Anchor stays alive while EITHER the run-down window is open
            # (still travelling to it) OR Tom is occupying it within dwell.
            if (window_ok and not occupying) or (occupying and occupy_ok):
                # Re-stamp a high-confidence sighting at the anchor so Tom
                # keeps pursuing / camping it. add_sighting merges with the
                # existing same-type same-tile suspicion, refreshing it.
                self.belief.add_sighting(self._anchor_pos, now)
                self.anchor_active = True
            else:
                # Run-down window expired without reaching, or occupy dwell
                # finished — give up the anchor.
                self._anchor_pos = None
                self._anchor_until_tick = -1
                self._anchor_occupy_until = -1
                self.anchor_active = False
        else:
            self.anchor_active = False

    def _observe_scent(self, world: World, now: int) -> None:
        """Read the scent gradient at Tom's position and, if strong enough,
        project a SCENT suspicion a few tiles in the strongest direction.

        Uses only Tom's own tile gradient — no access to Jerry's position.
        """
        scent = getattr(world, "scent", None)
        if scent is None:
            return
        grad = scent.gradient_at(world.tom.position)
        # grad is {"N":.., "S":.., "E":.., "W":..}
        direction, strength = max(grad.items(), key=lambda kv: kv[1])
        if strength < self.config.scent_threshold:
            return
        step = {
            "N": Position(0, -self.config.scent_projection_tiles),
            "S": Position(0, self.config.scent_projection_tiles),
            "E": Position(self.config.scent_projection_tiles, 0),
            "W": Position(-self.config.scent_projection_tiles, 0),
        }[direction]
        projected = world.tom.position + step
        self.belief.add_scent(
            projected, now,
            strength=strength * self.config.scent_strength_scale,
        )

    # ---- query --------------------------------------------------------

    def suggested_target(self, world: World) -> Position | None:
        """The position the Conductor would send Tom toward, or None if the
        belief is empty (Tom should patrol). This is observe-only in 6b:
        nothing is forced to USE this yet.
        """
        now = world.tick_count
        strongest = self.belief.strongest(now)
        if strongest is None:
            self.last_suggested_target = None
            self.last_suggested_type = None
            return None
        src, _conf = strongest
        self.last_suggested_target = src.position
        self.last_suggested_type = src.type
        return src.position

    def suggested_source(self, world: World):
        """Like suggested_target but returns the full (source, confidence)
        pair so callers can see the TYPE (for mode selection later in 6d).
        Returns None if belief is empty.
        """
        return self.belief.strongest(world.tick_count)

    # ---- patrol (Phase 6d) --------------------------------------------

    def patrol_target(self, world: World) -> Position:
        """Where the Conductor directs Tom when the belief is empty.

        Picks the least-recently-visited sector (excluding the one Tom is
        currently in) and returns a walkable tile near its centroid. This
        produces legible coverage sweeps rather than the random wandering
        of the base ScriptedTom patrol.

        Always returns a valid walkable Position (falls back to Tom's own
        tile if no walkable tile can be found near the target sector, which
        shouldn't happen on connected maps).
        """
        sectors = self._ensure_sectors(world)
        stalest = sectors.stalest_sector(exclude_current=world.tom.position)
        centroid = sectors.sector_centroid(stalest)
        target = self._nearest_walkable(world, centroid)
        return target if target is not None else world.tom.position

    def _nearest_walkable(self, world: World, center: Position) -> Position | None:
        """Find the nearest walkable tile to `center` within snap radius.

        Expanding-ring manhattan search. Returns None if nothing walkable
        is found within patrol_snap_radius (degenerate maps only).
        """
        if world.grid.is_walkable(center):
            return center
        r_max = self.config.patrol_snap_radius
        for r in range(1, r_max + 1):
            # Scan the manhattan ring at radius r
            for dx in range(-r, r + 1):
                dy_abs = r - abs(dx)
                for dy in ({dy_abs, -dy_abs} if dy_abs else {0}):
                    cand = Position(center.x + dx, center.y + dy)
                    if world.grid.is_walkable(cand):
                        return cand
        return None
