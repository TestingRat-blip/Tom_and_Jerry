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

    def reset(self) -> None:
        """Clear belief at episode start."""
        self.belief.clear()
        self.last_suggested_target = None
        self.last_suggested_type = None

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
        if world._tom_can_see_jerry():
            self.belief.add_sighting(world.jerry.position, now)

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

        # --- decay / prune ---
        self.belief.tick(now)

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
