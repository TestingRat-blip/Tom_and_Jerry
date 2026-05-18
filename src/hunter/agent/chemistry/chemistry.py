"""Chemistry state and per-tick update system.

A `Chemistry` instance holds five chemical *levels* and five matching
*accumulation buffers*. Events deposit into the buffer; the buffer
gradually transfers into the level; both decay.

This two-stage design (buffer → level) is borrowed from Vera's chemistry
v2 pattern. Without it, a single large event would push a chemical to
1.0 instantly. With it, even a sustained stream of events builds
chemicals smoothly and naturally.

Update order each tick:
  1. Passive trickle (cortisol grows when no Jerry; serotonin pressures toward 0.3)
  2. Decay levels and buffers (multiplicative)
  3. Apply event deltas (into buffers, not levels)
  4. Transfer buffer → level
  5. Apply cross-chemical interactions
  6. Clamp to [floor, ceiling]
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from src.env.world.world import Event, EventType
from src.hunter.agent.chemistry.config import (
    ChemicalAxisConfig,
    ChemistryConfig,
)


CHEMICAL_NAMES: tuple[str, ...] = (
    "adrenaline", "cortisol", "dopamine", "oxytocin", "serotonin",
)


@dataclass
class Chemistry:
    """Tom's chemical state. Five levels + five accumulation buffers."""
    adrenaline: float = 0.0
    cortisol: float = 0.0
    dopamine: float = 0.0
    oxytocin: float = 0.0
    serotonin: float = 0.0

    # Accumulation buffers — events deposit here, then transfer to levels
    _buf_adrenaline: float = 0.0
    _buf_cortisol: float = 0.0
    _buf_dopamine: float = 0.0
    _buf_oxytocin: float = 0.0
    _buf_serotonin: float = 0.0

    def snapshot(self) -> dict[str, float]:
        """Levels only (no buffers) — for replay frames."""
        return {name: float(getattr(self, name)) for name in CHEMICAL_NAMES}

    def reset(self) -> None:
        """Wipe everything for a new episode."""
        for name in CHEMICAL_NAMES:
            setattr(self, name, 0.0)
            setattr(self, f"_buf_{name}", 0.0)


def _axis(config: ChemistryConfig, name: str) -> ChemicalAxisConfig:
    return getattr(config, name)


class ChemistrySystem:
    """Per-tick chemistry update.

    Usage:
        chem = Chemistry()
        system = ChemistrySystem(ChemistryConfig())
        system.reset(chem)  # at start of episode
        # each tick:
        system.tick(chem, events=events, jerry_visible=True/False)
    """

    def __init__(self, config: ChemistryConfig | None = None):
        self.config = config or ChemistryConfig()
        self._event_lookup: dict[int, list[tuple[str, float]]] | None = None

    def reset(self, chem: Chemistry) -> None:
        chem.reset()

    def tick(
        self,
        chem: Chemistry,
        events: list[Event] | tuple[Event, ...] = (),
        jerry_visible: bool = False,
    ) -> None:
        """Advance chemistry one tick."""
        cfg = self.config

        # 1. Passive trickle
        if not jerry_visible:
            chem._buf_cortisol += cfg.cortisol_per_tick_when_no_jerry
        # Serotonin pressures toward 0.3 (mild positive baseline)
        if chem.serotonin < 0.3:
            chem._buf_serotonin += cfg.serotonin_baseline_pressure

        # 2. Decay levels and buffers
        for name in CHEMICAL_NAMES:
            axis = _axis(cfg, name)
            level = getattr(chem, name)
            buf = getattr(chem, f"_buf_{name}")
            setattr(chem, name, level * axis.decay_per_tick)
            setattr(chem, f"_buf_{name}", buf * axis.buffer_decay_per_tick)

        # 3. Apply event deltas (to buffers)
        if self._event_lookup is None:
            lookup: dict[int, list[tuple[str, float]]] = {}
            for ev_type, chem_name, delta in cfg.event_deltas:
                lookup.setdefault(int(ev_type), []).append((chem_name, delta))
            self._event_lookup = lookup

        for ev in events:
            # Filter — only Tom-relevant events affect Tom's chemistry
            # (NOISE_EMITTED is Tom-relevant regardless of actor)
            if ev.actor and ev.actor != "tom":
                if int(ev.type) != int(EventType.NOISE_EMITTED):
                    continue
            for chem_name, delta in self._event_lookup.get(int(ev.type), []):
                attr = f"_buf_{chem_name}"
                setattr(chem, attr, getattr(chem, attr) + delta)

        # 4. Transfer buffer → level
        for name in CHEMICAL_NAMES:
            axis = _axis(cfg, name)
            buf = getattr(chem, f"_buf_{name}")
            transfer = buf * axis.buffer_transfer_rate
            setattr(chem, name, getattr(chem, name) + transfer)
            setattr(chem, f"_buf_{name}", buf - transfer)

        # 5. Apply cross-chemical interactions
        # Note: we snapshot the BEFORE values so interactions don't cascade
        # within the same tick (i.e., A affecting B affecting A's same-tick value).
        before = {name: getattr(chem, name) for name in CHEMICAL_NAMES}
        for src, dst, coeff in cfg.interactions:
            delta = before[src] * coeff
            setattr(chem, dst, getattr(chem, dst) + delta)

        # 6. Clamp
        for name in CHEMICAL_NAMES:
            axis = _axis(cfg, name)
            level = getattr(chem, name)
            if level < axis.floor:
                level = axis.floor
            elif level > axis.ceiling:
                level = axis.ceiling
            setattr(chem, name, level)
            # Buffers also stay non-negative (negative deltas can push them below 0)
            buf = getattr(chem, f"_buf_{name}")
            if buf < 0:
                # Allow negative buffers up to a floor of -ceiling so that
                # suppression events (e.g., catch dropping cortisol) actually
                # work. Without this, the buffer floors at 0 and suppression
                # is silently discarded.
                buf = max(buf, -axis.ceiling)
                setattr(chem, f"_buf_{name}", buf)
