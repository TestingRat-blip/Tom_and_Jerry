"""The Conductor — the director half of Tom's two-brain architecture.

Per ADR-013, the Conductor replaces BFS-as-targeting. It maintains a
belief about where Jerry is (from observable signals only) and directs
Tom toward it. Tom's local brain (chemistry, drives, behavior tree)
still runs and colors execution.

Phase 6a: belief layer.
Phase 6b: Conductor scaffolding (observe-only).
Phase 6c: targeting handover (belief drives Tom's memory fields).
Phase 6d: sectors + directed patrol.
Phase 6e: hunt modes + chemistry override.
"""
from src.hunter.agent.conductor.belief import (
    BeliefConfig,
    SuspicionBelief,
    SuspicionSource,
    SuspicionType,
)
from src.hunter.agent.conductor.conductor import Conductor, ConductorConfig
from src.hunter.agent.conductor.modes import (
    HuntMode,
    ModeConfig,
    apply_chemistry_override,
    conductor_suggested_mode,
    decide_mode,
)
from src.hunter.agent.conductor.sectors import SectorConfig, SectorMap

__all__ = [
    "BeliefConfig",
    "SuspicionBelief",
    "SuspicionSource",
    "SuspicionType",
    "Conductor",
    "ConductorConfig",
    "SectorConfig",
    "SectorMap",
    "HuntMode",
    "ModeConfig",
    "decide_mode",
    "conductor_suggested_mode",
    "apply_chemistry_override",
]
