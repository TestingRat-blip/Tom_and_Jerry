"""The Conductor — the director half of Tom's two-brain architecture.

Per ADR-013, the Conductor replaces BFS-as-targeting. It maintains a
belief about where Jerry is (from observable signals only) and directs
Tom toward it. Tom's local brain (chemistry, drives, behavior tree)
still runs and colors execution.

Phase 6a: belief layer.
Phase 6b: Conductor scaffolding (observe-only).
"""
from src.hunter.agent.conductor.belief import (
    BeliefConfig,
    SuspicionBelief,
    SuspicionSource,
    SuspicionType,
)
from src.hunter.agent.conductor.conductor import Conductor, ConductorConfig

__all__ = [
    "BeliefConfig",
    "SuspicionBelief",
    "SuspicionSource",
    "SuspicionType",
    "Conductor",
    "ConductorConfig",
]
