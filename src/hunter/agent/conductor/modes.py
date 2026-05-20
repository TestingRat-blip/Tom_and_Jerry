"""Hunt modes + chemistry override (Phase 6e).

The Conductor issues Tom a directive of (target, mode). The MODE governs
*how* Tom executes the approach to the target — patiently, aggressively,
investigatively. This is the layer where Tom's temperament can sabotage
the Conductor's strategy: the Conductor SUGGESTS a mode from what it
believes; Tom's chemistry can OVERRIDE it.

The guiding rule (locked in the Phase 6 design doc):

    The Conductor suggests what's tactically available;
    the body decides whether it has the patience for it.

So a Conductor that holds a high-confidence sighting may suggest STALK
(close patiently, don't blow it) — but a highly adrenalized Tom upgrades
that to RUSH and over-commits, potentially blowing the ambush. That
over-commitment is a FEATURE: it's Tom's flaw and Jerry's manipulation
target (spike Tom's adrenaline → provoke a premature commit).

This module is pure: mode selection is a function of (belief type,
chemistry levels). It has no world/Tom dependency, so it's unit-testable
in isolation. The BEHAVIORAL realization of each mode lives in
ChemicalTom (how STALK vs RUSH actually move).

Scope note for 6e: STALK, RUSH, INVESTIGATE, PATROL are fully realized.
BAIT (the approach-then-retreat deception) is defined in the enum so the
vocabulary is complete, but its multi-tick maneuver is deferred to a
later batch; for now BAIT falls back to INVESTIGATE behavior. Cortisol's
"unlock BAIT" path is therefore also deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from src.hunter.agent.conductor.belief import SuspicionType


class HuntMode(IntEnum):
    """How Tom executes the approach to the Conductor's target."""
    PATROL = 0       # no live belief — sweep sectors (6d)
    INVESTIGATE = 1  # move toward a suspicion at a neutral pace
    STALK = 2        # hold at a distance, apply pressure, don't close all the way
    RUSH = 3         # close directly and commit ASAP (the over-committer)
    BAIT = 4         # approach then retreat to draw prey out (deferred; falls back to INVESTIGATE)


@dataclass(frozen=True)
class ModeConfig:
    """Thresholds for the chemistry override. Conservative for Stage 1:
    chemistry overrides the Conductor only at EXTREMES of arousal, so the
    Conductor's guidance usually stands but a strongly-provoked Tom errs.
    Co-evolution (Stage 2) tunes where these lines sit — they're the
    Tom-side override weights ADR-003 calls "conductor weights".
    """
    # Adrenaline at or above this upgrades the suggested mode to RUSH.
    # High default = Tom must be genuinely amped to override good guidance.
    rush_adrenaline_threshold: float = 0.65
    # Confidence at or above this (for a SIGHTING) lets the Conductor
    # suggest STALK rather than plain INVESTIGATE — we only stalk prey we're
    # fairly sure we've located.
    stalk_confidence_threshold: float = 0.6
    # Cortisol at/above this would unlock BAIT (deferred — see module docstring).
    bait_cortisol_threshold: float = 0.7


def conductor_suggested_mode(
    suspicion_type: SuspicionType | None,
    confidence: float,
    config: ModeConfig,
) -> HuntMode:
    """The mode the Conductor PROPOSES, from belief alone (no chemistry).

    - None type        -> PATROL (empty belief)
    - SIGHTING + high   -> STALK (we know roughly where it is; apply pressure)
    - SIGHTING + low    -> INVESTIGATE (go confirm)
    - NOISE / SCENT     -> INVESTIGATE (go check / follow)
    """
    if suspicion_type is None:
        return HuntMode.PATROL
    if suspicion_type == SuspicionType.SIGHTING:
        if confidence >= config.stalk_confidence_threshold:
            return HuntMode.STALK
        return HuntMode.INVESTIGATE
    # NOISE or SCENT
    return HuntMode.INVESTIGATE


def apply_chemistry_override(
    suggested: HuntMode,
    adrenaline: float,
    cortisol: float,
    config: ModeConfig,
) -> tuple[HuntMode, bool]:
    """Apply Tom's temperament to the Conductor's suggestion.

    Returns (final_mode, overridden) where `overridden` is True if chemistry
    changed the Conductor's suggestion (useful for diagnostics / replay
    overlay / the "Tom blew it" signal).

    The rule:
      - PATROL is never overridden (nothing to commit to).
      - High adrenaline upgrades any active-pursuit mode to RUSH. This is
        the over-commit: even a STALK suggestion (hold back!) becomes a
        headlong RUSH when Tom is amped enough.
      - (Deferred) High cortisol would unlock BAIT from a STALK suggestion.
    """
    if suggested == HuntMode.PATROL:
        return suggested, False

    # Over-commit: adrenaline upgrades to RUSH.
    if adrenaline >= config.rush_adrenaline_threshold and suggested != HuntMode.RUSH:
        return HuntMode.RUSH, True

    # (Deferred) cortisol -> BAIT unlock would go here.

    return suggested, False


def decide_mode(
    suspicion_type: SuspicionType | None,
    confidence: float,
    adrenaline: float,
    cortisol: float,
    config: ModeConfig,
) -> tuple[HuntMode, HuntMode, bool]:
    """Full mode decision: (final_mode, suggested_mode, overridden).

    Convenience wrapper: compute the Conductor's suggestion, then apply the
    chemistry override. Returns both the final and the suggested mode so
    callers can see when temperament diverged from strategy.
    """
    suggested = conductor_suggested_mode(suspicion_type, confidence, config)
    final, overridden = apply_chemistry_override(
        suggested, adrenaline, cortisol, config)
    return final, suggested, overridden
