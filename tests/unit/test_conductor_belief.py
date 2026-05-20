"""Unit tests for the Conductor's suspicion belief (Phase 6a).

Pure data-structure tests — no world, no Tom, no Redis. Covers:
  - source creation per type
  - exponential decay and the death floor
  - the per-type half-life targets from BeliefConfig
  - merge behavior (same-type within radius refreshes, doesn't duplicate)
  - the source cap
  - strongest()/live_sources()/is_empty() queries
  - reset/clear
"""
from __future__ import annotations

import pytest

from src.hunter.agent.conductor.belief import (
    BeliefConfig,
    SuspicionBelief,
    SuspicionSource,
    SuspicionType,
)
from src.utils.types import Position


# ---- creation ---------------------------------------------------------

def test_empty_belief_has_no_strongest():
    b = SuspicionBelief()
    assert b.strongest(now_tick=0) is None
    assert b.is_empty(now_tick=0)
    assert len(b) == 0


def test_add_sighting_creates_source():
    b = SuspicionBelief()
    b.add_sighting(Position(5, 5), now_tick=0)
    s = b.strongest(now_tick=0)
    assert s is not None
    src, conf = s
    assert src.type == SuspicionType.SIGHTING
    assert src.position == Position(5, 5)
    assert conf == pytest.approx(1.0)


def test_add_noise_creates_source():
    b = SuspicionBelief()
    b.add_noise(Position(3, 7), now_tick=0)
    src, conf = b.strongest(now_tick=0)
    assert src.type == SuspicionType.NOISE
    assert conf == pytest.approx(0.6)  # birth_noise default


def test_add_scent_creates_source():
    b = SuspicionBelief()
    b.add_scent(Position(1, 1), now_tick=0)
    src, conf = b.strongest(now_tick=0)
    assert src.type == SuspicionType.SCENT
    assert conf == pytest.approx(0.5)  # birth_scent default


# ---- intensity scaling ------------------------------------------------

def test_noise_intensity_scales_birth_confidence():
    b = SuspicionBelief()
    b.add_noise(Position(0, 0), now_tick=0, intensity=0.5)
    _, conf = b.strongest(now_tick=0)
    assert conf == pytest.approx(0.3)  # 0.6 * 0.5


def test_noise_intensity_clamped_to_sighting_birth():
    """A very loud noise can't out-trust a fresh sighting at birth."""
    b = SuspicionBelief()
    b.add_noise(Position(0, 0), now_tick=0, intensity=100.0)
    _, conf = b.strongest(now_tick=0)
    assert conf <= 1.0  # clamped to birth_sighting


def test_negative_intensity_floored_to_zero():
    """Negative intensity floors to 0 birth confidence. A zero-confidence
    source is below the death floor, so it's effectively dead on arrival —
    strongest() returns None. (A noise with no effective intensity should
    not create a real suspicion.)
    """
    b = SuspicionBelief()
    b.add_noise(Position(0, 0), now_tick=0, intensity=-5.0)
    assert b.strongest(now_tick=0) is None


# ---- decay ------------------------------------------------------------

def test_confidence_decays_over_time():
    b = SuspicionBelief()
    b.add_sighting(Position(0, 0), now_tick=0)
    c0 = b.strongest(now_tick=0)[1]
    c5 = b.strongest(now_tick=5)[1]
    c10 = b.strongest(now_tick=10)[1]
    assert c0 > c5 > c10


def test_source_dies_below_floor():
    """A noise source should die after enough ticks (fast decay)."""
    cfg = BeliefConfig()
    b = SuspicionBelief(cfg)
    b.add_noise(Position(0, 0), now_tick=0)
    # birth 0.6, decay 0.87/tick, floor 0.05
    # 0.6 * 0.87^n < 0.05  =>  0.87^n < 0.0833  =>  n > ln(0.0833)/ln(0.87) ~ 17.8
    # so by tick 20 it should be dead after a tick() prune
    b.tick(now_tick=20)
    assert b.is_empty(now_tick=20)
    assert len(b) == 0


def test_sighting_outlives_noise():
    """Same birth tick: a sighting should remain live long after a noise
    has died, because sightings decay slower.
    """
    b = SuspicionBelief()
    b.add_sighting(Position(0, 0), now_tick=0)
    b.add_noise(Position(20, 20), now_tick=0)
    b.tick(now_tick=20)
    live = b.live_sources(now_tick=20)
    types_live = {s.type for s, _ in live}
    assert SuspicionType.SIGHTING in types_live
    assert SuspicionType.NOISE not in types_live


# ---- half-life targets ------------------------------------------------

@pytest.mark.parametrize("adder,decay_attr,half_life", [
    ("add_noise", "decay_noise", 5),
    ("add_scent", "decay_scent", 12),
    ("add_sighting", "decay_sighting", 25),
])
def test_half_life_targets(adder, decay_attr, half_life):
    """Each type's decay rate should roughly halve confidence over its
    intended half-life (the comments in BeliefConfig).
    """
    cfg = BeliefConfig()
    b = SuspicionBelief(cfg)
    getattr(b, adder)(Position(0, 0), now_tick=0)
    c0 = b.strongest(now_tick=0)[1]
    c_half = b.strongest(now_tick=half_life)
    assert c_half is not None, f"{adder} died before its half-life"
    ratio = c_half[1] / c0
    # within 10% of 0.5
    assert ratio == pytest.approx(0.5, abs=0.1), \
        f"{adder} half-life ~{half_life}: ratio {ratio:.3f} not ~0.5"


# ---- merge ------------------------------------------------------------

def test_same_type_within_radius_merges():
    """Two noises close together should merge into one source, not two."""
    b = SuspicionBelief()
    b.add_noise(Position(5, 5), now_tick=0)
    b.add_noise(Position(6, 5), now_tick=1)  # within merge_radius=3
    assert len(b) == 1


def test_same_type_outside_radius_does_not_merge():
    b = SuspicionBelief()
    b.add_noise(Position(5, 5), now_tick=0)
    b.add_noise(Position(20, 20), now_tick=1)  # far away
    assert len(b) == 2


def test_different_types_do_not_merge():
    """A noise and a sighting at the same spot are distinct sources."""
    b = SuspicionBelief()
    b.add_noise(Position(5, 5), now_tick=0)
    b.add_sighting(Position(5, 5), now_tick=0)
    assert len(b) == 2


def test_merge_refreshes_position_and_age():
    """Merging moves the source toward the new signal and resets its age,
    so the merged source decays from the new reinforcement time.
    """
    b = SuspicionBelief()
    b.add_noise(Position(5, 5), now_tick=0)
    # let it decay a bit
    c_before = b.strongest(now_tick=3)[1]
    # reinforce with a nearby noise at tick 3
    b.add_noise(Position(6, 6), now_tick=3)
    src, c_after = b.strongest(now_tick=3)
    # position moved to the new signal
    assert src.position == Position(6, 6)
    # confidence refreshed back up (reinforcement resets age)
    assert c_after > c_before
    assert c_after == pytest.approx(0.6)  # full birth_noise again


def test_merge_takes_stronger_birth_confidence():
    """If a weaker source is reinforced by a stronger signal, it takes the
    stronger birth confidence.
    """
    b = SuspicionBelief()
    b.add_noise(Position(5, 5), now_tick=0, intensity=0.3)  # weak
    b.add_noise(Position(5, 5), now_tick=0, intensity=1.0)  # strong, merges
    _, conf = b.strongest(now_tick=0)
    assert conf == pytest.approx(0.6)  # took the stronger birth


# ---- cap --------------------------------------------------------------

def test_source_cap_enforced():
    """Adding more than max_sources distinct sources drops the weakest."""
    cfg = BeliefConfig(max_sources=3)
    b = SuspicionBelief(cfg)
    # Add 5 noise sources far apart (no merging), at increasing strengths
    # via intensity, so we know which should survive.
    positions = [Position(i * 10, 0) for i in range(5)]
    intensities = [0.2, 0.4, 0.6, 0.8, 1.0]
    for p, inten in zip(positions, intensities):
        b.add_noise(p, now_tick=0, intensity=inten)
    assert len(b) == 3
    # The three strongest (intensities 0.6, 0.8, 1.0) should survive
    live = b.live_sources(now_tick=0)
    surviving_positions = {s.position for s, _ in live}
    assert Position(40, 0) in surviving_positions  # intensity 1.0
    assert Position(30, 0) in surviving_positions  # intensity 0.8
    assert Position(20, 0) in surviving_positions  # intensity 0.6
    assert Position(0, 0) not in surviving_positions  # intensity 0.2, dropped


# ---- queries ----------------------------------------------------------

def test_strongest_returns_highest_confidence():
    b = SuspicionBelief()
    b.add_noise(Position(0, 0), now_tick=0)       # 0.6
    b.add_sighting(Position(10, 10), now_tick=0)  # 1.0
    src, conf = b.strongest(now_tick=0)
    assert src.type == SuspicionType.SIGHTING
    assert conf == pytest.approx(1.0)


def test_live_sources_sorted_strongest_first():
    b = SuspicionBelief()
    b.add_noise(Position(0, 0), now_tick=0)       # 0.6
    b.add_sighting(Position(10, 10), now_tick=0)  # 1.0
    b.add_scent(Position(5, 5), now_tick=0)       # 0.5
    live = b.live_sources(now_tick=0)
    confs = [c for _, c in live]
    assert confs == sorted(confs, reverse=True)
    assert len(live) == 3


def test_strongest_changes_as_sighting_decays_below_noise():
    """A fresh sighting starts strongest, but if a fresh noise keeps
    getting reinforced while the sighting decays, eventually noise wins.
    """
    b = SuspicionBelief()
    b.add_sighting(Position(0, 0), now_tick=0)  # 1.0, slow decay
    # Initially sighting wins
    assert b.strongest(now_tick=0)[0].type == SuspicionType.SIGHTING
    # Reinforce a noise repeatedly far away; check it can become strongest
    # right after reinforcement at a late tick when sighting has decayed.
    for t in range(0, 40, 2):
        b.add_noise(Position(20, 20), now_tick=t)
    # At tick 38, noise was just reinforced (0.6), sighting decayed
    # 1.0 * 0.973^38 ~ 0.354 — so noise (0.6) should now win.
    src, _ = b.strongest(now_tick=38)
    assert src.type == SuspicionType.NOISE


# ---- reset ------------------------------------------------------------

def test_clear_empties_belief():
    b = SuspicionBelief()
    b.add_sighting(Position(0, 0), now_tick=0)
    b.add_noise(Position(5, 5), now_tick=0)
    assert len(b) == 2
    b.clear()
    assert len(b) == 0
    assert b.is_empty(now_tick=0)


# ---- determinism / purity --------------------------------------------

def test_current_confidence_is_pure_function_of_tick():
    """Querying confidence at the same tick twice gives the same answer,
    and querying doesn't mutate state.
    """
    b = SuspicionBelief()
    b.add_sighting(Position(0, 0), now_tick=0)
    a = b.strongest(now_tick=7)[1]
    c = b.strongest(now_tick=7)[1]
    assert a == c
    # querying at tick 7 then tick 0 still gives the original (no mutation)
    assert b.strongest(now_tick=0)[1] == pytest.approx(1.0)
