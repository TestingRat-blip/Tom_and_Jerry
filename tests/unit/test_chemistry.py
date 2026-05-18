"""Unit tests for Chemistry + ChemistrySystem."""
from __future__ import annotations

import pytest

from src.env.world.world import Event, EventType
from src.hunter.agent.chemistry.chemistry import (
    CHEMICAL_NAMES,
    Chemistry,
    ChemistrySystem,
)
from src.hunter.agent.chemistry.config import (
    ChemicalAxisConfig,
    ChemistryConfig,
)


def _events_of(*types_with_actor):
    return [Event(type=t, actor=a) for t, a in types_with_actor]


# ---- basics ------------------------------------------------------------

def test_chemistry_starts_at_zero():
    c = Chemistry()
    for name in CHEMICAL_NAMES:
        assert getattr(c, name) == 0.0
        assert getattr(c, f"_buf_{name}") == 0.0


def test_snapshot_returns_levels_only():
    c = Chemistry()
    c.adrenaline = 0.7
    c._buf_adrenaline = 0.4
    snap = c.snapshot()
    assert snap["adrenaline"] == pytest.approx(0.7)
    # Buffer should NOT be in snapshot
    assert "_buf_adrenaline" not in snap


def test_reset_wipes_everything():
    c = Chemistry()
    c.adrenaline = 0.5
    c.cortisol = 0.9
    c._buf_dopamine = 0.4
    system = ChemistrySystem()
    system.reset(c)
    for name in CHEMICAL_NAMES:
        assert getattr(c, name) == 0.0
        assert getattr(c, f"_buf_{name}") == 0.0


# ---- decay -------------------------------------------------------------

def test_levels_decay_toward_zero_with_no_events():
    c = Chemistry()
    c.adrenaline = 1.0
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(20):
        system.tick(c, events=[], jerry_visible=True)  # jerry_visible to prevent cortisol creep
    # Adrenaline decays at 0.85/tick → after 20 ticks: 1.0 * 0.85^20 ≈ 0.04
    assert c.adrenaline < 0.1


def test_adrenaline_decays_faster_than_cortisol():
    """Per the config — adrenaline decays at 0.85, cortisol at 0.97."""
    c = Chemistry()
    c.adrenaline = 1.0
    c.cortisol = 1.0
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(10):
        system.tick(c, events=[], jerry_visible=True)
    assert c.adrenaline < c.cortisol


def test_serotonin_decays_very_slowly():
    """Serotonin should still be substantial after many ticks."""
    c = Chemistry()
    c.serotonin = 0.5
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(20):
        system.tick(c, events=[], jerry_visible=True)
    # serotonin at 0.995^20 = 0.905 of original = 0.4525, plus baseline pressure
    # may have nudged it slightly. Should be well above 0.4.
    assert c.serotonin > 0.4


# ---- event spikes ------------------------------------------------------

def test_seeing_jerry_spikes_adrenaline():
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    initial = c.adrenaline
    system.tick(c, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                jerry_visible=True)
    assert c.adrenaline > initial


def test_catching_jerry_floods_dopamine():
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    initial = c.dopamine
    system.tick(c, events=_events_of((EventType.TOM_CAUGHT_JERRY, "tom")),
                jerry_visible=True)
    # After one tick, dopamine should have a clear spike
    assert c.dopamine > initial + 0.2  # buffer transfer rate 0.5 * delta 0.8 = 0.4


def test_catching_jerry_drops_cortisol():
    """Catch should suppress cortisol (relief response)."""
    c = Chemistry()
    c.cortisol = 0.6
    system = ChemistrySystem(ChemistryConfig())
    system.tick(c, events=_events_of((EventType.TOM_CAUGHT_JERRY, "tom")),
                jerry_visible=True)
    assert c.cortisol < 0.6


def test_wall_bump_raises_cortisol():
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(10):
        system.tick(c, events=_events_of((EventType.TOM_BUMPED_WALL, "tom")),
                    jerry_visible=False)
    assert c.cortisol > 0.05


# ---- buffer dynamics ---------------------------------------------------

def test_single_event_does_not_saturate_chemical():
    """A single TOM_SAW_JERRY event should NOT push adrenaline to 1.0 in one tick.
    The buffer mechanism should spread the effect over several ticks.
    """
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    system.tick(c, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                jerry_visible=True)
    # Buffer delta 0.35, transfer rate 0.5 → first tick gets ~0.175
    assert c.adrenaline < 0.5


def test_repeated_events_build_chemical_smoothly():
    """Repeated TOM_SAW_JERRY events should ramp adrenaline up smoothly,
    not slam it to 1.0 immediately.
    """
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    levels = []
    for _ in range(10):
        system.tick(c, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                    jerry_visible=True)
        levels.append(c.adrenaline)
    # Monotonically increasing (or roughly so) — each tick higher than previous
    # within reasonable tolerance (last few may oscillate at saturation)
    assert levels[5] > levels[0]
    assert levels[-1] > levels[3]
    # Should approach but not exceed ceiling
    assert all(v <= 1.0 for v in levels)


def test_buffer_decays_when_unused():
    """A spike that doesn't fully transfer to level should decay from the buffer."""
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    # Inject a single event
    system.tick(c, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                jerry_visible=True)
    buf_after_event = c._buf_adrenaline
    assert buf_after_event > 0
    # Many quiet ticks should drain the buffer
    for _ in range(20):
        system.tick(c, events=[], jerry_visible=True)
    assert c._buf_adrenaline < buf_after_event * 0.1


# ---- cross-interactions ------------------------------------------------

def test_adrenaline_suppresses_cortisol():
    """Same starting cortisol, different adrenaline levels → adrenalized
    Tom should have lower cortisol after several ticks.
    """
    cfg = ChemistryConfig()
    c1 = Chemistry()
    c2 = Chemistry()
    c1.cortisol = 0.5
    c2.cortisol = 0.5
    c2.adrenaline = 0.8
    system = ChemistrySystem(cfg)
    for _ in range(5):
        system.tick(c1, events=[], jerry_visible=True)
        system.tick(c2, events=[], jerry_visible=True)
    # c2 had high adrenaline → cortisol should have dropped faster
    assert c2.cortisol < c1.cortisol


def test_serotonin_caps_adrenaline():
    """High serotonin should slow the rise of adrenaline."""
    cfg = ChemistryConfig()
    c1 = Chemistry()
    c2 = Chemistry()
    c2.serotonin = 0.9  # confident Tom
    system = ChemistrySystem(cfg)
    for _ in range(5):
        system.tick(c1, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                    jerry_visible=True)
        system.tick(c2, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                    jerry_visible=True)
    # c2 should have lower adrenaline despite seeing the same events
    assert c2.adrenaline < c1.adrenaline


def test_dopamine_raises_serotonin():
    """Dopamine flood should slowly build serotonin."""
    c = Chemistry()
    c.dopamine = 0.9
    initial_serotonin = c.serotonin
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(5):
        system.tick(c, events=[], jerry_visible=True)
    assert c.serotonin > initial_serotonin


# ---- passive behaviors -------------------------------------------------

def test_cortisol_grows_when_jerry_not_visible():
    """A long patrol without sighting should accumulate cortisol (frustration)."""
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(100):
        system.tick(c, events=[], jerry_visible=False)
    assert c.cortisol > 0.02


def test_cortisol_stable_when_jerry_visible():
    """When Tom can see Jerry, cortisol shouldn't grow from frustration."""
    c1 = Chemistry()
    c2 = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(100):
        system.tick(c1, events=[], jerry_visible=False)
        system.tick(c2, events=[], jerry_visible=True)
    assert c2.cortisol < c1.cortisol


# ---- clamping ---------------------------------------------------------

def test_chemicals_clamped_to_ceiling():
    """Many TOM_SAW_JERRY events shouldn't push adrenaline above 1.0."""
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())
    for _ in range(100):
        system.tick(c, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                    jerry_visible=True)
    assert c.adrenaline <= 1.0


def test_chemicals_clamped_to_floor():
    """Cortisol shouldn't go negative even after a big suppression event."""
    c = Chemistry()
    c.cortisol = 0.1
    system = ChemistrySystem(ChemistryConfig())
    # Catch event has cortisol delta of -0.5
    system.tick(c, events=_events_of((EventType.TOM_CAUGHT_JERRY, "tom")),
                jerry_visible=True)
    assert c.cortisol >= 0.0


# ---- integration: a full encounter -----------------------------------

def test_chemistry_episode_arc():
    """Simulate a plausible arc:
    1. 30 ticks patrol → cortisol creeps up
    2. Spot Jerry → adrenaline spikes
    3. Lose Jerry → adrenaline decays, cortisol continues to climb
    4. Catch Jerry → dopamine flood, cortisol crashes
    """
    c = Chemistry()
    system = ChemistrySystem(ChemistryConfig())

    # 1. Patrol
    for _ in range(30):
        system.tick(c, events=[], jerry_visible=False)
    patrol_cortisol = c.cortisol
    assert patrol_cortisol > 0.01

    # 2. Sighting
    for _ in range(5):
        system.tick(c, events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
                    jerry_visible=True)
    sighting_adrenaline = c.adrenaline
    assert sighting_adrenaline > 0.2

    # 3. Lost sight, more patrol
    for _ in range(30):
        system.tick(c, events=[], jerry_visible=False)
    assert c.adrenaline < sighting_adrenaline  # decayed
    # Cortisol may continue to rise OR fall slightly depending on adrenaline
    # suppression. We mainly check adrenaline went away.

    # 4. Catch
    pre_catch_cortisol = c.cortisol
    pre_catch_dopamine = c.dopamine
    system.tick(c, events=_events_of((EventType.TOM_CAUGHT_JERRY, "tom")),
                jerry_visible=True)
    assert c.dopamine > pre_catch_dopamine + 0.2
    assert c.cortisol < pre_catch_cortisol
