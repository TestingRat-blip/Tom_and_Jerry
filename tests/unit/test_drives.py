"""Unit tests for Drives + DriveSystem."""
from __future__ import annotations

import pytest

from src.env.world.world import Event, EventType
from src.hunter.agent.drives.config import DriveAxisConfig, DrivesConfig
from src.hunter.agent.drives.drives import DriveSystem, Drives
from src.utils.types import Position


def _events_of(*types_with_actor):
    """Build a list of Event objects from (event_type, actor) tuples."""
    return [Event(type=t, actor=a) for t, a in types_with_actor]


# ---- basics ------------------------------------------------------------

def test_drives_default_values():
    d = Drives()
    assert d.hunger == pytest.approx(0.4)
    assert d.aggression == pytest.approx(0.5)
    assert d.caution == pytest.approx(0.5)
    assert d.curiosity == pytest.approx(0.4)
    assert d.fatigue == pytest.approx(0.1)
    assert d.social_bond == pytest.approx(0.5)


def test_snapshot_returns_dict_of_floats():
    d = Drives()
    snap = d.snapshot()
    assert isinstance(snap, dict)
    assert set(snap.keys()) == {
        "hunger", "aggression", "caution", "curiosity", "fatigue", "social_bond"
    }
    for v in snap.values():
        assert isinstance(v, float)


def test_reset_restores_baselines():
    cfg = DrivesConfig()
    d = Drives(hunger=0.9, aggression=0.1, caution=0.99, curiosity=0.0,
               fatigue=0.8, social_bond=0.2)
    system = DriveSystem(cfg)
    system.reset(d)
    assert d.hunger == cfg.hunger.baseline
    assert d.aggression == cfg.aggression.baseline
    assert d.fatigue == cfg.fatigue.baseline


# ---- passive growth ----------------------------------------------------

def test_hunger_grows_passively_with_no_events():
    d = Drives()
    system = DriveSystem(DrivesConfig())
    initial = d.hunger
    # Run 50 ticks without events
    for _ in range(50):
        system.tick(d, events=[])
    assert d.hunger > initial


def test_fatigue_grows_only_when_moving():
    d_still = Drives()
    d_move = Drives()
    system = DriveSystem(DrivesConfig())
    for _ in range(50):
        system.tick(d_still, events=[], agent_moved=False)
        system.tick(d_move, events=[], agent_moved=True)
    # Moving Tom should have higher fatigue than still Tom
    assert d_move.fatigue > d_still.fatigue


# ---- decay -------------------------------------------------------------

def test_drives_decay_toward_baseline():
    """A drive pushed above baseline should drift back down without events."""
    cfg = DrivesConfig()
    d = Drives()
    d.aggression = 0.9  # well above baseline of 0.5
    system = DriveSystem(cfg)
    for _ in range(100):
        system.tick(d, events=[])
    assert d.aggression < 0.9
    assert d.aggression >= cfg.aggression.baseline - 0.05
    # The hunger axis grows passively, so it might end above baseline
    # but the aggression axis only drifts


def test_drives_decay_upward_when_below_baseline():
    """A drive pushed below baseline should drift back up."""
    cfg = DrivesConfig()
    d = Drives()
    d.caution = 0.05  # well below baseline of 0.5
    system = DriveSystem(cfg)
    for _ in range(200):
        system.tick(d, events=[])
    # Should have moved meaningfully toward baseline
    assert d.caution > 0.4


# ---- event deltas ------------------------------------------------------

def test_seeing_jerry_raises_aggression():
    d = Drives()
    system = DriveSystem(DrivesConfig())
    initial = d.aggression
    for _ in range(5):
        system.tick(
            d,
            events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
        )
    assert d.aggression > initial


def test_seeing_jerry_lowers_caution():
    d = Drives()
    system = DriveSystem(DrivesConfig())
    initial = d.caution
    for _ in range(5):
        system.tick(
            d,
            events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
        )
    assert d.caution < initial


def test_catching_jerry_drops_hunger():
    d = Drives()
    d.hunger = 0.8
    system = DriveSystem(DrivesConfig())
    system.tick(d, events=_events_of((EventType.TOM_CAUGHT_JERRY, "tom")))
    # Catch delta is -0.6 to hunger
    assert d.hunger < 0.4


def test_wall_bump_raises_fatigue():
    d = Drives()
    system = DriveSystem(DrivesConfig())
    initial = d.fatigue
    for _ in range(10):
        system.tick(d, events=_events_of((EventType.TOM_BUMPED_WALL, "tom")))
    assert d.fatigue > initial


def test_noise_from_jerry_raises_curiosity():
    """NOISE_EMITTED with actor='jerry' should still affect Tom's curiosity."""
    d = Drives()
    system = DriveSystem(DrivesConfig())
    initial = d.curiosity
    for _ in range(10):
        system.tick(d, events=_events_of((EventType.NOISE_EMITTED, "jerry")))
    assert d.curiosity > initial


def test_jerry_internal_events_dont_affect_tom():
    """Events like JERRY_ENTERED_LOCKER should not move Tom's drives."""
    d = Drives()
    system = DriveSystem(DrivesConfig())
    snapshot_before = d.snapshot()
    system.tick(d, events=_events_of(
        (EventType.JERRY_ENTERED_LOCKER, "jerry"),
        (EventType.JERRY_EXITED_LOCKER, "jerry"),
    ))
    snapshot_after = d.snapshot()
    # All drives should be very close to before (only passive growth and decay)
    # We check that the deltas are tiny — within decay/growth tolerances
    for name in snapshot_before:
        delta = abs(snapshot_after[name] - snapshot_before[name])
        # Decay is ~0.02 per tick * (baseline - current); for a Drives() at
        # baseline, the per-tick delta is near 0. Passive hunger growth is
        # 0.001 per tick. So all deltas should be < 0.01.
        assert delta < 0.01, f"{name} moved by {delta} on jerry-internal events"


# ---- clamping ----------------------------------------------------------

def test_drives_clamped_to_ceiling():
    """Forced high values should not exceed ceiling=1.0."""
    d = Drives()
    d.aggression = 0.95
    system = DriveSystem(DrivesConfig())
    # Bombard with TOM_SAW_JERRY (each is +0.05)
    for _ in range(50):
        system.tick(d, events=_events_of((EventType.TOM_SAW_JERRY, "tom")))
    assert d.aggression <= 1.0
    # And it should have hit the ceiling
    assert d.aggression > 0.9


def test_drives_clamped_to_floor():
    """Forced low values should not go below floor=0.0."""
    d = Drives()
    d.caution = 0.05
    system = DriveSystem(DrivesConfig())
    for _ in range(50):
        system.tick(d, events=_events_of((EventType.TOM_SAW_JERRY, "tom")))
    assert d.caution >= 0.0


# ---- integration --------------------------------------------------------

def test_full_episode_simulation():
    """Simulate a plausible episode and check the drive trajectory makes sense:
    early patrol → see Jerry → pursue (aggression up) → catch (hunger down).
    """
    d = Drives()
    system = DriveSystem(DrivesConfig())

    # 50 ticks of patrol (no events)
    for _ in range(50):
        system.tick(d, events=[], agent_moved=True)
    patrol_aggression = d.aggression
    patrol_fatigue = d.fatigue
    patrol_hunger = d.hunger
    # Hunger should have grown during patrol
    assert patrol_hunger > 0.4

    # 20 ticks of sighting Jerry
    for _ in range(20):
        system.tick(
            d,
            events=_events_of((EventType.TOM_SAW_JERRY, "tom")),
            agent_moved=True,
        )
    # Aggression should have risen, caution should have dropped
    assert d.aggression > patrol_aggression

    # Catch event
    pre_catch_hunger = d.hunger
    system.tick(d, events=_events_of((EventType.TOM_CAUGHT_JERRY, "tom")))
    # Hunger should drop dramatically after catch
    assert d.hunger < pre_catch_hunger - 0.4
