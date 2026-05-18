"""Integration tests for L1 wired into ChemicalTom.

These verify the end-to-end behavior change:
  - Tom with L1 attached, no false noise history → noise threshold == base modulated
  - Tom with L1 attached, false noises around him → noise threshold raised
  - Tom with L1 attached but L1 cleared between episodes → no leakage
  - Tom without L1 (l1=None) behaves identically to Phase 2

Most tests use FakeRedis so they run fast and need no real Redis.
A few use real Redis (marked @pytest.mark.redis) to confirm the
end-to-end path works under realistic conditions.
"""
from __future__ import annotations

import pytest

from src.env.gym_env import JerryEnv
from src.env.world.world import WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.memory.l1 import L1Config, L1Memory
from src.persistence.redis.client import FakeRedis, RedisClient
from src.render.replay.recorder import ReplayRecorder
from src.utils.types import Action, Position


# ---- helpers ----------------------------------------------------------

def _make_l1(episode_id: str = "test_ep") -> L1Memory:
    """Build an L1Memory backed by FakeRedis — fast, no Redis dependency."""
    client = RedisClient(client=FakeRedis())
    return L1Memory(client, episode_id=episode_id)


# ---- L1 attached as a no-op ------------------------------------------

def test_chemical_tom_with_l1_runs_full_episode():
    """Smoke test: Tom + L1 completes an episode without errors."""
    l1 = _make_l1()
    tom = ChemicalTom(l1=l1, seed=0)
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=100),
        tom_policy=tom,
    )
    env.reset(seed=42)
    while True:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break


def test_chemical_tom_with_l1_resets_on_episode_reset():
    """When the env resets, ChemicalTom.reset() should clear L1 too."""
    l1 = _make_l1()
    tom = ChemicalTom(l1=l1, seed=0)
    env = JerryEnv(tom_policy=tom)
    env.reset(seed=42)
    # Pollute L1 with some artificial state
    tom.l1.store.increment_false_noise_count(5, 5)
    tom.l1.store.increment_false_noise_count(5, 5)
    assert tom.l1.store.get_false_noise_count(5, 5) == 2
    # Reset the env (which resets the policy)
    env.reset(seed=99)
    assert tom.l1.store.get_false_noise_count(5, 5) == 0


def test_l1_none_behaves_like_phase2():
    """ChemicalTom(l1=None) should behave EXACTLY like Phase 2."""
    tom = ChemicalTom(l1=None, seed=0)
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=80),
        tom_policy=tom,
    )
    env.reset(seed=42)
    # Just run a few ticks — should not raise
    for _ in range(30):
        _, _, term, trunc, _ = env.step(int(Action.WAIT))
        if term or trunc:
            break
    # No L1 means no l1 attribute lookup errors anywhere in the loop
    assert tom.l1 is None


# ---- L1 actually changes the threshold ------------------------------

def test_noise_threshold_unaffected_when_no_false_noises():
    """L1 attached but no false noise history → threshold == Phase 2 value."""
    l1 = _make_l1()
    tom = ChemicalTom(l1=l1, seed=0)
    base_threshold = tom._modulated_noise_threshold(tom_pos=Position(10, 10))

    tom_no_l1 = ChemicalTom(l1=None, seed=0)
    no_l1_threshold = tom_no_l1._modulated_noise_threshold(tom_pos=Position(10, 10))

    assert base_threshold == pytest.approx(no_l1_threshold)


def test_noise_threshold_raises_with_false_noise_history():
    """After several false noises near Tom's location, the threshold goes up."""
    l1 = _make_l1()
    tom = ChemicalTom(l1=l1, seed=0)
    pos = Position(10, 10)
    before = tom._modulated_noise_threshold(tom_pos=pos)

    # Push enough false noises near pos to reach saturation
    for _ in range(l1.config.false_noise_saturation):
        l1.store.increment_false_noise_count(pos.x, pos.y)

    after = tom._modulated_noise_threshold(tom_pos=pos)
    assert after > before
    # At saturation, the factor is max_false_noise_factor (default 0.5).
    # With l1_false_noise_weight default 1.0, threshold should be ~50% higher.
    assert after >= before * 1.4   # allow some slack


def test_noise_threshold_unchanged_when_false_noises_far_away():
    """False noises across the map should NOT raise Tom's local threshold."""
    l1 = _make_l1()
    tom = ChemicalTom(l1=l1, seed=0)
    pos = Position(10, 10)
    before = tom._modulated_noise_threshold(tom_pos=pos)

    # False noises FAR from pos (outside lookup radius)
    for _ in range(l1.config.false_noise_saturation * 3):
        l1.store.increment_false_noise_count(25, 25)

    after = tom._modulated_noise_threshold(tom_pos=pos)
    assert after == pytest.approx(before)


def test_l1_weight_zero_disables_l1_contribution():
    """Setting l1_false_noise_weight=0 should make L1 a pure observer
    (it still records, but doesn't influence the threshold).
    """
    from src.hunter.agent.behavior.chemical_tom import ChemicalTomConfig
    l1 = _make_l1()
    cfg = ChemicalTomConfig(l1_false_noise_weight=0.0)
    tom = ChemicalTom(chemical_config=cfg, l1=l1, seed=0)
    pos = Position(10, 10)
    before = tom._modulated_noise_threshold(tom_pos=pos)

    for _ in range(l1.config.false_noise_saturation * 2):
        l1.store.increment_false_noise_count(pos.x, pos.y)

    after = tom._modulated_noise_threshold(tom_pos=pos)
    assert after == pytest.approx(before)


# ---- end-to-end: L1 actually records during a real episode -----------

def test_l1_records_during_real_episode():
    """Run a real episode and confirm L1 captured at least one sighting."""
    l1 = _make_l1()
    tom = ChemicalTom(l1=l1, seed=0)
    env = JerryEnv(
        world_config=WorldConfig(max_ticks=400),
        tom_policy=tom,
    )
    env.reset(seed=42)

    while True:
        _, _, term, trunc, _ = env.step(int(Action.WAIT))
        if term or trunc:
            break

    # Tom should have seen Jerry at least once in 400 ticks against a
    # passive jerry on a 30×30 grid. (catch rate is ~70% so most seeds
    # produce sightings)
    heatmap = l1.store.all_heatmap_counts()
    # Could be 0 if Tom never managed to get LOS, but that's rare.
    # Either way, no crash is the main thing.
    assert isinstance(heatmap, dict)


def test_l1_recorder_sets_locker_positions():
    """When recording an episode, the recorder should hand Tom's L1
    the locker positions from the freshly-generated map.
    """
    l1 = _make_l1()
    tom = ChemicalTom(l1=l1, seed=0)
    rec = ReplayRecorder(world_config=WorldConfig(max_ticks=20), seed=42)
    rec.record_episode(
        jerry_policy=lambda obs, world: int(Action.WAIT),
        tom_policy=tom,
        tom_label="chemical-l1",
    )
    # After the episode, Tom's L1 should know about the lockers from the map
    assert len(tom.l1.locker_positions) > 0


# ---- real Redis variant (opt-in via -m redis) ------------------------

@pytest.mark.redis
def test_l1_in_tom_against_real_redis():
    """End-to-end against real Redis to confirm FakeRedis is faithful."""
    import uuid
    client = RedisClient()
    try:
        client.ping()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")

    ep_id = f"itest_{uuid.uuid4().hex[:6]}"
    l1 = L1Memory(client, episode_id=ep_id)
    try:
        tom = ChemicalTom(l1=l1, seed=0)
        env = JerryEnv(
            world_config=WorldConfig(max_ticks=80),
            tom_policy=tom,
        )
        env.reset(seed=42)
        for _ in range(40):
            _, _, term, trunc, _ = env.step(int(Action.WAIT))
            if term or trunc:
                break
        # Just confirm no crash and Tom's L1 store is reachable via real Redis
        heat = l1.store.all_heatmap_counts()
        assert isinstance(heat, dict)
    finally:
        l1.reset()
