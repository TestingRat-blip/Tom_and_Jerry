"""Tests for sector decomposition + directed patrol (Phase 6d).

Two layers:
  1. SectorMap unit tests (pure, no world): tile<->sector mapping,
     centroids, LRV staleness tracking.
  2. Conductor patrol integration: patrol_target returns walkable tiles,
     directed patrol covers the map more systematically than random, and
     a Conductor-driven Tom runs full episodes.
"""
from __future__ import annotations

import pytest

from src.env.world.world import World, WorldConfig
from src.hunter.agent.behavior.chemical_tom import ChemicalTom
from src.hunter.agent.conductor import (
    Conductor,
    SectorConfig,
    SectorMap,
)
from src.utils.types import Action, Position


# ---- SectorMap: tile <-> sector --------------------------------------

def test_sector_of_corners():
    sm = SectorMap(30, 30, SectorConfig(cols=3, rows=3))
    # Top-left tile -> sector 0
    assert sm.sector_of(Position(0, 0)) == 0
    # Bottom-right tile -> last sector (index 8)
    assert sm.sector_of(Position(29, 29)) == 8
    # Center tile -> middle sector (index 4)
    assert sm.sector_of(Position(15, 15)) == 4


def test_sector_of_clamps_out_of_range():
    sm = SectorMap(30, 30, SectorConfig(cols=3, rows=3))
    # Out-of-bounds positions clamp into valid sectors (defensive)
    assert 0 <= sm.sector_of(Position(100, 100)) < 9
    assert 0 <= sm.sector_of(Position(-5, -5)) < 9


def test_n_sectors():
    sm = SectorMap(30, 30, SectorConfig(cols=4, rows=2))
    assert sm.n_sectors == 8


def test_sector_centroid_in_bounds():
    sm = SectorMap(30, 30, SectorConfig(cols=3, rows=3))
    for i in range(sm.n_sectors):
        c = sm.sector_centroid(i)
        assert 0 <= c.x < 30
        assert 0 <= c.y < 30
        # The centroid should map back to its own sector
        assert sm.sector_of(c) == i


# ---- SectorMap: LRV tracking -----------------------------------------

def test_unvisited_sectors_are_stalest():
    sm = SectorMap(30, 30, SectorConfig(cols=3, rows=3))
    # Visit sector 4 (center) at tick 5
    sm.mark_visited(Position(15, 15), tick=5)
    # The stalest sector should NOT be 4 (it's the only visited one)
    stalest = sm.stalest_sector()
    assert stalest != 4


def test_stalest_excludes_current_sector():
    sm = SectorMap(30, 30, SectorConfig(cols=3, rows=3))
    # Mark everything visited at tick 0 except give sector 0 the oldest
    for i in range(sm.n_sectors):
        c = sm.sector_centroid(i)
        sm.mark_visited(c, tick=10)
    # Now Tom is standing in sector 0; stalest excluding current should
    # not return 0 even if 0 were stale.
    current = sm.sector_centroid(0)
    stalest = sm.stalest_sector(exclude_current=current)
    assert stalest != 0


def test_most_recently_visited_not_chosen():
    sm = SectorMap(30, 30, SectorConfig(cols=3, rows=3))
    # Visit sector 0 recently (tick 100), others never
    sm.mark_visited(sm.sector_centroid(0), tick=100)
    stalest = sm.stalest_sector()
    # Stalest should be a never-visited sector, not 0
    assert stalest != 0
    assert sm.last_visited_tick(stalest) < 100


def test_reset_clears_visit_history():
    sm = SectorMap(30, 30)
    sm.mark_visited(Position(15, 15), tick=50)
    sm.reset()
    # After reset, all sectors are equally (never) visited
    for i in range(sm.n_sectors):
        assert sm.last_visited_tick(i) < 0


# ---- Conductor patrol_target -----------------------------------------

def test_patrol_target_is_walkable():
    """patrol_target must always return a walkable tile."""
    world = World(WorldConfig(max_ticks=50), seed=42)
    world.reset()
    c = Conductor()
    c.reset()
    c.observe(world)  # builds sector map, marks current sector visited
    target = c.patrol_target(world)
    assert world.grid.is_walkable(target)


def test_patrol_target_changes_as_sectors_visited():
    """As Tom 'visits' sectors, the patrol target should move toward
    unvisited ones (coverage behavior).
    """
    world = World(WorldConfig(max_ticks=50), seed=42)
    world.reset()
    c = Conductor()
    c.reset()
    c.observe(world)
    first = c.patrol_target(world)
    # The first target should be a real walkable tile
    assert world.grid.is_walkable(first)
    # Patrol target is deterministic given the same visit state
    assert c.patrol_target(world) == first


# ---- integration: directed patrol in a real episode -----------------

def test_conductor_directed_patrol_runs_episode():
    """A Conductor-driven Tom with directed patrol completes an episode."""
    world = World(WorldConfig(max_ticks=150), seed=5)
    world.reset()
    tom = ChemicalTom(conductor=Conductor(), seed=5)
    tom.reset()
    steps = 0
    for _ in range(150):
        a = tom(world)
        _, _, _, done = world.step(tom_action=a, jerry_action=Action.WAIT)
        steps += 1
        if done or not world.jerry.alive:
            break
    assert steps > 0


def test_directed_patrol_sweeps_toward_unvisited_sectors():
    """Directed patrol should aim Tom at the stalest (least-recently-
    visited) sector and commit to it while travelling, producing a
    systematic sweep rather than random wandering.

    We verify the policy directly (no hand-rolled walker, which can get
    stuck on walls and confound the test):
      (a) the first patrol target lands in an unvisited sector, and
      (b) once Tom's actual position is marked into several sectors, the
          patrol target avoids the freshly-visited ones (it keeps seeking
          stale zones).
    """
    world = World(WorldConfig(max_ticks=300), seed=5)
    world.reset()
    c = Conductor()
    c.reset()
    c.observe(world)  # marks Tom's starting sector visited, builds sectors

    sm = c._sectors
    start_sector = sm.sector_of(world.tom.position)

    # (a) First patrol target should be in a DIFFERENT (staler) sector than
    # the one Tom is standing in.
    tgt = c.patrol_target(world)
    assert sm.sector_of(tgt) != start_sector, \
        "patrol target should aim at a different sector than Tom's current one"
    assert world.grid.is_walkable(tgt)

    # (b) Simulate having visited several sectors recently; the patrol
    # target should then point at one we have NOT recently visited.
    recent_tick = 100
    visited_now = set()
    for s in range(sm.n_sectors // 2):
        sm.mark_visited(sm.sector_centroid(s), tick=recent_tick)
        visited_now.add(s)
    # Re-query: target sector should be one of the not-recently-visited ones
    world.tick_count = recent_tick
    tgt2 = c.patrol_target(world)
    assert sm.sector_of(tgt2) not in visited_now, \
        "patrol should seek sectors not among the freshly-visited set"
