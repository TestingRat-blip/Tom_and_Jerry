"""Tests for the Phase 8 scent additions: diffusion, cap, passive deposit.

These guard DORMANT infrastructure. Scent diffusion is config-gated off by
default (scent_diffusion_rate=0.0) and does not currently catch any statue on
its own (the occluded-pocket case it was built for is defeated by tortuous
geometry — see docs/PHASE8_SENSORY_FEAR_DESIGN.md). It is committed as correct,
verified machinery for the richer Phase 8 scent/fear model. Because it is
dormant, it is especially important that these invariants are tested — a silent
regression in unused code would otherwise go unnoticed until Phase 8 enables it.
"""
from __future__ import annotations

import numpy as np

from src.env.world.grid import Grid, TileType
from src.env.sensors.scent import ScentField
from src.utils.types import Position


def _open_grid(w: int = 7, h: int = 7) -> Grid:
    """A grid with walls only on the border, open interior."""
    tiles = np.zeros((h, w), dtype=np.int8)
    tiles[0, :] = int(TileType.WALL)
    tiles[-1, :] = int(TileType.WALL)
    tiles[:, 0] = int(TileType.WALL)
    tiles[:, -1] = int(TileType.WALL)
    return Grid(width=w, height=h, tiles=tiles)


def test_diffusion_off_by_default_is_legacy():
    """diffusion_rate=0.0 → field only decays + deposits, never spreads."""
    g = _open_grid()
    sf = ScentField(g, decay_per_tick=1.0, diffusion_rate=0.0, floor=0.0)
    sf.deposit_at(Position(3, 3), amount=1.0)
    sf.tick()
    # No spread: center keeps all of it, neighbors stay zero.
    assert sf.strength_at(Position(3, 3)) == 1.0
    assert sf.strength_at(Position(3, 2)) == 0.0
    assert sf.strength_at(Position(2, 3)) == 0.0


def test_diffusion_conserves_mass_in_interior():
    """With decay off, diffusion neither creates nor destroys scent in an
    all-open interior (mass conserved)."""
    g = _open_grid()
    sf = ScentField(g, decay_per_tick=1.0, diffusion_rate=0.1, cap=10.0, floor=0.0)
    sf.deposit_at(Position(3, 3), amount=1.0)
    total0 = float(sf.field.sum())
    for _ in range(5):
        sf.tick()
    total1 = float(sf.field.sum())
    assert abs(total1 - total0) < 1e-5, f"mass drifted: {total0} -> {total1}"


def test_diffusion_spreads_outward():
    """Scent spreads to neighbors over successive ticks."""
    g = _open_grid()
    sf = ScentField(g, decay_per_tick=1.0, diffusion_rate=0.1, cap=10.0, floor=0.0)
    sf.deposit_at(Position(3, 3), amount=1.0)
    sf.tick()
    near = sf.strength_at(Position(3, 2))
    assert near > 0.0, "scent did not spread to adjacent tile"
    # The center should have lost some to its neighbors.
    assert sf.strength_at(Position(3, 3)) < 1.0


def test_diffusion_does_not_cross_walls():
    """Scent never bleeds into a wall tile."""
    g = _open_grid()
    # Put a wall directly north of the deposit point.
    g.tiles[2, 3] = int(TileType.WALL)
    sf = ScentField(g, decay_per_tick=1.0, diffusion_rate=0.2, cap=10.0, floor=0.0)
    sf.deposit_at(Position(3, 3), amount=1.0)
    for _ in range(5):
        sf.tick()
    assert sf.strength_at(Position(3, 2)) == 0.0, "scent leaked into a wall"


def test_cap_limits_accumulation():
    """Repeated same-tile deposits saturate at the configured cap."""
    g = _open_grid()
    sf = ScentField(g, decay_per_tick=1.0, diffusion_rate=0.0, cap=2.5, floor=0.0)
    for _ in range(10):
        sf.deposit_at(Position(3, 3), amount=1.0)
    assert sf.strength_at(Position(3, 3)) == 2.5


def test_passive_deposit_amount_override():
    """deposit_at(amount=...) overrides the default movement deposit."""
    g = _open_grid()
    sf = ScentField(g, deposit_amount=1.0, diffusion_rate=0.0, floor=0.0)
    sf.deposit_at(Position(3, 3), amount=0.1)  # weak passive emission
    assert abs(sf.strength_at(Position(3, 3)) - 0.1) < 1e-6
