"""Sector decomposition + patrol policy for the Conductor (Phase 6d).

Sectors are the Conductor's COARSE spatial vocabulary. The grid is split
into an N×M zoning; the Conductor reasons about "which sector to sweep"
when its belief is empty, and Tom navigates tile-by-tile within/across
sectors (sectors never replace local navigation — see ADR-013 / the
Phase 6 design doc's sector boundary note).

This module is self-contained and testable in isolation: it knows the
grid dimensions and walkability, nothing about Tom or the belief.

Patrol policy for Stage 1 (scripted): LEAST-RECENTLY-VISITED. The
Conductor tracks when Tom was last actually inside each sector, and
directs patrol toward the stalest sector. This produces legible coverage
sweeps a human watching a replay reads as "Tom is methodically searching."

The interesting upgrade (Stage 2): weight sector staleness by L2
historical heatmap, so Tom preferentially sweeps places Jerry has
historically been. That's deferred — it's exactly the kind of thing that
should become LEARNED rather than hand-tuned, and wiring L2 into patrol
now would couple two systems prematurely.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.types import Position


@dataclass(frozen=True)
class SectorConfig:
    """How the grid is divided into sectors.

    5x5 (was 3x3): with 3x3 on a 30x30 map, each sector is 10x10 tiles and
    the Conductor patrols to sector CENTROIDS — leaving a motionless prey in a
    sector's corner ~7 tiles from where Tom ever goes, outside his effective
    sweep. That let "statues" hide in unswept corners for a whole night. 5x5
    makes sectors 6x6 tiles, so centroid visits actually bring Tom within
    sight of the whole sector, closing the open-corner coverage gap. (Does not
    fix LOS-occluded pockets — those are a separate perception problem for the
    Phase 8 sensory model.)
    """
    cols: int = 5   # sectors across (x)
    rows: int = 5   # sectors down (y)


class SectorMap:
    """Maps tiles <-> sectors and tracks per-sector last-visited ticks for
    least-recently-visited patrol.

    Construct with the grid dimensions. The caller drives it with
    mark_visited(tom_position, tick) each tick and queries
    stalest_sector_target(...) for a patrol destination.
    """

    def __init__(self, grid_width: int, grid_height: int,
                 config: SectorConfig | None = None):
        self.config = config or SectorConfig()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cols = self.config.cols
        self.rows = self.config.rows
        self.n_sectors = self.cols * self.rows
        # last_visited_tick[sector_index] -> tick (or -inf-ish if never)
        self._last_visited: list[int] = [-10**9] * self.n_sectors

    # ---- tile <-> sector ----------------------------------------------

    def sector_of(self, p: Position) -> int:
        """Return the sector index for a tile. Clamped to valid range."""
        # Which column / row band does this tile fall in?
        col = (p.x * self.cols) // max(1, self.grid_width)
        row = (p.y * self.rows) // max(1, self.grid_height)
        col = min(max(col, 0), self.cols - 1)
        row = min(max(row, 0), self.rows - 1)
        return row * self.cols + col

    def sector_centroid(self, sector_index: int) -> Position:
        """The centroid tile of a sector (its geometric center)."""
        row = sector_index // self.cols
        col = sector_index % self.cols
        # Center of the column band
        cx = int((col + 0.5) * self.grid_width / self.cols)
        cy = int((row + 0.5) * self.grid_height / self.rows)
        cx = min(max(cx, 0), self.grid_width - 1)
        cy = min(max(cy, 0), self.grid_height - 1)
        return Position(cx, cy)

    # ---- visit tracking -----------------------------------------------

    def mark_visited(self, tom_position: Position, tick: int) -> None:
        """Record that Tom was inside this position's sector at `tick`.

        'Visited' means Tom's ACTUAL position fell in the sector — not that
        the Conductor pointed at it. This keeps the LRV sweep honest: a
        sector counts as covered only once Tom actually got there.
        """
        s = self.sector_of(tom_position)
        self._last_visited[s] = tick

    def last_visited_tick(self, sector_index: int) -> int:
        return self._last_visited[sector_index]

    # ---- patrol policy ------------------------------------------------

    def stalest_sector(self, exclude_current: Position | None = None) -> int:
        """Return the index of the least-recently-visited sector.

        If exclude_current is given, the sector Tom is currently in is
        skipped (no point patrolling to where you already are). Ties broken
        by lowest index for determinism.
        """
        current_sector = (
            self.sector_of(exclude_current) if exclude_current is not None else -1
        )
        best_idx = None
        best_tick = None
        for idx in range(self.n_sectors):
            if idx == current_sector:
                continue
            t = self._last_visited[idx]
            if best_tick is None or t < best_tick:
                best_tick = t
                best_idx = idx
        # If every other sector was excluded (1-sector edge case), fall back
        # to the current sector.
        if best_idx is None:
            return current_sector if current_sector >= 0 else 0
        return best_idx

    def reset(self) -> None:
        """Clear visit history at episode start."""
        self._last_visited = [-10**9] * self.n_sectors
