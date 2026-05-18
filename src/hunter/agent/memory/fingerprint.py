"""Fingerprinting for L2 memory retrieval.

Two map fingerprints (fine, coarse) and one jerry fingerprint. All
deterministic — same inputs always produce the same hash.

These are stable hex strings stored in the L2 episode_summaries table
and queried against during warm-start.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.env.world.world import Grid
from src.utils.types import Position


# ---- map fingerprints --------------------------------------------------

def fingerprint_map_fine(grid: Grid) -> str:
    """SHA-256 of the canonical serialization of the grid state.

    Two grids are equal iff their fine fingerprints match. Lockers and
    vent pairs are sorted before hashing so set ordering doesn't matter.
    """
    h = hashlib.sha256()
    # Dimensions
    h.update(f"w{grid.width}h{grid.height}".encode())

    # Tile grid — flatten row-major
    # Grid stores tiles as a numpy array; .tolist() gives nested Python lists
    h.update(b"tiles:")
    for row in grid.tiles.tolist():
        h.update(bytes(row))  # row is list[int]; bytes() works for 0-255 ints

    # Sorted locker positions
    h.update(b"lockers:")
    for p in sorted(grid.locker_positions, key=lambda p: (p.x, p.y)):
        h.update(f"({p.x},{p.y})".encode())

    # Sorted vent pairs (vents are directed in the dict but undirected
    # in meaning, so we canonicalize each pair as the lexicographically
    # smaller endpoint first)
    h.update(b"vents:")
    canonical_pairs = set()
    for a, b in grid.vent_links.items():
        pair = tuple(sorted([(a.x, a.y), (b.x, b.y)]))
        canonical_pairs.add(pair)
    for pair in sorted(canonical_pairs):
        h.update(f"{pair[0]}-{pair[1]}".encode())

    return h.hexdigest()


def fingerprint_map_coarse(grid: Grid) -> str:
    """Hash of the map's STRUCTURAL signature, not its content.

    Coarse fingerprint matches across different random maps that share
    the same shape. Use this for "I've fought this Jerry on similar
    maps before" generalization, not "I've fought this Jerry on THIS
    map before" (that's the fine fingerprint).
    """
    wall_count = int((grid.tiles == 1).sum())  # tile value 1 = wall
    locker_count = len(grid.locker_positions)
    vent_pair_count = len({
        frozenset([(a.x, a.y), (b.x, b.y)])
        for a, b in grid.vent_links.items()
    })

    signature = (
        f"w{grid.width}h{grid.height}"
        f"_walls{wall_count}"
        f"_lockers{locker_count}"
        f"_vents{vent_pair_count}"
    )
    return hashlib.sha256(signature.encode()).hexdigest()


def fingerprint_map(grid: Grid) -> tuple[str, str]:
    """Convenience: return (fine, coarse) at once."""
    return fingerprint_map_fine(grid), fingerprint_map_coarse(grid)


# ---- jerry fingerprints -----------------------------------------------

def fingerprint_jerry(policy: Any, label: str | None = None) -> str:
    """Stable identifier for a Jerry policy.

    Resolution order:
      1. If `label` is given explicitly, use it (caller knows best —
         e.g. for evolved Jerrys with generation+individual ids).
      2. If `policy` has a `model_path` or `path` attribute pointing at
         a file, hash the file bytes (PPO checkpoint case).
      3. Otherwise fall back to the class name. This covers scripted
         policies, lambdas (which get "function"), and random policies.

    Returns a short hex string, prefixed by type for human readability:
      "label:custom_v3"        explicit label
      "file:abc123def..."      file hash (16 hex chars from SHA-256)
      "class:ScriptedJerry"    class name fallback
    """
    if label is not None:
        return f"label:{label}"

    # PPO-style policy with a model file
    path_attr = None
    for attr in ("model_path", "path", "_model_path"):
        if hasattr(policy, attr):
            v = getattr(policy, attr)
            if v is not None:
                path_attr = v
                break

    if path_attr is not None:
        p = Path(path_attr)
        if p.exists() and p.is_file():
            return f"file:{_hash_file(p)[:16]}"
        # Path was given but file isn't there — fall back to the string
        return f"path:{path_attr}"

    # Class name fallback
    cls_name = type(policy).__name__
    return f"class:{cls_name}"


def _hash_file(path: Path) -> str:
    """SHA-256 of file contents, returned as hex."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        # Stream in 64KB chunks so we don't load huge checkpoints into RAM
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
