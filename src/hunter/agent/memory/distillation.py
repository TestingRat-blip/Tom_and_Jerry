"""L1 → L2 distillation.

At episode end, we summarize the L1 per-encounter memory into a single
`EpisodeSummary` that gets persisted to SQLite. Next episode's Tom can
query L2, retrieve relevant past summaries, and warm-start his fresh L1.

This module is PURE — given an L1Memory, World, and outcome metadata,
it produces an EpisodeSummary without touching any storage.

What we keep (and why):
  - Top 10 sighting tiles → next-episode heatmap priors
  - All non-zero locker suspicions → next-episode locker priors
  - Top 10 false-noise hotspots → next-episode false-noise priors
  - Aggregate noise counts (total + verified) → noise-reliability stat
  - `ticks_to_first_sight` → "how findable is Jerry on this map" signal

What we drop (and why):
  - Individual noise records (too noisy, not useful at L2 grain)
  - The pending-noise queue (meaningful only within an episode)
  - Per-tick chemistry/drives (not Phase 4's job)
"""
from __future__ import annotations

from typing import Any

from src.env.world.world import Grid
from src.hunter.agent.memory.fingerprint import (
    fingerprint_jerry,
    fingerprint_map,
)
from src.hunter.agent.memory.l1 import L1Memory
from src.persistence.sqlite.l2_store import EpisodeSummary


# Default tops to keep in the summary. Small enough to stay tiny on disk,
# large enough to capture the meaningful signal.
DEFAULT_HEATMAP_TOP_N = 10
DEFAULT_FALSE_NOISE_TOP_N = 10


def distill_l1_to_summary(
    l1: L1Memory,
    grid: Grid,
    jerry_policy: Any,
    outcome: str,
    total_ticks: int,
    total_jerry_reward: float,
    ticks_to_first_sight: int | None,
    tom_label: str = "",
    jerry_label: str | None = None,
    heatmap_top_n: int = DEFAULT_HEATMAP_TOP_N,
    false_noise_top_n: int = DEFAULT_FALSE_NOISE_TOP_N,
    notes: dict | None = None,
) -> EpisodeSummary:
    """Build an EpisodeSummary from the current L1 state.

    Call at episode END, before L1.reset(). The L1 contents at this
    point reflect everything Tom learned this life.

    Args:
        l1: live L1Memory at episode end
        grid: the world's grid (for fingerprinting)
        jerry_policy: the Jerry policy used this episode (for fingerprinting)
        outcome: "caught" | "survived" | "timeout"
        total_ticks: episode length
        total_jerry_reward: Jerry's cumulative reward this episode
        ticks_to_first_sight: tick of Tom's first sighting, or None
        tom_label: label of the Tom policy (analysis only)
        jerry_label: explicit Jerry label, or None to fingerprint automatically
        heatmap_top_n: how many top sighting tiles to keep
        false_noise_top_n: how many top false-noise tiles to keep
        notes: optional free-form blob for additional context
    """
    fine_fp, coarse_fp = fingerprint_map(grid)
    jerry_fp = fingerprint_jerry(jerry_policy, label=jerry_label)

    # Top-N sighting tiles (Position, count) → (x, y, count) tuples
    heatmap_hottest = l1.heatmap_hottest(top_n=heatmap_top_n)
    heatmap_top = [(p.x, p.y, c) for p, c in heatmap_hottest]

    # All non-zero locker sightings, sorted by count desc
    all_lockers = l1.store.all_locker_sightings()
    lockers_sorted = sorted(all_lockers.items(), key=lambda kv: kv[1], reverse=True)
    lockers = [(x, y, c) for (x, y), c in lockers_sorted if c > 0]

    # Top-N false-noise tiles, sorted by count desc
    all_false = l1.store.all_false_noise_counts()
    false_sorted = sorted(all_false.items(), key=lambda kv: kv[1], reverse=True)
    false_noise_top = [(x, y, c) for (x, y), c in false_sorted[:false_noise_top_n]]

    # Aggregate noise stats: total events recorded + how many got verified
    total_noise_events = sum(all_false.values())
    verified_noise_count = _count_verified_noises(l1)

    return EpisodeSummary(
        map_fingerprint_fine=fine_fp,
        map_fingerprint_coarse=coarse_fp,
        jerry_fingerprint=jerry_fp,
        outcome=outcome,
        total_ticks=total_ticks,
        total_jerry_reward=total_jerry_reward,
        ticks_to_first_sight=ticks_to_first_sight,
        tom_label=tom_label,
        heatmap_top=heatmap_top,
        lockers=lockers,
        false_noise_top=false_noise_top,
        total_noise_events=total_noise_events,
        verified_noise_count=verified_noise_count,
        notes=notes or {},
    )


def _count_verified_noises(l1: L1Memory) -> int:
    """Scan all noise records in L1's store, count verified ones.

    Verified means: a sighting near the noise occurred within the
    verification window. Used as a "noise reliability" indicator —
    when total_noise_events > 0, verified/total tells you the
    signal:noise ratio for this episode.
    """
    # The L1Store exposes individual noise records by tick, not as a
    # batch. For Phase 4 we walk a small set of likely ticks: the
    # `_pending` queue captures pending unverified noises, but verified
    # ones have already been dropped from it. We rely on the verified
    # bit on each NoiseRecord which is stored in the noise:{tick} hash.
    #
    # Since L1's verification window is small (default 12 ticks), most
    # episodes will have on the order of dozens of noise records, not
    # thousands. We scan all noise keys for this episode.
    count = 0
    pattern = l1.store.client.ns("l1", l1.store.episode_id, "noise", "*")
    for key in l1.store.client.scan_iter(match=pattern, count=100):
        # Skip the noise_index hash and the meta hash, which share the
        # same prefix — those have no ":tick" suffix-only segment.
        # The noise:{tick} keys end in a numeric tick value.
        suffix = key.rsplit(":", 1)[-1]
        if not suffix.isdigit():
            continue
        verified = l1.store.client.hget(key, "verified")
        if verified == "1":
            count += 1
    return count
