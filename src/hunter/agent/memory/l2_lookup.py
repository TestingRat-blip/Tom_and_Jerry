"""L2 retrieval — turning past episode summaries into a WarmStart bundle.

The query shape is a fine-then-coarse cascade:
  1. Find episodes matching (current_map_fine, current_jerry) — exact matches
  2. Find episodes matching (current_map_coarse, current_jerry) — similar maps,
     excluding ones already counted at the fine layer

Each retrieved episode contributes weighted prior counts to the WarmStart.
Weights decay exponentially with age (most-recent = full weight).
Fine matches weigh more than coarse matches.

The WarmStart is then handed to L1Memory.apply_warm_start() before the
episode begins. L1 stores those priors in parallel "warm" keys; queries
combine warm + in-episode counts.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.persistence.sqlite.l2_store import EpisodeSummary, L2Store


# ---- behavioral stance (memory-driven adaptation) ---------------------

@dataclass
class StrategicStance:
    """What memory says about HOW to hunt this Jerry on this map.

    Derived from the behavioral signatures of past episodes (los_break_count
    etc.). The Conductor applies this at episode start to deploy counters
    like the hold-on-LOS-break run-down — but ONLY when memory says this
    prey warrants it, rather than always-on.
    """
    # Episodes this stance was aggregated from (0 = no memory → neutral).
    episode_count: int = 0
    # Mean LOS-breaks per past episode against this Jerry/map.
    mean_los_breaks: float = 0.0
    # Whether to deploy the hold-on-LOS-break run-down this episode.
    deploy_hold_on_los_break: bool = False
    # Top cover spots Jerry vanished into, weighted (x, y) -> weight.
    los_break_hotspots: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def is_neutral(self) -> bool:
        """True when memory has nothing to say (no behavioral history)."""
        return self.episode_count == 0


# Threshold: mean LOS-breaks per episode above which we deploy the run-down.
# A cover-dancer racks up many LOS-breaks; a normal evader a handful. Tuned
# conservatively so the run-down only deploys against genuine LOS-breakers.
DEFAULT_LOS_BREAK_DEPLOY_THRESHOLD = 5.0


# ---- defaults: weights and bounds -------------------------------------

# Per-episode age decay (multiplicative). 0.95 → episode-10-ago weighs ~0.6,
# episode-100-ago weighs ~0.006. Recent dominates without erasing history.
DEFAULT_DECAY_BASE = 0.95

# Fine matches weigh more than coarse matches. With these defaults a fine
# match at episode-1-ago weighs 0.95, a coarse match at episode-1-ago
# weighs 0.38 — fine dominates ~2.5x when both exist.
DEFAULT_FINE_WEIGHT = 1.0
DEFAULT_COARSE_WEIGHT = 0.4

# Maximum number of past episodes to pull from each query layer.
DEFAULT_FINE_LIMIT = 20
DEFAULT_COARSE_LIMIT = 20

# Saturation cap on warm-start counts. Without this, a locker that's
# been suspicious across 50 episodes ends up with an enormous prior
# count that dominates fresh in-episode data forever. Priors should
# BIAS Tom, not DICTATE him.
DEFAULT_MAX_PRIOR_COUNT = 10


@dataclass(slots=True)
class WarmStart:
    """Aggregated priors for one new episode, built from past summaries.

    Each dict maps (x, y) → weighted prior count. The counts are floats
    because they're sums-of-weights; L1 stores them as floats too.

    Empty WarmStart means "no useful prior history" — Tom starts fresh.
    """
    heatmap: dict[tuple[int, int], float] = field(default_factory=dict)
    lockers: dict[tuple[int, int], float] = field(default_factory=dict)
    false_noise: dict[tuple[int, int], float] = field(default_factory=dict)

    # Diagnostics: how many past episodes contributed at each tier
    fine_episode_count: int = 0
    coarse_episode_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.heatmap or self.lockers or self.false_noise)

    @property
    def total_episodes(self) -> int:
        return self.fine_episode_count + self.coarse_episode_count


@dataclass(frozen=True, slots=True)
class L2LookupConfig:
    """Tunable knobs for the lookup cascade.

    Defaults are deliberately modest in v1. Phase 6+ may evolve these
    per archetype.
    """
    decay_base: float = DEFAULT_DECAY_BASE
    fine_weight: float = DEFAULT_FINE_WEIGHT
    coarse_weight: float = DEFAULT_COARSE_WEIGHT
    fine_limit: int = DEFAULT_FINE_LIMIT
    coarse_limit: int = DEFAULT_COARSE_LIMIT
    max_prior_count: float = DEFAULT_MAX_PRIOR_COUNT


class L2Lookup:
    """Builds WarmStart bundles from the L2 store.

    Construct once per Tom; call `build_warm_start(map_fp_fine, map_fp_coarse,
    jerry_fp)` at the start of each episode.
    """

    def __init__(self, store: L2Store, config: L2LookupConfig | None = None):
        self.store = store
        self.config = config or L2LookupConfig()

    def build_warm_start(
        self,
        map_fp_fine: str,
        map_fp_coarse: str,
        jerry_fp: str,
    ) -> WarmStart:
        """Run the fine-then-coarse cascade and aggregate weighted priors."""
        cfg = self.config

        # Stage 1: fine matches (exact map + jerry)
        fine_summaries = self.store.query_fine(
            map_fp_fine=map_fp_fine,
            jerry_fp=jerry_fp,
            limit=cfg.fine_limit,
        )

        # Stage 2: coarse matches (similar map + jerry), excluding the
        # fine fingerprint we just queried (no double-counting)
        coarse_summaries = self.store.query_coarse(
            map_fp_coarse=map_fp_coarse,
            jerry_fp=jerry_fp,
            limit=cfg.coarse_limit,
            exclude_fine=map_fp_fine,
        )

        warm = WarmStart(
            fine_episode_count=len(fine_summaries),
            coarse_episode_count=len(coarse_summaries),
        )

        # Aggregate, weighting by (tier_weight × age_decay)
        heat: defaultdict[tuple[int, int], float] = defaultdict(float)
        lockers: defaultdict[tuple[int, int], float] = defaultdict(float)
        false_noise: defaultdict[tuple[int, int], float] = defaultdict(float)

        self._fold_summaries(
            fine_summaries, cfg.fine_weight, cfg.decay_base,
            heat, lockers, false_noise,
        )
        self._fold_summaries(
            coarse_summaries, cfg.coarse_weight, cfg.decay_base,
            heat, lockers, false_noise,
        )

        # Cap each entry at max_prior_count
        cap = cfg.max_prior_count
        warm.heatmap = {k: min(cap, v) for k, v in heat.items()}
        warm.lockers = {k: min(cap, v) for k, v in lockers.items()}
        warm.false_noise = {k: min(cap, v) for k, v in false_noise.items()}

        return warm

    def behavioral_stance(
        self,
        map_fp_fine: str,
        map_fp_coarse: str,
        jerry_fp: str,
        deploy_threshold: float = DEFAULT_LOS_BREAK_DEPLOY_THRESHOLD,
    ) -> StrategicStance:
        """Aggregate past behavioral signatures into a StrategicStance.

        Reads los_break_count / los_break_hotspots from past episodes (same
        fine→coarse cascade as build_warm_start) and decides whether to
        deploy the hold-on-LOS-break run-down: if this Jerry historically
        breaks LOS often (mean above deploy_threshold), deploy it. This is
        what makes the run-down SELECTIVE — memory-deployed against genuine
        cover-dancers, not always-on.
        """
        cfg = self.config
        fine = self.store.query_fine(
            map_fp_fine=map_fp_fine, jerry_fp=jerry_fp, limit=cfg.fine_limit)
        coarse = self.store.query_coarse(
            map_fp_coarse=map_fp_coarse, jerry_fp=jerry_fp,
            limit=cfg.coarse_limit, exclude_fine=map_fp_fine)
        summaries = fine + coarse

        if not summaries:
            return StrategicStance()  # neutral — no memory

        total_breaks = sum(s.los_break_count for s in summaries)
        mean_breaks = total_breaks / len(summaries)

        # Aggregate hotspots, weighting fine over coarse.
        hotspots: defaultdict[tuple[int, int], float] = defaultdict(float)
        for s in fine:
            for x, y, c in s.los_break_hotspots:
                hotspots[(x, y)] += c * cfg.fine_weight
        for s in coarse:
            for x, y, c in s.los_break_hotspots:
                hotspots[(x, y)] += c * cfg.coarse_weight

        return StrategicStance(
            episode_count=len(summaries),
            mean_los_breaks=mean_breaks,
            deploy_hold_on_los_break=mean_breaks >= deploy_threshold,
            los_break_hotspots=dict(hotspots),
        )

    @staticmethod
    def _fold_summaries(
        summaries: list[EpisodeSummary],
        tier_weight: float,
        decay_base: float,
        heat: defaultdict,
        lockers: defaultdict,
        false_noise: defaultdict,
    ) -> None:
        """Add each summary's contribution into the running aggregates.

        Position in `summaries` is the episode age (0 = most recent).
        """
        for age, summary in enumerate(summaries):
            weight = tier_weight * (decay_base ** age)
            for x, y, count in summary.heatmap_top:
                heat[(x, y)] += weight * count
            for x, y, count in summary.lockers:
                lockers[(x, y)] += weight * count
            for x, y, count in summary.false_noise_top:
                false_noise[(x, y)] += weight * count
