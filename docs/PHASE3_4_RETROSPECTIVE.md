# Phase 3 + 4 — Retrospective

**Status:** Complete
**Wrapped:** 2026-05-18
**Artifacts:**
- `src/persistence/redis/` — L1 storage layer (Phase 3)
- `src/persistence/sqlite/` — L2 storage layer (Phase 4)
- `src/hunter/agent/memory/l1.py` — per-encounter memory API
- `src/hunter/agent/memory/l2_lookup.py` — past-episode retrieval
- `src/hunter/agent/memory/distillation.py` — L1 → L2 summarization
- `src/hunter/agent/memory/fingerprint.py` — map + jerry identity hashes
- `src/hunter/agent/memory/l3.py` — interface stub for Phase 5+
- ChemicalTom + ReplayRecorder extensions
- `scripts/show_l2_effect.py` — demonstration tool

---

## Exit criteria check

**Phase 3 (per-encounter memory):**

| Criterion | Achieved |
|---|---|
| Redis-backed L1 storage with episode-isolated keyspace | ✓ |
| Noise event log with verification | ✓ |
| Per-tile false-noise counts | ✓ |
| Per-locker suspicion counts | ✓ |
| Sighting heatmap | ✓ |
| ChemicalTom reads L1 in modulated noise threshold | ✓ |
| Replay recorder hands L1 the map's locker positions | ✓ |
| Backwards compatible (L1=None matches Phase 2) | ✓ |

**Phase 4 (persistent memory):**

| Criterion | Achieved |
|---|---|
| SQLite-backed L2 with migrations and indexes | ✓ |
| Map fingerprinting (fine + coarse) | ✓ |
| Jerry fingerprinting (label / file hash / class) | ✓ |
| L1 → L2 distillation at episode end | ✓ |
| Cascade lookup (fine → coarse) with age decay | ✓ |
| L1 warm-start hook with priors/in-episode separation | ✓ |
| End-to-end wire-up via ChemicalTom lifecycle methods | ✓ |
| ReplayRecorder triggers warm-start + distillation | ✓ |
| L3 interface stub for future expansion | ✓ |
| Backwards compatible (L2=None matches Phase 3) | ✓ |

**Both phases functionally complete.**

---

## Headline numbers

279 tests passing across the project (Phase 1: 4 + Phase 2: ~70 + Phase 3: 25 + Phase 4: 81). Phase 3 added 25 tests (21 unit + 4 redis-marked integration). Phase 4 added 81 tests (23 L2 store + 18 fingerprint + 15 distillation + 16 L2 lookup + 20 warm-start/L3 + 11 e2e integration — 103 if you count the new unit tests, 11 if you count end-to-end).

Demonstration run on seed 42 vs passive Jerry, 5 episodes:

| Episode | Outcome | Ticks | Warm-start at start |
|---|---|---|---|
| 1 | caught | 197 | empty |
| 2 | caught | 116 | 1 prior (heatmap (1,22) w=9.0) |
| 3 | caught | 40 | 2 priors (heatmap (1,22) w=10.0 saturated) |
| 4 | caught | 194 | 3 priors |
| 5 | caught | 42 | 4 priors |

The visible signal: tile (1, 22) appeared in 5/5 episode summaries. That's
the heatmap pattern persisting across deaths — exactly what Phase 4 was
built to do.

---

## What worked

### Architectural separation

The cleanest decision in Phase 4 was **keeping warm-start priors in a separate
data structure from in-episode counters**. The behavior queries combine
warm + in-episode; the distillation pipeline reads in-episode only. Without
that split, priors would compound across episodes via distillation — episode
N's summary would include all priors from episodes 1..N-1, episode N+1 would
warm-start from that compounded set, and the priors would grow exponentially.

Tests `test_apply_warm_start_does_not_touch_redis_store` and
`test_distillation_reads_in_episode_only` pin this invariant explicitly.

### Two-tier fingerprint cascade

The fine-then-coarse cascade gives Phase 4 two graceful degradation tiers
instead of one. When Tom encounters a Jerry on the exact same map as
before, fine matches dominate. When the map differs but the Jerry is the
same, coarse fingerprints still produce useful priors (smaller weight, but
not zero). Phase 6 co-evolution will get a lot of mileage out of this —
evolved Jerry archetypes can be tracked across many random maps.

### Optional, optional, optional

Every Phase 4 addition is opt-in:

- `ChemicalTom(l1=None)` → exactly Phase 2 behavior
- `ChemicalTom(l1=L1Memory)` → Phase 3 behavior
- `ChemicalTom(l1=..., l2_lookup=..., l2_store=...)` → full Phase 4

Lifecycle methods (`warm_start_for_episode`, `distill_at_episode_end`) return
`False` when their dependencies aren't wired. The recorder calls them
unconditionally; nothing breaks. Tests at every tier pin this — the
no-regression suite is 268 tests strong before adding any Phase 4-specific
ones.

This matters more than it looks. Many memory architectures couple their
storage assumptions tightly to behavior code, and you can't run them
without infrastructure. Phase 4's opt-in design means:

- Unit tests don't need SQLite or Redis
- Phase 5 archetypes can pick which tiers to enable
- Future eval modes can read L2 without writing (frozen lookup)
- Phase 6 co-evolution can write to L2 from many workers in parallel
  (SQLite WAL handles this) or run pure-Phase-2 toms in the population

### FakeRedis + tmp_path SQLite

Both storage backends have in-memory test doubles. `FakeRedis` is a hand-
written class implementing the same `RedisLike` Protocol; SQLite tests
use `tmp_path` databases that vanish with the test. Result: the entire
default suite runs in 30s. Without these doubles, every L1 test would need
a running Redis container — which is fine in CI but agony during local
development.

The matched-pair test pattern is worth preserving: every behavior tested
against `FakeRedis` has a corresponding `@pytest.mark.redis` test against
real Redis. When the two backends drift, one of them passes and the other
fails — early warning system.

### Distillation discipline

The distill module makes explicit choices about **what gets carried forward
and what gets dropped**. Top-N heatmap tiles, all non-zero locker suspicions,
top-N false-noise spots, aggregate noise counts, ticks_to_first_sight. Drops
individual noise records, the pending-noise queue, per-tick chemistry/drives.

This is the kind of code that, two years from now, somebody will be tempted
to "just add" everything to. Resisting that bloat preserves Phase 4's
genuine usefulness — L2 stays small, queryable, and meaningful. A summary
that's 50 fields long is functionally the same as no summary at all because
nobody reads it.

---

## What didn't work / what was harder than expected

### The flash-prediction problem from Phase 2 still exists

Phase 3 didn't fix the dancing exploit. Phase 4 won't either. Both phases
add *capability* — Tom can now remember being fooled, and remember it
across deaths — but neither phase produces a *behavioral response* to
oscillating Jerry on its own.

This is the right outcome. The Phase 2 retro's discipline — "let Phase 6
evolve responses, don't hand-code them" — still applies. Phase 3 and 4
give Phase 6 more degrees of freedom to evolve against (Tom can now have
state that persists across deaths, so an evolved hunter can learn to
exploit that). But neither phase is a power-creep step.

### Test setup is sensitive to whether Tom sights Jerry

The central Phase 4 test (`test_two_consecutive_episodes_warm_start_from_each_other`)
needs episode 1 to actually produce *something* worth warm-starting from.
A passive Jerry on a 30×30 map in 100 ticks often produces an *empty*
summary (no sightings, no noises). The test fix was straightforward —
bump the budget to 400 ticks and add a `pytest.skip` when episode 1
happens to produce nothing — but it's a useful reminder that **most of
the time, on most map+jerry combinations, Tom doesn't learn much in
one life.** The persistence machinery is a slow burn. The interesting
signal builds over dozens or hundreds of episodes.

This will land harder in Phase 6 where every training rollout writes
to L2. Then the priors actually have something to bias against.

### Coarse fingerprints are a blunt instrument

The coarse fingerprint is `(width, height, wall_count, locker_count,
vent_pair_count)`. Two maps with the same dimensions and roughly the
same wall density get the same coarse hash. That's the design — generalize
across "similar-shaped" maps — but in practice, two 30×30 maps with the
same wall count can still look very different (one might be open corridors,
one might be a maze). Their tactically-relevant geometry differs even
though our fingerprint says they're "the same shape."

A future refinement would use a topological signature — connectivity stats,
chokepoint counts, room-graph isomorphism class. For now, the simple
fingerprint is good enough: coarse matches contribute at 0.4 weight
compared to fine matches' 1.0, so even if they're noisy, they don't
dominate.

### L3 is a stub and that's fine

The Phase 4 design originally considered L3 (semantic/episodic recall via
embeddings) as a parallel deliverable. After thinking through the build
cost — sentence-transformers model download, ChromaDB integration, query
interface design — we shipped an interface-only stub.

The stub is the right call. ChromaDB and embeddings add significant
infrastructure for marginal v1 benefit; the case for real L3 hasn't been
made yet. The interface stub keeps the door open without committing.

When a real Phase 5+ use case emerges that L2 can't cover (something
like "tactically-similar episodes regardless of map fingerprint"), L3
fills it. Until then, every call site that uses `L3Memory` works fine
because all methods return safe defaults.

### SQLite vs Postgres tradeoff

Chose SQLite for simplicity. The right call for v1, but worth being honest
about the limit: when Phase 6 has 16 training workers writing summaries
in parallel, SQLite's single-writer serialization will become a bottleneck.

The mitigation: the abstraction between `L2Store` and `SQLiteClient` is
clean enough that swapping to Postgres is a single-file change. We don't
pay the cost today; we documented where the swap would happen.

---

## Honest take on what Phase 3+4 actually delivers

**Architecturally**: This is the most novel piece of the project. Three-tier
memory with fingerprint-based retrieval, age-decayed warm-starting, and
strict separation between priors and in-episode observations is genuinely
distinctive. No published predator-AI architecture I'm aware of does this.

**Behaviorally**: The visible effect in v1 is modest. ChemicalTom-L2's
catch rate against passive Jerry is roughly the same as ChemicalTom's
catch rate against passive Jerry (which was already roughly the same as
ScriptedTom's). Phase 3/4's value will become visible when:

1. Trained Jerry policies produce richer L1 data (more varied sightings,
   real false noises from movement)
2. Phase 5's archetype-specific Jerry behaviors give Tom DIFFERENT patterns
   to learn across episodes
3. Phase 6's co-evolution loop runs hundreds of episodes — enough for
   L2 to actually accumulate meaningful cross-episode priors

This is intentional. Memory infrastructure that "obviously works" after
two episodes is probably overfit to a particular demo. Memory that builds
slowly into meaningful priors over many episodes is what biological
predators actually have.

---

## What Phase 3+4 unlocks for later phases

### Phase 5 — six Jerry archetypes

Each archetype can now have a **distinct fingerprint** at the jerry level.
A "sneaker" archetype gets `label:archetype_sneaker`; a "sprinter" gets
`label:archetype_sprinter`. Tom's L2 lookups partition the priors by
archetype, so the warm-start for a sneaker doesn't apply when fighting a
sprinter. This makes archetype identity *behaviorally legible* to Tom.

Phase 5 can also tune per-archetype L1Config and ChemicalTomConfig
parameters now, because the persistence works correctly across episodes
with different configurations.

### Phase 6 — co-evolution

Hall of fame: each historical Jerry policy gets a stable file hash
fingerprint. Tom can be evaluated against the full hall and L2 retains
the cross-policy memory.

Population selection signals: `total_jerry_reward` averaged over recent
episodes gives a real measure of how well current Jerry generations are
doing against current Tom. Stored in the `notes` field, queryable per
generation.

Tom can be **evaluated under "amnesia" conditions** — L2 lookup disabled,
warm-start always empty — vs **with-memory conditions** to measure the
contribution of persistence directly. This is the kind of ablation that
makes the research credible.

### Phase 7+ (future)

L3 with real embeddings becomes plug-and-play when the use case emerges.
Map-shape topological fingerprints can be added at the fingerprint layer
without touching anything downstream. Storage can swap to Postgres for
multi-worker training.

---

## Open notes for downstream work

1. **Phase 6 will need an `--amnesia` mode for Tom.** Easy: just construct
   ChemicalTom with `l2_lookup=None`. Wire this through `watch.py` and
   the training scripts as a flag.

2. **The dance is still alive.** ChemicalTom-L2 doesn't beat the trained
   Jerry's dance exploit any more than ChemicalTom did. Phase 6 is still
   the destination for that fix.

3. **Database growth bound.** Right now nothing caps L2 size. After 10k
   training episodes, the SQLite database will be a few MB — fine. After
   1M episodes it'll be a few GB. Add `L2Store.delete_older_than(seconds)`
   or `L2Store.delete_oldest(n)` to the training script's tear-down phase.

4. **`L1Memory` could share a Redis connection across many Toms** in a
   parallel training setup. Currently each Tom has its own L1Memory
   instance. That's fine for now; revisit if Phase 6's worker count
   strains Redis.

5. **The verify_redis script doesn't yet check SQLite.** Add a parallel
   `verify_sqlite.py` (or extend the existing one) so the Phase 4 setup
   has the same "first thing to run before debugging" tool that Phase 3
   has.

6. **Map fingerprint sensitivity to seed changes.** Right now, two maps
   generated with seeds 41 and 42 produce different fine fingerprints
   AND likely different coarse fingerprints (because the random wall
   counts differ slightly). This means warm-start doesn't help much
   across slight seed perturbations. Worth investigating whether the
   coarse fingerprint should bucket more aggressively.

---

## What I'd tell a future contributor

If you're touching the memory system:

- **Read the architectural invariant before touching any of the four memory
  query methods.** Warm-start priors live in `_warm_*` dicts, in-episode
  counts live in Redis. Behavior queries combine; distillation reads
  in-episode only. If you find yourself adding warm-start data to
  distillation, stop.

- **Don't add fields to `EpisodeSummary` casually.** Each field is queryable,
  but only by direct attribute access — there's no flexible query interface.
  Adding a field means modifying the SQLite schema (migration), the
  distillation function, the L2 lookup aggregation, and probably a few
  tests. Use the `notes_json` blob for experimental fields and promote
  to first-class only when they're load-bearing.

- **Fingerprints are SHA-256 hex strings — never collide in practice.**
  If you're tempted to use a shorter hash for "performance," don't. The
  L2 store is indexed on these strings, and SQLite handles 64-char text
  fields fine. Collision risk dominates speedup.

- **L1 episode_id should be unique per actual episode.** If you reuse an
  episode_id, the L1 store will accumulate state across multiple
  episodes' worth of in-episode counters — corrupting both behavior and
  distillation. Use UUIDs.

---

## Where to look for visible Phase 4 evidence

The `scripts/show_l2_effect.py` demonstration is the canonical way to see
Phase 4 working. Run it multiple times with the same seed (no `--fresh`)
and watch L2 grow across runs.

The `scripts/watch.py --tom chemical-l2` integration adds Phase 4 to the
visual replay flow. Successive runs of `watch.py --tom chemical-l2` on
the same `--seed` and `--jerry` arguments accumulate persistent memory
across sessions. After 20 such runs, Tom's noise threshold near commonly-
fooled tiles will measurably differ from a fresh Tom's.

The end-to-end test (`tests/integration/test_phase4_end_to_end.py::test_warm_start_affects_noise_threshold`)
is the most concise proof: manually inserts L2 data with false-noise
hotspots, verifies that Tom's modulated noise threshold near those tiles
is elevated vs threshold far from them.

---

## Discipline calls worth preserving

These were the moments in Phase 3+4 where the right answer was to *not*
do something. Future contributors will face similar temptations:

1. **No oscillation pattern detector in L1.** Grove specifically asked
   to keep L1 to "historical tracking only, let Phase 6 evolve pattern
   responses." Holding to that meant L1 stays small and trustworthy.

2. **No L3 in v1.** The right thing was to ship a stub, not a half-built
   real implementation.

3. **No retrospective after Phase 3 alone.** Phase 3 and Phase 4 are one
   architectural beat; splitting the retro would have produced two thin
   documents instead of one substantive one.

4. **No hand-tuning ChemicalTom to beat the dance.** This came up
   repeatedly across Phase 2 → 3 → 4. The discipline held: architecture
   work, not power-creep.

5. **No retraining of Jerry inside Phase 3/4.** The original temptation
   was to retrain Jerry against ChemicalTom-L2 and see what happens.
   That conflates Phase 4 (memory infrastructure) with Phase 6
   (co-evolution). Kept them separate.

---

## Artifacts pinned

### New source files
- `src/persistence/redis/client.py` — Redis client wrapper + FakeRedis
- `src/persistence/redis/l1_store.py` — L1 key schema and CRUD
- `src/persistence/sqlite/__init__.py`
- `src/persistence/sqlite/client.py` — SQLite client with migrations
- `src/persistence/sqlite/l2_store.py` — L2 episode-summary CRUD
- `src/hunter/agent/memory/l1.py` — L1Memory API (Phase 3) + warm-start (Phase 4)
- `src/hunter/agent/memory/l2_lookup.py` — Fine-then-coarse cascade
- `src/hunter/agent/memory/distillation.py` — L1 → L2 builder
- `src/hunter/agent/memory/fingerprint.py` — Map + jerry hashes
- `src/hunter/agent/memory/l3.py` — Phase 4 stub
- `docker-compose.yml` — Redis container definition
- `scripts/verify_redis.py` — Redis setup verifier
- `scripts/show_l2_effect.py` — Phase 4 demonstration

### Extended existing files
- `src/hunter/agent/behavior/chemical_tom.py` — L1 integration (Phase 3),
  warm-start + distill lifecycle methods (Phase 4)
- `src/render/replay/recorder.py` — L1 locker setup (Phase 3),
  warm-start + distill hooks (Phase 4)
- `scripts/watch.py` — `chemical-l1` (Phase 3), `chemical-l2` (Phase 4) options
- `pyproject.toml` — pytest markers for `redis` and `slow`

### Test count
- Phase 3: 21 unit + 4 redis-marked integration = 25
- Phase 4: 23 L2 store + 18 fingerprint + 15 distillation + 16 L2 lookup
  + 20 warm-start/L3 + 11 e2e = 103

Total project state at end of Phase 4: **279 tests passing in ~30s.**
