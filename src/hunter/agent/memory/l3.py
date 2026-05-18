"""L3Memory — semantic/episodic recall (Phase 4 STUB).

Phase 4 ships an interface-only stub. The full implementation will use
ChromaDB + embeddings to support fuzzy retrieval like "this map feels
like that one where Jerry preferred the southwest corner."

L2 already provides retrieval by exact and structural map fingerprints.
L3's job is to handle the cases L2 can't:

  - Cross-map jerry behavior patterns: "I've seen this jerry route
    south-then-east-then-vent on three different maps."
  - Episode-shape similarity: "this episode's first 30 ticks feel
    like ep_47, where Jerry surprised me with a noisemaker at tick 35."

The interface below is what Phase 5+ code can already use. Method calls
return safe defaults (empty list, None, no-ops). When we plug in real
embeddings, callers don't change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class L3Config:
    """Tunable knobs for the eventual real L3.

    Defaults are placeholders. When ChromaDB lands these become live.
    """
    # Number of similar episodes to retrieve in a query.
    top_k: int = 5

    # Cosine similarity threshold below which results are filtered out.
    # Higher = stricter matching, fewer false positives.
    min_similarity: float = 0.6

    # Embedding model name — sentence-transformers convention.
    # Pinned here so Phase 5+ knows the dependency without surprises.
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass(slots=True)
class EpisodeMemory:
    """A single recalled episodic memory. Phase 4: never constructed.

    Phase 5+ will populate `summary_id` (pointing at the L2 row) and
    a `similarity` score from the embedding query.
    """
    summary_id: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)


class L3Memory:
    """STUB. All methods are no-ops in Phase 4.

    Tom and the rest of the codebase can hold an `L3Memory` reference
    and call its methods unconditionally — the stub returns safe defaults
    so the call sites don't need conditionals. When the real implementation
    lands, behavior changes without code changes elsewhere.
    """

    def __init__(self, config: L3Config | None = None):
        self.config = config or L3Config()
        # The stub holds no state. The real implementation will own a
        # Chroma client and an embedding model.
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """True iff the real L3 backend is wired in. Always False in Phase 4."""
        return self._enabled

    # ---- write side --------------------------------------------------

    def index_episode(self, summary_id: str, embedding_text: str,
                      metadata: dict | None = None) -> None:
        """Add a finished episode to the L3 index.

        Phase 4: no-op. Phase 5+: embed `embedding_text` and store the
        vector alongside `summary_id` + metadata in ChromaDB.
        """
        # Intentionally empty.
        return None

    # ---- read side ---------------------------------------------------

    def recall_similar(
        self,
        query_text: str,
        top_k: int | None = None,
    ) -> list[EpisodeMemory]:
        """Retrieve up to top_k past episodes most similar to query_text.

        Phase 4: returns []. Phase 5+: embeds query_text and finds the
        nearest neighbors above the similarity threshold.
        """
        return []

    def recall_for_map_and_jerry(
        self,
        map_summary: str,
        jerry_summary: str,
        top_k: int | None = None,
    ) -> list[EpisodeMemory]:
        """Convenience query: find episodes with similar map+jerry context.

        Phase 4: returns []. Phase 5+: combines map and jerry text into
        a single embedding query.
        """
        return []

    # ---- maintenance -------------------------------------------------

    def count(self) -> int:
        """How many episodes are indexed. Always 0 in Phase 4."""
        return 0

    def clear(self) -> None:
        """Wipe the L3 index. Phase 4: no-op."""
        return None
