"""Systems whose behaviour is known by construction, for the eval calibration fleet.

Nothing here ships in the wheel. See docs/EVAL_CALIBRATION_FLEET_DESIGN.md.
"""
from __future__ import annotations

from datetime import UTC, datetime

from recall.types import Chunk, ScoredChunk


class ScriptedEmbedder:
    """An embedder that records the last text it was asked to encode.

    The store needs the QUERY TEXT to decide what to return, and `query_dense` receives only a
    vector. Rather than encode the query into the vector and decode it back (which
    `tests/fakes.py:VectorKeyedFakeStore` does, at the cost of unreadable fixtures), the
    embedder and the store share this object. `HybridRetriever._retrieve_legs` calls
    `embed_query` BEFORE `query_dense` on every search, so `last_query` is always the query
    currently being served.

    `name` is set explicitly to satisfy the `Embedder` protocol, whose docstring documents it as
    identifying the backend, used in logging and evals. Two call sites read it, and it is worth
    being precise about which one does what.

    `_score_config` builds the returned `AblationResult` with
    `embedder=embedding_profile_id(embedder)` (`harness.py:167`), using this raw, unwrapped
    instance, never the `TimedEmbedder` wrapper. Without `name`, `embedding_profile_id` would
    fall through `profile` (also absent here) to `type(embedder).__name__`, so the label would
    read "ScriptedEmbedder", not "TimedEmbedder".

    The label "TimedEmbedder" is reachable, but through a different field entirely:
    `HybridRetriever` builds `RetrievalDiagnostics.embedding_profile` from `self._embedder`,
    which `_score_config` sets to the `TimedEmbedder` wrapper. `TimedEmbedder.name` is a property
    delegating to `self._inner.name`; if `ScriptedEmbedder` had no `name`, that property access
    would raise `AttributeError`, which `embedding_profile_id`'s own `getattr(obj, "name", None)`
    swallows, falling back to `type(embedder).__name__` on the wrapper this time. That is the
    one field that would actually say "TimedEmbedder", and the fleet never reads it.

    So the attribute is kept for a plainer reason: `ScriptedEmbedder` claims to satisfy the
    `Embedder` protocol, and an implementation missing a property the protocol documents is a
    fake that fails its own interface quietly.
    """

    dim = 4
    name = "scripted"

    def __init__(self) -> None:
        self.last_query: str | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts:
            self.last_query = texts[-1]
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class QueryKeyedStore:
    """A store whose dense leg is scripted per QUERY TEXT.

    `script` maps a query string to the rows it retrieves, in order, as `(chunk_id, score)`.
    `chunk_id` is the `file:ord` form `harness._key` rebuilds from chunk metadata, so a member
    writes the same string it puts in `relevant_ids`.

    Two things are controlled at once and both matter:

      * the ORDER of the ids, which drives precision, recall, MRR and nDCG;
      * the dense SCORES, which drive `gap_warning`. `HybridRetriever` computes it as
        `max(dense_scores) < 0.50` (`recall/guards.py`), so a member picks scores either side
        of 0.50 to drive it deterministically.

    The sparse legs return nothing. The fleet drives `_score_config` with `fusion="dense"`,
    which sets `use_sparse=False`, so RRF runs over a single list and preserves the scripted
    order exactly. Under a hybrid fusion the fused score would reorder the hits and every
    closed form in `members.py` would silently stop holding.
    """

    def __init__(
        self, embedder: ScriptedEmbedder, script: dict[str, list[tuple[str, float]]]
    ) -> None:
        self._embedder = embedder
        self._script = script

    @staticmethod
    def _chunk(chunk_id: str) -> Chunk:
        # `harness._key` rebuilds the id as f"{metadata['file']}:{metadata['ord']}" and raises
        # KeyError on the empty metadata `tests/fakes.py:FakeStore` supplies, so splitting the
        # id back into its two parts here is load-bearing, not cosmetic.
        file, _, ordinal = chunk_id.rpartition(":")
        return Chunk(
            id=chunk_id,
            source="fleet",
            text=f"text-{chunk_id}",
            metadata={"file": file, "ord": int(ordinal)},
        )

    def query_dense(self, vector, k, source=None):
        query = self._embedder.last_query
        if query not in self._script:
            raise KeyError(
                f"no script for query {query!r}. A fleet member must script every query it "
                f"asks about: an unscripted query returning no rows would score as a total "
                f"retrieval failure, which is a fixture bug that reads exactly like a defect."
            )
        return [
            ScoredChunk(chunk=self._chunk(cid), score=score)
            for cid, score in self._script[query][:k]
        ]

    def query_sparse(self, text, k, source=None, vec=None):
        return []

    def query_learned_sparse(self, weights, k, profile_id, source=None, vec=None):
        return []

    def newest_indexed_at(self):
        return datetime.now(UTC)
