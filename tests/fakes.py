"""Minimal duck-typed store and embedder for retriever tests. No database."""

from __future__ import annotations

from datetime import UTC, datetime

from recall.types import Chunk, ScoredChunk


class FakeEmbedder:
    dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]


class FakeStore:
    def __init__(
        self,
        dense: list[tuple[str, float]] | None = None,
        sparse: list[tuple[str, float]] | None = None,
        learned: list[tuple[str, float]] | None = None,
    ) -> None:
        self._dense = dense or []
        self._sparse = sparse or []
        self._learned = learned or []
        self.sparse_vec_used: list[float] | None = None
        self.learned_vec_used: list[float] | None = None
        self.cosines_calls: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
        #: `source` as received on each `query_dense` call, in call order. Dense is queried on
        #: every leg regardless of `sparse_backend`, so this is the one place a caller scoped to
        #: one source can check that BOTH the query leg and the history leg honoured that scope.
        self.dense_sources: list[str | None] = []

    @staticmethod
    def _hits(rows: list[tuple[str, float]]) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=Chunk(id=cid, source="s", text=f"text-{cid}", metadata={}),
                score=score,
            )
            for cid, score in rows
        ]

    def query_dense(self, vector, k, source=None):
        self.dense_sources.append(source)
        return self._hits(self._dense[:k])

    def query_sparse(self, text, k, source=None, vec=None):
        self.sparse_vec_used = vec
        return self._hits(self._sparse[:k])

    def query_learned_sparse(self, weights, k, profile_id, source=None, vec=None):
        self.learned_vec_used = vec
        return self._hits(self._learned[:k])

    def cosines_for(self, ids, vec):
        self.cosines_calls.append((tuple(ids), tuple(vec)))
        return {cid: 0.77 for cid in ids}

    def newest_indexed_at(self):
        return datetime.now(UTC)


class VectorKeyedFakeStore(FakeStore):
    """A `FakeStore` whose dense leg varies BY QUERY VECTOR, unlike the base class, which returns
    the same fixed rows no matter what vector it is asked to search.

    Needed to build a scenario where the query's dense leg and the history's dense leg must return
    genuinely different rows, for example to pin `gap_warning` to the query leg specifically: with
    `FakeStore`'s fixed dense leg, the query and history variants are indistinguishable, so a
    mutation that fed the history leg into the gap computation would still pass.

    A vector with no entry in `dense_by_vector` reads as no dense hits, matching the base class's
    behaviour for any leg the caller did not script.
    """

    def __init__(
        self,
        dense_by_vector: dict[tuple[float, ...], list[tuple[str, float]]],
        sparse: list[tuple[str, float]] | None = None,
        learned: list[tuple[str, float]] | None = None,
    ) -> None:
        super().__init__(sparse=sparse, learned=learned)
        self._dense_by_vector = dense_by_vector

    def query_dense(self, vector, k, source=None):
        self.dense_sources.append(source)
        return self._hits(self._dense_by_vector.get(tuple(vector), [])[:k])


class FakeExtractionEngine:
    """A truth extraction engine that returns scripted output and counts its calls.

    The call counter is the point. Extraction is cached, and a cache that silently stopped
    answering would look identical to one that never stopped — the same output, quietly
    re-paid for. Assert on `call_count` to pin whether the cache answered.

    `responses` maps a file name to the raw string the engine returns for it. Unscripted
    files get `default`, which is a well formed empty batch, so a test that only cares about
    one file does not have to script the rest of the corpus.
    """

    engine_id = "tests.fake_extraction"
    model_id = "fake"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        default: str = '{"claims": []}',
        revision: str = "fake-v1",
    ) -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.revision = revision
        #: Every prompt received, in call order.
        self.calls: list[object] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def run(self, prompt) -> str:
        self.calls.append(prompt)
        return self.responses.get(prompt.file, self.default)
