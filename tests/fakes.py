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
