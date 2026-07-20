from __future__ import annotations

from datetime import datetime, timedelta, timezone

from recall.embeddings import Embedder
from recall.guards import DEFAULT_GAP_THRESHOLD, gap_warning, staleness
from recall.rerank import Reranker
from recall.store import PgVectorStore
from recall.types import RetrievalResult, ScoredChunk


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse several best-first ID rankings into one score map (Reciprocal Rank Fusion).

    Each input list is an independent ranking, best first. Every id accrues
    ``1 / (k + rank)`` from each list it appears in; `k` (default 60, the standard RRF
    damping constant — unrelated to the caller's result-count `k`) softens the weight of
    top ranks so no single ranking dominates. The returned dict is UNSORTED; callers sort
    by value descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    """Hybrid dense + sparse retrieval with the self-recall honesty guards.

    Fuses pgvector cosine search (dense) and Postgres full-text search (sparse) via
    Reciprocal Rank Fusion, then annotates the result with a corpus-gap warning and a
    staleness report.

    Tunables:
      gap_threshold: dense cosine below which the corpus is treated as lacking an answer.
      max_age:       index age beyond which results are flagged stale.
      candidate_k:   how many candidates each of dense/sparse contributes before fusion.
      use_sparse:    include the sparse full-text leg in fusion; False = dense-only (ablations).
    """

    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
        max_age: timedelta = timedelta(days=2),
        candidate_k: int = 20,
        use_sparse: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._gap_threshold = gap_threshold
        self._max_age = max_age
        self._candidate_k = candidate_k
        self._use_sparse = use_sparse

    def search(self, query: str, k: int = 5, source: str | None = None) -> RetrievalResult:
        """Retrieve the top-`k` chunks for `query` (optionally filtered to one `source`).

        `k` must be >= 1 (a negative k would silently slice from the wrong end).

        `gap_warning` is computed from the DENSE cosine scores only (not the fused ranks),
        so a purely lexical / sparse-only match still reports a gap — the intended "honest
        about what it doesn't know" behavior. Each hit's `score` is its true dense cosine
        similarity, including hits that arrived via the sparse leg.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        qvec = self._embedder.embed([query])[0]
        dense = self._store.query_dense(qvec, k=self._candidate_k, source=source)
        sparse = (
            self._store.query_sparse(query, k=self._candidate_k, source=source, vec=qvec)
            if self._use_sparse
            else []
        )

        fused = _rrf([[h.chunk.id for h in dense], [h.chunk.id for h in sparse]])
        by_id = {h.chunk.id: h for h in dense}
        for h in sparse:
            by_id.setdefault(h.chunk.id, h)  # sparse hits carry their true cosine (vec=qvec)
        dense_score = {h.chunk.id: h.score for h in dense}

        # Rerank the WHOLE fused candidate pool, then truncate to k — slicing to k first would
        # hide a relevant doc sitting just below the fused cutoff from the cross-encoder, which
        # is exactly the doc reranking exists to rescue.
        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        hits = [
            ScoredChunk(
                chunk=by_id[cid].chunk,
                score=dense_score.get(cid, by_id[cid].score),
                indexed_at=by_id[cid].indexed_at,
            )
            for cid in ranked_ids
        ]
        if self._reranker is not None:
            hits = self._reranker.rerank(query, hits)
        hits = hits[:k]

        gap = gap_warning(list(dense_score.values()), self._gap_threshold)
        stale = staleness(self._store.newest_indexed_at(), datetime.now(timezone.utc), self._max_age)
        return RetrievalResult(query=query, hits=hits, gap_warning=gap, staleness=stale)
