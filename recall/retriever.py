from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from recall.embeddings import Embedder
from recall.fusion import leg_confidence, weighted_rrf
from recall.guards import DEFAULT_GAP_THRESHOLD, gap_warning, staleness
from recall.rerank import Reranker
from recall.store import PgVectorStore
from recall.types import RetrievalResult, ScoredChunk

#: Default candidate pool per retrieval leg before fusion. Exposed as a module constant (not only a
#: signature default) so the eval harness references the SAME number instead of a hardcoded copy:
#: this value BINDS the depth curve — the fused pool holds at most ``2 * candidate_k`` distinct
#: chunks before truncation to k, so hit@k stops rising once k reaches the pool. A curve run past it
#: measures the pool, not the depth, so the eval must use exactly this default.
DEFAULT_CANDIDATE_K = 20


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse several best-first ID rankings into one score map (Reciprocal Rank Fusion).

    The plain, equal-weight case of `recall.fusion.weighted_rrf` — kept as a thin wrapper
    (rather than inlining the formula here again) because `tests/test_hybrid_fusion_contract.py`
    imports this name directly and pins its behaviour. Uniform weights reproduce the shipped
    ORDER exactly: every score is scaled by the same constant, which cannot reorder anything.
    The returned dict is UNSORTED; callers sort by value descending.
    """
    return weighted_rrf(rankings, k=k)


@dataclass(frozen=True)
class LegProbe:
    """One search's raw per-leg evidence, handed to an optional observer.

    Exists so a diagnostic can read the legs the REAL pipeline produced. Reconstructing them
    in an eval harness would measure a copy of the retriever rather than the retriever.

    `sparse_ranks` carries `ts_rank` — the sparse leg's own ranking score. It is NOT
    `[h.score for h in sparse]`: those are dense cosines, because `query_sparse` rescores its
    hits against the query vector so lexical-only hits stay comparable downstream.
    """

    query: str
    dense: list[ScoredChunk]        # dense candidates, best-first; score = cosine
    sparse: list[ScoredChunk]       # sparse candidates, best-first by ts_rank; score = cosine
    sparse_ranks: list[float]       # ts_rank per sparse hit, same order as `sparse`
    fused: list[ScoredChunk]        # post-fusion, pre-rerank, pre-truncation


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
      use_dense:     include the dense vector leg in fusion; False = sparse-only (ablations).
      fusion:        'rrf' (default, shipped) weights both legs equally; 'wrrf' weights each leg
                     by its per-query decisiveness (recall.fusion.leg_confidence). Equal
                     decisiveness makes the two identical.
      probe:         optional observer called once per search with the raw per-leg candidates.
                     Diagnostics only — it cannot influence the result, and the default (None)
                     leaves the query path byte-identical.

    ``use_dense=False`` is an ABLATION SWITCH, not a serving mode. The query is still embedded
    (the sparse leg reports each hit's true cosine), but `gap_warning` is computed from the
    DENSE candidate scores, and with no dense leg there are none — so it fires on every query,
    because `gap_warning` treats an empty candidate set as a gap. That is the fail-closed
    direction and it is deliberate, but it makes the flag uninformative on this arm: it means
    "no dense evidence was gathered", not "the corpus lacks an answer". Ablation results should
    read the hits, not the warning.
    """

    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
        max_age: timedelta = timedelta(days=2),
        candidate_k: int = DEFAULT_CANDIDATE_K,
        use_sparse: bool = True,
        use_dense: bool = True,
        fusion: str = "rrf",
        probe: Callable[[LegProbe], None] | None = None,
    ) -> None:
        if not (use_dense or use_sparse):
            raise ValueError("at least one of use_dense / use_sparse must be True")
        if fusion not in ("rrf", "wrrf"):
            raise ValueError(f"fusion must be 'rrf' or 'wrrf', got {fusion!r}")
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._gap_threshold = gap_threshold
        self._max_age = max_age
        self._candidate_k = candidate_k
        self._use_sparse = use_sparse
        self._use_dense = use_dense
        self._fusion = fusion
        self._probe = probe

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
        dense = (
            self._store.query_dense(qvec, k=self._candidate_k, source=source)
            if self._use_dense
            else []
        )
        sparse_ranks: list[float] = []
        if self._use_sparse:
            if self._probe is not None:
                # `with_rank` is passed ONLY when probing: store doubles in the test suite (and
                # any third-party PgVectorStore-shaped object) implement the 4-argument form.
                sparse, sparse_ranks = self._store.query_sparse(
                    query, k=self._candidate_k, source=source, vec=qvec, with_rank=True
                )
            else:
                sparse = self._store.query_sparse(
                    query, k=self._candidate_k, source=source, vec=qvec
                )
        else:
            sparse = []

        dense_ranking = [h.chunk.id for h in dense]
        sparse_ranking = [h.chunk.id for h in sparse]
        if self._fusion == "wrrf":
            # Each leg is scored on its OWN units; `leg_confidence` is affine-invariant, which is
            # what makes a cosine leg and a ts_rank leg comparable. `sparse_ranks` is only
            # populated when probing, so fall back to the sparse hits' cosines otherwise — both
            # are that leg's ordering evidence, and the z-score is scale-free either way.
            c_dense = leg_confidence([h.score for h in dense])
            c_sparse = leg_confidence(sparse_ranks or [h.score for h in sparse])
            total = max(c_dense, 0.0) + max(c_sparse, 0.0)
            weights = (
                [max(c_dense, 0.0) / total, max(c_sparse, 0.0) / total] if total > 0 else [0.5, 0.5]
            )
        else:
            weights = [0.5, 0.5]
        fused = weighted_rrf([dense_ranking, sparse_ranking], weights=weights)
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
        if self._probe is not None:
            self._probe(
                LegProbe(
                    query=query,
                    dense=list(dense),
                    sparse=list(sparse),
                    sparse_ranks=list(sparse_ranks),
                    fused=list(hits),
                )
            )
        if self._reranker is not None:
            hits = self._reranker.rerank(query, hits)
        hits = hits[:k]

        gap = gap_warning(list(dense_score.values()), self._gap_threshold)
        stale = staleness(self._store.newest_indexed_at(), datetime.now(timezone.utc), self._max_age)
        return RetrievalResult(query=query, hits=hits, gap_warning=gap, staleness=stale)
