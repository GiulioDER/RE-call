from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import os
import re
import time

from recall.embeddings import Embedder, embed_query, embedding_profile_id
from recall.guards import DEFAULT_GAP_THRESHOLD, gap_warning, staleness
from recall.rerank import Reranker
from recall.store import PgVectorStore
from recall.types import RetrievalDiagnostics, RetrievalResult, ScoredChunk


def _rescored(hit: ScoredChunk, score: float) -> ScoredChunk:
    """The hit with a new score, keeping every other field.

    `replace` for the real dataclass, so a field added to `ScoredChunk` later cannot be dropped
    the way `first_indexed_at` silently was. The fallback is not decoration: stores and their
    hits are duck-typed by tests and downstream adapters, and `replace` raises TypeError on a
    non-dataclass, so without it this would turn a working search into a crash — the exact
    hazard `recall.trust` goes out of its way to avoid one layer up.
    """
    if is_dataclass(hit) and not isinstance(hit, type):
        return replace(hit, score=score)
    return ScoredChunk(
        chunk=hit.chunk,
        score=score,
        indexed_at=getattr(hit, "indexed_at", None),
        first_indexed_at=getattr(hit, "first_indexed_at", None),
    )

#: Default candidate pool per retrieval leg before fusion. Exposed as a module constant (not only a
#: signature default) so the eval harness references the SAME number instead of a hardcoded copy:
#: this value BINDS the depth curve — the fused pool holds at most ``2 * candidate_k`` distinct
#: chunks before truncation to k, so hit@k stops rising once k reaches the pool. A curve run past it
#: measures the pool, not the depth, so the eval must use exactly this default.
DEFAULT_CANDIDATE_K = 20

#: Which retriever fills the sparse leg.
#:
#: `lexical` is Postgres full-text (`ts_rank`) and remains the DEFAULT, so nothing about existing
#: behaviour changes by adding this. `splade` REPLACES it with learned sparse — the clean A/B, and
#: the shape MTRAGEval's rank-2 system used (one learned sparse retriever, not an ensemble).
#: `both` fuses three legs; it is declared here rather than discovered later so that running it is
#: a pre-registered arm and not a post-hoc rescue after seeing the scores.
SPARSE_BACKENDS = frozenset({"lexical", "splade", "both"})

#: The most distinct chunks a fused search hands the cross-encoder.
#:
#: 🔑 MEASURED, not chosen for tidiness. Reranking a 547-candidate pool LOST 0.0513 R@100 where
#: reranking 200 GAINED 0.0226: the cross-encoder degrades as the pool widens, which
#: `closed-hypothesis-recall-rerank-pool-interaction-2026-08-05` recorded as "not more to select
#: from, more rope". Fusing two queries roughly doubles the pool, so without this cap the measured
#: gain reverses.
FUSED_RERANK_POOL_CAP = 100

#: Longest history CONCATENATION a fused search accepts, in characters.
#:
#: Matches `recall_mcp.service.MAX_QUERY_CHARS` so the library and the server agree on what an
#: over-long query is. Measured against the concatenation, which is built here and therefore never
#: passes the server's own check on the incoming query.
#:
#: ⚠️ The encoder truncates at 512 tokens regardless, so a history past roughly 2,000 characters
#: is already being clipped by the tokenizer. That clipping was PRESENT IN THE MEASUREMENT, so it
#: is part of the measured system rather than something introduced here. This budget bounds the
#: REQUEST, not the encoded query, and the two are different limits.
FUSED_HISTORY_MAX_CHARS = 4096


#: A leading MTRAG speaker tag: a pipe, the speaker name (no pipes inside it), a closing pipe, a
#: colon, then optional whitespace. Anchored to the start of the line so a line that merely begins
#: with a pipe and contains some later colon (a pasted markdown table row, a literal pipe in the
#: text) is not mistaken for a tag.
_SPEAKER_TAG = re.compile(r"^\|[^|]*\|:\s*")


def build_history_query(history: "Sequence[str]") -> str:
    """The benchmark's `full` variant: prior turns, newline joined, speaker tags removed.

    Owned here rather than left to callers so that two installations cannot send different
    concatenations and both call it the measured configuration.

    The tag match is anchored (`^\\|[^|]*\\|:\\s*`) rather than the looser "starts with a pipe and
    contains a colon somewhere" check the benchmark's `strip_speaker` used: that looser rule strips
    everything up to the FIRST colon on any line beginning with a pipe, which also eats real content
    on lines that merely start with a pipe (a pasted table row, a literal pipe). The two rules were
    compared against all 5,016 lines of the real MTRAG data, across all three query files and all
    four domains, and they disagree on zero of them: the tightened rule is bit-identical to the
    measured one on that corpus and only diverges on inputs the benchmark never contained.
    """
    lines = []
    for turn in history:
        for line in str(turn).splitlines():
            stripped = _SPEAKER_TAG.sub("", line, count=1).strip()
            if stripped:
                lines.append(stripped)
    return "\n".join(lines)


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


@dataclass(frozen=True)
class _Legs:
    """One query's retrieval legs, before any fusion.

    The seam `search` and `search_fused` share. Extracted rather than duplicated because two
    copies of this pipeline would drift, and the drift would be invisible: both would still
    return plausible hits.
    """

    qvec: list[float]
    dense: list[ScoredChunk]
    sparse: list[ScoredChunk]
    learned: list[ScoredChunk]
    timings: dict[str, float]


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
        sparse_backend: str = "lexical",
        sparse_encoder: object | None = None,
        retrieval_profile: str = "legacy",
        index_generation: str = "legacy",
    ) -> None:
        if not (use_dense or use_sparse):
            raise ValueError("at least one of use_dense / use_sparse must be True")
        if sparse_backend not in SPARSE_BACKENDS:
            raise ValueError(
                f"sparse_backend must be one of {sorted(SPARSE_BACKENDS)}, got {sparse_backend!r}"
            )
        wants_learned = sparse_backend in ("splade", "both")
        if wants_learned and sparse_encoder is None:
            raise ValueError(
                f"sparse_backend={sparse_backend!r} needs a sparse_encoder to encode queries. "
                f"Falling back to lexical would hand an operator who asked for learned sparse a "
                f"keyword search, and a set of numbers that look like a SPLADE result."
            )
        if wants_learned and os.environ.get("RECALL_ENV") == "production":
            raise RuntimeError(
                "the learned sparse leg is not available under RECALL_ENV=production: its "
                "sidecar (recall_sparse_v1) has no foreign key to the chunk table and the "
                "erasure paths (generations.forget, control_plane.erase_sources_from_pending) "
                "predate it, so a forgotten chunk would leave its SPLADE weights behind. Those "
                "weights are term weights over the vocabulary, i.e. partially reconstructable "
                "content. Wiring erasure is the precondition for lifting this refusal. "
                "See docs/superpowers/specs/2026-08-06-learned-sparse-splade-design.md."
            )
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._gap_threshold = gap_threshold
        self._max_age = max_age
        self._candidate_k = candidate_k
        self._use_sparse = use_sparse
        self._use_dense = use_dense
        self._sparse_backend = sparse_backend
        self._sparse_encoder = sparse_encoder
        self._retrieval_profile = retrieval_profile
        self._index_generation = index_generation

    def _retrieve_legs(
        self, query: str, source: str | None, report_vec: list[float] | None = None
    ) -> _Legs:
        """Run every enabled leg for one query.

        `report_vec` overrides which vector the SPARSE legs report their cosine against, without
        changing what they rank by. `search` leaves it None and gets the pre-existing behaviour.
        `search_fused` passes the QUERY's vector when retrieving for the HISTORY variant, so that
        every returned hit's score is on one basis: `trust.py` feeds `hit.score` to a calibrated
        confidence, and a cosine against a different string would silently mean something else.
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()
        qvec = embed_query(self._embedder, query)
        timings["query_embedding"] = (time.perf_counter() - started) * 1000.0
        reporting = qvec if report_vec is None else report_vec

        started = time.perf_counter()
        dense = (
            self._store.query_dense(qvec, k=self._candidate_k, source=source)
            if self._use_dense
            else []
        )
        timings["dense_retrieval"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        # `lexical` and `both` include the ts_rank leg; `splade` REPLACES it. That replacement is
        # the point of the arm — MTRAGEval's rank 1 reports that adding retrievers HURT once the
        # strong one worked, because unique documents from the weak leg land at ranks 37-54 and
        # inject fusion noise. So this is a swap by default, not an addition.
        wants_lexical = self._use_sparse and self._sparse_backend in ("lexical", "both")
        sparse = (
            self._store.query_sparse(query, k=self._candidate_k, source=source, vec=reporting)
            if wants_lexical
            else []
        )
        timings["sparse_retrieval"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        learned: list[ScoredChunk] = []
        if self._use_sparse and self._sparse_backend in ("splade", "both"):
            encoder = self._sparse_encoder
            assert encoder is not None  # guaranteed by __init__; re-stated for the type checker
            weights = encoder.encode([query])[0]  # type: ignore[attr-defined]
            if weights:
                learned = self._store.query_learned_sparse(
                    weights,
                    k=self._candidate_k,
                    profile_id=encoder.profile.profile_id,  # type: ignore[attr-defined]
                    source=source,
                    vec=reporting,
                )
            # An empty query encoding is NOT an error here, unlike in the store: a query of pure
            # stopwords legitimately produces no terms. The leg contributes nothing and says so
            # by way of `learned_sparse_terms` below, rather than by raising on a valid question.
        timings["learned_sparse_retrieval"] = (time.perf_counter() - started) * 1000.0
        return _Legs(qvec=qvec, dense=dense, sparse=sparse, learned=learned, timings=timings)

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
        legs = self._retrieve_legs(query, source)
        timings = dict(legs.timings)
        dense, sparse, learned = legs.dense, legs.sparse, legs.learned

        started = time.perf_counter()
        fused = _rrf(
            [
                [h.chunk.id for h in dense],
                [h.chunk.id for h in sparse],
                [h.chunk.id for h in learned],
            ]
        )
        by_id = {h.chunk.id: h for h in dense}
        for h in sparse:
            by_id.setdefault(h.chunk.id, h)  # sparse hits carry their true cosine (vec=qvec)
        for h in learned:
            by_id.setdefault(h.chunk.id, h)
        dense_score = {h.chunk.id: h.score for h in dense}

        # Rerank the WHOLE fused candidate pool, then truncate to k — slicing to k first would
        # hide a relevant doc sitting just below the fused cutoff from the cross-encoder, which
        # is exactly the doc reranking exists to rescue.
        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        # `replace`, not a freshly-listed ScoredChunk. Enumerating the carried fields by hand
        # silently DROPPED `first_indexed_at` the day it was added: every production hit reached
        # the trust layer with it None, fell back to `indexed_at`, and the entire point-in-time
        # fix was inert on the only read path that matters — with the suite still green, because
        # no test built a hit through a retriever. Copying the hit and overriding the one field
        # that actually changes cannot lose a field added later.
        hits = [_rescored(by_id[cid], dense_score.get(cid, by_id[cid].score)) for cid in ranked_ids]
        timings["fusion"] = (time.perf_counter() - started) * 1000.0
        reranking_ran = self._reranker is not None
        started = time.perf_counter()
        if self._reranker is not None:
            hits = self._reranker.rerank(query, hits)
        timings["reranking"] = (time.perf_counter() - started) * 1000.0
        hits = hits[:k]

        gap = gap_warning(list(dense_score.values()), self._gap_threshold)
        stale = staleness(self._store.newest_indexed_at(), datetime.now(timezone.utc), self._max_age)
        return RetrievalResult(
            query=query,
            hits=hits,
            gap_warning=gap,
            staleness=stale,
            diagnostics=RetrievalDiagnostics(
                embedding_profile=embedding_profile_id(self._embedder),
                retrieval_profile=self._retrieval_profile,
                index_generation=self._index_generation,
                candidate_pool_size=self._candidate_k,
                reranking_ran=reranking_ran,
                stage_ms={key: round(value, 3) for key, value in timings.items()},
            ),
        )

    def search_fused(
        self,
        query: str,
        history: "Sequence[str]",
        k: int = 5,
        source: str | None = None,
    ) -> RetrievalResult:
        """Retrieve for `query` fused with a concatenation of `history`, then rerank once.

        Reproduces the `mq_nested2_nogold` arm measured on 2026-08-06/07: +0.0084 nDCG@5
        (MiniLM, Holm-significant) and +0.0842 R@100 over single-query search, measured at
        `candidate_k=100` with a reranker on MTRAG-human dev. Other settings are untested rather
        than merely different.

        ⚠️ The gain is CONDITIONAL on reranking. Raw, this arm scores 0.0447 nDCG@5 BELOW
        `search()`. That is why a missing reranker is refused instead of warned about.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if self._reranker is None:
            raise ValueError(
                "search_fused requires a reranker: the fused arm was measured at +0.0084 nDCG@5 "
                "WITH one and at -0.0447 WITHOUT one, so serving it unreranked would be a "
                "measurably worse system than search(). Pass a reranker, or call search()."
            )
        if not history:
            raise ValueError(
                "search_fused requires a non-empty history; call search() for single-query "
                "retrieval rather than passing an empty history and getting it implicitly"
            )
        history_query = build_history_query(history)
        if len(history_query) > FUSED_HISTORY_MAX_CHARS:
            raise ValueError(
                f"history concatenation is {len(history_query)} characters, over the "
                f"{FUSED_HISTORY_MAX_CHARS} budget. It is refused rather than truncated: a "
                f"truncated history is a configuration that was never measured. Send fewer turns."
            )
        if not history_query:
            raise ValueError(
                "history contained no usable text after stripping speaker tags and blank turns"
            )
        raise NotImplementedError("fusion lands in Task 5")
