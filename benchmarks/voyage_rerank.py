"""A Voyage cross-encoder reranker satisfying RE-call's `Reranker` protocol.

RE-call's `HybridRetriever` retrieves a wide candidate pool (dense + sparse, `candidate_k` each),
RRF-fuses it, reranks the WHOLE pool, and only then truncates to `k`. So a reranker can rescue an
answer that ranked below the `k` cutoff — which is exactly the cat1 failure the benchmark exposed.
This plugs Voyage's `rerank-2.5` cross-encoder into that slot (docs: benchmarks/VOYAGE_REFERENCE.md).

Like `recall.rerank.CrossEncoderReranker`, this REORDERS ONLY: each returned `ScoredChunk` keeps its
original `.score` (the dense cosine) and `.indexed_at`. The Voyage relevance score is in different
units; leaking it into `.score` would corrupt the trust layer's cosine thresholds and confidence.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from recall.types import ScoredChunk


class VoyageReranker:
    """Reorder retrieved hits by Voyage `rerank-2.5` relevance. Reorders, never rescores.

    `client` is injectable (any object with ``rerank(query, documents, model, top_k) -> results``)
    so the reordering logic is unit-testable without the network or the `voyage` extra; when it is
    None the real `voyageai.Client` is built lazily on first use.
    """

    def __init__(
        self,
        model: str = "rerank-2.5",
        api_key: str | None = None,
        top_k: int | None = None,
        max_document_chars: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.top_k = top_k
        if max_document_chars is not None and max_document_chars < 1:
            raise ValueError("max_document_chars must be positive when set")
        self.max_document_chars = max_document_chars
        self._client = client
        #: Guards the lazy build in `_voyage_client`. ONE instance is reached concurrently:
        #: `benchmarks.beam.run` builds a single system before its conversation loop, that system
        #: resolves a single reranker and passes it into every `retrieve`, and `_run_pool` maps the
        #: scoring closure over a `ThreadPoolExecutor` (8 workers by default) with the retrieval
        #: inside the worker. `if x is None: x = ...` is a check-then-set, so without this each
        #: racing thread built its own `voyageai.Client` and all but one were overwritten and
        #: dropped, still holding an HTTP connection pool nobody closes.
        #:
        #: The same guard as `benchmarks.llm.OpenRouterLLM._client_lock`, for the same reason and
        #: with the same reasoning about why the build stays lazy; that comment is the longer one.
        self._client_lock = threading.Lock()
        # Resolve the key eagerly (mirrors VoyageEmbedder) so a missing key fails at construction,
        # not after a run has started — UNLESS a client was injected, which needs no key.
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if self._client is None and not self._api_key:
            raise RuntimeError("VoyageReranker needs VOYAGE_API_KEY (env) or an explicit api_key")

    def _voyage_client(self) -> Any:
        # Taken unconditionally rather than as the second half of a double-check: the unlocked
        # fast path would save an uncontended acquire (tens of nanoseconds) off a reranking call
        # that crosses the network, and would buy a publication argument with no settled answer on
        # a free-threaded build, which this package's 3.14 support puts in range.
        #
        # The client is returned from INSIDE the critical section, so even the read of an
        # already-built client is guarded. That costs nothing measurable here and means this
        # method carries no unlocked read to reason about.
        with self._client_lock:
            if self._client is None:
                import voyageai

                self._client = voyageai.Client(api_key=self._api_key)
            return self._client

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        limit = self.top_k if self.top_k is not None else len(hits)
        documents = [
            h.chunk.text[: self.max_document_chars] if self.max_document_chars else h.chunk.text
            for h in hits
        ]
        result = self._voyage_client().rerank(query, documents, model=self.model, top_k=limit)
        # `result.results` is sorted by descending relevance; each item's `.index` points back into
        # `documents`. Reorder the ORIGINAL ScoredChunk objects — identity preserved, scores intact.
        return [hits[item.index] for item in result.results]


class BlendedVoyageReranker:
    """Blend the original hybrid rank with the Voyage reranker rank.

    Voyage relevance scores remain isolated from trust scores. This wrapper combines only the
    original and reranked positions, then returns the original hit objects unchanged.
    """

    def __init__(
        self,
        model: str = "rerank-2.5",
        api_key: str | None = None,
        rank_weight: float = 0.5,
        max_document_chars: int | None = None,
        client: Any | None = None,
    ) -> None:
        if not 0.0 <= rank_weight <= 1.0:
            raise ValueError("rank_weight must be between 0 and 1")
        self.rank_weight = rank_weight
        self._voyage = VoyageReranker(
            model=model,
            api_key=api_key,
            max_document_chars=max_document_chars,
            client=client,
        )

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        reranked = self._voyage.rerank(query, hits)
        baseline_rank = {hit.chunk.id: index for index, hit in enumerate(hits)}
        voyage_rank = {hit.chunk.id: index for index, hit in enumerate(reranked)}
        reciprocal_rank_constant = 60.0
        scored = [
            (
                self.rank_weight / (reciprocal_rank_constant + voyage_rank[hit.chunk.id])
                + (1.0 - self.rank_weight)
                / (reciprocal_rank_constant + baseline_rank[hit.chunk.id]),
                -baseline_rank[hit.chunk.id],
                hit,
            )
            for hit in hits
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored]
