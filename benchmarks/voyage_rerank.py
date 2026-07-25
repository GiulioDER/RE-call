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
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self._client = client
        # Resolve the key eagerly (mirrors VoyageEmbedder) so a missing key fails at construction,
        # not after a run has started — UNLESS a client was injected, which needs no key.
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if self._client is None and not self._api_key:
            raise RuntimeError("VoyageReranker needs VOYAGE_API_KEY (env) or an explicit api_key")

    def _voyage_client(self) -> Any:
        if self._client is None:
            import voyageai

            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        limit = self.top_k if self.top_k is not None else len(hits)
        documents = [h.chunk.text for h in hits]
        result = self._voyage_client().rerank(query, documents, model=self.model, top_k=limit)
        # `result.results` is sorted by descending relevance; each item's `.index` points back into
        # `documents`. Reorder the ORIGINAL ScoredChunk objects — identity preserved, scores intact.
        return [hits[item.index] for item in result.results]
