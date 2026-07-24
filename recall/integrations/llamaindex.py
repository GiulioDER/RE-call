"""A LlamaIndex retriever backed by RE-call's trust layer.

``RecallRetriever`` is a drop-in ``llama_index.core`` retriever, so RE-call can be the retriever in
any LlamaIndex query engine, chat engine or agent. Like the LangChain adapter, it differs from an
ordinary vector retriever in exactly one way, and it is the whole point of RE-call:

    **When the trust layer abstains, this returns no nodes — not a best-effort neighbour.**

It runs ``trusted_search`` (verdict + confidence + provenance per hit, valid hits ordered first, an
explicit abstention when none remain) and maps the decision onto LlamaIndex's contract: an
abstention becomes an empty ``list[NodeWithScore]``, so a query engine synthesises from nothing
rather than from a stale, superseded or unentailed memory. Each node carries the trust signal in
``metadata`` (``recall_verdict``, ``recall_confidence``, ``recall_cosine``, ``superseded_by`` …);
its ``score`` is the cosine similarity.

Requires the ``llamaindex`` extra::

    pip install "recall-rag[llamaindex]"

Typical use::

    from recall.integrations.llamaindex import RecallRetriever

    retriever = RecallRetriever.from_store(store, embedder, k=5)
    nodes = retriever.retrieve("how many requests per second can a client make?")
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.rerank import Reranker
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.types import TrustedHit, TrustedResult

try:
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise ModuleNotFoundError(
        'RecallRetriever requires llama-index-core. Install it with: pip install "recall-rag[llamaindex]"'
    ) from exc

#: A query -> trust-evaluated result. Injectable so the adapter is testable without a database.
SearchFn = Callable[[str], TrustedResult]


def _hit_to_node(hit: TrustedHit) -> NodeWithScore:
    """Map one trusted hit onto a scored LlamaIndex node, carrying the trust signal in metadata.

    The chunk's own metadata is preserved; the ``recall_*`` and provenance keys are layered on top
    (and win on a collision). The node ``score`` is the cosine similarity; the calibrated trust
    confidence rides in ``metadata['recall_confidence']``.
    """
    prov = hit.provenance
    val = hit.validity
    metadata: dict[str, Any] = dict(hit.chunk.metadata)
    metadata.update(
        recall_verdict=hit.verdict,
        recall_confidence=hit.confidence,
        recall_cosine=hit.cosine,
        source=prov.source,
        file=prov.file,
        ord=prov.ord,
        indexed_at=prov.indexed_at.isoformat() if prov.indexed_at is not None else None,
        superseded_by=val.superseded_by,
        valid_from=val.valid_from.isoformat() if val.valid_from is not None else None,
        valid_until=val.valid_until.isoformat() if val.valid_until is not None else None,
    )
    node = TextNode(id_=hit.chunk.id, text=hit.chunk.text, metadata=metadata)
    return NodeWithScore(node=node, score=hit.cosine)


class RecallRetriever(BaseRetriever):
    """LlamaIndex retriever that returns trust-evaluated nodes, or nothing when it abstains.

    Construct with :meth:`from_store` for the common case; the ``search_fn`` seam keeps the adapter
    independent of a live store (and trivially testable). When ``return_abstention_reason`` is set,
    an abstention yields a single empty node whose metadata carries ``recall_abstained`` and
    ``recall_reason`` instead of an empty list — for a query engine that must *say* it does not know.
    """

    def __init__(
        self,
        search_fn: SearchFn,
        *,
        return_abstention_reason: bool = False,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self._search_fn = search_fn
        self._return_abstention_reason = return_abstention_reason
        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_store(
        cls,
        store: PgVectorStore,
        embedder: Embedder,
        *,
        k: int = 5,
        source: str | None = None,
        calibration: Calibration | None = None,
        reranker: Reranker | None = None,
        entailment: Any | None = None,
        return_abstention_reason: bool = False,
        callback_manager: CallbackManager | None = None,
    ) -> RecallRetriever:
        """Build a retriever that calls :func:`recall.trust.trusted_search` on each query.

        ``store`` and ``embedder`` are captured by reference; their lifecycle is the caller's.
        ``entailment`` is the optional near-miss judge (see :mod:`recall.entailment`), left untyped
        here so importing the adapter does not pull in the ``entail`` extra.
        """

        def _search(query: str) -> TrustedResult:
            return trusted_search(
                store,
                embedder,
                query,
                k=k,
                source=source,
                calibration=calibration,
                reranker=reranker,
                entailment=entailment,
            )

        return cls(
            _search,
            return_abstention_reason=return_abstention_reason,
            callback_manager=callback_manager,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result = self._search_fn(query_bundle.query_str)
        if result.abstained:
            if self._return_abstention_reason:
                node = TextNode(
                    text="",
                    metadata={"recall_abstained": True, "recall_reason": result.reason},
                )
                return [NodeWithScore(node=node, score=0.0)]
            return []
        return [_hit_to_node(hit) for hit in result.hits]
