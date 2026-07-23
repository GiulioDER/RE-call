"""A LangChain retriever backed by RE-call's trust layer.

``RecallRetriever`` is a drop-in ``langchain_core`` retriever, so RE-call can be the ``retriever=``
behind any chain, agent or ``create_retrieval_chain`` pipeline. It differs from an ordinary vector
retriever in exactly one way, and it is the whole point of RE-call:

    **When the trust layer abstains, this returns no documents — not a best-effort neighbour.**

A plain similarity retriever always hands back its top-k, so a chain cites the closest vector even
when that memory is stale, superseded, or does not actually answer the question (the stale hit is
often the *highest*-cosine one). ``RecallRetriever`` runs ``trusted_search`` — verdict + confidence
+ provenance per hit, valid hits ordered first, an explicit abstention when none remain — and maps
its decision onto LangChain's contract: an abstention becomes an empty result, so the chain gets
nothing rather than a confident wrong memory. Each returned ``Document`` carries the trust signal
in ``metadata`` (``recall_verdict``, ``recall_confidence``, ``recall_cosine``, ``superseded_by`` …)
so a downstream prompt or reranker can use it.

Requires the ``langchain`` extra::

    pip install "recall-rag[langchain]"

Typical use::

    from recall.integrations.langchain import RecallRetriever

    retriever = RecallRetriever.from_store(store, embedder, k=5)
    docs = retriever.invoke("how many requests per second can a client make?")
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
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise ModuleNotFoundError(
        'RecallRetriever requires langchain-core. Install it with: pip install "recall-rag[langchain]"'
    ) from exc

#: A query -> trust-evaluated result. Injectable so the adapter is testable without a database.
SearchFn = Callable[[str], TrustedResult]


def _hit_to_document(hit: TrustedHit) -> Document:
    """Map one trusted hit onto a LangChain ``Document``, carrying the trust signal in metadata.

    The chunk's own metadata is preserved; the ``recall_*`` and provenance keys are layered on top
    (and win on a collision), so a downstream consumer can read the verdict without re-deriving it.
    """
    prov = hit.provenance
    val = hit.validity
    metadata: dict[str, Any] = dict(hit.chunk.metadata)
    metadata.update(
        recall_verdict=hit.verdict,
        recall_confidence=hit.confidence,
        recall_cosine=hit.cosine,
        chunk_id=hit.chunk.id,
        source=prov.source,
        file=prov.file,
        ord=prov.ord,
        indexed_at=prov.indexed_at.isoformat() if prov.indexed_at is not None else None,
        superseded_by=val.superseded_by,
        valid_from=val.valid_from.isoformat() if val.valid_from is not None else None,
        valid_until=val.valid_until.isoformat() if val.valid_until is not None else None,
    )
    return Document(page_content=hit.chunk.text, metadata=metadata)


class RecallRetriever(BaseRetriever):
    """LangChain retriever that returns trust-evaluated memories, or nothing when it abstains.

    Construct with :meth:`from_store` for the common case; the ``search_fn`` field is the seam that
    keeps the adapter independent of a live store (and trivially testable). When
    ``return_abstention_reason`` is set, an abstention yields a single empty ``Document`` whose
    metadata carries ``recall_abstained`` and ``recall_reason`` instead of an empty list — useful
    when a chain needs to *say* it does not know rather than silently retrieve nothing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    search_fn: SearchFn
    return_abstention_reason: bool = False

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
    ) -> RecallRetriever:
        """Build a retriever that calls :func:`recall.trust.trusted_search` on each query.

        ``store`` and ``embedder`` are captured by reference; their lifecycle is the caller's — a
        server holds one open ``PgVectorStore`` (with a pool) for the process. ``entailment`` is the
        optional near-miss judge (see :mod:`recall.entailment`), left untyped here so importing the
        adapter does not pull in the ``entail`` extra.
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

        return cls(search_fn=_search, return_abstention_reason=return_abstention_reason)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = self.search_fn(query)
        if result.abstained:
            if self.return_abstention_reason:
                return [
                    Document(
                        page_content="",
                        metadata={"recall_abstained": True, "recall_reason": result.reason},
                    )
                ]
            return []
        return [_hit_to_document(hit) for hit in result.hits]
