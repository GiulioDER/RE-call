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
from recall.evidence import (
    EvidenceBundle,
    EvidencePolicy,
    build_evidence_bundle,
    render_evidence_prompt,
)
from recall.integrations import result_trust_metadata, trust_metadata
from recall.rerank import Reranker
from recall.retriever import DocumentExpansionPolicy, StructuralExpansionPolicy
from recall.store import PgVectorStore
from recall.trust import marked_text, servable_hits, trusted_search
from recall.trust_policy import TrustPolicy
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


def _hit_to_document(hit: TrustedHit, result: TrustedResult | None = None) -> Document:
    """Map one trusted hit onto a LangChain ``Document``, carrying the trust signal in metadata.

    The chunk's own metadata is preserved; the ``recall_*`` and provenance keys are layered on top
    (and win on a collision), so a downstream consumer can read the verdict without re-deriving it.
    """
    metadata: dict[str, Any] = dict(hit.chunk.metadata)
    metadata.update(trust_metadata(hit))
    if result is not None:
        metadata.update(
            embedding_profile=result.diagnostics.embedding_profile,
            retrieval_profile=result.diagnostics.retrieval_profile,
            index_generation=result.diagnostics.index_generation,
            **result_trust_metadata(result),
        )
    return Document(page_content=marked_text(hit), metadata=metadata)


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
    #: Serve hits the trust layer refused. Off by default: `TrustedResult.hits` is `ok +
    #: rest`, so forwarding it wholesale handed a chain the very superseded memory this
    #: library exists to demote — and did so only when some OTHER hit happened to be ok,
    #: which is not a rule anyone chose. When on, each untrusted hit's `page_content`
    #: carries an in-band warning, because out-of-band metadata is what failed here.
    include_untrusted: bool = False

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
        include_untrusted: bool = False,
        policy: TrustPolicy | None = None,
        document_expansion: DocumentExpansionPolicy | None = None,
        structural_expansion: StructuralExpansionPolicy | None = None,
    ) -> RecallRetriever:
        """Build a retriever that calls :func:`recall.trust.trusted_search` on each query.

        ``store`` and ``embedder`` are captured by reference; their lifecycle is the caller's — a
        server holds one open ``PgVectorStore`` (with a pool) for the process. ``entailment`` is the
        optional near-miss judge (see :mod:`recall.entailment`), left untyped here so importing the
        adapter does not pull in the ``entail`` extra. ``include_untrusted`` opts a live-store
        retriever into serving trust-refused hits (with an in-band warning) — the same escape hatch
        the constructor exposes, reachable through the documented factory.
        ``document_expansion`` opts relational queries into source-scoped retrieval before the
        trust layer evaluates the complete candidate set.
        ``structural_expansion`` adds bounded ordinal neighbors and the terminal section.
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
                policy=policy,
                document_expansion=document_expansion,
                structural_expansion=structural_expansion,
            )

        return cls(
            search_fn=_search,
            return_abstention_reason=return_abstention_reason,
            include_untrusted=include_untrusted,
        )

    def evidence(self, query: str, *, policy: EvidencePolicy | None = None) -> EvidenceBundle:
        """The generator-neutral evidence bundle for ``query``, over the same ``search_fn``.

        Additive: ``invoke`` / ``_get_relevant_documents`` are untouched and a caller that never
        asks for evidence sees no change. This exists because a chain that stuffs ``Document``
        objects into a prompt has already lost the boundary — the trust verdict rides in
        ``metadata`` and the stock ``stuff_documents_chain`` renders ``page_content`` alone.

        ``include_untrusted`` is deliberately NOT honoured here. It is an escape hatch for a
        caller who wants to see what the trust layer refused; the evidence bundle is the input to
        a generator, and letting a constructor flag widen it would make "what may be cited" a
        setting rather than a rule.
        """
        return build_evidence_bundle(self.search_fn(query), policy or EvidencePolicy())

    def evidence_prompt(
        self, query: str, *, policy: EvidencePolicy | None = None
    ) -> tuple[str, str]:
        """The fixed system instruction and the delimited user data message for ``query``."""
        return render_evidence_prompt(self.evidence(query, policy=policy))

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
        return [
            _hit_to_document(hit, result)
            for hit in servable_hits(result.hits, include_untrusted=self.include_untrusted)
        ]
