"""Which of a caller's own questions get answered from text the corpus already replaced.

One `trusted_search` call carries both sides of the comparison. The highest-cosine hit is what a
plain top-k retriever would return; `hits[0]` is what RE-call serves, because verdict-ok hits are
ordered first. Deriving the baseline from the same call rather than a second retrieval path is
what makes the resulting claim defensible.
"""
from __future__ import annotations

from collections.abc import Callable

from recall.embeddings import Embedder
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.types import TrustedResult
from recall_consistency.findings import StaleAnswer

SearchFn = Callable[..., TrustedResult]


def probe(
    store: PgVectorStore,
    embedder: Embedder,
    questions: list[str],
    k: int = 5,
    search: SearchFn = trusted_search,
) -> list[StaleAnswer]:
    """Report every question whose nearest match is superseded, or that RE-call refused."""
    found: list[StaleAnswer] = []
    for index, question in enumerate(questions, start=1):
        failure: str | None = None
        try:
            result = search(store, embedder, question, k=k)
        except Exception as exc:
            # Broad on purpose: the trust layer refuses for several unrelated reasons and the
            # operator's next move is the same for all of them, so the position is what they need.
            # Only the type name escapes this block. The exception object does not.
            failure = type(exc).__name__
        if failure is not None:
            # Raised OUTSIDE the except block on purpose. `recall.trust_policy.TrustRefusal`
            # excludes the query from its payload by construction, "so there is no sanitiser to
            # forget to call", but a psycopg, embedder or reranker failure makes no such promise.
            # `raise ... from exc` would print its message through the default excepthook, and
            # even `from None` leaves the original reachable through `__context__`, because
            # Python fills that in implicitly at the raise. Raising here, with the handler
            # already exited, leaves both `__cause__` and `__context__` empty.
            #
            # The cost is real: a connection failure loses its detail. The position and the type
            # name are what the operator needs to drop a question and re-run, and the query text
            # is the caller's data, owed the same protection as anything else in the corpus.
            raise RuntimeError(
                f"search failed for question {index} of {len(questions)} ({failure})"
            )
        if not result.hits:
            continue
        # Ties break toward the hit this report is about: a superseded one. `hits` arrives
        # ok-first and `max` keeps the first maximum, so a plain `cosine` key hides a superseded
        # hit that tied with a current one, and a key that only asks "not ok" hands a three-way
        # tie to whichever non-ok verdict happened to come first. A tie means a plain retriever
        # could return either, so the superseded one is a nearest match and belongs in the report.
        nearest = max(
            result.hits,
            key=lambda hit: (hit.cosine, hit.validity.superseded_by is not None,
                             hit.verdict != "ok"),
        )
        superseded_by = nearest.validity.superseded_by or ""
        if not superseded_by and not result.abstained:
            continue  # the nearest match is current and RE-call served it: nothing to report
        found.append(
            StaleAnswer(
                question=question,
                plain_top_file=nearest.provenance.file or nearest.provenance.source,
                plain_top_superseded_by=superseded_by,
                trusted_verdict=result.hits[0].verdict,
                trusted_abstained=result.abstained,
            )
        )
    return found
