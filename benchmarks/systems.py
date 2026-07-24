"""`MemorySystem` protocol + the RE-call adapter for the head-to-head benchmark.

The benchmark runs two memory systems through an IDENTICAL LLM generator (see
`benchmarks.pipeline.run_question`) on LOCOMO — only the retrieved memory differs between systems.
`MemorySystem` is the seam that makes that swap possible: `run_question` takes a bare
``retrieve(question) -> str`` callable, and any `MemorySystem.retrieve` satisfies it directly.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemorySystem(Protocol):
    """One competitor in the head-to-head benchmark.

    ``ingest`` indexes a single LOCOMO conversation; ``retrieve`` answers one question against
    whatever was last ingested. Both are per-conversation — LOCOMO's conversations are unrelated
    worlds, and the benchmark harness ingests exactly once per conversation before scoring its
    questions.
    """

    name: str

    def ingest(self, conversation: dict[str, Any]) -> None: ...

    def retrieve(self, question: str) -> str: ...


class RecallSystem:
    """RE-call adapter: index each dialogue turn, retrieve via the trust layer.

    Reuses the exact LOCOMO indexing/search machinery `recall.eval.locomo` already validates —
    `index_conversation` for ingest, `trusted_search` for retrieve — rather than reimplementing
    either. Each conversation gets its own tenant (``bench-{sample_id}``) in a benchmark-only table
    (``bench_locomo_chunks``), mirroring the isolation `recall.eval.locomo.run` already relies on
    to keep one conversation's turns from leaking into another's answers.

    Returns an EMPTY STRING from `retrieve` when the trust layer abstains. That
    abstention-propagates-as-empty-context behaviour is the single thing this whole benchmark
    exists to measure: the downstream generator sees no memories and must emit its own refusal
    token, exactly as it would for a real caller with no LLM in RE-call's path.
    """

    name = "recall"

    def __init__(self, dsn: str, embedder_name: str = "fastembed", k: int = 5) -> None:
        from recall.eval.locomo import _make_embedder

        self._dsn = dsn
        self._k = k
        self._embedder_name = embedder_name
        self._embedder = _make_embedder(embedder_name)
        self._tenant: str | None = None

    def ingest(self, conversation: dict[str, Any]) -> None:
        from recall.eval.locomo import index_conversation
        from recall.store import PgVectorStore

        # `conversation` here is one LOCOMO item as loaded from locomo10.json: it carries
        # `sample_id` and `qa` alongside the nested `conversation` object (`session_N` turns,
        # `speaker_a`/`speaker_b`) that `index_conversation` actually indexes. Passing the outer
        # item straight into `index_conversation` would find zero `session_` keys and silently
        # index nothing.
        self._tenant = f"bench-{conversation.get('sample_id')}"
        inner = conversation["conversation"]
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table="bench_locomo_chunks"
        ) as store:
            index_conversation(store, self._embedder, inner)

    def retrieve(self, question: str) -> str:
        from recall.store import PgVectorStore
        from recall.trust import trusted_search

        if self._tenant is None:
            raise RuntimeError("RecallSystem.retrieve() called before ingest()")
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table="bench_locomo_chunks"
        ) as store:
            result = trusted_search(store, self._embedder, question, k=self._k)
            if result.abstained:
                return ""
            return "\n".join(hit.chunk.text for hit in result.hits)
