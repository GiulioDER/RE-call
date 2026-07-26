"""RE-call adapter shaped for BEAM: index dialogue turns, retrieve dated memories.

Mirrors `benchmarks.systems.RecallSystem` — same store, same trust layer, same
abstention-propagates-as-empty-context behaviour — and differs only where BEAM differs from
LOCOMO: turns arrive as a flat `role`/`content`/`date` list rather than `session_N` blocks, and
retrieval must hand back memories carrying their DATE, because the vendored answerer prompt
prefixes each memory with `[YYYY-MM-DD]` and BEAM's temporal and event-ordering categories are
scored on getting those dates right.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import Conversation

#: Benchmark-only table, isolated from the LOCOMO arm's `bench_locomo_chunks` so a BEAM run cannot
#: contaminate — or be contaminated by — an accuracy run in flight.
BEAM_TABLE = "bench_beam_chunks"

#: Upstream retrieves `--top-k 200` and reports the `top_200` cutoff as its headline. Matching it
#: is what makes our cell comparable: a smaller budget would flatter RE-call's token cost while
#: measuring a different retrieval task.
DEFAULT_TOP_K = 200

#: Chunks embedded per ONNX call. The library default is 512, which is right for a corpus of short
#: uniform notes and badly wrong here: ONNX pads every document in a batch to the LONGEST one, and
#: BEAM turns vary from one line to several hundred. A single 512-wide flush was measured stuck
#: inside one `onnxruntime.run` for over thirteen minutes at 4.5 GB resident, having written zero
#: chunks — on a 12 GB laptop that is what took the machine down, and on a 12-core VPS it still
#: pushed load average from 7 to 17 while a live host served traffic.
#:
#: 32 is a benchmark-local flush size, NOT a library change: the vectors, the chunks and the rows
#: written are byte-identical, only the number computed per call differs. Padding waste falls with
#: the square of the batch width, so this is ~16x less wasted compute for the same result.
EMBED_BATCH = 32


def _turn_document(turn: dict[str, str], index: int) -> str:
    """One dialogue turn as a standalone markdown document.

    Role and date go in the BODY, not into metadata: they are frequently the answer. BEAM's
    `temporal_reasoning` and `event_ordering` questions turn on when something was said, and
    `contradiction_resolution` turns on which speaker said it. A chunk that has been stripped of
    both cannot answer those categories, and the score would be measuring the harness's document
    format rather than the retriever.
    """
    date = turn.get("date") or "unknown date"
    speaker = "User" if turn["role"] == "user" else "Assistant"
    return f"# {speaker} — {date}\n\n{speaker}: {turn['content']}\n"


def _filename(index: int) -> str:
    """Zero-padded so lexical order matches chronological order in any directory listing."""
    return f"turn_{index:06d}.md"


class BeamRecallSystem:
    """RE-call over one BEAM conversation at a time.

    One tenant per conversation (`beam-{size}-{idx}`), cleared at ingest. Without the clear, a
    second run against the same database indexes every turn a second time under fresh temp paths —
    the ids differ, so the upsert never fires — and top-k silently fills with duplicates, shrinking
    the effective context. The LOCOMO adapter learned this the hard way; the same guard is here.
    """

    name = "recall"

    def __init__(
        self,
        dsn: str,
        embedder_name: str = "fastembed",
        k: int = DEFAULT_TOP_K,
        reranker_name: str = "none",
        table: str = BEAM_TABLE,
        candidate_k: int | None = None,
    ) -> None:
        from benchmarks.systems import resolve_embedder, resolve_reranker

        self._dsn = dsn
        self._k = k
        # The fused candidate pool `trusted_search` ranks within. It defaults to 20 library-wide
        # (`recall.retriever.DEFAULT_CANDIDATE_K`), which at k=200 silently caps retrieval at ~20
        # chunks no matter what k says — measured: a k=200 query over a 1,548-chunk conversation
        # came back with 21 memories. Left at the default, this arm would have shown the answerer
        # a twentieth of the context Mem0's published run showed its own, then reported the
        # resulting deficit as a retrieval result. The pool is therefore sized to the ask and
        # never below it.
        self._candidate_k = max(candidate_k if candidate_k is not None else k, k)
        self._embedder_name = embedder_name
        self._embedder = resolve_embedder(embedder_name)
        self._reranker_name = reranker_name
        self._reranker = resolve_reranker(reranker_name)
        self._table = table
        self._tenant: str | None = None
        #: filename -> turn date, so a retrieved chunk can be handed back with its date.
        self._dates: dict[str, str] = {}

    def describe(self) -> dict[str, Any]:
        """This arm's configuration for the results artifact. Carries no secret (not the DSN)."""
        return {
            "system": self.name,
            "k": self._k,
            "candidate_k": self._candidate_k,
            "embedder": {"name": self._embedder_name, "model": self._embedder.name},
            "reranker": self._reranker_name,
            "table": self._table,
            "tenant": self._tenant,
        }

    def ingest(self, conversation: Conversation) -> int:
        """Materialise every turn as a document and index it. Returns the turn count."""
        from recall.index import Indexer
        from recall.store import PgVectorStore

        self._tenant = f"beam-{conversation.chat_size}-{conversation.index}".lower()
        self._dates = {}
        workspace = Path(tempfile.mkdtemp(prefix="beam-"))
        try:
            for i, turn in enumerate(conversation.turns):
                name = _filename(i)
                (workspace / name).write_text(_turn_document(turn, i), encoding="utf-8")
                self._dates[name] = turn.get("date", "")
            with PgVectorStore(
                self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
            ) as store:
                store.ensure_schema()
                self._clear(store)
                Indexer(store, self._embedder, batch_chunks=EMBED_BATCH).index_path(workspace)
            return len(conversation.turns)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @staticmethod
    def _clear(store: Any) -> int:
        sources = sorted({chunk.source for chunk in store.iter_chunks()})
        if sources:
            store.delete_sources(sources)
        return len(sources)

    def retrieve(self, question: str) -> list[dict[str, str]]:
        """Top-k memories as ``{"memory", "created_at"}`` dicts, best rank first.

        Returns an EMPTY LIST when the trust layer abstains. The vendored answerer prompt then
        renders "(No memories available)" and — per its own rule 4 — must emit the refusal string,
        which is exactly the behaviour BEAM's `abstention` category rewards and every other
        category punishes. Propagating the abstention as empty context rather than quietly falling
        back to unfiltered hits is the single design decision this benchmark exists to price.
        """
        from recall.store import PgVectorStore
        from recall.trust import trusted_search

        if self._tenant is None:
            raise RuntimeError("BeamRecallSystem.retrieve() called before ingest()")
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
        ) as store:
            result = trusted_search(
                store,
                self._embedder,
                question,
                k=self._k,
                reranker=self._reranker,
                candidate_k=self._candidate_k,
            )
            if result.abstained:
                return []
            memories: list[dict[str, str]] = []
            for hit in result.hits:
                filename = hit.chunk.metadata.get("file", "")
                memories.append(
                    {
                        "memory": hit.chunk.text,
                        "created_at": self._dates.get(Path(filename).name, ""),
                    }
                )
            return memories
