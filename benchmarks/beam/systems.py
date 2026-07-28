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


#: BEAM stamps turns with a human-readable `time_anchor` like "March-01-2024". Two things in the
#: VENDORED answerer prompt consume that value, and both break on it:
#:
#:   1. it SORTS the memories by `created_at` as a plain string, so "April" precedes "January"
#:      precedes "March" — the chronological order the prompt promises is scrambled;
#:   2. it renders `created_at[:10]`, which truncates "March-01-2024" to "March-01-2".
#:
#: The prompt then instructs the model to "prefer the more recent one" on contradictions and to
#: "pay attention to dates" on temporal questions. Handing it a scrambled order and a mangled
#: label sabotages exactly those instructions — and `knowledge_update` (-0.192) and
#: `temporal_reasoning` (-0.133) were this arm's two worst categories against Mem0, whose own
#: pipeline hands the same prompt ISO 8601 timestamps that sort and render correctly.
#:
#: So the anchor is normalised to ISO on the way in. This is a defect fix, not a tuning choice:
#: it removes a handicap that was ours alone and makes the two arms comparable on the dimension
#: the prompt actually reads.
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _iso_date(anchor: str) -> str:
    """"March-01-2024" -> "2024-03-01". Unparseable input returns "" rather than a guess.

    An unrecognised anchor must NOT fall through as its raw self: a mix of ISO and non-ISO keys
    in one sort is worse than no key at all, because it orders confidently and wrongly. Empty
    sorts first and renders blank, which the prompt already handles.
    """
    parts = (anchor or "").replace("/", "-").split("-")
    if len(parts) != 3:
        return ""
    month, day, year = (p.strip() for p in parts)
    if month.lower() not in _MONTHS:
        # Already ISO (YYYY-MM-DD)? Keep it.
        if len(month) == 4 and month.isdigit():
            return anchor.strip()
        return ""
    if not (day.isdigit() and year.isdigit() and len(year) == 4):
        return ""
    return f"{year}-{_MONTHS[month.lower()]:02d}-{int(day):02d}"


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


#: How many top-ranked trusted hits the entailment guard actually judges. The guard is a QNLI
#: cross-encoder (82 M params) and BEAM chunks average 620 tokens, so judging all ~160 trusted
#: hits for all 300 questions is ~5 PFLOPs — the same CPU arithmetic that already took a laptop
#: down once this session. 25 is ~0.8 PFLOPs, a couple of hours on the VPS.
#:
#: Judging the TOP of the ranking is also the right place to spend the budget rather than a
#: concession: the abstention decision turns on whether the BEST evidence entails an answer, and
#: a hit ranked 100th that the top 25 did not support is not going to rescue it. Hits beyond the
#: cap keep their existing verdict and still reach the answerer as context — the guard changes
#: WHETHER RE-call abstains and what it trusts most, not how much context the answerer sees.
ENTAILMENT_TOP_N = 25


class TopNEntailment:
    """Wraps an `EntailmentJudge`, judging only the first `n` candidates.

    Returns True (keep) for everything past the cap. That is deliberate and it is the
    conservative direction: an unjudged hit keeps whatever verdict it already had, so the cap can
    only make the guard demote FEWER hits than a full pass would. It cannot manufacture an
    abstention, and it cannot make RE-call look better by hiding a bad hit — any error it
    introduces understates the guard's effect rather than overstating it.
    """

    def __init__(self, inner: Any, n: int = ENTAILMENT_TOP_N) -> None:
        self._inner = inner
        self._n = n
        #: Judged/skipped counts, reported in the artifact so the cap is visible in the results
        #: rather than only in this docstring.
        self.judged = 0
        self.skipped = 0

    def judge(self, query: str, texts: list[str]) -> list[bool]:
        head, tail = texts[: self._n], texts[self._n :]
        self.judged += len(head)
        self.skipped += len(tail)
        return (self._inner.judge(query, head) if head else []) + [True] * len(tail)


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
        entailment_top_n: int = 0,
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
        # OFF unless asked for: `trusted_search` ships the guard disabled, and the as-shipped arm
        # must measure what ships. A non-zero cap builds the QNLI judge ONCE here rather than per
        # query — the cross-encoder loads ~300 MB of weights, and paying that per question would
        # dominate the measurement it is supposed to inform.
        self._entailment_top_n = entailment_top_n
        self._entailment: Any | None = None
        if entailment_top_n:
            from recall.entailment import QnliEntailmentJudge

            self._entailment = TopNEntailment(QnliEntailmentJudge(), entailment_top_n)
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
            "entailment": (
                {
                    "guard": "qnli-cross-encoder",
                    "top_n": self._entailment_top_n,
                    "candidates_judged": self._entailment.judged,
                    "candidates_skipped_by_cap": self._entailment.skipped,
                }
                if self._entailment is not None
                else {"guard": "off"}
            ),
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
                # ISO so the vendored prompt's string sort is chronological and its
                # `created_at[:10]` render is a whole date rather than "March-01-2".
                self._dates[name] = _iso_date(turn.get("date", ""))
            with PgVectorStore(
                self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
            ) as store:
                store.ensure_schema()
                self._clear(store)
                Indexer(store, self._embedder, batch_chunks=EMBED_BATCH).index_path(workspace)
                # Fail LOUD on an empty index rather than let the run discover it as scores.
                # Every beam table was found emptied between two probes, with no process running
                # and no error anywhere. Had that happened mid-scoring, retrieval would have
                # returned nothing, the answerer would have refused, and every affected question
                # would have been recorded as a legitimate 0.00 — a corrupted result wearing the
                # face of a clean one. An ingest that indexed turns but stored no chunk is not a
                # low score, it is a broken run, and it must stop the run to say so.
                stored = sum(1 for _ in store.iter_chunks())
                if conversation.turns and not stored:
                    raise RuntimeError(
                        f"ingest of {self._tenant} wrote ZERO chunks from "
                        f"{len(conversation.turns)} turns — the index is empty, so every question "
                        f"for this conversation would score 0.00 without an error. Refusing to "
                        f"continue: check the store and the table before re-running."
                    )
            return len(conversation.turns)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @staticmethod
    def _clear(store: Any) -> int:
        sources = sorted({chunk.source for chunk in store.iter_chunks()})
        if sources:
            store.delete_sources(sources)
        return len(sources)

    def top_cosine(self, question: str) -> float:
        """Best raw cosine for `question`, BEFORE the trust layer's threshold is applied.

        Calibration needs the unfiltered score distribution: sampling through `retrieve` would
        only ever observe scores that already cleared the current threshold, so the fit would
        chase its own tail and certify whatever it started from.
        """
        from recall.store import PgVectorStore

        if self._tenant is None:
            raise RuntimeError("BeamRecallSystem.top_cosine() called before ingest()")
        vector = self._embedder.embed([question])[0]
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
        ) as store:
            hits = store.query_dense(vector, k=5)
        return max((h.score for h in hits), default=0.0)

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
                entailment=self._entailment,
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
