"""Score one arm over a frozen manifest, corpus-agnostically, into a resumable ledger.

The runner never learns which corpus it is scoring. It takes frozen questions, their query texts,
and a `SearchFn`, and emits `QuestionRecord`s. Every corpus therefore gets the same definition of
what a record is, which is the reason `hit@5` can be compared across them at all.

Three things happen per question, in this order, and the order is the guarantee:

1. `FrozenQuestion.verify` on the query text, BEFORE the search. A corpus that drifted under a
   frozen manifest fails on the question that drifted, rather than producing a plausible number
   for a question that is no longer the frozen one, and it fails before paying for an embedding.
2. The search, timed.
3. The row, appended and flushed. A crash costs one question.

Resume is `recall.eval.resume`, the single mechanism, keyed on `(corpus, question_id)` because a
question id is unique within its corpus and not across them.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from recall.eval.promotion.manifest import FrozenQuestion
from recall.eval.promotion.records import (
    QuestionRecord,
    build_record,
    record_from_dict,
    record_to_dict,
)
from recall.eval.resume import append_rows, read_ledger

#: The ledger's resume key. `(corpus, question_id)` for the reason in the module docstring.
LEDGER_ID_FIELDS = ("corpus", "question_id")


@dataclass(frozen=True)
class ArmConfig:
    """The immutable identity of one arm, stamped onto every record it produces.

    Everything here is a field the gate's evidence must be able to distinguish two arms by. A
    baseline and a candidate that differ in nothing but `label` are a null difference, and proving
    the gate refuses to promote one is what this harness is first used for.
    """

    label: str
    embedding_profile_id: str
    retrieval_profile: str
    generation: str
    candidate_pool: int
    #: `EmbeddingProfile.fingerprint()`. Not a record field — the record carries the profile ID,
    #: which is what a human reads — but it keys the arm's artifacts on the COMPLETE immutable
    #: identity, so two arms sharing a profile ID and differing in artifact digest cannot land in
    #: the same ledger.
    embedding_fingerprint: str = ""

    def artifact_stem(self) -> str:
        """A filename stem carrying the arm's full identity, not just its label."""
        fingerprint = self.embedding_fingerprint[:16] or "nofingerprint"
        return f"{self.label}.{self.embedding_profile_id}.{fingerprint}"


@dataclass(frozen=True)
class SearchOutcome:
    """What one retrieval returned, in the terms a record stores.

    `dense_cosine` and `confidence` are NaN when nothing was retrieved: there is no cosine, and
    0.0 would read as "maximally dissimilar", a measurement that was never taken.
    """

    retrieved_chunk_ids: tuple[str, ...]
    dense_cosine: float
    confidence: float
    trust_verdict: str
    reranking_status: str
    #: Stage name to milliseconds, WITHOUT `total`. The runner adds `total` from its own clock, so
    #: a search function cannot under-report the wall time the harness actually spent.
    stage_timings: dict[str, float] = field(default_factory=dict)


SearchFn = Callable[[str], SearchOutcome]


@dataclass(frozen=True)
class ArmResult:
    records: tuple[QuestionRecord, ...]
    scored_now: int
    resumed: int
    repaired_truncated_tail: bool


def score_arm(
    frozen: Sequence[FrozenQuestion],
    queries: Mapping[str, str],
    search: SearchFn,
    arm: ArmConfig,
    ledger_path: Path,
    *,
    resume: bool = True,
) -> ArmResult:
    """Score every frozen question not already in the ledger. Returns ALL records, not just new.

    Returning the whole set, resumed rows included, is what makes a resumed run and a fresh run
    produce the same gate input. A version that returned only this invocation's rows would gate on
    whatever fraction of the manifest happened to survive the last crash, and would look identical
    from the outside.
    """
    existing = (
        read_ledger(ledger_path, id_field=LEDGER_ID_FIELDS, repair_truncated_tail=True)
        if resume
        else read_ledger(None, id_field=LEDGER_ID_FIELDS)
    )
    by_identity: dict[tuple[str, str], QuestionRecord] = {}
    for row in existing.rows:
        record = record_from_dict(row)
        by_identity[(record.corpus, record.question_id)] = record
    resumed = len(by_identity)

    scored_now = 0
    for question in sorted(frozen, key=lambda q: (q.corpus, q.question_id)):
        identity = (question.corpus, question.question_id)
        if identity in by_identity:
            continue
        if question.question_id not in queries:
            raise KeyError(
                f"{question.corpus}/{question.question_id} is in the frozen manifest but its "
                f"adapter supplied no query text for it. Scoring the rest and reporting success "
                f"would silently shrink the evidence base."
            )
        query = queries[question.question_id]
        question.verify(query)
        started = time.perf_counter()
        outcome = search(query)
        total_ms = (time.perf_counter() - started) * 1000.0
        if "total" in outcome.stage_timings:
            raise ValueError(
                f"{question.corpus}/{question.question_id}: the search function reported its own "
                f"'total' stage timing. `total` is the harness's wall clock, so a search that "
                f"supplies one could under-report the time the harness actually spent."
            )
        record = build_record(
            question_id=question.question_id,
            corpus=question.corpus,
            expected_relevance_labels=question.expected_relevance_labels,
            retrieved_chunk_ids=tuple(outcome.retrieved_chunk_ids),
            rank_positions=tuple(range(1, len(outcome.retrieved_chunk_ids) + 1)),
            dense_cosine=outcome.dense_cosine,
            confidence=outcome.confidence,
            trust_verdict=outcome.trust_verdict,
            embedding_profile_id=arm.embedding_profile_id,
            retrieval_profile=arm.retrieval_profile,
            generation=arm.generation,
            candidate_pool=arm.candidate_pool,
            reranking_status=outcome.reranking_status,
            stage_timings={**outcome.stage_timings, "total": total_ms},
            # Copied from the frozen manifest unchanged. This is what makes the row traceable to
            # a question that was frozen before any candidate existed, and `aggregate` refuses an
            # arm whose rows disagree with the manifest here.
            input_hash=question.input_hash,
        )
        append_rows(ledger_path, [record_to_dict(record)])
        by_identity[identity] = record
        scored_now += 1

    return ArmResult(
        records=tuple(
            by_identity[(question.corpus, question.question_id)]
            for question in sorted(frozen, key=lambda q: (q.corpus, q.question_id))
        ),
        scored_now=scored_now,
        resumed=resumed,
        repaired_truncated_tail=bool(existing.truncated),
    )
