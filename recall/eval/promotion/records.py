"""The per-question record: one row of promotion evidence, with a fixed field list.

The field list is CLOSED. `RECORD_FIELDS` is the whole schema, `record_from_dict` refuses a row
carrying anything else, and a test asserts the dataclass and the tuple agree. That strictness is
the point: every adapter in `adapters.py` writes into the same ledger, the aggregator reads all of
them without knowing which corpus a row came from, and a per-adapter extra field is how a schema
turns into five schemas that happen to share a name.

Two hashes, and they cover different halves on purpose:

`input_hash` is copied from the frozen manifest, unchanged. It is what makes a row traceable to a
question that was frozen before any candidate existed — see `manifest.py`.

`output_hash` covers the retrieval OUTPUT plus the configuration identity that produced it:
retrieved ids, rank positions, cosine, confidence, verdict, the embedding profile, the retrieval
profile, the generation, the candidate pool, the reranking status, and `input_hash` itself.

**`stage_timings` is deliberately excluded from `output_hash`.** Timings are not reproducible —
two runs of the identical configuration over the identical corpus differ in the third decimal on
any machine — so including them would make the hash differ on every rerun and answer no question
at all. Excluded, `output_hash` answers the question worth asking: *did these two arms return the
same thing?* That is exactly the check the baseline-equals-candidate proof needs, and a hash that
changed with the clock could not perform it.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

_DOMAIN = b"recall-promotion-question-record-v1\n"

#: The complete schema, in declaration order. Nothing else is a record field.
RECORD_FIELDS: tuple[str, ...] = (
    "question_id",
    "corpus",
    "expected_relevance_labels",
    "retrieved_chunk_ids",
    "rank_positions",
    "dense_cosine",
    "confidence",
    "trust_verdict",
    "embedding_profile_id",
    "retrieval_profile",
    "generation",
    "candidate_pool",
    "reranking_status",
    "stage_timings",
    "input_hash",
    "output_hash",
)

#: The subset `output_hash` covers. `stage_timings` is absent by design (see the module
#: docstring); `output_hash` is absent because a digest cannot cover itself.
_HASHED_FIELDS: tuple[str, ...] = (
    "question_id",
    "corpus",
    "expected_relevance_labels",
    "retrieved_chunk_ids",
    "rank_positions",
    "dense_cosine",
    "confidence",
    "trust_verdict",
    "embedding_profile_id",
    "retrieval_profile",
    "generation",
    "candidate_pool",
    "reranking_status",
    "input_hash",
)

#: Reranking status is a three-state, not a boolean. "ran" and "not_configured" are both healthy;
#: "failed" is a reranker that was configured, was asked, and did not answer — a run that scores
#: those rows as if the reranker had never been requested is comparing two different systems while
#: labelling them one.
RERANKING_STATUSES = frozenset({"ran", "not_configured", "failed"})


@dataclass(frozen=True)
class QuestionRecord:
    """One question's retrieval outcome under one arm."""

    question_id: str
    corpus: str
    expected_relevance_labels: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    #: 1-based rank of each id in `retrieved_chunk_ids`, same length and same order. Carried
    #: rather than inferred from position because a reranked list and its pre-rerank pool are both
    #: legitimate things to record, and only the ranks say which one this is.
    rank_positions: tuple[int, ...]
    #: Cosine of the TOP hit, pre-rerank. NaN when nothing was retrieved: there is no cosine, and
    #: 0.0 would read as "maximally dissimilar", a measurement that was never taken.
    dense_cosine: float
    #: Calibrated confidence of the top hit, on the same convention. NaN when nothing was
    #: retrieved.
    confidence: float
    #: The top hit's trust verdict, or `"abstained"` when the trust layer returned nothing to
    #: trust. `"abstained"` is not one of `recall.types.Verdict` deliberately — a verdict
    #: describes a hit, and abstention is the absence of one.
    trust_verdict: str
    embedding_profile_id: str
    retrieval_profile: str
    generation: str
    candidate_pool: int
    reranking_status: str
    #: Stage name to milliseconds. The only place latency enters a record; `aggregate.py` reads
    #: `total` from it, and refuses a row that has no `total`.
    stage_timings: dict[str, float]
    input_hash: str
    output_hash: str

    def __post_init__(self) -> None:
        if len(self.rank_positions) != len(self.retrieved_chunk_ids):
            raise ValueError(
                f"{self.corpus}/{self.question_id}: {len(self.rank_positions)} rank positions for "
                f"{len(self.retrieved_chunk_ids)} retrieved ids — they are parallel sequences."
            )
        if self.reranking_status not in RERANKING_STATUSES:
            raise ValueError(
                f"reranking_status must be one of {sorted(RERANKING_STATUSES)}, "
                f"got {self.reranking_status!r}"
            )
        if self.candidate_pool < 0:
            raise ValueError("candidate_pool cannot be negative")
        if "total" not in self.stage_timings:
            raise ValueError(
                f"{self.corpus}/{self.question_id}: stage_timings has no 'total'. Latency is a "
                f"gate input; a row that cannot report its own is not evidence."
            )

    @property
    def abstained(self) -> bool:
        return self.trust_verdict == "abstained"

    @property
    def answerable(self) -> bool:
        return bool(self.expected_relevance_labels)

    @property
    def total_ms(self) -> float:
        return float(self.stage_timings["total"])


def _jsonable(value: Any) -> Any:
    """JSON cannot represent NaN portably, and `json.dumps` emits a bare `NaN` token that is not
    valid JSON. NaN is a real value here — "nothing was retrieved, so there is no cosine" — so it
    is rendered as the string `"nan"` rather than as `null`, which would be indistinguishable from
    a field the writer forgot to set."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _unjsonable_float(value: Any) -> float:
    if value == "nan":
        return float("nan")
    return float(value)


def record_to_dict(record: QuestionRecord) -> dict[str, Any]:
    return {field: _jsonable(getattr(record, field)) for field in RECORD_FIELDS}


def record_from_dict(payload: Mapping[str, Any]) -> QuestionRecord:
    missing = [field for field in RECORD_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"question record is missing fields: {missing}")
    extra = [key for key in payload if key not in RECORD_FIELDS]
    if extra:
        raise ValueError(
            f"question record carries fields outside the schema: {sorted(extra)}. The schema is "
            f"closed — an adapter that needs a new field must add it to RECORD_FIELDS, where "
            f"every other adapter and the aggregator can see it."
        )
    return QuestionRecord(
        question_id=payload["question_id"],
        corpus=payload["corpus"],
        expected_relevance_labels=tuple(payload["expected_relevance_labels"]),
        retrieved_chunk_ids=tuple(payload["retrieved_chunk_ids"]),
        rank_positions=tuple(int(rank) for rank in payload["rank_positions"]),
        dense_cosine=_unjsonable_float(payload["dense_cosine"]),
        confidence=_unjsonable_float(payload["confidence"]),
        trust_verdict=payload["trust_verdict"],
        embedding_profile_id=payload["embedding_profile_id"],
        retrieval_profile=payload["retrieval_profile"],
        generation=payload["generation"],
        candidate_pool=int(payload["candidate_pool"]),
        reranking_status=payload["reranking_status"],
        stage_timings={key: float(value) for key, value in payload["stage_timings"].items()},
        input_hash=payload["input_hash"],
        output_hash=payload["output_hash"],
    )


def output_digest(
    *,
    question_id: str,
    corpus: str,
    expected_relevance_labels: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    rank_positions: Sequence[int],
    dense_cosine: float,
    confidence: float,
    trust_verdict: str,
    embedding_profile_id: str,
    retrieval_profile: str,
    generation: str,
    candidate_pool: int,
    reranking_status: str,
    input_hash: str,
) -> str:
    """The `output_hash` for one row. Keyword-only, so a caller cannot transpose two fields.

    Retrieved ids and ranks are NOT sorted: their order IS the result. Labels are not sorted
    either, because `input_hash` already covers them in sorted form and re-sorting here would let
    two different orderings of the same evidence collide.
    """
    payload = {
        "question_id": question_id,
        "corpus": corpus,
        "expected_relevance_labels": list(expected_relevance_labels),
        "retrieved_chunk_ids": list(retrieved_chunk_ids),
        "rank_positions": [int(rank) for rank in rank_positions],
        "dense_cosine": _jsonable(float(dense_cosine)),
        "confidence": _jsonable(float(confidence)),
        "trust_verdict": trust_verdict,
        "embedding_profile_id": embedding_profile_id,
        "retrieval_profile": retrieval_profile,
        "generation": generation,
        "candidate_pool": int(candidate_pool),
        "reranking_status": reranking_status,
        "input_hash": input_hash,
    }
    if set(payload) != set(_HASHED_FIELDS):
        raise AssertionError(
            "output_digest's payload has drifted from _HASHED_FIELDS; the hash would silently "
            "start covering a different set of fields than the module documents."
        )
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def build_record(**kwargs: Any) -> QuestionRecord:
    """Construct a record, computing `output_hash` rather than accepting one.

    There is no way to pass `output_hash` in: a caller that could supply it could supply a wrong
    one, and a self-reported digest that nobody recomputes is decoration.
    """
    if "output_hash" in kwargs:
        raise TypeError("output_hash is computed, never supplied")
    hashed = {field: kwargs[field] for field in _HASHED_FIELDS}
    return QuestionRecord(**kwargs, output_hash=output_digest(**hashed))


def _assert_schema_matches_dataclass() -> None:
    declared = tuple(field.name for field in fields(QuestionRecord))
    if declared != RECORD_FIELDS:
        raise AssertionError(
            f"QuestionRecord fields {declared} do not match RECORD_FIELDS {RECORD_FIELDS}"
        )


_assert_schema_matches_dataclass()
