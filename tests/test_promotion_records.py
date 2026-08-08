"""The closed per-question record schema, and what its two hashes cover."""
from __future__ import annotations

import math

import pytest

from recall.eval.promotion.records import (
    RECORD_FIELDS,
    build_record,
    record_from_dict,
    record_to_dict,
)


def _kwargs(**overrides):
    base = dict(
        question_id="q01",
        corpus="labelled",
        expected_relevance_labels=("a.md:0",),
        retrieved_chunk_ids=("a.md:0", "b.md:0"),
        rank_positions=(1, 2),
        dense_cosine=0.81,
        confidence=0.77,
        trust_verdict="ok",
        embedding_profile_id="bge-small-symmetric-v1",
        retrieval_profile="fast",
        generation="g1",
        candidate_pool=20,
        reranking_status="not_configured",
        stage_timings={"dense": 3.0, "total": 11.5},
    )
    base.update(overrides)
    return base


def test_the_schema_is_exactly_the_sixteen_declared_fields() -> None:
    assert RECORD_FIELDS == (
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


def test_a_row_carrying_a_field_outside_the_schema_is_refused() -> None:
    payload = record_to_dict(build_record(input_hash="h", **_kwargs()))
    payload["notes"] = "a field one adapter wanted"
    with pytest.raises(ValueError, match="outside the schema"):
        record_from_dict(payload)


def test_output_hash_ignores_stage_timings_so_two_runs_of_one_config_agree() -> None:
    """Timings are not reproducible; a hash that moved with the clock could not compare arms."""
    fast = build_record(input_hash="h", **_kwargs(stage_timings={"total": 9.0}))
    slow = build_record(input_hash="h", **_kwargs(stage_timings={"total": 900.0}))
    assert fast.output_hash == slow.output_hash


def test_output_hash_changes_when_the_retrieval_changes() -> None:
    original = build_record(input_hash="h", **_kwargs())
    reordered = build_record(
        input_hash="h", **_kwargs(retrieved_chunk_ids=("b.md:0", "a.md:0"))
    )
    assert original.output_hash != reordered.output_hash


def test_output_hash_changes_when_the_configuration_changes() -> None:
    """Two arms that returned the same ids under different profiles are not the same evidence."""
    fast = build_record(input_hash="h", **_kwargs(retrieval_profile="fast"))
    quality = build_record(input_hash="h", **_kwargs(retrieval_profile="quality"))
    assert fast.output_hash != quality.output_hash


def test_output_hash_cannot_be_supplied() -> None:
    with pytest.raises(TypeError, match="computed, never supplied"):
        build_record(input_hash="h", output_hash="whatever i like", **_kwargs())


def test_a_row_with_no_total_timing_is_refused() -> None:
    with pytest.raises(ValueError, match="stage_timings has no 'total'"):
        build_record(input_hash="h", **_kwargs(stage_timings={"dense": 3.0}))


def test_ranks_and_ids_must_be_parallel() -> None:
    with pytest.raises(ValueError, match="parallel sequences"):
        build_record(input_hash="h", **_kwargs(rank_positions=(1,)))


def test_an_unknown_reranking_status_is_refused() -> None:
    """"failed" and "not_configured" are different systems and must not share a label."""
    with pytest.raises(ValueError, match="reranking_status must be one of"):
        build_record(input_hash="h", **_kwargs(reranking_status="maybe"))


def test_nan_survives_the_json_round_trip_as_nan_and_not_as_null() -> None:
    """Nothing retrieved means there is no cosine. `null` would be indistinguishable from a
    field the writer forgot to set, and 0.0 would read as maximally dissimilar."""
    record = build_record(
        input_hash="h",
        **_kwargs(
            retrieved_chunk_ids=(),
            rank_positions=(),
            dense_cosine=math.nan,
            confidence=math.nan,
            trust_verdict="abstained",
        ),
    )
    restored = record_from_dict(record_to_dict(record))
    assert math.isnan(restored.dense_cosine) and math.isnan(restored.confidence)
    assert restored.abstained


def test_a_round_trip_preserves_the_record() -> None:
    record = build_record(input_hash="h", **_kwargs())
    assert record_from_dict(record_to_dict(record)) == record
