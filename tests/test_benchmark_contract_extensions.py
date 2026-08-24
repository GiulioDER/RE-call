import pytest

from benchmarks.artifact_contract import (
    validate_evidence_cost_contract,
    validate_operational_claim_separation,
    validate_routing_experiment,
)
from benchmarks.evidence_curve import (
    EVIDENCE_BUDGETS,
    evidence_cost_curve,
    evidence_cost_curve_from_artifacts,
)
from benchmarks.evidence_tokens import PinnedReaderTokenizer, truncate_evidence_context
from benchmarks.operational import (
    OperationalMeasurement,
    attach_operational_metrics,
    measure_staged_indexing,
    run_operational_benchmark,
)


def test_exact_evidence_contract_requires_pinned_reader_tokenizer() -> None:
    payload = {
        "tokenizer_metadata": {
            "tokenizer_id": "wrong",
            "tokenizer_revision": "fixture",
            "tokenizer_hash": "fixture",
        },
        "outcomes": [],
    }
    with pytest.raises(ValueError, match="cl100k_base"):
        validate_evidence_cost_contract(payload)


def test_budgeted_evidence_contract_requires_tokenizer_metadata() -> None:
    with pytest.raises(ValueError, match="tokenizer_metadata"):
        validate_evidence_cost_contract(
            {"outcomes": [{"evidence_budget": 128}]}
        )


def test_pinned_reader_tokenizer_records_reproducible_identity() -> None:
    tokenizer = PinnedReaderTokenizer()
    first = tokenizer.metadata()
    second = tokenizer.metadata()
    assert first == second
    assert first["tokenizer_id"] == "cl100k_base"
    assert first["tokenizer_revision"] == "tiktoken-0.13.0"
    assert len(first["tokenizer_hash"]) == 64
    assert tokenizer.count_tokens("alpha beta") == tokenizer.count_tokens("alpha beta")


def test_evidence_cost_curve_uses_fixed_ladder_and_never_invents_quality() -> None:
    points = evidence_cost_curve(
        [
            {"evidence_tokens_exact": 100},
            {"evidence_tokens_exact": 900},
        ]
    )
    assert tuple(point["budget_tokens"] for point in points) == EVIDENCE_BUDGETS
    assert points[0]["n"] == 1
    assert points[0]["measured_budget"] is False
    assert points[0]["quality_measurement"] == "observed_within_budget"
    assert points[0]["accuracy"]["rate"] is None
    assert points[0]["citation_metrics"]["available"] is False
    assert all(left["coverage"] <= right["coverage"] for left, right in zip(points, points[1:]))


def test_evidence_curve_retains_class_route_and_raw_budget_records() -> None:
    points = evidence_cost_curve(
        [
            {
                "question_id": "q1",
                "query_class": "lookup",
                "routing_profile": "fast",
                "evidence_tokens_exact": 100,
                "input_tokens_exact": 140,
                "evidence_budget": 128,
                "correct": True,
                "abstained": False,
            }
        ]
    )
    point = points[0]
    assert point["by_query_class"]["lookup"]["n"] == 1
    assert point["by_routing_profile"]["fast"]["n"] == 1
    assert point["input_tokens_exact"]["max"] == 140
    assert point["records"][0]["question_id"] == "q1"


def test_budgeted_curve_uses_the_matching_budget_arm() -> None:
    points = evidence_cost_curve(
        [
            {"evidence_tokens_exact": 120, "evidence_budget": 128, "correct": True},
            {"evidence_tokens_exact": 240, "evidence_budget": 256, "correct": False},
        ]
    )
    assert points[0]["measured_budget"] is True
    assert points[0]["n"] == 1
    assert points[1]["n"] == 1


def test_artifact_contract_requires_the_preregistered_curve_shape() -> None:
    with pytest.raises(ValueError, match="budget ladder"):
        validate_evidence_cost_contract(
            {
                "evidence_cost": {
                    "claim_family": "evidence_cost",
                    "curve": [{"budget_tokens": 256, "records": [], "measured_budget": False}],
                }
            }
        )


def test_budget_artifact_curve_requires_paired_question_identity() -> None:
    artifacts = [
        {
            "config": {"evidence_budget": budget},
            "outcomes": [
                {
                    "question_id": "q1",
                    "evidence_budget": budget,
                    "evidence_tokens_exact": min(budget, 10),
                    "input_tokens_exact": min(budget + 10, 20),
                    "correct": True,
                    "abstained": False,
                }
            ],
        }
        for budget in EVIDENCE_BUDGETS
    ]
    points = evidence_cost_curve_from_artifacts(artifacts)
    assert all(point["pairing_complete"] for point in points)
    assert all(point["paired_question_ids"] == ["q1"] for point in points)


def test_operational_artifact_is_separate_from_quality_claims() -> None:
    artifact = attach_operational_metrics({}, OperationalMeasurement(snapshot_load_ms=12.5))
    validate_operational_claim_separation(artifact)
    assert artifact["operational_metrics"]["retrieval_quality_claim"] is False

    bad = {"operational_metrics": {"claim_family": "quality", "retrieval_quality_claim": True}}
    with pytest.raises(ValueError, match="claim_family"):
        validate_operational_claim_separation(bad)


def test_staged_operational_measurement_reports_all_operational_stages() -> None:
    measurement = measure_staged_indexing(
        lambda: "lexical",
        lambda: "semantic",
        lambda: "snapshot",
        lambda: "warm",
        atomic_cutover=lambda: True,
        recovery=lambda: True,
    )
    assert measurement.lexical_ready_ms is not None
    assert measurement.semantic_ready_ms is not None
    assert measurement.snapshot_load_ms is not None
    assert measurement.atomic_cutover_ok is True
    assert measurement.recovery_ok is True


def test_operational_runner_returns_a_separate_claim_family() -> None:
    artifact = run_operational_benchmark(
        lambda: None,
        lambda: None,
        lambda: None,
        lambda: None,
        configuration={"host": "fixture"},
    )
    assert artifact["benchmark"] == "recall-operational-v1"
    validate_operational_claim_separation(artifact)
    assert artifact["configuration"]["host"] == "fixture"


def test_exact_budget_truncation_counts_the_rendered_wrapper() -> None:
    tokenizer = PinnedReaderTokenizer()
    context = "alpha beta gamma delta epsilon"
    budget = tokenizer.count_tokens(f"<memories>\n{context}\n</memories>") - 1
    truncated = truncate_evidence_context(context, budget, tokenizer)
    assert tokenizer.count_tokens(f"<memories>\n{truncated}\n</memories>") <= budget
    assert truncated != context


def test_routing_artifact_is_versioned() -> None:
    validate_routing_experiment(
        {
            "routing_experiment": {
                "mode": "shadow",
                "classifier_version": "query-class-v1",
                "policy_version": "routing-v1",
            }
        }
    )
    with pytest.raises(ValueError, match="policy_version"):
        validate_routing_experiment(
            {
                "routing_experiment": {
                    "mode": "shadow",
                    "classifier_version": "query-class-v1",
                    "policy_version": "old",
                }
            }
        )
