from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.eval.reasoning_session6 import (
    SYSTEMS,
    TASKS,
    load_fixture,
    reasoning_session6_metrics,
)


def test_reasoning_session6_fixture_covers_required_task_classes() -> None:
    fixture = load_fixture()

    assert {case.task for case in fixture.cases} == set(TASKS)
    assert {case.corpus_kind for case in fixture.cases} == {"synthetic", "real"}
    assert set(fixture.preregistered_thresholds) >= {
        "answer_accuracy_min",
        "citation_precision_min",
        "unsupported_claim_rate_max",
        "heldout_answer_accuracy_min",
    }


def test_reasoning_session6_every_case_has_every_ablation_and_control() -> None:
    fixture = load_fixture()

    for case in fixture.cases:
        assert [observation.system for observation in case.observations] == list(SYSTEMS)


def test_reasoning_session6_artifact_matches_harness() -> None:
    expected = json.loads(
        Path("results/reasoning_session6_controls.json").read_text(encoding="utf-8")
    )

    assert reasoning_session6_metrics() == expected


def test_reasoning_session6_rejects_answer_text_in_metadata(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    raw["memories"][0]["metadata"]["leaky_label"] = raw["cases"][0]["expected_answer_facts"][0]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="answer text leaked into metadata"):
        load_fixture(path)


def test_reasoning_session6_rejects_missing_observation_system(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    del raw["cases"][0]["observations"]["nearest_neighbor"]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="observation systems mismatch"):
        load_fixture(path)


def test_reasoning_session6_rejects_non_integer_count_fields(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    raw["cases"][0]["observations"]["retrieval_full_planner"]["model_calls"] = "2"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="model_calls"):
        load_fixture(path)


def test_reasoning_session6_rejects_boolean_latency(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    raw["cases"][0]["observations"]["current_retrieval"]["latency_ms"] = True
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="latency_ms must be a non-negative JSON number"):
        load_fixture(path)


def test_reasoning_session6_rejects_missing_provider_identity(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    del raw["cases"][0]["observations"]["current_retrieval"]["provider_id"]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(KeyError, match="provider_id"):
        load_fixture(path)


def test_reasoning_session6_rejects_unknown_literal_fields(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    raw["cases"][0]["task"] = "surprise_task"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="task must be one of"):
        load_fixture(path)


def test_reasoning_session6_controls_gate_claimed_improvement() -> None:
    metrics = reasoning_session6_metrics()
    status = metrics["threshold_status"]

    assert status["all_passed"] is True
    checks = status["passes"]
    assert checks["answer_accuracy_threshold"] is True
    assert checks["citation_precision_threshold"] is True
    assert checks["correct_abstention_threshold"] is True
    assert checks["false_abstention_threshold"] is True
    assert checks["proposal_precision_threshold"] is True
    assert checks["proposal_recall_threshold"] is True
    assert checks["contradiction_detection_threshold"] is True
    assert checks["nearest_neighbor_control_not_enough"] is True
    assert checks["shuffled_edge_control_drops"] is True
    assert checks["removed_edge_control_drops"] is True
    assert checks["heldout_survives"] is True


def test_reasoning_session6_threshold_gate_consumes_metric_thresholds() -> None:
    artifact = reasoning_session6_metrics()
    thresholds = dict(artifact["threshold_status"]["thresholds"])
    thresholds["citation_precision_min"] = 1.01
    fixture = load_fixture()
    fixture = type(fixture)(
        version=fixture.version,
        index_generation=fixture.index_generation,
        preregistered_thresholds=thresholds,
        memories=fixture.memories,
        cases=fixture.cases,
    )

    status = reasoning_session6_metrics(fixture)["threshold_status"]

    assert status["passes"]["citation_precision_threshold"] is False
    assert status["all_passed"] is False


def test_reasoning_session6_multi_hop_gate_uses_multi_hop_slice(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    for case in raw["cases"]:
        if case["task"] == "multi_hop":
            case["observations"]["retrieval_full_planner"]["answer_correct"] = False
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    fixture = load_fixture(path)
    status = reasoning_session6_metrics(fixture)["threshold_status"]

    assert status["passes"]["answer_accuracy_threshold"] is True
    assert status["passes"]["multi_hop_improves"] is False


def test_reasoning_session6_removed_edge_gate_uses_graph_dependent_slices(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    for case in raw["cases"]:
        obs = case["observations"]["removed_edges_control"]
        if case["task"] in {"multi_hop", "supersession_recovery"}:
            obs["answer_correct"] = case["observations"]["retrieval_full_planner"]["answer_correct"]
        else:
            obs["answer_correct"] = False
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    fixture = load_fixture(path)
    status = reasoning_session6_metrics(fixture)["threshold_status"]

    assert status["passes"]["answer_accuracy_threshold"] is True
    assert status["passes"]["removed_edge_control_drops"] is False


def test_reasoning_session6_shuffled_edge_gate_uses_supersession_slice(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    for case in raw["cases"]:
        obs = case["observations"]["shuffled_edges_control"]
        if case["task"] == "supersession_recovery":
            obs["answer_correct"] = case["observations"]["retrieval_full_planner"]["answer_correct"]
        else:
            obs["answer_correct"] = False
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    fixture = load_fixture(path)
    status = reasoning_session6_metrics(fixture)["threshold_status"]

    assert status["passes"]["answer_accuracy_threshold"] is True
    assert status["passes"]["shuffled_edge_control_drops"] is False


def test_reasoning_session6_keeps_unsupported_inference_from_counting_as_gain() -> None:
    metrics = reasoning_session6_metrics()["metrics"]

    assert (
        metrics["retrieval_full_planner"]["unsupported_claim_rate"]
        <= metrics["current_retrieval"]["unsupported_claim_rate"]
    )
    assert metrics["retrieval_full_planner"]["unsupported_claim_rate"] == 0.0


def test_reasoning_session6_reports_per_query_audit_controls() -> None:
    artifact = reasoning_session6_metrics()

    assert artifact["audit"]["per_query_results"].startswith("observations are stored per case")
    assert artifact["per_query_error_taxonomy"]["multi_hop_rollout_owner"]["current_retrieval"]
    assert set(artifact["audit"]["controls"]) == {
        "nearest_neighbor",
        "shuffled_edges_control",
        "removed_edges_control",
    }


def test_reasoning_session6_records_ablation_and_provider_identities() -> None:
    artifact = reasoning_session6_metrics()

    assert artifact["ablation_matrix"]["retrieval_full_planner"]["uses_planner"] is True
    assert artifact["ablation_matrix"]["nearest_neighbor"]["control"] is True
    assert (
        artifact["metrics"]["retrieval_full_planner"]["cross_generation_reproducibility"]
        is None
    )
    assert artifact["provider_generation_identities"]["retrieval_full_planner"] == [
        {"provider_id": "offline-control", "generation_id": "gen-a"}
    ]


def test_reasoning_session6_proposal_precision_is_case_scoped(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    raw["cases"][0]["observations"]["retrieval_full_planner"]["proposed_edges"] = [
        {"from_memory_id": "m_search_v1", "to_memory_id": "m_search_v2"}
    ]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    fixture = load_fixture(path)
    metrics = reasoning_session6_metrics(fixture)["metrics"]["retrieval_full_planner"]

    assert metrics["proposal_precision"] == 0.5
    assert metrics["proposal_recall"] == 1.0


def test_reasoning_session6_contradiction_precision_penalizes_false_positives(
    tmp_path: Path,
) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session6.json").read_text(encoding="utf-8"))
    raw["cases"][0]["observations"]["retrieval_full_planner"]["detected_contradiction"] = True
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    fixture = load_fixture(path)
    metrics = reasoning_session6_metrics(fixture)["metrics"]["retrieval_full_planner"]

    assert metrics["contradiction_detection_precision"] == 0.5
