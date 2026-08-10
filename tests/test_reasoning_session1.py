from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.eval.reasoning_session1 import baseline_metrics, load_fixture


def test_reasoning_fixture_covers_every_session1_case_type() -> None:
    fixture = load_fixture()

    assert {case.category for case in fixture.cases} == {
        "direct_answer",
        "multi_hop",
        "near_miss",
        "contradiction",
        "missing_supersession",
        "ambiguous_entity",
        "empty_corpus",
        "stale_corpus",
    }
    assert {case.expected_outcome for case in fixture.cases} == {
        "answer",
        "abstain",
        "needs_clarification",
        "needs_review",
    }


def test_baseline_metrics_pin_the_pre_reasoning_observations() -> None:
    metrics = baseline_metrics()

    assert metrics["direct_hit_rate"] > metrics["multi_hop_complete_support_rate"]
    assert metrics["near_miss_false_confident_rate"] == 1.0
    assert metrics["missing_supersession_unresolved_rate"] == 1.0


def test_checked_in_baseline_artifact_matches_the_harness() -> None:
    expected = json.loads(
        Path("results/reasoning_session1_baseline.json").read_text(encoding="utf-8")
    )

    assert baseline_metrics() == expected


def test_fixture_loader_rejects_non_boolean_abstention(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session1.json").read_text(encoding="utf-8"))
    raw["cases"][0]["baseline_retrieval"]["abstained"] = "false"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline_retrieval\\.abstained"):
        load_fixture(path)


def test_fixture_loader_rejects_unknown_memory_references(tmp_path: Path) -> None:
    raw = json.loads(Path("recall/eval/reasoning_session1.json").read_text(encoding="utf-8"))
    raw["cases"][0]["baseline_retrieval"]["retrieved_memory_ids"].append("missing-memory")
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown memory id"):
        load_fixture(path)


def test_missing_supersession_is_a_proposal_not_authored_metadata() -> None:
    fixture = load_fixture()
    memory_by_id = {memory["id"]: memory for memory in fixture.memories}
    cases = [case for case in fixture.cases if case.category == "missing_supersession"]

    assert cases, "fixture must include a missing supersession case"
    for case in cases:
        assert case.expected_proposals, f"{case.id} has no proposed relation"
        involved = {
            proposal["from_memory_id"]
            for proposal in case.expected_proposals
        } | {
            proposal["to_memory_id"]
            for proposal in case.expected_proposals
        }
        for memory_id in involved:
            metadata = memory_by_id[memory_id]["metadata"]
            assert "supersedes" not in metadata
            assert "superseded_by" not in metadata


def test_fixture_labels_are_separate_from_baseline_outputs() -> None:
    fixture = load_fixture()

    for case in fixture.cases:
        assert case.expected_outcome in {
            "answer",
            "abstain",
            "needs_clarification",
            "needs_review",
        }
        assert isinstance(case.baseline_retrieval.abstained, bool)
        if case.expected_outcome in {"needs_clarification", "needs_review"}:
            assert not case.baseline_retrieval.abstained


def test_missing_supersession_metric_uses_baseline_exposure_not_label_presence() -> None:
    fixture = load_fixture()
    cases = []
    for case in fixture.cases:
        if case.category == "missing_supersession":
            cases.append(
                type(case)(
                    id=case.id,
                    category=case.category,
                    question=case.question,
                    expected_outcome=case.expected_outcome,
                    supporting_memory_ids=case.supporting_memory_ids,
                    baseline_retrieval=type(case.baseline_retrieval)(
                        retrieved_memory_ids=(),
                        abstained=True,
                        reason="no memory retrieved at all",
                    ),
                    expected_proposals=case.expected_proposals,
                )
            )
        else:
            cases.append(case)
    modified = type(fixture)(
        version=fixture.version,
        index_generation=fixture.index_generation,
        memories=fixture.memories,
        cases=tuple(cases),
    )

    assert baseline_metrics(modified)["missing_supersession_unresolved_rate"] == 0.0


def test_reasoning_contract_records_the_required_safety_invariants() -> None:
    text = Path("docs/REASONING_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Every reasoning output must distinguish authored evidence from inferred structure.",
        "Inferred relationships are proposals.",
        "No untrusted corpus text may enter system instructions",
        "Reasoning graphs are immutable per index generation.",
        "Session 1 does not infer corpus metadata.",
    ]
    for phrase in required:
        assert phrase in text
