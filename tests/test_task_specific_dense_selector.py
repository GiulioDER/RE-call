from __future__ import annotations

from scripts.run_task_specific_dense_selector import (
    prepare_rows,
    summarize_scored_rows,
    validation_gate,
)


def test_prepare_rows_drops_every_repeated_answer_hash() -> None:
    rows = [
        {
            "id": "a",
            "expected_answerability": "answerable",
            "source_sha256": "1" * 64,
            "answer_span_sha256": "repeat",
        },
        {
            "id": "b",
            "expected_answerability": "answerable",
            "source_sha256": "2" * 64,
            "answer_span_sha256": "repeat",
        },
        {
            "id": "c",
            "expected_answerability": "answerable",
            "source_sha256": "3" * 64,
            "answer_span_sha256": "unique",
        },
        {
            "id": "control",
            "expected_answerability": "unanswerable",
            "source_sha256": None,
            "answer_span_sha256": "NOT_FOUND",
        },
    ]

    prepared = prepare_rows(rows)

    assert [row["id"] for row in prepared] == ["c"]
    assert prepared[0]["split"] in {"train", "validation", "internal_test"}


def test_summary_counts_dense_to_selector_gains_and_losses() -> None:
    rows = [
        {
            "candidates": [
                {"chunk_id": "wrong", "gold_source": False, "exact_span": False},
                {"chunk_id": "right", "gold_source": True, "exact_span": True},
            ],
            "trained_order": ["right", "wrong"],
            "base_order": ["wrong", "right"],
        },
        {
            "candidates": [
                {"chunk_id": "right-2", "gold_source": True, "exact_span": True},
                {"chunk_id": "wrong-2", "gold_source": False, "exact_span": False},
            ],
            "trained_order": ["wrong-2", "right-2"],
            "base_order": ["right-2", "wrong-2"],
        },
        {
            "candidates": [
                {"chunk_id": "right-3", "gold_source": True, "exact_span": True},
                {"chunk_id": "wrong-3", "gold_source": False, "exact_span": False},
            ],
            "trained_order": ["right-3", "wrong-3"],
            "base_order": ["wrong-3", "right-3"],
        },
    ]

    summary = summarize_scored_rows(rows)

    assert summary["trained_exact_by_cutoff"]["1"] == 2
    assert summary["dense_exact_by_cutoff"]["1"] == 2
    assert summary["exact_rank1_gains"] == 1
    assert summary["exact_rank1_losses"] == 1
    assert summary["changed_rank1_exact_precision"] == 0.5
    assert summary["memberships_preserved"] == 3


def test_validation_gate_requires_both_exact_and_gold_improvement() -> None:
    passing = {
        "rows": 33,
        "memberships_preserved": 33,
        "dense_exact_by_cutoff": {"1": 4},
        "trained_exact_by_cutoff": {"1": 6},
        "dense_gold_by_cutoff": {"1": 7},
        "trained_gold_by_cutoff": {"1": 9},
        "exact_rank1_losses": 1,
        "gold_rank1_losses": 0,
        "changed_rank1_exact_precision": 0.5,
        "changed_rank1_gold_precision": 0.6,
        "pools_reordered_vs_base": 3,
    }

    decision, checks = validation_gate(passing, weight_drift=0.001)

    assert decision == "PROCEED_INTERNAL_TEST"
    assert all(checks.values())

    failing = dict(passing)
    failing["trained_gold_by_cutoff"] = {"1": 8}
    decision, checks = validation_gate(failing, weight_drift=0.001)
    assert decision == "STOP_TASK_SPECIFIC_SELECTOR_VALIDATION"
    assert checks["gold_rank1_improvement"] is False
