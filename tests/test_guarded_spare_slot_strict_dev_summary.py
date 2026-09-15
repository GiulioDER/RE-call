from __future__ import annotations

from scripts.summarize_guarded_spare_slot_strict_dev import summarize


def _addition(*, gold: bool, sparse_rank: int = 1, margin: float = 0.1, cross: float = 1.0) -> dict:
    return {
        "lane": "dual_leg",
        "sparse_rank": sparse_rank,
        "is_gold_source": gold,
        "features": {"source_margin": margin, "cross_leg_fraction": cross},
    }


def test_summary_scores_only_first_addition_and_keeps_controls_separate() -> None:
    """Red proof: the summary module did not exist before the strict policy was frozen."""
    payload = {
        "decision": "READY_FOR_POLICY_FIT",
        "parity_rows": 3,
        "selection_parity_mismatches": 1,
        "gold_source_additions": 2,
        "rows": [
            {
                "expected_answerability": "answerable",
                "additions": [_addition(gold=True), _addition(gold=True)],
            },
            {
                "expected_answerability": "answerable",
                "additions": [_addition(gold=False, sparse_rank=3)],
            },
            {
                "expected_answerability": "unanswerable",
                "additions": [_addition(gold=False, cross=0.25)],
            },
        ],
    }

    result = summarize(payload)

    assert result["accepted_total"] == 1
    assert result["accepted_gold"] == 1
    assert result["accepted_answerable_non_gold"] == 0
    assert result["accepted_controls"] == 0
    assert result["later_gold_not_considered"] == 1
