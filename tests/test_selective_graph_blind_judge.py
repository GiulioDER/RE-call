from __future__ import annotations

import pytest

from scripts.run_selective_graph_blind_judge import _pairs, _parse, _totals


def _answer_row(identifier: str, complete: bool, category: int = 4) -> dict:
    return {
        "id": identifier,
        "category": category,
        "question": "question",
        "gold": ["D1:1"],
        "gold_citations": ["D1:1"] if complete else [],
        "answer": "answer",
    }


def test_pair_sampling_includes_all_changes_and_fixed_unchanged_sample() -> None:
    baseline = [_answer_row(f"q{index:03d}", index < 10) for index in range(20)]
    selective = [_answer_row(f"q{index:03d}", index < 15) for index in range(20)]
    payload = {"arms": {"baseline": {"rows": baseline}, "selective_tail": {"rows": selective}}}

    pairs = _pairs(payload)

    assert len(pairs) == 20
    assert sum(pair["bucket"] == "rescue" for pair in pairs) == 5
    assert sum(pair["bucket"] == "regression" for pair in pairs) == 0
    assert sum(pair["bucket"] == "unchanged" for pair in pairs) == 15
    assert [pair["id"] for pair in pairs] == [pair["id"] for pair in _pairs(payload)]


def test_judge_parser_rejects_out_of_range_scores() -> None:
    raw = '{"correctness_a": 3, "support_a": 1, "completeness_a": 1, "unsupported_claims_a": 0, "correctness_b": 1, "support_b": 1, "completeness_b": 1, "unsupported_claims_b": 0}'

    with pytest.raises(ValueError, match="correctness_a"):
        _parse(raw)


def test_totals_follow_blinded_label_mapping() -> None:
    row = {
        "baseline_label": "b",
        "scores": {
            "correctness_a": 2,
            "support_a": 2,
            "completeness_a": 2,
            "unsupported_claims_a": 0,
            "correctness_b": 1,
            "support_b": 1,
            "completeness_b": 1,
            "unsupported_claims_b": 2,
        },
    }

    assert _totals(row) == (3, 6, 2, 0)
