from __future__ import annotations

from scripts.summarize_query_anchor_dev import summarize


def _payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "query_pool_sha256": "a" * 64,
        "generation_id": "generation",
        "calibration_id": "calibration",
        "pipeline_fingerprint": "b" * 64,
        "corpus_fingerprint": "c" * 64,
        "rows": rows,
    }


def _row(index: int, *, exact: bool, control: bool = False) -> dict[str, object]:
    return {
        "query_index": index,
        "expected_answerability": "unanswerable_control" if control else "answerable",
        "base_count": 0,
        "base_exact_span_covered": False,
        "proposal": {
            "chunk_contains_exact_span": exact,
            "is_gold_source": exact,
            "source_pool_contains_exact_span": exact,
            "anchor_features": {
                "anchor_count": 3,
                "zero_document_frequency_anchors": 0 if not control else 1,
                "chunk_coverage_fraction": 2.0 / 3.0,
            },
        },
    }


def test_summary_freezes_policy_after_sufficient_parity_and_exact_proposals() -> None:
    rows = [_row(index, exact=index < 5) for index in range(5)]
    rows.extend(
        {
            "query_index": index,
            "expected_answerability": "answerable",
            "base_count": 0,
            "base_exact_span_covered": False,
            "proposal": None,
        }
        for index in range(5, 450)
    )
    rows.append(_row(450, exact=False, control=True))

    result = summarize(_payload(rows))

    assert result["decision"] == "POLICY_FROZEN"
    assert result["accepted_total"] == 5
    assert result["accepted_exact_span_gains"] == 5
    assert result["accepted_controls"] == 0


def test_summary_fails_closed_below_parity_floor() -> None:
    rows = [_row(index, exact=True) for index in range(5)]

    result = summarize(_payload(rows))

    assert result["decision"] == "INSUFFICIENT_DEVELOPMENT_PARITY"
    assert result["accepted_total"] == 0
