from __future__ import annotations

from scripts.diagnose_query_anchor_holdout import summarize


def test_summary_counts_proposal_and_anchor_blockers() -> None:
    rows = [
        {"expected_answerability": "answerable", "has_proposal": False},
        {
            "expected_answerability": "answerable",
            "has_proposal": True,
            "features": {
                "anchor_count": 3,
                "zero_document_frequency_anchors": 0,
                "chunk_coverage_fraction": 2.0 / 3.0,
            },
            "chunk_contains_exact_span": True,
            "is_gold_source": True,
            "source_pool_contains_exact_span": True,
        },
        {
            "expected_answerability": "unanswerable",
            "has_proposal": True,
            "features": {
                "anchor_count": 3,
                "zero_document_frequency_anchors": 1,
                "chunk_coverage_fraction": 1.0 / 3.0,
            },
            "chunk_contains_exact_span": False,
            "is_gold_source": False,
            "source_pool_contains_exact_span": False,
        },
    ]

    result = summarize(rows)

    assert result["guarded_proposals"] == 2
    assert result["no_guarded_proposal"] == 1
    assert result["eligible_proposals"] == 1
    assert result["proposal_exact_spans"] == 1
