from __future__ import annotations

from scripts.run_direct_query_anchor_dev import summarize


def test_summary_requires_two_exact_gains_and_no_controls() -> None:
    answerable = {
        "expected_answerability": "answerable",
        "pool_gold_source": True,
        "pool_exact_span": True,
        "dense_gold_source": True,
        "dense_exact_span": True,
        "sparse_gold_source": False,
        "sparse_exact_span": False,
        "selected": True,
        "selected_gold_source": True,
        "selected_exact_span": True,
        "selected_source_pool_exact_span": True,
    }
    control = {
        key: False for key in answerable if key != "expected_answerability"
    }
    control["expected_answerability"] = "unanswerable"

    result = summarize([answerable, answerable, control])

    assert result["decision"] == "PROCEED_FRESH_HOLDOUT"
    assert result["selected_exact_span_precision"] == 1.0
    assert result["selected_controls"] == 0
