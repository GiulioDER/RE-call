from __future__ import annotations

import pytest

from scripts.run_anchor_boosted_query_dev import (
    anchor_boosted_query,
    changed_rank_one_candidate,
    summarize,
)


def _item(chunk_id: str, source: str, text: str) -> dict[str, object]:
    return {"chunk_id": chunk_id, "source": source, "text": text}


def test_anchor_boost_preserves_question_and_repeats_rare_terms_in_query_order() -> None:
    """The representation must add no vocabulary and preserve question semantics.

    Red proof targets ``anchor_boosted_query``. Returning the selected anchors in hash tie break
    order instead of their original query order fails the final string assertion.
    """

    pool = [
        _item("a", "a.md", "alpha beta gamma evidence"),
        _item("b", "b.md", "common evidence appears here"),
    ]

    boosted, anchors = anchor_boosted_query(
        "What gamma alpha beta evidence was recorded?", pool
    )

    assert anchors == ("gamma", "alpha", "beta")
    assert boosted == (
        "What gamma alpha beta evidence was recorded? gamma alpha beta"
    )


def test_anchor_boost_refuses_a_corpus_absent_anchor() -> None:
    pool = [_item("a", "a.md", "alpha beta evidence")]

    boosted, anchors = anchor_boosted_query(
        "What alpha beta velnora evidence was recorded?", pool
    )

    assert boosted is None
    assert anchors == ()


def test_changed_rank_one_candidate_requires_a_new_safe_top_item() -> None:
    original = [_item("old", "a.md", "old")]
    transformed = [_item("new", "b.md", "new")]

    assert changed_rank_one_candidate(original, transformed, eligible=True) == transformed[0]
    assert changed_rank_one_candidate(original, original, eligible=True) is None
    assert changed_rank_one_candidate(original, transformed, eligible=False) is None


def _row(
    *,
    answerable: bool,
    eligible: bool,
    original_gold: int | None,
    original_exact: int | None,
    transformed_gold: int | None,
    transformed_exact: int | None,
    selected: bool = False,
    selected_gold: bool = False,
    selected_exact: bool = False,
) -> dict[str, object]:
    return {
        "expected_answerability": "answerable" if answerable else "unanswerable",
        "eligible": eligible,
        "original_min_gold_rank": original_gold,
        "original_min_exact_rank": original_exact,
        "transformed_min_gold_rank": transformed_gold,
        "transformed_min_exact_rank": transformed_exact,
        "selected": selected,
        "selected_gold_source": selected_gold,
        "selected_exact_span": selected_exact,
    }


def _baseline_rows() -> list[dict[str, object]]:
    gold_ranks = [1] * 6 + [5] + [10] + [20] * 2 + [None] * 5
    exact_ranks = [1] * 3 + [5] * 2 + [10] * 3 + [20] * 2 + [None] * 5
    answerable = [
        _row(
            answerable=True,
            eligible=index < 14,
            original_gold=gold_ranks[index],
            original_exact=exact_ranks[index],
            transformed_gold=gold_ranks[index],
            transformed_exact=exact_ranks[index],
        )
        for index in range(15)
    ]
    controls = [
        _row(
            answerable=False,
            eligible=False,
            original_gold=None,
            original_exact=None,
            transformed_gold=None,
            transformed_exact=None,
        )
        for _ in range(15)
    ]
    return answerable + controls


def test_summary_proceeds_only_for_precise_changed_candidates() -> None:
    rows = _baseline_rows()
    rows[3]["selected"] = True
    rows[3]["selected_gold_source"] = True
    rows[3]["selected_exact_span"] = True
    rows[3]["transformed_min_exact_rank"] = 1
    rows[4]["selected"] = True
    rows[4]["selected_gold_source"] = True
    rows[4]["selected_exact_span"] = True
    rows[4]["transformed_min_exact_rank"] = 1

    summary = summarize(rows)

    assert summary["changed_rank_one_exact_spans"] == 2
    assert summary["changed_rank_one_exact_precision"] == 1.0
    assert summary["decision"] == "PROCEED_FRESH_HOLDOUT"


def test_summary_refuses_baseline_drift() -> None:
    rows = _baseline_rows()
    rows[0]["original_min_exact_rank"] = None

    with pytest.raises(RuntimeError, match="original dense apparatus mismatch"):
        summarize(rows)
