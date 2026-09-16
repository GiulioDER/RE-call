from __future__ import annotations

import pytest

from scripts.run_anchor_only_aux_view_dev import (
    anchor_only_query,
    deduplicated_union,
    summarize,
)


def _item(chunk_id: str, source: str, text: str) -> dict[str, object]:
    return {"chunk_id": chunk_id, "source": source, "text": text}


def test_anchor_only_view_contains_only_safe_anchors_in_question_order() -> None:
    """The auxiliary view must not inherit the full primary question.

    Red proof targets ``anchor_only_query``. Prefixing the original question to the three anchors
    fails the exact query assertion.
    """

    pool = [
        _item("a", "a.md", "alpha beta gamma evidence"),
        _item("b", "b.md", "common evidence appears here"),
    ]

    auxiliary, anchors = anchor_only_query(
        "What gamma alpha beta evidence was recorded?", pool
    )

    assert anchors == ("gamma", "alpha", "beta")
    assert auxiliary == "gamma alpha beta"


def test_anchor_only_view_refuses_a_corpus_absent_anchor() -> None:
    pool = [_item("a", "a.md", "alpha beta evidence")]

    auxiliary, anchors = anchor_only_query(
        "What alpha beta velnora evidence was recorded?", pool
    )

    assert auxiliary is None
    assert anchors == ()


def test_union_preserves_original_and_appends_unique_auxiliary_items() -> None:
    original = [_item("a", "a.md", "a"), _item("b", "b.md", "b")]
    auxiliary = [_item("b", "b.md", "b"), _item("c", "c.md", "c")]

    combined = deduplicated_union(original, auxiliary, cutoff=2)

    assert [item["chunk_id"] for item in combined] == ["a", "b", "c"]


def _row(
    *,
    answerable: bool,
    eligible: bool,
    original_gold: int | None,
    original_exact: int | None,
    auxiliary_gold: int | None,
    auxiliary_exact: int | None,
    union_gold: bool,
    union_exact: bool,
    incremental_gold: bool = False,
    incremental_exact: bool = False,
) -> dict[str, object]:
    return {
        "expected_answerability": "answerable" if answerable else "unanswerable",
        "eligible": eligible,
        "original_min_gold_rank": original_gold,
        "original_min_exact_rank": original_exact,
        "auxiliary_min_gold_rank": auxiliary_gold,
        "auxiliary_min_exact_rank": auxiliary_exact,
        "union_top20_gold_source": union_gold,
        "union_top20_exact_span": union_exact,
        "incremental_gold_source": incremental_gold,
        "incremental_exact_span": incremental_exact,
        "unique_auxiliary_chunks_added": 4 if eligible else 0,
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
            auxiliary_gold=None,
            auxiliary_exact=None,
            union_gold=gold_ranks[index] is not None,
            union_exact=exact_ranks[index] is not None,
        )
        for index in range(15)
    ]
    controls = [
        _row(
            answerable=False,
            eligible=False,
            original_gold=None,
            original_exact=None,
            auxiliary_gold=None,
            auxiliary_exact=None,
            union_gold=False,
            union_exact=False,
        )
        for _ in range(15)
    ]
    return answerable + controls


def test_summary_proceeds_when_auxiliary_view_adds_two_exact_rows() -> None:
    rows = _baseline_rows()
    for index in (10, 11):
        rows[index]["auxiliary_min_gold_rank"] = 5
        rows[index]["auxiliary_min_exact_rank"] = 5
        rows[index]["union_top20_gold_source"] = True
        rows[index]["union_top20_exact_span"] = True
        rows[index]["incremental_gold_source"] = True
        rows[index]["incremental_exact_span"] = True

    summary = summarize(rows)

    assert summary["incremental_exact_span_rows"] == 2
    assert summary["union_top20_exact_span_reachable"] == 12
    assert summary["decision"] == "PROCEED_ONE_SLOT_SELECTOR"


def test_summary_stops_when_only_one_exact_row_is_added() -> None:
    rows = _baseline_rows()
    rows[10]["auxiliary_min_gold_rank"] = 5
    rows[10]["auxiliary_min_exact_rank"] = 5
    rows[10]["union_top20_gold_source"] = True
    rows[10]["union_top20_exact_span"] = True
    rows[10]["incremental_gold_source"] = True
    rows[10]["incremental_exact_span"] = True

    assert summarize(rows)["decision"] == "STOP_LEXICAL_ANCHOR_QUERY_CONSTRUCTION"


def test_summary_refuses_baseline_drift() -> None:
    rows = _baseline_rows()
    rows[0]["original_min_exact_rank"] = None

    with pytest.raises(RuntimeError, match="original dense apparatus mismatch"):
        summarize(rows)
