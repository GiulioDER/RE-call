from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_colbert_dense_selector_dev import (
    apply_scores,
    load_scores,
    summarize,
    write_offload_inputs,
)


def _candidate(chunk_id: str, *, exact: bool = False, gold: bool = False) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source": f"{chunk_id}.md",
        "text": chunk_id,
        "dense_score": 1.0,
        "gold_source": gold,
        "exact_span": exact,
    }


def _row(
    index: int,
    *,
    answerable: bool,
    eligible: bool,
    original_gold: int | None,
    original_exact: int | None,
) -> dict[str, object]:
    candidates = [
        _candidate(
            f"{index}-{rank}",
            exact=answerable and rank == original_exact,
            gold=answerable and rank == original_gold,
        )
        for rank in range(1, 21)
    ]
    return {
        "query_index": index,
        "query": f"question {index}",
        "expected_answerability": "answerable" if answerable else "unanswerable",
        "anchors": ["a", "b", "c"] if eligible else [],
        "eligible": eligible,
        "original_min_gold_rank": original_gold,
        "original_min_exact_rank": original_exact,
        "candidates": candidates,
    }


def _baseline_rows() -> list[dict[str, object]]:
    gold_ranks = [1] * 6 + [5] + [10] + [20] * 2 + [None] * 5
    exact_ranks = [1] * 3 + [5] * 2 + [10] * 3 + [20] * 2 + [None] * 5
    answerable = [
        _row(
            index,
            answerable=True,
            eligible=index < 14,
            original_gold=gold_ranks[index],
            original_exact=exact_ranks[index],
        )
        for index in range(15)
    ]
    controls = [
        _row(
            index + 15,
            answerable=False,
            eligible=False,
            original_gold=None,
            original_exact=None,
        )
        for index in range(15)
    ]
    return answerable + controls


def _scores(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    return {
        str(row["query_index"]): {
            str(candidate["chunk_id"]): float(20 - rank)
            for rank, candidate in enumerate(row["candidates"])
        }
        for row in rows
        if row["eligible"]
    }


def test_offload_inputs_exclude_ineligible_queries(tmp_path: Path) -> None:
    """The model surface must not score controls or receive gold annotations.

    Red proof removes the ineligible row guard in ``write_offload_inputs``. The count assertion
    then reports two queries and 40 pairs instead of one query and 20 pairs.
    """

    rows = [
        _row(1, answerable=True, eligible=True, original_gold=1, original_exact=1),
        _row(2, answerable=False, eligible=False, original_gold=None, original_exact=None),
    ]
    counts = write_offload_inputs(rows, tmp_path)

    assert counts == {"queries": 1, "documents": 20, "pairs": 20}
    assert "question 2" not in (tmp_path / "queries.jsonl").read_text(encoding="utf-8")
    pair_text = (tmp_path / "pairs.jsonl").read_text(encoding="utf-8")
    assert '"qid": "2"' not in pair_text
    assert "gold_source" not in (tmp_path / "docs.jsonl").read_text(encoding="utf-8")


def test_apply_scores_uses_stable_maxsim_order_and_preserves_dense_scores() -> None:
    """Red proof: reversing the dense IDs before ``rerank_order`` flips the tied first two IDs."""

    rows = [_row(1, answerable=True, eligible=True, original_gold=1, original_exact=1)]
    candidates = rows[0]["candidates"]
    original_scores = {candidate["chunk_id"]: candidate["dense_score"] for candidate in candidates}
    scores = _scores(rows)
    scores["1"]["1-1"] = 99.0
    scores["1"]["1-2"] = 99.0

    measured = apply_scores(rows, scores)[0]

    assert [item["chunk_id"] for item in measured["reranked_candidates"][:2]] == ["1-1", "1-2"]
    assert {
        item["chunk_id"]: item["dense_score"] for item in measured["reranked_candidates"]
    } == original_scores


def test_apply_scores_refuses_incomplete_membership() -> None:
    """Red proof fills a missing score with negative infinity instead of refusing it.

    The ``pytest.raises`` assertion then fails because incomplete model output is accepted.
    """

    rows = [_row(1, answerable=True, eligible=True, original_gold=1, original_exact=1)]
    scores = _scores(rows)
    scores["1"].pop("1-20")

    with pytest.raises(RuntimeError, match="score membership mismatch"):
        apply_scores(rows, scores)


def test_load_scores_requires_model_identity(tmp_path: Path) -> None:
    """Red proof removes the checkpoint comparison and accepts ``wrong/model`` without raising."""

    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps(
            {
                "_header": True,
                "arm": "li_colbertv2",
                "checkpoint": "wrong/model",
                "licence": "mit",
                "deployable": True,
                "fastembed_version": "1.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="wrong model identity"):
        load_scores(path)


def test_summary_proceeds_only_when_every_colbert_gate_passes() -> None:
    """Red proof: changing the exact rank one floor from five to six returns the stop verdict."""

    rows = _baseline_rows()
    scores = _scores(rows)
    for index in (3, 4):
        scores[str(index)][f"{index}-5"] = 100.0
    measured = apply_scores(rows, scores)

    summary = summarize(measured)

    assert summary["maxsim_exact_span_by_cutoff"]["1"] == 5
    assert summary["changed_rank1_exact_rows"] == 2
    assert summary["changed_rank1_exact_precision"] == 1.0
    assert summary["decision"] == "PROCEED_FRESH_COLBERT_SELECTOR_VALIDATION"


def test_summary_stops_when_changed_rank1_precision_is_below_half() -> None:
    """Red proof lowers the frozen precision gate to 0.40 and returns the proceed verdict."""

    rows = _baseline_rows()
    scores = _scores(rows)
    for index in (3, 4):
        scores[str(index)][f"{index}-5"] = 100.0
    for index in (5, 6, 7):
        scores[str(index)][f"{index}-2"] = 100.0

    summary = summarize(apply_scores(rows, scores))

    assert summary["changed_rank1_exact_rows"] == 2
    assert summary["changed_rank1_rows"] == 5
    assert summary["changed_rank1_exact_precision"] == pytest.approx(0.4)
    assert summary["decision"] == "STOP_GENERIC_RERANKING_ON_THIS_COHORT"


def test_summary_refuses_baseline_drift() -> None:
    """Red proof disables ``validate_baseline`` and accepts a rank one count mismatch."""

    rows = _baseline_rows()
    rows[0]["original_min_exact_rank"] = 3

    with pytest.raises(RuntimeError, match="original dense apparatus mismatch"):
        summarize(apply_scores(rows, _scores(rows)))
