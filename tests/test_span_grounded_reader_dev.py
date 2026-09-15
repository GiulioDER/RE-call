from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_span_grounded_reader_dev import (
    EXPECTED_CONFIG,
    EXPECTED_LICENCE,
    EXPECTED_MODEL,
    apply_reader,
    load_reader_output,
    prepare_input,
    summarize,
)


def _candidate(index: int, rank: int, *, exact: bool = False, gold: bool = False) -> dict[str, object]:
    return {
        "chunk_id": f"{index}-{rank}",
        "source": f"source-{index}-{rank}.md",
        "text": f"chunk {index} rank {rank} answer text",
        "dense_score": float(21 - rank),
        "gold_source": gold,
        "exact_span": exact,
    }


def _row(index: int, *, answerable: bool, eligible: bool, gold_rank: int | None, exact_rank: int | None) -> dict[str, object]:
    candidates = [
        _candidate(
            index,
            rank,
            exact=answerable and rank == exact_rank,
            gold=answerable and rank in {gold_rank, exact_rank},
        )
        for rank in range(1, 21)
    ]
    return {
        "query_index": index,
        "query": f"question {index}",
        "answer_span": f"chunk {index} rank {exact_rank} answer text" if exact_rank else "missing answer",
        "expected_answerability": "answerable" if answerable else "unanswerable",
        "eligible": eligible,
        "original_min_gold_rank": gold_rank,
        "original_min_exact_rank": exact_rank,
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
            gold_rank=gold_ranks[index],
            exact_rank=exact_ranks[index],
        )
        for index in range(15)
    ]
    controls = [
        _row(index + 15, answerable=False, eligible=False, gold_rank=None, exact_rank=None)
        for index in range(15)
    ]
    return answerable + controls


def _null(qid: int) -> dict[str, object]:
    return {
        "qid": str(qid),
        "selected_doc_id": None,
        "answer": "",
        "start": None,
        "end": None,
        "margin": -1.0,
        "candidates_scored": 20,
    }


def _selection(row: dict[str, object], rank: int) -> dict[str, object]:
    candidate = row["candidates"][rank - 1]
    text = candidate["text"]
    return {
        "qid": str(row["query_index"]),
        "selected_doc_id": candidate["chunk_id"],
        "answer": text,
        "start": 0,
        "end": len(text),
        "margin": 2.0,
        "candidates_scored": 20,
    }


def test_prepare_input_excludes_every_gold_and_eligibility_field(tmp_path: Path) -> None:
    """Red proof includes ``eligible`` and fails on the serialized private label."""
    rows = _baseline_rows()
    output = tmp_path / "input.jsonl"
    counts = prepare_input({"rows": rows}, output)

    text = output.read_text(encoding="utf-8")
    assert counts == {"queries": 30, "pairs": 600}
    assert "gold_source" not in text
    assert "exact_span" not in text
    assert "expected_answerability" not in text
    assert "answer_span" not in text
    assert "eligible" not in text


def test_load_reader_output_requires_frozen_model_and_config(tmp_path: Path) -> None:
    """Red proof accepts a wrong revision and fails to raise at the intended identity gate."""
    output = tmp_path / "scores.jsonl"
    output.write_text(
        json.dumps(
            {
                "_header": True,
                "model": EXPECTED_MODEL,
                "revision": "wrong",
                "licence": EXPECTED_LICENCE,
                "transformers_version": "1",
                "torch_version": "2",
                "config": EXPECTED_CONFIG,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="wrong model identity"):
        load_reader_output(output)


def test_apply_reader_refuses_nonliteral_quote() -> None:
    """Red proof skips offset verification and accepts text the selected chunk does not contain."""
    row = _baseline_rows()[0]
    result = _selection(row, 1)
    result["answer"] = "fabricated"

    with pytest.raises(RuntimeError, match="exact selected chunk substring"):
        apply_reader([row], {"0": result})


def _passing_results(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    results = {str(row["query_index"]): _null(int(row["query_index"])) for row in rows}
    for index in range(7):
        rank = int(rows[index]["original_min_gold_rank"] or 1)
        if index in {3, 4}:
            rank = 5
        results[str(index)] = _selection(rows[index], rank)
    return results


def test_summary_proceeds_only_when_every_reader_gate_passes() -> None:
    """Red proof raises the exact bearing floor to six and returns the stop verdict."""
    rows = _baseline_rows()
    measured = apply_reader(rows, _passing_results(rows))

    summary = summarize(measured)

    assert summary["anchor_gated_reader"]["selected_gold_source_rows"] == 7
    assert summary["anchor_gated_reader"]["selected_exact_bearing_rows"] == 5
    assert summary["selected_exact_gains"] == 2
    assert summary["selected_exact_losses"] == 0
    assert summary["decision"] == "PROCEED_FRESH_SPAN_GROUNDED_READER_VALIDATION"


def test_summary_stops_on_control_activation_or_exact_loss() -> None:
    """Red proof removes both safety terms and incorrectly returns the proceed verdict."""
    rows = _baseline_rows()
    results = _passing_results(rows)
    results["0"] = _selection(rows[0], 2)
    results["5"] = _selection(rows[5], 10)
    results["7"] = _selection(rows[7], 10)
    measured = apply_reader(rows, results)
    measured[15]["gated_selection"] = _selection(rows[15], 1)

    summary = summarize(measured)

    assert summary["anchor_gated_reader"]["control_activations"] == 1
    assert summary["selected_exact_losses"] == 1
    assert summary["decision"] == "STOP_OFF_THE_SHELF_READER_ON_THIS_COHORT"


def test_summary_refuses_baseline_drift() -> None:
    """Red proof disables collection validation and lets the wrong dense baseline pass."""
    rows = _baseline_rows()
    rows[0]["original_min_exact_rank"] = 3

    with pytest.raises(RuntimeError, match="original dense apparatus mismatch"):
        summarize(apply_reader(rows, _passing_results(rows)))
