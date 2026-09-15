from __future__ import annotations

from scripts.run_live_guarded_spare_slot_extractive_holdout import aggregate, score_query


def _item(chunk_id: str, source: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "source": source, "text": text}


def test_extractive_scorer_uses_exact_span_without_a_judge() -> None:
    """Red proof: the deterministic holdout scorer did not exist before this test."""
    query = {
        "expected_answerability": "answerable",
        "gold_sources": ["gold.md"],
        "answer_span": "The frozen answer has eight exact words in this sentence.",
    }
    base = [_item("base", "other.md", "A related but incorrect passage.")]
    candidate = [
        *base,
        _item(
            "added",
            "gold.md",
            "Context. The frozen answer has eight exact words in this sentence. More context.",
        ),
    ]

    row = score_query(query, base, candidate)

    assert row["base_exact_span_covered"] is False
    assert row["candidate_exact_span_covered"] is True
    assert row["exact_span_gain"] is True
    assert row["source_hit_gain"] is True
    assert row["added_exact_span_items"] == 1
    assert row["base_prefix_preserved"] is True


def test_extractive_gate_rejects_any_control_activation() -> None:
    control = {
        "expected_answerability": "unanswerable",
        "gold_sources": [],
        "answer_span": "NOT_FOUND",
    }
    row = score_query(
        control,
        [],
        [_item("unsafe", "unrelated.md", "Evidence admitted for an absent identifier.")],
    )

    result = aggregate([row])

    assert result["metrics"]["control_activations"] == 1
    assert result["decision"] == "FAIL_CONTROL_ACTIVATION"
