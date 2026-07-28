"""The blind labelling set must actually be blind, and the verdict reader must not invent data.

The annotator is the author of one of the two systems being compared. Unblinded labels would be
worthless regardless of how carefully they were made — "he graded his own system" ends the
conversation. So the properties below are not stylistic: they are what makes the result publishable.
"""
from __future__ import annotations

import pytest

from benchmarks.labelling.build_beam_labelling import build
from benchmarks.labelling.score_beam_labels import mcnemar_exact, read_verdict, score

OURS = [
    {"question_id": "1M_0_q0_abstention", "question_type": "abstention", "question": "Q0?",
     "rubric": ["no info about X"], "generated_answer": "I don't know.", "score": 1.0},
    {"question_id": "1M_0_q1_temporal_reasoning", "question_type": "temporal_reasoning",
     "question": "Q1?", "rubric": ["7 days"], "generated_answer": "14 days", "score": 0.0},
    {"question_id": "1M_0_q2_summarization", "question_type": "summarization", "question": "Q2?",
     "rubric": ["a summary"], "generated_answer": "summary", "score": 1.0},
]
THEIRS = [
    {"question_id": "1M_0_q0_abstention", "question_type": "abstention", "question": "Q0?",
     "rubric": ["no info about X"], "ground_truth_answer": "no info",
     "cutoff_results": {"top_200": {"generated_answer": "It was blue.", "score": 0.0}}},
    {"question_id": "1M_0_q1_temporal_reasoning", "question_type": "temporal_reasoning",
     "question": "Q1?", "rubric": ["7 days"], "ground_truth_answer": "7 days",
     "cutoff_results": {"top_200": {"generated_answer": "7 days", "score": 1.0}}},
    {"question_id": "1M_0_q2_summarization", "question_type": "summarization", "question": "Q2?",
     "rubric": ["a summary"], "ground_truth_answer": "a summary",
     "cutoff_results": {"top_200": {"generated_answer": "summary", "score": 1.0}}},
    {"question_id": "1M_9_q9_unmatched", "question_type": "abstention", "question": "Qx?",
     "rubric": ["x"], "ground_truth_answer": "x",
     "cutoff_results": {"top_200": {"generated_answer": "x", "score": 1.0}}},
]


def _build(**kw):
    kw.setdefault("seed", 0)
    kw.setdefault("category", None)
    kw.setdefault("disagreements_only", False)
    kw.setdefault("limit", None)
    return build(OURS, THEIRS, **kw)


def test_the_csv_never_names_the_system_or_the_llm_verdict() -> None:
    rows, _ = _build()
    for row in rows:
        assert set(row) == {
            "item", "category", "question", "gold_answer",
            "predicted_answer", "your_verdict_Y_or_N",
        }
        blob = " ".join(str(v) for v in row.values()).lower()
        assert "recall" not in blob and "mem0" not in blob


def test_the_key_carries_the_mapping_the_csv_withholds() -> None:
    rows, key = _build()
    assert len(key) == len(rows)
    assert {v["arm"] for v in key.values()} == {"recall", "mem0"}


def test_every_question_contributes_exactly_one_row_per_arm() -> None:
    _, key = _build()
    per_question: dict[str, list[str]] = {}
    for meta in key.values():
        per_question.setdefault(meta["question_id"], []).append(meta["arm"])
    assert per_question, "fixture produced nothing"
    for qid, arms in per_question.items():
        assert sorted(arms) == ["mem0", "recall"], qid


def test_questions_only_one_side_answered_are_dropped() -> None:
    """A question the other arm never saw cannot be part of a paired comparison."""
    _, key = _build()
    assert all(m["question_id"] != "1M_9_q9_unmatched" for m in key.values())


def test_disagreements_only_drops_questions_both_arms_scored_alike() -> None:
    _, all_key = _build()
    _, dis_key = _build(disagreements_only=True)
    assert {m["question_id"] for m in all_key.values()} >= {m["question_id"] for m in dis_key.values()}
    # q2 was 1.0 for both arms -> carries no information -> excluded
    assert all(m["question_id"] != "1M_0_q2_summarization" for m in dis_key.values())


def test_the_shuffle_is_seeded_so_a_rebuild_reproduces_the_sheet() -> None:
    a_rows, a_key = _build(seed=7)
    b_rows, b_key = _build(seed=7)
    assert a_rows == b_rows and a_key == b_key
    c_rows, _ = _build(seed=8)
    assert [r["item"] for r in a_rows] == [r["item"] for r in c_rows]  # items are positional
    assert a_rows != c_rows, "a different seed must produce a different order"


def test_category_filter_restricts_to_one_axis() -> None:
    rows, key = _build(category="abstention")
    assert rows and all(r["category"] == "abstention" for r in rows)


def test_gold_shows_every_rubric_nugget() -> None:
    """BEAM grades against nuggets and 'covers only some' is wrong — show them all."""
    rows, _ = _build()
    assert any("no info about X" in r["gold_answer"] for r in rows)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Y", True), ("yes", True), ("Yes + extra detail", True),
        ("N", False), ("no", False),
        ("partial missing the second item", False),
        ("yes missing advices from friend", False),   # the annotator's own wording, reads incorrect
        ("no gold answer", None),                     # dataset defect, NOT a system error
        ("not sure golden is vague", None),
        ("", None), ("   ", None),
    ],
)
def test_verdicts_are_read_the_way_the_previous_round_read_them(raw, expected) -> None:
    assert read_verdict(raw) is expected


def test_undecidable_rows_are_excluded_from_the_denominator() -> None:
    _, key = _build()
    items = sorted(key, key=int)
    labels = [{"item": items[0], "your_verdict_Y_or_N": "no gold answer"}]
    labels += [{"item": i, "your_verdict_Y_or_N": "Y"} for i in items[1:]]
    result = score(labels, key)
    assert result["excluded_undecidable"] == 1
    assert result["labelled"] == len(items) - 1


def test_mcnemar_uses_only_discordant_pairs() -> None:
    assert mcnemar_exact(0, 0) is None          # nothing to separate the systems
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == pytest.approx(1.0)
