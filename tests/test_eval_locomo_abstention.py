"""The abstention ablation's question-partitioning logic.

The scoring itself needs a live pgvector and a cross-encoder and is run by hand; these pin the one
piece of pure logic that decides WHICH questions land in which arm — a silent bug here would score
adversarial questions as answerable (or drop evidence-less ones into the retrieval arm), quietly
corrupting both rates the experiment reports.
"""
from __future__ import annotations

import random

from recall.eval.locomo_abstention import MODES, _partition_questions


def _q(cat, question="q?", evidence=None):
    d = {"category": cat, "question": question}
    if evidence is not None:
        d["evidence"] = evidence
    return d


def test_answerable_needs_evidence() -> None:
    # An answerable question with no gold evidence cannot be scored for retrieval and must not
    # enter the answerable arm — it would otherwise count as a false-abstain with no ground truth.
    qa = [
        _q(1, "has evidence", ["D1:1"]),
        _q(2, "no evidence"),
        _q(4, "empty evidence", []),
    ]
    answerable, adversarial = _partition_questions(qa, 0, random.Random(0))
    assert [a["question"] for a in answerable] == ["has evidence"]
    assert adversarial == []


def test_adversarial_is_category_five_regardless_of_evidence() -> None:
    # Category 5 questions carry `adversarial_answer`, not `evidence`. They must all be collected
    # for the abstention arm even though they have no evidence key.
    qa = [_q(5, "adv1"), _q(5, "adv2"), _q(1, "ans", ["D1:1"])]
    answerable, adversarial = _partition_questions(qa, 0, random.Random(0))
    assert len(adversarial) == 2
    assert len(answerable) == 1


def test_answerable_sample_caps_count_deterministically() -> None:
    qa = [_q(1, f"q{i}", ["D1:1"]) for i in range(100)]
    a1, _ = _partition_questions(qa, 40, random.Random(0))
    a2, _ = _partition_questions(qa, 40, random.Random(0))
    assert len(a1) == 40
    # Same seed -> same sample, so the false-abstain denominator is reproducible run to run.
    assert [q["question"] for q in a1] == [q["question"] for q in a2]


def test_answerable_sample_zero_means_all() -> None:
    qa = [_q(1, f"q{i}", ["D1:1"]) for i in range(12)]
    answerable, _ = _partition_questions(qa, 0, random.Random(0))
    assert len(answerable) == 12


def test_sample_larger_than_pool_keeps_all() -> None:
    qa = [_q(1, f"q{i}", ["D1:1"]) for i in range(5)]
    answerable, _ = _partition_questions(qa, 40, random.Random(0))
    assert len(answerable) == 5


def test_questions_without_text_are_dropped() -> None:
    qa = [_q(1, "", ["D1:1"]), _q(5, "")]
    answerable, adversarial = _partition_questions(qa, 0, random.Random(0))
    assert answerable == []
    assert adversarial == []


def test_modes_are_the_expected_four() -> None:
    assert MODES == ("default", "calibrated", "entail", "both")
