"""2x2 per rung, lambda-pricing, and the H1 gate that can kill the benchmark.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

`answered_answerable` is deliberately not called `correct_answer`: v1 has no judge, so answering
an answerable question counts as success WITHOUT the content being checked. Every v1 accuracy is
therefore an upper bound, and the field name is where that is enforced.
"""
from __future__ import annotations

import pytest

from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    RING_ORIGINAL,
    Instance,
)
from benchmarks.ladder.score import (
    Cell,
    confusion_by_ring,
    correct_abstain_rate,
    h1_verdict,
    lambda_cost,
    paired_difference_ci,
)


def _i(instance_id: str, label: str, ring: int, pair_id: str) -> Instance:
    return Instance(
        instance_id=instance_id, corpus="locomo", source_question_id=pair_id,
        question="q", label=label, ring=ring,
        excised_doc_ids=("g",) if label == LABEL_UNANSWERABLE else (),
        gold_doc_ids=("g",), pair_id=pair_id,
    )


def test_unanswerable_abstention_is_a_correct_abstain():
    cells = confusion_by_ring([_i("a", LABEL_UNANSWERABLE, 0, "p1")], {"a": True})
    assert cells[0].correct_abstain == 1
    assert cells[0].false_answer == 0


def test_unanswerable_answered_is_a_false_answer():
    cells = confusion_by_ring([_i("a", LABEL_UNANSWERABLE, 0, "p1")], {"a": False})
    assert cells[0].false_answer == 1


def test_answerable_abstention_is_a_false_abstain():
    cells = confusion_by_ring([_i("o", LABEL_ANSWERABLE, RING_ORIGINAL, "p1")], {"o": True})
    assert cells[RING_ORIGINAL].false_abstain == 1


def test_answerable_answered_counts_as_answered_not_correct():
    cells = confusion_by_ring([_i("o", LABEL_ANSWERABLE, RING_ORIGINAL, "p1")], {"o": False})
    assert cells[RING_ORIGINAL].answered_answerable == 1
    assert not hasattr(cells[RING_ORIGINAL], "correct_answer")


def test_instances_with_no_recorded_response_are_skipped_not_counted_as_abstentions():
    """A missing row is missing data. Counting it as an abstention would flatter a crashed run."""
    assert confusion_by_ring([_i("a", LABEL_UNANSWERABLE, 0, "p1")], {}) == {}


def test_lambda_weights_a_false_answer_more_heavily_as_lambda_rises():
    cell = Cell(correct_abstain=0, false_answer=1, false_abstain=1, answered_answerable=0)
    assert lambda_cost(cell, 10.0) > lambda_cost(cell, 1.0)


def test_lambda_one_weights_the_two_errors_equally():
    a = Cell(correct_abstain=0, false_answer=2, false_abstain=0, answered_answerable=0)
    b = Cell(correct_abstain=0, false_answer=0, false_abstain=2, answered_answerable=0)
    assert lambda_cost(a, 1.0) == lambda_cost(b, 1.0)


def test_correct_abstain_rate_is_over_unanswerable_instances_only():
    cell = Cell(correct_abstain=3, false_answer=1, false_abstain=5, answered_answerable=5)
    assert correct_abstain_rate(cell) == 0.75


def test_h1_passes_only_when_the_ci_excludes_zero_and_the_gap_is_large():
    assert h1_verdict(0.30, 0.20, 0.40) == "PASS"
    assert h1_verdict(0.30, -0.05, 0.60) == "FAIL"   # CI includes zero
    assert h1_verdict(0.05, 0.01, 0.09) == "FAIL"    # significant but below threshold


def test_paired_difference_uses_only_questions_present_at_both_rungs():
    instances = [
        _i("p1#d0", LABEL_UNANSWERABLE, 0, "p1"),
        _i("p1#dmax", LABEL_UNANSWERABLE, RING_MAX, "p1"),
        _i("p2#d0", LABEL_UNANSWERABLE, 0, "p2"),  # no d=max partner
    ]
    abstained = {"p1#d0": False, "p1#dmax": True, "p2#d0": False}
    diff, low, high = paired_difference_ci(instances, abstained, 0, RING_MAX, iterations=200)
    assert diff == 1.0  # the one paired question flips 0 -> 1
    assert low <= diff <= high


def test_no_shared_questions_raises_instead_of_returning_a_fake_null():
    """(0.0, 0.0, 0.0) is bit-identical to a tightly-measured null, and h1_verdict reads it as
    FAIL — the kill condition. Absent data must never be publishable as a result."""
    instances = [
        _i("p1#d0", LABEL_UNANSWERABLE, 0, "p1"),
        _i("p2#dmax", LABEL_UNANSWERABLE, RING_MAX, "p2"),  # different pair_id: no overlap
    ]
    abstained = {"p1#d0": False, "p2#dmax": True}
    with pytest.raises(ValueError, match="no question appears at BOTH"):
        paired_difference_ci(instances, abstained, 0, RING_MAX, iterations=50)


def test_an_empty_response_map_raises_rather_than_reporting_a_verdict():
    instances = [
        _i("p1#d0", LABEL_UNANSWERABLE, 0, "p1"),
        _i("p1#dmax", LABEL_UNANSWERABLE, RING_MAX, "p1"),
    ]
    with pytest.raises(ValueError, match="no question appears at BOTH"):
        paired_difference_ci(instances, {}, 0, RING_MAX, iterations=50)


def test_a_single_shared_question_still_computes(tmp_path=None):
    """The guard must reject EMPTY, not merely small — n=1 is weak evidence, not absent data."""
    instances = [
        _i("p1#d0", LABEL_UNANSWERABLE, 0, "p1"),
        _i("p1#dmax", LABEL_UNANSWERABLE, RING_MAX, "p1"),
    ]
    abstained = {"p1#d0": False, "p1#dmax": True}
    diff, low, high = paired_difference_ci(instances, abstained, 0, RING_MAX, iterations=50)
    assert diff == 1.0
