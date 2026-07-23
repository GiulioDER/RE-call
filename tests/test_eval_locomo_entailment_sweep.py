"""The entailment-sweep's pure logic: threshold math and entailment-column detection.

The model passes and the DB reads are exercised by hand; these pin the two places a silent bug
would corrupt the ROC — the abstain-at-threshold rule (off-by-one on the inequality inverts the
whole curve) and reading the entailment class from a model's config (a wrong column scores
contradictions and reports them as answers).
"""
from __future__ import annotations

import pytest

from recall.eval.locomo_entailment_sweep import (
    THRESHOLD_GRID,
    QuestionScores,
    _abstains_at,
    _entailment_index,
    sweep,
)


def test_abstains_when_best_score_below_threshold() -> None:
    q = QuestionScores(is_adversarial=True, max_entail=0.4)
    assert _abstains_at(q, 0.5) is True
    assert _abstains_at(q, 0.3) is False


def test_abstains_at_exact_threshold_is_false() -> None:
    # apply_entailment keeps an ok hit whose score >= threshold, so equality does NOT abstain.
    q = QuestionScores(is_adversarial=False, max_entail=0.5)
    assert _abstains_at(q, 0.5) is False


def test_no_ok_hit_abstains_at_every_threshold() -> None:
    # max_entail None = the default path already abstained; the judge cannot un-abstain it.
    q = QuestionScores(is_adversarial=True, max_entail=None)
    assert all(_abstains_at(q, t) for t in THRESHOLD_GRID)


def test_none_is_not_a_zero_score() -> None:
    # A real score of 0.0 abstains only above threshold 0; None abstains at 0 too. Keeping them
    # distinct is the whole reason max_entail is Optional.
    real_zero = QuestionScores(is_adversarial=False, max_entail=0.0)
    no_hit = QuestionScores(is_adversarial=False, max_entail=None)
    assert _abstains_at(real_zero, 0.0) is False
    assert _abstains_at(no_hit, 0.0) is True


def test_sweep_computes_separation_and_covers_grid() -> None:
    scores = [
        QuestionScores(is_adversarial=True, max_entail=0.2),   # abstains except at t<=0.2
        QuestionScores(is_adversarial=True, max_entail=0.9),   # abstains only at t>0.9
        QuestionScores(is_adversarial=False, max_entail=0.95),  # rarely abstains (answerable, good)
        QuestionScores(is_adversarial=False, max_entail=0.99),
    ]
    points = sweep(scores)
    assert len(points) == len(THRESHOLD_GRID)
    for p in points:
        adv = p["adversarial_abstention"]["rate"]
        ans = p["answerable_false_abstain"]["rate"]
        assert p["separation"] == pytest.approx(round(adv - ans, 4))
        assert p["adversarial_abstention"]["n"] == 2
        assert p["answerable_false_abstain"]["n"] == 2


def test_entailment_index_reads_label_not_position() -> None:
    # Column order differs across NLI checkpoints; the index must come from the label text.
    assert _entailment_index({0: "contradiction", 1: "entailment", 2: "neutral"}) == 1
    assert _entailment_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0
    assert _entailment_index({0: "ENTAILMENT", 1: "NEUTRAL"}) == 0


def test_entailment_index_raises_when_absent() -> None:
    with pytest.raises(ValueError, match="no entailment label"):
        _entailment_index({0: "LABEL_0", 1: "LABEL_1"})
