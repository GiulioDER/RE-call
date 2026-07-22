"""The calibration threshold must not be scored on the samples it was fitted to.

`best_threshold` minimises misclassification on the set it is given, so evaluating it on that
same set reports the optimiser's own objective, not the threshold's ability to generalise. On
separable data that is 0.00 by arithmetic. These tests pin the leave-one-out split that makes
the published number an out-of-sample one.
"""
import math


from recall.eval.calibrate import loo_threshold_rates


def test_in_sample_fcr_is_zero_by_construction_on_separable_data():
    """The motivating failure: why the old number could not fail."""
    from recall.calibration import best_threshold
    from recall.eval.metrics import false_confident_rate

    ans = [0.80, 0.82, 0.85, 0.88]
    unans = [0.20, 0.22, 0.25, 0.28]
    thr = best_threshold(ans, unans)
    assert false_confident_rate([u < thr for u in unans]) == 0.0  # fitted and scored on `unans`


def test_loo_confirms_the_guard_side_on_separable_data():
    """The unanswerable side does generalise when the classes are well separated."""
    fcr, _ = loo_threshold_rates([0.80, 0.82, 0.85, 0.88], [0.20, 0.22, 0.25, 0.28])
    assert fcr == 0.0


def test_the_threshold_now_has_margin_on_the_answerable_side():
    """Regression guard for the defect this rule replaced.

    The old `best_threshold` minimised misclassification, and the cheapest way to keep every
    answerable sample above the boundary is to put the boundary exactly ON the lowest one. That
    left zero margin: holding out the minimum answerable sample raised the refit boundary above
    it, so leave-one-out false-abstain was `1/n_answerable` even on PERFECTLY separable data —
    and at runtime any real answer below the weakest calibration sample abstained.

    Bisecting the gap leaves room on both sides, so no fold misclassifies its held-out sample.
    """
    ans = [0.80, 0.82, 0.85, 0.88]
    fcr, false_abstain = loo_threshold_rates(ans, [0.20, 0.22, 0.25, 0.28])
    assert false_abstain == 0.0
    assert fcr == 0.0


def test_loo_exposes_a_threshold_that_does_not_generalise():
    """Overlapping classes: the in-sample optimum flatters itself, LOO does not.

    With an unanswerable sample sitting inside the answerable range, refitting without it
    moves the boundary below it, so the held-out sample is (correctly) counted as a miss the
    in-sample number hides.
    """
    ans = [0.50, 0.60, 0.70, 0.80]
    unans = [0.10, 0.20, 0.30, 0.65]
    fcr, _ = loo_threshold_rates(ans, unans)
    assert fcr > 0.0


def test_loo_needs_at_least_two_samples_in_the_held_out_class():
    """Leaving one out of a single-sample class fits on nothing — report NaN, not a score.

    The two classes are independent: one side can be cross-validated while the other cannot.
    """
    fcr, false_abstain = loo_threshold_rates([0.8, 0.9], [0.2])
    assert math.isnan(fcr)  # only one unanswerable sample
    assert not math.isnan(false_abstain)  # two answerable samples -> the other side still runs

    fcr2, false_abstain2 = loo_threshold_rates([0.8], [0.2, 0.3])
    assert math.isnan(false_abstain2)
    assert not math.isnan(fcr2)


def test_loo_empty_is_nan():
    fcr, false_abstain = loo_threshold_rates([], [])
    assert math.isnan(fcr) and math.isnan(false_abstain)
