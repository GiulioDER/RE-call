from recall.eval.calibrate import best_threshold


def test_best_threshold_separates_cleanly():
    ans = [0.80, 0.90, 0.85]
    unans = [0.20, 0.30, 0.10]
    thr = best_threshold(ans, unans)
    assert all(a >= thr for a in ans)
    assert all(u < thr for u in unans)


def test_best_threshold_bisects_the_overlap_instead_of_collapsing_onto_one_class():
    """With no clean gap the boundary splits the overlap rather than sitting on a sample.

    The old rule returned one of the observed values — necessarily the lowest answerable one,
    which is exactly why it carried no margin. The midpoint is the least-bad boundary when the
    two classes genuinely overlap.
    """
    ans = [0.60, 0.70]
    unans = [0.55, 0.65]
    thr = best_threshold(ans, unans)
    assert min(unans) < thr < max(ans)


def test_best_threshold_leaves_margin_on_the_answerable_side():
    """The defect this rule replaces: the boundary must sit BELOW the weakest answerable sample,
    so a real answer scoring slightly under it is still served."""
    ans = [0.80, 0.90, 0.85]
    unans = [0.20, 0.30, 0.10]
    assert best_threshold(ans, unans) < min(ans)


def test_a_single_answerable_outlier_does_not_relocate_the_threshold():
    """One unlucky retrieval must not move the operating point (issue #26).

    Needs a realistic calibration set: the floor is the 5th percentile, so at least ~20 samples
    are required before it excludes anything at all (see the small-n test below).
    """
    ans = [0.80 + i * 0.002 for i in range(40)]
    unans = [0.10 + i * 0.002 for i in range(20)]
    baseline = best_threshold(ans, unans)
    with_outlier = best_threshold([0.31] + ans, unans)
    assert abs(with_outlier - baseline) <= 0.01


def test_small_calibration_sets_cannot_be_outlier_robust_and_do_not_pretend_to_be():
    """A 5% tail is not identifiable from a handful of samples, so the floor collapses onto the
    minimum. Pinned rather than hidden: a corpus with 14 answerable queries (the shipped one)
    gets margin from bisecting the gap, but NOT outlier robustness.
    """
    ans = [0.80, 0.85, 0.90]
    unans = [0.10, 0.20, 0.30]
    baseline = best_threshold(ans, unans)
    with_outlier = best_threshold([0.31] + ans, unans)
    assert with_outlier < baseline  # the outlier does move it, at this sample size
    assert with_outlier < 0.31      # but margin below the outlier still survives
