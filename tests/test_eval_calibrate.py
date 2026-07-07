from recall.eval.calibrate import best_threshold


def test_best_threshold_separates_cleanly():
    ans = [0.80, 0.90, 0.85]
    unans = [0.20, 0.30, 0.10]
    thr = best_threshold(ans, unans)
    assert all(a >= thr for a in ans)
    assert all(u < thr for u in unans)


def test_best_threshold_returns_a_candidate_on_overlap():
    ans = [0.60, 0.70]
    unans = [0.55, 0.65]
    thr = best_threshold(ans, unans)
    assert thr in {round(x, 3) for x in ans + unans}
