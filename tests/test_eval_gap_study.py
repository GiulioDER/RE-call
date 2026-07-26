"""The analysis that decides the study.

`gap = voyage - bge` is mechanically anti-correlated with `bge` through the ceiling alone: a
corpus where the local model already scores 0.9 cannot show a +0.28 gap. So every candidate
predictor will correlate with the gap whether or not it explains anything, and the only question
worth asking is whether it explains variance *beyond the local model's own score*.

These tests exist to prove the analysis can tell those two situations apart. If it cannot, every
number the study produces is unfalsifiable.
"""
from __future__ import annotations

import math
import random

import pytest

from recall.eval.gap_study import holm_adjust, partial_spearman, permutation_p, spearman


def test_spearman_is_one_for_a_perfectly_monotone_relation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ranks_rather_than_fits_a_line():
    # y = x**3 is monotone but wildly non-linear. Spearman must see 1.0; Pearson would not.
    # The predictors here have no reason to be linear in the gap, so rank correlation is the
    # honest default rather than a robustness afterthought.
    x = [1, 2, 3, 4, 5]
    assert spearman(x, [v**3 for v in x]) == pytest.approx(1.0)


def test_spearman_handles_ties_with_average_ranks():
    # Ties are not exotic here: several corpora can share a hit@5 to two decimals.
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_partial_spearman_is_near_zero_when_the_relation_runs_entirely_through_the_control():
    """The load-bearing test: a spurious predictor must be caught.

    `x` and `y` are both driven by `z` and share nothing else. Their raw correlation is high and
    means nothing — this is exactly the shape of a vocabulary metric that correlates with the gap
    only because both track the local model's score. Partialling `z` out must collapse it.
    """
    rng = random.Random(20260726)
    z = [rng.gauss(0, 1) for _ in range(40)]
    x = [v + rng.gauss(0, 0.3) for v in z]
    y = [v + rng.gauss(0, 0.3) for v in z]

    assert abs(spearman(x, y)) > 0.8              # looks like a strong finding
    assert abs(partial_spearman(x, y, z)) < 0.35  # and survives nothing


def test_partial_spearman_keeps_a_relation_that_does_not_run_through_the_control():
    """The other half: a real predictor must survive. A test that only ever returns zero would
    pass the test above and be useless."""
    rng = random.Random(20260726)
    z = [rng.gauss(0, 1) for _ in range(40)]
    x = [rng.gauss(0, 1) for _ in range(40)]
    y = [zi + 2.0 * xi + rng.gauss(0, 0.2) for zi, xi in zip(z, x)]

    assert abs(partial_spearman(x, y, z)) > 0.8


def test_permutation_p_is_small_for_a_real_partial_relation_and_large_for_a_spurious_one():
    # n here is deliberately ~20, the study's actual size, so the test exercises the regime the
    # analysis will really run in rather than a comfortable one.
    rng = random.Random(7)
    z = [rng.gauss(0, 1) for _ in range(20)]
    real_x = [rng.gauss(0, 1) for _ in range(20)]
    real_y = [zi + 2.0 * xi + rng.gauss(0, 0.2) for zi, xi in zip(z, real_x)]
    spurious_x = [v + rng.gauss(0, 0.3) for v in z]
    spurious_y = [v + rng.gauss(0, 0.3) for v in z]

    assert permutation_p(real_x, real_y, z, n_perm=2000, seed=1) < 0.01
    assert permutation_p(spurious_x, spurious_y, z, n_perm=2000, seed=1) > 0.10


def test_permutation_p_is_deterministic_for_a_given_seed():
    rng = random.Random(3)
    z = [rng.gauss(0, 1) for _ in range(15)]
    x = [rng.gauss(0, 1) for _ in range(15)]
    y = [rng.gauss(0, 1) for _ in range(15)]
    assert permutation_p(x, y, z, n_perm=500, seed=42) == permutation_p(x, y, z, n_perm=500, seed=42)


def test_holm_adjust_is_step_down_not_plain_bonferroni():
    # Holm multiplies the smallest p by m, the next by m-1, and so on — uniformly less
    # conservative than Bonferroni, which would multiply every one by m. With three predictors
    # against ~20 corpora that difference decides whether anything is reportable at all.
    assert holm_adjust([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_adjust_enforces_monotonicity():
    # A later (larger raw) p can never end up adjusted below an earlier one; Holm carries the
    # running maximum forward. Without it the ordering of findings could invert under correction.
    # m=3. Sorted: 0.005*3=0.015, 0.04*2=0.08, 0.5*1=0.5 — already increasing, so the running
    # maximum changes nothing here; the result is returned in the INPUT order, not sorted order.
    assert holm_adjust([0.04, 0.005, 0.5]) == pytest.approx([0.08, 0.015, 0.5])


def test_holm_adjust_carries_the_running_maximum_forward():
    # m=3. Sorted: 0.01*3=0.03, 0.02*2=0.04, 0.021*1=0.021 — the third would come back BELOW the
    # second. Holm carries the max forward so it becomes 0.04, otherwise correction could invert
    # the ordering of findings and make the weakest result look like the strongest.
    assert holm_adjust([0.01, 0.02, 0.021]) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_adjust_caps_at_one():
    assert all(p <= 1.0 for p in holm_adjust([0.5, 0.6, 0.9]))


def test_holm_adjust_is_empty_for_no_tests():
    assert holm_adjust([]) == []


def test_analysis_functions_are_nan_safe_on_degenerate_input():
    # A constant column has no ranks to correlate. NaN is the honest answer; 0.0 would read as
    # "measured, and there is no relation".
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(partial_spearman([1, 1, 1], [1, 2, 3], [3, 2, 1]))
