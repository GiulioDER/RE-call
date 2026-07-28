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

from recall.eval.gap_study import (
    analyse_records,
    headroom_capture,
    holm_adjust,
    partial_spearman,
    permutation_p,
    predictor_correlations,
    spearman,
)


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


def test_headroom_capture_equalises_two_corpora_at_very_different_ceilings():
    """Why this response variable exists, in one assertion.

    Corpus A: local 0.50 -> cloud 0.75, a raw gap of +0.25. Corpus B: local 0.90 -> cloud 0.95, a
    raw gap of +0.05 — five times smaller, and it would rank near the bottom on the raw gap. But
    both cloud models captured exactly half the room that was left. The raw gap partly measures
    how much headroom a corpus happened to have; this measures what was done with it.
    """
    assert headroom_capture(0.50, 0.75) == pytest.approx(0.5)
    assert headroom_capture(0.90, 0.95) == pytest.approx(0.5)


def test_headroom_capture_is_signed_when_the_cloud_model_loses():
    # §8 publishes a configuration where the cloud embedder loses. A response variable that could
    # not represent that would quietly drop the honest half of the result.
    assert headroom_capture(0.60, 0.50) == pytest.approx(-0.25)


def test_headroom_capture_is_nan_at_a_saturated_ceiling():
    # No room left means the question "how much of the room was captured" has no answer. Zero
    # would read as "the cloud model achieved nothing", which is a different claim.
    assert math.isnan(headroom_capture(1.0, 1.0))


def test_predictor_correlations_reports_each_unordered_pair_once():
    pairs = predictor_correlations({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [4, 3, 2, 1]})
    assert set(pairs) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_predictor_correlations_exposes_predictors_that_are_the_same_measurement():
    """The claim "three distinct hypotheses" is empirical, so it gets checked rather than asserted.

    If the predictors cluster, Holm over three is also the wrong correction — it assumes
    independent tests and is over-conservative for correlated ones.
    """
    pairs = predictor_correlations({
        "oov": [0.1, 0.2, 0.3, 0.4, 0.5],
        "twin": [0.11, 0.21, 0.31, 0.41, 0.51],   # the same measurement, rescaled
        "other": [0.5, 0.1, 0.4, 0.2, 0.3],
    })
    assert pairs[("oov", "twin")] == pytest.approx(1.0)
    assert abs(pairs[("oov", "other")]) < 0.5


def _record(name, local, cloud, oov, overlap, crowd):
    return {"status": "ok", "dataset": name,
            "scores": {"local": {"hybrid": local}, "cloud": {"hybrid": cloud}},
            "predictors": {"oov_rate": oov, "query_overlap": overlap, "crowding": crowd}}


def test_analyse_records_runs_the_preregistered_analysis_end_to_end():
    # oov_rate predicts the gap and is INDEPENDENT of the local score. That independence is the
    # point: making oov a function of i alongside local makes the two collinear, the control then
    # absorbs the predictor entirely, and the partial correlation destabilises. A predictor that
    # only moves with the null model cannot be shown to beat it, by construction.
    rng = random.Random(11)
    records = []
    for i in range(18):
        local = 0.30 + 0.03 * i
        oov = rng.random()
        cloud = min(0.99, local + 0.30 * oov + rng.gauss(0, 0.01))
        records.append(_record(f"c{i}", local, cloud, oov, rng.random(), rng.random()))

    out = analyse_records(records, arm="hybrid")
    assert out["n"] == 18
    assert out["predictors"]["oov_rate"]["holm_p"] < 0.05
    assert out["predictors"]["query_overlap"]["holm_p"] > 0.05
    # The null model is reported as a number, not assumed: how much the local score alone explains.
    assert "null_spearman" in out
    # Collinearity with the control is reported per predictor. Near +/-1 the partial correlation
    # is arithmetically unstable and the study cannot answer the question for that predictor —
    # which must be visible rather than showing up as a confidently wrong p-value.
    assert abs(out["predictors"]["oov_rate"]["control_collinearity"]) < 0.9
    # Both response variables, always — the secondary is not contingent on the primary's result.
    assert "headroom" in out["responses"]


def test_analyse_records_ignores_failed_records_and_says_so():
    records = [_record(f"c{i}", 0.3, 0.5, 0.4, 0.5, 0.6) for i in range(4)]
    records.append({"status": "failed", "dataset": "broken", "error": "404"})
    out = analyse_records(records, arm="hybrid")
    assert out["n"] == 4
    assert out["excluded"] == ["broken"]
    # n=4 is far under the floor; a null here means underpowered and the output must carry that.
    assert out["underpowered"] is True


def test_analyse_records_surfaces_haystack_size_as_a_measured_confound():
    """A common subsample cap compresses haystack size across corpora; it does not equalise it.

    Corpora smaller than the cap are used whole, so document counts still range about 5.5x. A
    bigger haystack is a harder haystack and can move the gap on its own, so the residual is
    reported as a number rather than carried in prose as a caveat nobody can check.
    """
    rng = random.Random(5)
    records = []
    for i in range(16):
        n_docs = 4000 + 1000 * i          # the haystack, deliberately varied
        local = 0.5 + rng.gauss(0, 0.02)  # and NOT confounded with the local score
        cloud = min(0.99, local + 0.00002 * n_docs + rng.gauss(0, 0.005))
        rec = _record(f"c{i}", local, cloud, rng.random(), rng.random(), rng.random())
        rec["predictors"]["n_documents"] = n_docs
        records.append(rec)

    out = analyse_records(records, arm="hybrid")
    assert abs(out["haystack_confound"]["spearman"]) > 0.5
    assert "n_documents" in out["haystack_confound"]["range"]


def test_analyse_records_reports_no_haystack_confound_when_sizes_do_not_track_the_gap():
    rng = random.Random(9)
    records = []
    for i in range(16):
        local = 0.4 + 0.02 * i
        rec = _record(f"c{i}", local, min(0.99, local + 0.1 + rng.gauss(0, 0.02)),
                      rng.random(), rng.random(), rng.random())
        rec["predictors"]["n_documents"] = rng.randint(4000, 20000)
        records.append(rec)
    assert abs(analyse_records(records, arm="hybrid")["haystack_confound"]["spearman"]) < 0.5
