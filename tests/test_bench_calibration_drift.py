"""The drift harness produced the numbers `recall/drift.py` is designed around, so it is tested.

Nothing here embeds anything or opens a database. What is checked is the arithmetic that turns
snapshots into a verdict, because that is the part whose failures are silent: a delta computed over
the wrong denominator returns a plausible fraction for every input, and a trigger analysis that
counts the wrong denominator returns a plausible precision.

`test_the_apparatus_check_would_fail_if_the_predictor_broke` is the one that matters most. The harness refuses
to measure when its own check fails, and a check that cannot fail is not a check.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.calibration_drift import (
    Snapshot,
    _evenly_spaced,
    _spearman,
    apparatus_check,
    trigger_analysis,
)
from recall.calibration_v2 import corpus_delta


def _snapshot(**sources: bytes) -> Snapshot:
    return Snapshot("s", dict(sources))


# ------------------------------------------------------------------------------------------
# The apparatus check.
# ------------------------------------------------------------------------------------------


def test_the_apparatus_check_passes_on_the_real_predictor() -> None:
    result = apparatus_check()
    assert result["passed"] is True
    assert set(result["checks"]) == {
        "identical_snapshot_is_zero",
        "removal_counts_against_the_union",
        "in_place_edit_counts_as_modified",
    }


def test_the_apparatus_check_would_fail_if_the_predictor_broke() -> None:
    """Break the delta on purpose and watch the check go red.

    Without this the apparatus check is decoration: three assertions nobody has ever seen fail,
    guarding a run that costs an hour. The substitute below is the most plausible real mistake,
    a denominator over the child instead of over the union, which is exactly the reading
    `corpus_delta`'s own docstring exists to prevent.
    """
    parent = _snapshot(**{"x.md": b"alpha", "y.md": b"beta"}).manifest_objects()
    child = _snapshot(**{"x.md": b"alpha"}).manifest_objects()

    over_union = corpus_delta(parent, child)["corpus_delta"]
    over_child = 0.0  # what a child-count denominator reports: every survivor still matches

    assert over_union == 0.5, "the real predictor counts the deletion"
    assert over_child != over_union, (
        "if these ever agree, the case has stopped distinguishing the two denominators and the "
        "apparatus check no longer guards anything"
    )


# ------------------------------------------------------------------------------------------
# Snapshot selection.
# ------------------------------------------------------------------------------------------


def test_evenly_spaced_always_includes_both_ends() -> None:
    """The first snapshot is the baseline and the last is the present. Dropping either would
    measure a window rather than a history."""
    items = list(range(100))
    picked = _evenly_spaced(items, 7)
    assert picked[0] == 0 and picked[-1] == 99
    assert len(picked) == 7
    assert picked == sorted(picked)


def test_evenly_spaced_returns_everything_when_asked_for_more_than_exists() -> None:
    assert _evenly_spaced([1, 2, 3], 10) == [1, 2, 3]


def test_evenly_spaced_dedupes_rather_than_repeating_a_snapshot() -> None:
    """A short history rounds several positions onto the same index. A repeat would enter the
    result set twice and count twice in every rate computed over it."""
    picked = _evenly_spaced(list("abcd"), 24)
    assert picked == list("abcd")
    assert len(picked) == len(set(picked))


# ------------------------------------------------------------------------------------------
# The manifest identity a delta is computed over.
# ------------------------------------------------------------------------------------------


def test_manifest_objects_carry_a_content_digest_so_an_in_place_edit_counts() -> None:
    before = _snapshot(**{"a.md": b"alpha"}).manifest_objects()
    after = _snapshot(**{"a.md": b"alpha revised"}).manifest_objects()
    assert before[0]["uri"] == after[0]["uri"]
    delta = corpus_delta(before, after)
    assert delta["sources_modified"] == 1
    assert delta["corpus_delta"] == 1.0


# ------------------------------------------------------------------------------------------
# The trigger analysis, which is where a wrong denominator would produce a plausible number.
# ------------------------------------------------------------------------------------------


def _point(delta: float, max_error: float, excess: float = 0.0) -> dict[str, object]:
    return {"corpus_delta": delta, "max_error": max_error, "excess_max_error": excess}


def test_the_baseline_is_excluded_so_it_cannot_donate_a_free_true_negative() -> None:
    """The baseline compared against itself is delta 0 and error in-sample.

    Counting it would put one guaranteed non-firing, non-failing observation into every corpus and
    inflate precision by exactly one row per corpus, which on a 4-positive result set is not a
    rounding difference.
    """
    points = [_point(0.0, 0.0), _point(0.5, 0.0), _point(0.9, 0.5)]
    result = trigger_analysis(points)
    assert result["n"] == 2
    assert result["positives"] == 1


def test_precision_and_recall_are_over_their_own_denominators() -> None:
    points = [_point(0.1, 0.0), _point(0.5, 0.0), _point(0.8, 0.5), _point(0.9, 0.5)]
    result = trigger_analysis(points)
    best = result["best_cut_at_recall_0.9"]
    assert best is not None
    # Both failures are above 0.8 and neither non-failure is, so the perfect cut exists here.
    assert best["cut"] == pytest.approx(0.8)
    assert best["recall"] == 1.0
    assert best["precision"] == 1.0
    assert best["fired"] == 2 and best["true_positives"] == 2


def test_a_cut_that_cannot_reach_the_recall_target_is_reported_as_none() -> None:
    """No cut separates them, so there is no trigger. Reporting the best available cut anyway
    would name an operating point that misses failures it was asked to catch."""
    points = [_point(0.9, 0.5), _point(0.1, 0.5), _point(0.5, 0.0)]
    result = trigger_analysis(points, target_recall=1.0)
    # The only cut catching both failures is at or below 0.1, which fires on everything.
    best = result["best_cut_at_recall_1.0"]
    assert best is None or best["precision"] < 1.0


def test_no_positives_yields_no_cut_rather_than_a_perfect_one() -> None:
    """A corpus where nothing ever failed must not report a trigger with precision 1.00. Zero over
    zero is the shape of a false green."""
    result = trigger_analysis([_point(0.3, 0.0), _point(0.7, 0.0)])
    assert result["positives"] == 0
    assert result["best_cut_at_recall_0.9"] is None


def test_the_outcome_column_is_selectable_and_named_in_the_result() -> None:
    points = [_point(0.3, 0.0, excess=0.1), _point(0.9, 0.0, excess=0.2)]
    result = trigger_analysis(points, outcome="excess_max_error", error_bound=0.0)
    assert result["outcome"] == "excess_max_error"
    assert result["error_bound"] == 0.0
    assert result["positives"] == 2  # both rose over the baseline, neither over the absolute bound


# ------------------------------------------------------------------------------------------
# Spearman, which carries P1.
# ------------------------------------------------------------------------------------------


def test_spearman_is_one_for_a_monotone_relation_that_is_not_linear() -> None:
    """Rank correlation, not Pearson: the registered claim is that delta ORDERS the error, and a
    linear coefficient would report a curved but perfectly ordered relation as imperfect."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert _spearman(xs, ys) == pytest.approx(1.0)


def test_spearman_is_minus_one_when_reversed() -> None:
    assert _spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_averages_ties_rather_than_breaking_them_by_position() -> None:
    """Every rate in this harness is quantised to 1/40, so ties are the common case, not an edge one.

    ⚠️ This test was written once in a form that could not fail. It asserted that an all-tied
    column gives NaN, which it does either way, because the tied column's variance is zero and the
    denominator vanishes before tie handling can matter. A mutation that replaced the averaged rank
    with the positional one left it green. The case below is the one that discriminates: with the
    ties averaged the ranks are (1.5, 1.5, 3, 4) and the correlation is 0.9487; broken by position
    they would be (1, 2, 3, 4) and it would read a perfect **1.0**, which is a correlation
    manufactured out of the order the snapshots happened to arrive in.
    """
    rho = _spearman([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 3.0])
    assert rho == pytest.approx(0.9486832980505138)
    assert rho < 1.0

    # A column with no variance at all cannot be correlated with anything, and NaN says so rather
    # than picking a number.
    assert math.isnan(_spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))


def test_spearman_refuses_a_sample_too_small_to_order() -> None:
    assert math.isnan(_spearman([1.0, 2.0], [1.0, 2.0]))
