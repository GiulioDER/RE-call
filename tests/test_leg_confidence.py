from __future__ import annotations

import pytest

from recall.eval.legconf import leg_confidence, more_decisive


def test_leg_confidence_is_affine_invariant():
    """The load-bearing property: cosine and ts_rank live on different scales, and the
    trigger compares them directly. If conf() were not affine-invariant, that comparison
    would measure the units rather than the decisiveness."""
    base = [0.90, 0.50, 0.40, 0.35, 0.20]
    expected = leg_confidence(base)
    for a, b in [(2.0, 0.0), (0.5, 0.0), (1.0, 10.0), (3.0, -7.5), (1000.0, 42.0)]:
        scaled = [a * s + b for s in base]
        assert leg_confidence(scaled) == pytest.approx(expected, rel=1e-9)


def test_leg_confidence_is_higher_for_a_peaked_leg():
    peaked = [0.9, 0.2, 0.2, 0.2, 0.2]
    flat = [0.5, 0.5, 0.5, 0.5, 0.4]
    assert leg_confidence(peaked) > leg_confidence(flat)


def test_leg_confidence_is_zero_when_flat():
    assert leg_confidence([0.4, 0.4, 0.4, 0.4]) == 0.0


@pytest.mark.parametrize("scores", [[], [0.7]])
def test_leg_confidence_is_zero_without_spread(scores):
    """Empty leg or a single candidate: no spread exists, so there is no decisiveness to
    report. Zero means 'no information', and because the trigger uses a STRICT >, a leg in
    this state can never fire it."""
    assert leg_confidence(scores) == 0.0


def test_leg_confidence_is_never_negative():
    # the max is always >= the mean, so the z-score of the max cannot be negative
    for scores in ([0.1, 0.9], [-5.0, -1.0, -3.0], [0.5, 0.5, 0.51]):
        assert leg_confidence(scores) >= 0.0


def test_more_decisive_compares_at_a_common_depth():
    """A sparse leg that is genuinely peaked but short must be able to beat a longer, flatter
    dense leg. At its natural (untruncated) length the sparse leg would lose to the dense leg's
    larger n even with zero real signal — the whole point of the common-depth comparison is that
    it doesn't."""
    sparse = [0.9, 0.1, 0.1]
    dense = [0.5] * 17 + [0.55, 0.6, 0.65]
    assert more_decisive(sparse, dense) is True


def test_more_decisive_is_not_fooled_by_leg_length():
    """Regression guard for the defect itself: two legs drawn from the SAME distribution but at
    different lengths must compare the same as their equal-length prefixes, once truncated to a
    common depth."""
    dense = [i / 20 for i in range(20)]
    sparse = [i / 20 for i in range(5)]
    assert more_decisive(sparse, dense) == (
        leg_confidence(sorted(sparse, reverse=True)[:5])
        > leg_confidence(sorted(dense, reverse=True)[:5])
    )


@pytest.mark.parametrize(
    ("sparse", "dense"),
    [
        ([], [0.5, 0.4]),
        ([0.9], [0.5, 0.4]),
        ([], []),
    ],
)
def test_more_decisive_false_when_common_depth_below_two(sparse, dense):
    assert more_decisive(sparse, dense) is False


def test_more_decisive_does_not_mutate_inputs():
    sparse = [0.9, 0.1, 0.1]
    dense = [0.5] * 17 + [0.55, 0.6, 0.65]
    sparse_copy = list(sparse)
    dense_copy = list(dense)
    more_decisive(sparse, dense)
    assert sparse == sparse_copy
    assert dense == dense_copy
