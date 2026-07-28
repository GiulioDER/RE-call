from __future__ import annotations

import math

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


def test_more_decisive_truncates_from_the_top_not_the_bottom():
    """Pins the direction of the `sorted(..., reverse=True)[:m]` truncation. `sparse` is the
    shorter leg (len 3), so `m == 3` and the WHOLE of `sparse` is used regardless of which end
    the slice is taken from — only `dense`'s truncation direction is actually exercised here.

    `dense`'s top 3 candidates are an exact tie (leg_confidence == 0.0, maximally undecisive);
    its bottom 3 are a sharp single-winner spike ([0.0, 0.0, 1.0], leg_confidence ==
    sqrt(2) ~= 1.4142, the theoretical ceiling for n=3 per Samuelson's inequality). `sparse`'s
    confidence (~1.3887) sits strictly between those two: above dense's top-3 confidence, below
    dense's bottom-3 confidence. So the correct (top) truncation must yield True, and the wrong
    (bottom) truncation must yield False — a full verdict flip, not a marginal wobble.

    This test fails if the slice is taken from the wrong end.
    """
    sparse = [0.9, 0.5, 0.4]
    dense = [50.0, 50.0, 50.0, 0.0, 0.0, 1.0]

    assert leg_confidence(sorted(dense, reverse=True)[:3]) == 0.0
    assert leg_confidence(sorted(dense)[:3]) == pytest.approx(math.sqrt(2))
    assert leg_confidence(sparse) == pytest.approx(1.3887301496588274)

    assert more_decisive(sparse, dense) is True


def test_more_decisive_is_not_fooled_by_leg_length():
    """Length independence: the common-depth truncation must never let extra candidates on the
    longer leg move the verdict, no matter how many are appended — as long as they are LOW
    enough to never enter that leg's top-m slice.

    `sparse` (peaked, n=4) is unambiguously more decisive than `dense` (tightly clustered
    around 0.5, n=4) with a comfortable margin (~1.687 vs ~1.342, not a float-level tie).
    Appending a pile of very-low-scoring candidates to `dense` must not change that verdict,
    because `m = min(len(sparse), len(dense))` stays pinned to `len(sparse) == 4` and the top-4
    of the extended dense leg is identical to the top-4 of the original.
    """
    sparse = [0.9, 0.6, 0.55, 0.5]
    dense = [0.52, 0.50, 0.49, 0.51]
    dense_padded = dense + [-1000.0, -2000.0, -3000.0, -50.0, -75.0]

    assert more_decisive(sparse, dense) is True
    assert more_decisive(sparse, dense) == more_decisive(sparse, dense_padded)


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
