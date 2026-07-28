from __future__ import annotations

import pytest

from recall.eval.legconf import leg_confidence


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
