"""Percentiles are nearest-rank, and one rank matters.

`int(q * n)` looks like a percentile index and is not: it IS the 1-based nearest rank, so using
it to subscript a 0-based list returns the next sample up. The error is invisible in review —
the expression is short, plausible, and produces a number of the right magnitude — and it is
invisible in the output too, because a tail one rank worse than reality still looks like a tail.

The tell is that p99 becomes indistinguishable from max on any sample of 100.
"""
from __future__ import annotations

import math

import pytest

from recall.observability import percentile


def _nearest_rank(sorted_samples: list[float], q: float) -> float:
    """Reference implementation, written from the definition rather than from the code."""
    n = len(sorted_samples)
    rank = math.ceil(q * n)  # 1-based
    return sorted_samples[min(n, max(1, rank)) - 1]


@pytest.mark.parametrize("q", [0.5, 0.95, 0.99])
@pytest.mark.parametrize("n", [1, 2, 3, 10, 99, 100, 101, 1000])
def test_percentile_matches_the_definition(n, q):
    """Against the definition, at every size — not against a remembered index expression."""
    samples = [float(i) for i in range(1, n + 1)]
    assert percentile(samples, q) == pytest.approx(_nearest_rank(samples, q))


def test_p99_of_a_hundred_samples_is_not_the_maximum():
    """The concrete symptom, stated as its own test because it is the one you can eyeball.

    With the off-by-one, p99 of 1..100 returns 100.0 — the largest sample — so the reported p99
    can never distinguish "1% of requests are slow" from "one request was slow".
    """
    samples = [float(i) for i in range(1, 101)]

    assert percentile(samples, 0.99) == 99.0
    assert percentile(samples, 0.99) != max(samples)
    assert percentile(samples, 0.95) == 95.0
    assert percentile(samples, 0.50) == 50.0


def test_percentiles_are_ordered_and_within_the_sample():
    """Invariants that hold whatever the convention: monotone in q, and never invented."""
    samples = sorted([0.4, 1.1, 1.2, 3.0, 7.5, 9.9, 12.0, 40.0, 41.0, 900.0])

    p50, p95, p99 = (percentile(samples, q) for q in (0.50, 0.95, 0.99))

    assert p50 <= p95 <= p99
    assert all(v in samples for v in (p50, p95, p99)), "a nearest-rank percentile is a real sample"
    assert percentile(samples, 1.0) == max(samples)


def test_degenerate_inputs():
    assert math.isnan(percentile([], 0.95))
    assert percentile([42.0], 0.99) == 42.0
    assert percentile([1.0, 2.0], 0.0) == 1.0  # q=0 clamps to the first sample, not to index -1


def test_the_scale_report_uses_the_same_convention():
    """The eval module carried a second copy of the formula, so it carried the same bug.

    Pinned here because fixing one copy and publishing from the other is exactly how the wrong
    number reached results/*.md in the first place.
    """
    from recall.eval.scale import _percentiles

    samples = [float(i) for i in range(1, 101)]
    got = _percentiles(samples)

    assert got == {"p50": 50.0, "p95": 95.0, "p99": 99.0}
