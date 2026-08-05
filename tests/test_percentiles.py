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


def test_the_shared_latency_report_uses_the_same_convention():
    """The helper both eval harnesses publish through — not the harnesses themselves.

    They each published `{"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * len(lat))]}`, the same
    1-based-rank-as-0-based-index error in both slots. This file pins the FORMULA; that
    `labelled` and `longmemeval_perq` actually publish through it is a separate claim and is
    pinned end to end in tests/test_eval_latency_percentiles.py.

    n=100 is the size that makes the failure unmissable rather than arguable.
    """
    from recall.eval.metrics import latency_report

    samples = [float(i) for i in range(1, 101)]

    assert latency_report(samples) == {"p50": 50.0, "p95": 95.0}
    # The old expressions, evaluated here so the disagreement is visible rather than asserted:
    # index 50 -> 51.0 and index 95 -> 96.0, both one rank above the definition.
    assert samples[len(samples) // 2] == 51.0
    assert samples[int(0.95 * len(samples))] == 96.0


@pytest.mark.parametrize("n", [2, 20, 44])
def test_each_slot_is_pinned_at_a_size_where_it_actually_discriminates(n):
    """p50 and p95 fail at DIFFERENT sizes, so no single fixture pins both.

    Measured: `n // 2` is one rank high for every EVEN n, while `int(0.95 * n)` is one rank high
    only when n is a multiple of 20. A fixture at n=2 separates the two p50 implementations and
    says nothing whatever about p95, because there the old and new p95 index coincide — so a
    docstring claiming n=2 "separates the two implementations" would be true of one slot and
    false of the other. n=20 is the smallest size at which the p95 slot can fail. n=44 is the
    PEPs arm, where p50 moves and p95 does not, which is why that published table needed its p50
    annotated and not its p95.
    """
    from recall.eval.metrics import latency_report

    samples = [float(i) for i in range(1, n + 1)]
    got = latency_report(samples)

    assert got["p50"] == _nearest_rank(samples, 0.50)
    assert got["p95"] == _nearest_rank(samples, 0.95)
    # And the disagreement with the replaced expression, at each size, asserted rather than implied.
    assert (got["p50"] != samples[len(samples) // 2]) is (n % 2 == 0)
    assert (got["p95"] != samples[int(0.95 * n)]) is (n % 20 == 0)


def test_the_percentile_helper_can_return_an_unrounded_sample():
    """`ndigits=None`, because rounding twice is not rounding once.

    `percentile` rounds to 3 dp by default, and a publisher that quantises to 0.1 ms on top of
    that disagrees with a single round on about one value in two hundred. `latency_report` says
    it publishes "to 0.1 ms", so it has to be able to round exactly once, or that is a label the
    function does not honour.
    """
    x = 1402.6496351897192

    assert percentile([x], 0.50, ndigits=None) == x
    assert percentile([x], 0.50) == round(x, 3)          # the default is unchanged
    assert round(percentile([x], 0.50, ndigits=None), 1) == round(x, 1) == 1402.6
    assert round(percentile([x], 0.50), 1) == 1402.7, "the double round this parameter avoids"


def test_the_latency_report_rounds_once():
    """The same value through the shipped helper — the assertion that would have failed."""
    from recall.eval.metrics import latency_report

    x = 1402.6496351897192
    assert latency_report([x]) == {"p50": 1402.6, "p95": 1402.6} == {"p50": round(x, 1),
                                                                    "p95": round(x, 1)}


def test_the_latency_report_takes_an_unsorted_sample_and_reports_nothing_on_an_empty_one():
    """It sorts internally — the callers hand it raw per-question timings, not a sorted list.

    A percentile helper that silently assumes sorted input is a landmine: the wrong answer it
    returns is still a plausible latency. And an empty sample reports `{}`, which is what both
    harnesses already published for a run with no timed queries — a latency of 0.0 ms would read
    as an instantaneous system rather than as an unmeasured one.
    """
    from recall.eval.metrics import latency_report

    assert latency_report([30.0, 10.0, 20.0]) == latency_report([10.0, 20.0, 30.0])
    assert latency_report([30.0, 10.0, 20.0]) == {"p50": 20.0, "p95": 30.0}
    assert latency_report([]) == {}
