"""The paired statistics, tested at the degenerate cases rather than only the happy one.

Each test below was watched to fail against a deliberately broken version of the function, because
a test nobody has watched fail has not been tested. The mutations used are named in each docstring.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.agent_ab.stats import (
    MIN_PAIRS,
    compare_binary,
    compare_continuous,
    mcnemar_exact,
    paired_bootstrap,
    wilcoxon_signed_rank,
)


def test_bootstrap_refuses_a_constant_sample():
    """Mutation: returning `(d, d)` instead of None. A zero-width interval reads as precision."""

    assert paired_bootstrap([2.0] * 20) is None


def test_bootstrap_refuses_a_single_pair():
    assert paired_bootstrap([1.0]) is None


def test_bootstrap_brackets_a_known_mean():
    deltas = [-5.0, -4.0, -6.0, -5.0, -7.0, -3.0, -5.0, -4.0]
    interval = paired_bootstrap(deltas)
    assert interval is not None
    low, high = interval
    assert low < sum(deltas) / len(deltas) < high
    assert high < 0.0, "an all-negative sample must not produce an interval spanning zero"


def test_mcnemar_uses_only_discordant_pairs():
    """Concordant pairs carry no information; a version that counted them would shift p."""

    on = [True, True, False, False, False, False, False, False]
    off = [True, True, True, True, True, True, False, False]
    p, b, c = mcnemar_exact(on, off)
    assert (b, c) == (0, 4)
    assert p is not None and p < 0.13


def test_mcnemar_is_undefined_when_every_pair_agrees():
    """Mutation: returning 1.0. That asserts 'no difference' from evidence that says nothing."""

    p, b, c = mcnemar_exact([True] * 10, [True] * 10)
    assert p is None
    assert (b, c) == (0, 0)


def test_mcnemar_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_exact([True], [True, False])


def test_wilcoxon_is_undefined_for_all_zero_differences():
    """scipy returns nan with a warning here, which serialises as a number. None does not."""

    assert wilcoxon_signed_rank([0.0] * 10) is None


def test_compare_binary_refuses_below_the_pair_floor():
    """Under MIN_PAIRS the exact test cannot reach significance at any split of the pairs."""

    pairs = [(False, True)] * (MIN_PAIRS - 1)
    result = compare_binary("trap", pairs)
    assert result.p_value is None
    assert "sample size" in result.note
    # The rates are still reported: they are descriptive and do not claim significance.
    assert result.on_mean == 0.0
    assert result.off_mean == 1.0


def test_compare_binary_reports_a_clean_separation():
    pairs = [(False, True)] * 10
    result = compare_binary("trap", pairs)
    assert result.p_value is not None and result.p_value < 0.01
    assert result.delta_mean == -1.0
    assert "on-only=0, off-only=10" in result.note


def test_compare_continuous_drops_missing_measurements_rather_than_zeroing_them():
    """Mutation: treating None as 0.0. That reports an unmeasured session as a free one."""

    pairs: list[tuple[float, float]] = [(100.0, 200.0)] * 8
    pairs.append((None, 200.0))  # type: ignore[arg-type]
    result = compare_continuous("input_tokens", pairs)
    assert result.n_pairs == 8
    assert "1 pair(s) dropped" in result.note
    assert result.on_mean == 100.0


def test_compare_continuous_detects_a_consistent_reduction():
    pairs = [(90.0, 150.0), (95.0, 160.0), (80.0, 140.0), (100.0, 170.0),
             (85.0, 155.0), (92.0, 148.0), (88.0, 162.0), (97.0, 151.0)]
    result = compare_continuous("input_tokens", pairs)
    assert result.delta_mean is not None and result.delta_mean < 0
    assert result.p_value is not None and result.p_value < 0.05
    assert result.delta_ci is not None and result.delta_ci[1] < 0


def test_result_serialises_without_nan():
    result = compare_binary("trap", [(True, True)] * 10)
    payload = result.to_dict()
    assert payload["p_value"] is None
    assert payload["delta_ci"] is None
    assert not any(
        isinstance(v, float) and math.isnan(v) for v in payload.values()
    ), "a nan in the artifact is not valid JSON and reads as a measured value"
