import json
import math

import pytest

from recall.eval.metrics import (
    false_confident_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a", "c"], 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a", "b", "c"], ["b"], 1) == 0.0  # top-1 is 'a'
    assert precision_at_k(["a"], ["a"], 0) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "x"], 3) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], ["a"], 2) == 1.0
    assert recall_at_k(["a"], [], 2) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], ["b"]) == pytest.approx(0.5)
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["a", "b"], ["c"]) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "b"], ["a"], 2) == pytest.approx(1.0)
    # relevant item at rank 2: dcg = 1/log2(3), idcg = 1/log2(2) = 1.0
    assert ndcg_at_k(["a", "b"], ["b"], 2) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(["a", "b"], ["z"], 2) == 0.0


def test_false_confident_rate():
    # 2 of 4 unanswerable queries had gap_warning=False (wrongly confident)
    assert false_confident_rate([True, True, False, False]) == pytest.approx(0.5)
    assert false_confident_rate([True, True]) == 0.0  # guard fired on all -> good


def test_false_confident_rate_is_nan_on_empty():
    """No unanswerable queries is NOT a perfect guard score.

    0.0-on-empty reads as "the guard never failed" — an unearned pass published beside
    genuinely measured rates. Every other rate in this module returns NaN so the publisher
    renders 'n/a'; this one must agree.
    """
    assert math.isnan(false_confident_rate([]))


def test_superseded_trust_rate():
    from recall.eval.metrics import superseded_trust_rate

    assert superseded_trust_rate([True, True, False, False]) == 0.5
    assert math.isnan(superseded_trust_rate([]))  # a rate with no data is NOT a (perfect) score
    assert superseded_trust_rate([False]) == 0.0


def test_successor_accuracy_and_abstention_accuracy():
    from recall.eval.metrics import abstention_accuracy, fraction_true, successor_accuracy

    assert successor_accuracy([True, False]) == 0.5
    assert math.isnan(successor_accuracy([]))
    assert abstention_accuracy([True, True]) == 1.0
    assert math.isnan(abstention_accuracy([]))
    assert fraction_true([True, False, False, False]) == 0.25


def test_near_miss_false_confident_rate():
    from recall.eval.metrics import near_miss_false_confident_rate

    # flag True = the system answered a near-miss query confidently (the failure)
    assert near_miss_false_confident_rate([True, True, False, False]) == 0.5
    assert near_miss_false_confident_rate([False, False]) == 0.0  # abstained on all -> good
    assert math.isnan(near_miss_false_confident_rate([]))  # no data is NOT a perfect score


def test_false_abstain_rate():
    from recall.eval.metrics import false_abstain_rate

    # flag True = the system abstained on an ANSWERABLE query (the regression to watch)
    assert false_abstain_rate([True, False, False, False]) == 0.25
    assert false_abstain_rate([False, False]) == 0.0
    assert math.isnan(false_abstain_rate([]))


def test_bootstrap_ci_brackets_the_point_estimate():
    from recall.eval.metrics import bootstrap_ci, fraction_true

    flags = [True] * 7 + [False] * 3  # point estimate 0.70
    lo, hi = bootstrap_ci(flags, n=2000, seed=1)
    assert lo <= fraction_true(flags) <= hi
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < hi  # a non-degenerate sample yields a real interval


def test_bootstrap_ci_is_deterministic_for_a_seed():
    from recall.eval.metrics import bootstrap_ci

    flags = [True, False, True, True, False, False, True, False]
    assert bootstrap_ci(flags, seed=42) == bootstrap_ci(flags, seed=42)


def test_bootstrap_ci_degenerate_and_empty():
    from recall.eval.metrics import bootstrap_ci

    assert bootstrap_ci([True, True, True]) == (1.0, 1.0)  # no variance -> point interval
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)  # no data -> no interval


def test_wilson_ci_reports_uncertainty_where_the_bootstrap_cannot():
    """The reason this metric exists: a degenerate sample still has uncertainty.

    Every resample of an all-True sample is all-True, so the percentile bootstrap returns
    [1.00, 1.00] — "certain" from n=2. Wilson is derived from the normal approximation to the
    binomial rather than from resampling, so it widens as n shrinks instead of collapsing.
    """
    from recall.eval.metrics import bootstrap_ci, wilson_ci

    assert bootstrap_ci([True, True]) == (1.0, 1.0)  # the failure mode being replaced
    lo, hi = wilson_ci([True, True])
    assert lo < hi
    assert hi == pytest.approx(1.0)
    assert lo < 0.5  # n=2 is nearly no evidence; the band must say so


def test_wilson_ci_is_wider_for_less_data():
    from recall.eval.metrics import wilson_ci

    lo_small, hi_small = wilson_ci([True] * 4)
    lo_big, hi_big = wilson_ci([True] * 40)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_wilson_ci_brackets_the_point_estimate_and_stays_in_range():
    from recall.eval.metrics import fraction_true, wilson_ci

    flags = [True] * 7 + [False] * 3
    lo, hi = wilson_ci(flags)
    assert 0.0 <= lo <= fraction_true(flags) <= hi <= 1.0


def test_wilson_ci_known_value():
    """Cross-check against the closed form: 7/10 at 95% is [0.3968, 0.8922] (3 s.f.)."""
    from recall.eval.metrics import wilson_ci

    lo, hi = wilson_ci([True] * 7 + [False] * 3)
    assert lo == pytest.approx(0.3968, abs=1e-3)
    assert hi == pytest.approx(0.8922, abs=1e-3)


def test_wilson_ci_empty_is_nan():
    from recall.eval.metrics import wilson_ci

    lo, hi = wilson_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_nan_to_null_sanitises_the_mapping_types_this_repo_actually_returns():
    """`proposal_precision_recall` returns a MappingProxyType, not a dict.

    A sanitiser that only recognises `dict` passes the proxy through untouched, so the NaN
    survives and `json.dumps` then dies on the proxy itself rather than on the NaN.
    """
    from types import MappingProxyType

    from recall.eval.metrics import nan_to_null
    from recall.reasoning_proposals import proposal_precision_recall

    proxy = nan_to_null(MappingProxyType({"precision": float("nan")}))
    assert proxy == {"precision": None}

    scored = nan_to_null({"precision_recall": proposal_precision_recall([], {("a", "b")})})
    assert scored["precision_recall"]["precision"] is None
    assert json.dumps(scored, allow_nan=False)


def test_nan_to_null_sanitises_sets_and_nested_tuples():
    from recall.eval.metrics import nan_to_null

    assert nan_to_null({"counts": (1, float("-inf"))}) == {"counts": [1, None]}
    assert nan_to_null(frozenset({1, 2})) == [1, 2]


def test_nan_to_null_raises_on_a_cycle_rather_than_nulling_it_away():
    """Loud, not quiet. Returning None for the cyclic member would publish a valid LOOKING
    artifact with data silently replaced, and `write_json` treats an existing file as a
    completed corpus. A RecursionError is unacceptable, but so is a plausible wrong file:
    raise in json.dumps' own vocabulary instead."""
    from recall.eval.metrics import nan_to_null

    cyclic = {"precision": float("nan")}
    cyclic["self"] = cyclic

    with pytest.raises(ValueError, match="Circular reference detected"):
        nan_to_null(cyclic)


def test_nan_to_null_keeps_a_shared_subtree_on_both_paths():
    """The cycle guard is path scoped, so the same object referenced twice as SIBLINGS is not
    a cycle and must survive on both paths rather than being pruned to None."""
    from recall.eval.metrics import nan_to_null

    shared = {"rate": float("nan"), "n": 3}

    assert nan_to_null({"a": shared, "b": shared}) == {
        "a": {"rate": None, "n": 3},
        "b": {"rate": None, "n": 3},
    }


def test_nan_to_null_keeps_the_order_of_ordered_set_like_views():
    """`dict.keys()` and `dict.items()` register as collections.abc.Set but are ORDERED.
    Sorting them buys no determinism they did not already have and destroys insertion order."""
    from recall.eval.metrics import nan_to_null

    source = {"zeta": 1, "alpha": 2, "mid": float("nan")}

    assert nan_to_null(source.keys()) == ["zeta", "alpha", "mid"]
    assert nan_to_null(source.items()) == [["zeta", 1], ["alpha", 2], ["mid", None]]


def test_nan_to_null_leaves_a_non_finite_mapping_key_for_the_serializer_to_reject():
    """Keys are out of scope by design: the NaN key survives and `allow_nan=False` rejects it,
    which is the loud failure we want rather than a silently rewritten key."""
    from recall.eval.metrics import nan_to_null

    sanitised = nan_to_null({float("nan"): "x"})

    assert [type(k) for k in sanitised] == [float]
    with pytest.raises(ValueError, match="Out of range float"):
        json.dumps(sanitised, allow_nan=False)


def test_nan_to_null_orders_any_unordered_set_deterministically():
    """Not just `set`/`frozenset`. A hash backed `collections.abc.Set` has the same per process
    random order, so the rule keys off "is it an ordered view", not off the concrete type."""
    import collections.abc

    from recall.eval.metrics import nan_to_null

    class HashBackedSet(collections.abc.Set):
        def __init__(self, items):
            self._items = set(items)

        def __contains__(self, item):
            return item in self._items

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

    assert nan_to_null(HashBackedSet(["delta", "beta", "gamma", "alpha"])) == [
        "alpha",
        "beta",
        "delta",
        "gamma",
    ]


def test_nan_to_null_orders_sets_deterministically():
    """Set iteration order varies per process under hash randomisation, and both callers write
    git tracked artifacts, so an unordered list would rewrite the file on every run."""
    from recall.eval.metrics import nan_to_null

    assert nan_to_null({"d": {"scifact", "nfcorpus", "fiqa"}}) == {
        "d": ["fiqa", "nfcorpus", "scifact"]
    }


def test_nan_to_null_leaves_strings_and_finite_numbers_alone():
    from recall.eval.metrics import nan_to_null

    assert nan_to_null({"arm": "recall", "k": 5, "rate": 0.42}) == {
        "arm": "recall",
        "k": 5,
        "rate": 0.42,
    }
