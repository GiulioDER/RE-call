from __future__ import annotations

import pytest

from recall.fusion import leg_confidence, weighted_rrf


def _rrf_reference(rankings, k=60):
    """The shipped unweighted formula, inlined so the equivalence test cannot drift with it."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _order(s):
    return sorted(s, key=lambda c: (-s[c], c))


def test_equal_weights_preserve_the_shipped_ORDER():
    """The backward-compatibility guarantee. Equal weights scale every fused score by the same
    constant, which cannot reorder anything — so today's behaviour is a special case of the new
    function, and shipping this cannot silently change what existing users get."""
    a = ["d1", "d2", "d3", "d4"]
    b = ["d3", "d1", "d9", "d2"]
    ref = _rrf_reference([a, b])
    got = weighted_rrf([a, b], weights=[0.5, 0.5])

    assert set(got) == set(ref)
    assert _order(got) == _order(ref)


def test_leg_confidence_is_importable_from_the_new_home():
    """`leg_confidence` moved to `recall.fusion` verbatim — this is the proof the new home
    actually works, not just that the name resolves at import time."""
    assert leg_confidence([0.9, 0.1, 0.1]) > 0.0


def test_weights_none_is_uniform():
    a, b = ["d1", "d2"], ["d2", "d3"]
    assert weighted_rrf([a, b]) == pytest.approx(weighted_rrf([a, b], weights=[0.5, 0.5]))


def test_a_heavier_leg_pulls_its_own_ranking_up():
    """The whole point: when one leg is trusted more, its ordering should dominate the prefix."""
    dense = ["d_top", "x", "y"]
    sparse = ["s_top", "x", "y"]
    dense_heavy = weighted_rrf([dense, sparse], weights=[0.9, 0.1])
    sparse_heavy = weighted_rrf([dense, sparse], weights=[0.1, 0.9])

    assert dense_heavy["d_top"] > dense_heavy["s_top"]
    assert sparse_heavy["s_top"] > sparse_heavy["d_top"]


def test_a_zero_weight_leg_contributes_nothing():
    a, b = ["d1", "d2"], ["d3"]
    got = weighted_rrf([a, b], weights=[1.0, 0.0])
    assert got["d3"] == 0.0
    assert got["d1"] > 0.0


def test_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        weighted_rrf([["a"], ["b"]], weights=[1.0])


def test_empty_rankings_give_empty_scores():
    assert weighted_rrf([[], []], weights=[0.5, 0.5]) == {}
