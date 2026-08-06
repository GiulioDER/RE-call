"""`rerank_order` must reproduce `CrossEncoderReranker.rerank`'s ordering exactly.

If it does not, every reranked number this harness reports is a number RE-call would never
compute, and nothing about the output would look wrong.
"""

from __future__ import annotations

import pytest

from benchmarks.mtrag.rerank_offload import rerank_order


def test_candidates_are_ordered_by_descending_score() -> None:
    """The basic contract, and the direction is the whole point: higher score ranks first."""
    candidates = ["a", "b", "c"]
    scores = {"a": 0.1, "b": 9.0, "c": 1.0}

    assert rerank_order(candidates, scores) == ["b", "c", "a"]


def test_ties_keep_their_fused_order() -> None:
    """`rerank()` uses a STABLE sort, so equal scores retain their retrieval ranking.

    Cross-encoder logits collide often enough for this to matter, and any other tie rule makes
    the offload diverge from the in-process reranker on exactly the pairs a metric cut at 5 or 10
    is most sensitive to.
    """
    candidates = ["first", "second", "third"]
    scores = {"first": 1.0, "second": 1.0, "third": 1.0}

    assert rerank_order(candidates, scores) == ["first", "second", "third"]


def test_a_candidate_without_a_score_raises_instead_of_scoring_zero() -> None:
    """A missing score is a broken pipeline, not a zero.

    Defaulting to 0.0 would park the candidate in the middle of the ranking, which looks entirely
    plausible: cross-encoder logits straddle zero, so the unscored document lands among the
    mediocre ones rather than at an obviously wrong extreme.
    """
    with pytest.raises(KeyError, match="no score"):
        rerank_order(["a", "b"], {"a": 1.0})


def test_negative_scores_order_correctly() -> None:
    """MiniLM logits are unbounded and frequently negative, so the sort must not assume positives."""
    candidates = ["a", "b", "c"]
    scores = {"a": -8.0, "b": -1.0, "c": -11.0}

    assert rerank_order(candidates, scores) == ["b", "a", "c"]
