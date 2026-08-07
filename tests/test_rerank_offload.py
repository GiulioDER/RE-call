"""`rerank_order` must reproduce `CrossEncoderReranker.rerank`'s ordering exactly.

If it does not, every reranked number this harness reports is a number RE-call would never
compute, and nothing about the output would look wrong.
"""

from __future__ import annotations

import pytest

from benchmarks.mtrag.rerank_offload import compare_orderings, rerank_order


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


def test_compare_orderings_reports_no_failure_when_orders_match():
    failure, tie = compare_orderings(
        local=["a", "b", "c"],
        offloaded=["a", "b", "c"],
        local_by_id={"a": 3.0, "b": 2.0, "c": 1.0},
        task_id="t1",
    )
    assert failure is None
    assert tie is False


def test_compare_orderings_flags_a_top_k_order_difference():
    failure, tie = compare_orderings(
        local=["a", "b", "c"],
        offloaded=["b", "a", "c"],
        local_by_id={"a": 3.0, "b": 2.0, "c": 1.0},
        task_id="t1",
    )
    assert failure == {"task_id": "t1", "why": "top-10 order differs"}
    assert tie is False


def test_compare_orderings_reports_a_deep_tie_rather_than_a_failure():
    """Past the metric cutoffs a swap is a near-tie and is information, not failure. The first
    version of this gate demanded exact ordering over the whole pool and COULD NOT PASS: CUDA and
    CPU do not produce bit-identical floats."""
    local = [f"d{i}" for i in range(120)]
    offloaded = local[:100] + [local[101], local[100]] + local[102:]
    failure, tie = compare_orderings(
        local=local,
        offloaded=offloaded,
        local_by_id={c: float(1000 - i) for i, c in enumerate(local)},
        task_id="t1",
    )
    assert failure is None
    assert tie is True
