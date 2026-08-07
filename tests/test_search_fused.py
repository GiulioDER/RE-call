"""`search_fused` refuses rather than degrades.

The measured gain is CONDITIONAL on reranking: raw, this arm is worse by 0.0447 nDCG@5 and
tripped three preregistered ranking vetoes. RE-call ships with the reranker OFF by default, so an
operator enabling fusion without one would silently get a worse system than `search()`.
"""

from __future__ import annotations

import pytest

from recall.retriever import HybridRetriever
from tests.fakes import FakeEmbedder, FakeStore


class _StubReranker:
    def rerank(self, query, hits):
        return list(hits)


def _retriever(reranker=None):
    return HybridRetriever(
        FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)]),
        FakeEmbedder(),
        reranker=reranker,
        sparse_backend="lexical",
    )


def test_fused_search_without_a_reranker_is_refused() -> None:
    """The gain does not exist without reranking, so serving it without one is a worse system."""
    with pytest.raises(ValueError, match="reranker"):
        _retriever().search_fused("q", ["earlier turn"])


def test_empty_history_is_refused_rather_than_silently_becoming_a_single_query() -> None:
    """A caller wanting single-query behaviour should call `search`, not get it by accident.

    Matched on "non-empty history", text unique to this guard's message. A later guard's message
    also contains the word "history" (it fires on a history that stripped to no usable text), so
    matching on that word alone would still pass if this guard were deleted and execution fell
    through to the later one.
    """
    with pytest.raises(ValueError, match="non-empty history"):
        _retriever(_StubReranker()).search_fused("q", [])


def test_an_over_budget_history_is_refused_and_names_both_lengths() -> None:
    """Refused, never truncated.

    A truncated history is a configuration the benchmark never tested, served under the measured
    configuration's name. Same principle as `resolve_reranker` refusing an unparseable flag rather
    than reading it as "off".
    """
    with pytest.raises(ValueError, match="4096"):
        _retriever(_StubReranker()).search_fused("q", ["x" * 5000])


def test_k_below_one_is_refused() -> None:
    """The message must name the actual value, not just the expected one, so a caller passing

    `k=-3` and a caller passing `k=0` get distinguishable text rather than identical, ungreppable
    ones.
    """
    with pytest.raises(ValueError, match="k must be >= 1") as exc_info:
        _retriever(_StubReranker()).search_fused("q", ["earlier"], k=0)
    assert "got 0" in str(exc_info.value)


def test_a_bare_string_history_is_refused_rather_than_iterated_character_by_character() -> None:
    """A bare `str` satisfies `Sequence[str]` at runtime, so the type annotation alone does not

    stop a caller from passing one turn as a string instead of a one-element list. Without a
    guard, `build_history_query` would iterate it character by character and join each character
    with a newline, and retrieval would silently proceed on that garbage.
    """
    with pytest.raises(ValueError, match="list|sequence"):
        _retriever(_StubReranker()).search_fused("q", "an earlier turn")
