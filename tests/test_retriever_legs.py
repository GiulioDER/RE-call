"""`_retrieve_legs` is the seam `search` and `search_fused` share.

If it drifts from what `search` used to do inline, every existing caller changes behaviour
silently. These tests pin the seam itself rather than the methods built on it.
"""

from __future__ import annotations

from recall.retriever import HybridRetriever, _Legs
from tests.fakes import FakeEmbedder, FakeStore


def test_retrieve_legs_reports_sparse_cosines_against_the_query_by_default() -> None:
    """Default behaviour must match what `search` did inline: sparse hits carry the query cosine."""
    store = FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)])
    retriever = HybridRetriever(store, FakeEmbedder(), sparse_backend="lexical")

    legs = retriever._retrieve_legs("what is x", source=None)

    assert isinstance(legs, _Legs)
    assert store.sparse_vec_used == legs.qvec


def test_retrieve_legs_can_report_against_a_different_vector() -> None:
    """`search_fused` needs the HISTORY variant's sparse legs to report the QUERY's cosine.

    Without this the returned scores mix two bases, and `cal.confidence()` in trust.py silently
    receives a cosine measured against a different string.
    """
    store = FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)])
    retriever = HybridRetriever(store, FakeEmbedder(), sparse_backend="lexical")
    other = [0.5] * FakeEmbedder().dim

    legs = retriever._retrieve_legs("history text", source=None, report_vec=other)

    assert store.sparse_vec_used == other
    assert legs.qvec != other  # the leg still RANKS by its own embedding
