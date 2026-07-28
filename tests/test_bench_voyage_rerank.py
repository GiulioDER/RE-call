"""VoyageReranker — reorder-only, identity-preserving, offline via an injected fake client."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from benchmarks.voyage_rerank import VoyageReranker
from recall.types import Chunk, ScoredChunk


class _FakeResult:
    def __init__(self, indices: list[int]) -> None:
        self.results = [_FakeItem(i) for i in indices]


class _FakeItem:
    def __init__(self, index: int) -> None:
        self.index = index
        self.relevance_score = 1.0


class _FakeClient:
    """Records the call and returns a caller-specified reordering of the input documents."""

    def __init__(self, order: list[int]) -> None:
        self.order = order
        self.calls: list[dict[str, Any]] = []

    def rerank(self, query: str, documents: list[str], model: str, top_k: int) -> _FakeResult:
        self.calls.append({"query": query, "documents": documents, "model": model, "top_k": top_k})
        return _FakeResult(self.order[:top_k])


def _hit(chunk_id: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=chunk_id, source="mem", text=text, metadata={}),
        score=score,
        indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_empty_hits_return_unchanged_without_calling_the_client() -> None:
    client = _FakeClient(order=[])
    reranker = VoyageReranker(client=client)
    assert reranker.rerank("q", []) == []
    assert client.calls == []


def test_reorders_original_objects_by_the_returned_index_order() -> None:
    hits = [_hit("a", "alpha", 0.9), _hit("b", "bravo", 0.8), _hit("c", "charlie", 0.7)]
    client = _FakeClient(order=[2, 0, 1])  # charlie, alpha, bravo
    out = VoyageReranker(client=client).rerank("q", hits)

    assert [h.chunk.id for h in out] == ["c", "a", "b"]
    # identity preserved — reordering, not reconstruction
    assert out[0] is hits[2] and out[1] is hits[0] and out[2] is hits[1]
    # scores and timestamps untouched (the trust layer reads .score as a cosine)
    assert [h.score for h in out] == [0.7, 0.9, 0.8]
    assert all(h.indexed_at == hits[0].indexed_at for h in out)


def test_passes_query_texts_and_model_to_the_client() -> None:
    hits = [_hit("a", "alpha", 0.9), _hit("b", "bravo", 0.8)]
    client = _FakeClient(order=[0, 1])
    VoyageReranker(model="rerank-2.5", client=client).rerank("how many rps?", hits)

    call = client.calls[0]
    assert call["query"] == "how many rps?"
    assert call["documents"] == ["alpha", "bravo"]
    assert call["model"] == "rerank-2.5"
    assert call["top_k"] == 2  # defaults to len(hits): rerank ALL, trust layer truncates to k


def test_explicit_top_k_returns_exactly_that_many_still_identity_preserving() -> None:
    hits = [_hit("a", "alpha", 0.9), _hit("b", "bravo", 0.8), _hit("c", "charlie", 0.7)]
    client = _FakeClient(order=[2, 0, 1])
    out = VoyageReranker(top_k=2, client=client).rerank("q", hits)

    assert [h.chunk.id for h in out] == ["c", "a"]
    assert out[0] is hits[2] and out[1] is hits[0]
    assert client.calls[0]["top_k"] == 2


def test_missing_key_and_no_client_raises_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageReranker()
