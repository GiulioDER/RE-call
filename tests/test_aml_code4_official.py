"""Contracts for the frozen CAMBench Coding candidate.

RED receipt: this file was first run against commit
``2fdd3005da55210920a286768046cf61ff8aaf62``. Collection failed because
``recall_aml.code4`` did not exist. The baseline also had no Code4 profile or hosted variant.
Those are the exact omissions these tests discriminate.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from recall.embeddings import EmbeddingProfile, VoyageEmbedder
from recall.embedding_registry import registered_profile
from recall.types import Chunk, ScoredChunk
from recall_aml.code4 import (
    BM25_PROFILE,
    EMBEDDING_PROFILE,
    WORD_WINDOW_SIZE,
    WORD_WINDOW_STRIDE,
    rank_bm25_chunks,
    word_windows,
)
from recall_aml.retrieval import HostedRetriever
from recall_aml.models import AddRequest, Message
from recall_aml.service import build_chunks
from recall_aml.variants import variant


class _Embedder:
    dim = 3

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _Reranker:
    def rerank(self, _query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return hits


class _Store:
    def __init__(self) -> None:
        self.sparse_calls = 0
        self.dense_calls = 0
        self.exact_dense_calls = 0
        self.chunks = [
            Chunk(
                "z-dense-earlier-window",
                "source",
                "unrelated dense candidate",
                {"source_session_id": "sessions/a.jsonl", "segment": 0},
            ),
            Chunk(
                "a-lexical-later-window",
                "source",
                "parser parser regression fix",
                {"source_session_id": "sessions/b.jsonl", "segment": 0},
            ),
        ]

    def query_dense(self, _vector: list[float], k: int) -> list[ScoredChunk]:
        self.dense_calls += 1
        return [ScoredChunk(self.chunks[0], 0.9)][:k]

    def query_dense_exact(self, _vector: list[float], k: int) -> list[ScoredChunk]:
        self.exact_dense_calls += 1
        return [ScoredChunk(self.chunks[0], 0.9)][:k]

    def query_sparse(self, _query: str, k: int, vec=None) -> list[ScoredChunk]:
        self.sparse_calls += 1
        return [ScoredChunk(self.chunks[0], 0.8)][:k]

    def iter_chunks(self, batch_size: int = 1000):
        del batch_size
        yield from self.chunks

    def explicit_superseded_chunk_ids(self) -> frozenset[str]:
        return frozenset()


def test_code4_profile_and_variant_freeze_the_promoted_candidate() -> None:
    profile = registered_profile(EMBEDDING_PROFILE)
    behavior = variant("C6_code4_exact_bm25")
    historical = variant("C5_code4_bm25")

    assert profile.model_name == "voyage-code-4"
    assert profile.dimension == 1024
    assert profile.query_mode == "query"
    assert profile.passage_mode == "document"
    assert behavior.embedding_profile == EMBEDDING_PROFILE
    assert behavior.canonical_bm25 is True
    assert behavior.word_window_size == WORD_WINDOW_SIZE == 160
    assert behavior.word_window_stride == WORD_WINDOW_STRIDE == 120
    assert behavior.exact_dense is True
    assert behavior.stable_window_order is True
    assert behavior.content_only_windows is True
    assert historical.exact_dense is False
    assert historical.stable_window_order is False
    assert historical.content_only_windows is False
    assert behavior.compiler is False
    assert behavior.facets is False
    assert behavior.reranker is False
    assert behavior.learned_sparse is False


def test_code4_profile_sends_voyage_query_and_document_input_types(monkeypatch) -> None:
    """RED: VoyageEmbedder previously exposed only symmetric, untyped embedding calls."""
    calls: list[dict[str, object]] = []

    class _Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed(self, texts, *, model, input_type=None):
            calls.append({"texts": list(texts), "model": model, "input_type": input_type})
            return SimpleNamespace(embeddings=[[1.0, 0.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "voyageai", SimpleNamespace(Client=_Client))
    monkeypatch.setattr("recall._voyage_http.Client", _Client)
    identity = EmbeddingProfile(
        profile_id=EMBEDDING_PROFILE,
        model_name="voyage-code-4",
        artifact_digest="hosted-unverified",
        dimension=3,
        query_mode="query",
        passage_mode="document",
    )
    embedder = VoyageEmbedder(api_key="secret", identity=identity)
    calls.clear()

    embedder.embed_query("find parser fix")
    embedder.embed_passages(["parser implementation"])

    assert [call["input_type"] for call in calls] == ["query", "document"]


def test_word_windows_overlap_and_keep_the_final_tail_once() -> None:
    words = [f"w{index}" for index in range(10)]

    assert word_windows(" ".join(words), size=4, stride=3) == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]


def test_code4_add_windows_the_measured_content_stream_across_message_boundaries() -> None:
    request = AddRequest(
        request_id="request",
        user_id="user",
        session_id="session",
        messages=[
            Message(role="user", content="alpha beta gamma"),
            Message(role="assistant", content="delta epsilon zeta"),
        ],
    )

    chunks = build_chunks(
        request,
        [],
        embedding_profile=EMBEDDING_PROFILE,
        word_window_size=6,
        word_window_stride=4,
        content_only_windows=True,
        stable_window_identity=True,
    )

    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma delta epsilon zeta"
    ]
    assert "role:" not in chunks[0].text
    assert "content:" not in chunks[0].text
    assert {chunk.metadata["embedding_profile"] for chunk in chunks} == {
        EMBEDDING_PROFILE
    }
    assert {chunk.metadata["lexical_profile"] for chunk in chunks} == {BM25_PROFILE}
    assert [chunk.metadata["word_start"] for chunk in chunks] == [0]


def test_code4_window_identity_is_independent_of_transport_request_id() -> None:
    def ids(request_id: str) -> list[str]:
        request = AddRequest(
            request_id=request_id,
            user_id="user",
            session_id="sessions/task/session.jsonl",
            messages=[Message(role="user", content="alpha beta gamma")],
        )
        return [
            chunk.id
            for chunk in build_chunks(
                request,
                [],
                embedding_profile=EMBEDDING_PROFILE,
                word_window_size=2,
                word_window_stride=1,
                content_only_windows=True,
                stable_window_identity=True,
            )
        ]

    assert ids("transport-a") == ids("transport-b")


def test_frozen_bm25_uses_repeated_query_terms_and_deterministic_ties() -> None:
    chunks = [
        Chunk("b", "source", "parser fix", {}),
        Chunk("a", "source", "parser parser regression", {}),
        Chunk("c", "source", "unrelated", {}),
    ]

    ranked = rank_bm25_chunks(chunks, "parser parser the", k=100)

    assert [hit.chunk.id for hit in ranked] == ["a", "b"]
    assert ranked[0].score > ranked[1].score > 0
    assert BM25_PROFILE == "canonical-bm25-k1-1.5-b0.75-v1"


def test_code4_bm25_ties_follow_session_then_window_order_not_chunk_id() -> None:
    chunks = [
        Chunk(
            "a-id-but-later-session",
            "source",
            "parser fix",
            {"source_session_id": "sessions/z.jsonl", "segment": 0},
        ),
        Chunk(
            "z-id-but-earlier-session",
            "source",
            "parser fix",
            {"source_session_id": "sessions/a.jsonl", "segment": 0},
        ),
    ]

    ranked = rank_bm25_chunks(chunks, "parser", k=100, stable_ties=True)

    assert [hit.chunk.id for hit in ranked] == [
        "z-id-but-earlier-session",
        "a-id-but-later-session",
    ]


def test_code4_stable_ties_refuse_missing_window_identity() -> None:
    with pytest.raises(ValueError, match="source_session_id"):
        rank_bm25_chunks(
            [Chunk("opaque", "source", "parser fix", {})],
            "parser",
            k=100,
            stable_ties=True,
        )


def test_code4_retrieval_uses_canonical_bm25_not_postgres_sparse() -> None:
    store = _Store()
    retriever = HostedRetriever(_Embedder(), _Reranker())

    run = retriever.search(
        store,
        "parser regression",
        [],
        rerank=False,
        canonical_bm25=True,
        exact_dense=True,
        stable_window_order=True,
    )

    assert store.sparse_calls == 0
    assert store.dense_calls == 0
    assert store.exact_dense_calls == 1
    assert [hit.chunk.id for hit in run.hits] == [
        "z-dense-earlier-window",
        "a-lexical-later-window",
    ]
