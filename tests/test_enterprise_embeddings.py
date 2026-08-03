from pathlib import Path

import pytest

from recall.cache import EmbeddingCache, embed_with_cache
from recall.embeddings import (
    EmbeddingProfile,
    Qwen3EmbeddingEmbedder,
    artifact_tree_sha256,
    embed_passages,
    embed_query,
    embedding_profile_id,
    verify_artifact,
)


class AsymmetricFake:
    dim = 2
    name = "fake"
    profile = EmbeddingProfile(
        "fake-v1", "fake", "a" * 64, 2, "query", "passage"
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, texts):
        self.calls.append("legacy")
        return [[0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.calls.append("query")
        return [1.0, 0.0]

    def embed_passages(self, texts):
        self.calls.append("passage")
        return [[0.0, 1.0] for _ in texts]


def test_asymmetric_helpers_and_profile_identity() -> None:
    embedder = AsymmetricFake()
    assert embed_query(embedder, "q") == [1.0, 0.0]
    assert embed_passages(embedder, ["p"]) == [[0.0, 1.0]]
    assert embedder.calls == ["query", "passage"]
    assert embedding_profile_id(embedder) == "fake-v1"


def test_cache_separates_query_and_passage_vectors(tmp_path: Path) -> None:
    embedder = AsymmetricFake()
    with EmbeddingCache(tmp_path / "vectors.sqlite") as cache:
        query = embed_with_cache(embedder, ["same"], cache, purpose="query")
        passage = embed_with_cache(embedder, ["same"], cache, purpose="passage")
        assert query != passage
        assert embedder.calls == ["query", "passage"]
        embed_with_cache(embedder, ["same"], cache, purpose="query")
        assert embedder.calls == ["query", "passage"]


def test_artifact_checksum_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"weights")
    digest = artifact_tree_sha256(tmp_path)
    assert verify_artifact(tmp_path, digest) == tmp_path.resolve()
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_artifact(tmp_path, "0" * 64)


def test_qwen_normalizes_after_dimension_truncation() -> None:
    class _Model:
        def encode(self, _texts, **_kwargs):
            return [[3.0, 4.0]]

    embedder = object.__new__(Qwen3EmbeddingEmbedder)
    embedder._model = _Model()
    embedder._batch_size = 1

    vector = embedder._encode(["query"])[0]

    assert vector == pytest.approx([0.6, 0.8])
