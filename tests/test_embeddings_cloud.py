import os

import pytest

import recall.embeddings
from recall.embeddings import Embedder, VoyageEmbedder, resolve_embedder

requires_voyage = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"), reason="no VOYAGE_API_KEY"
)


def test_voyage_requires_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        VoyageEmbedder()


@requires_voyage
def test_voyage_roundtrip():
    emb = VoyageEmbedder()
    vecs = emb.embed(["hello world"])
    assert isinstance(emb, Embedder)
    assert emb.dim > 0 and len(vecs) == 1 and len(vecs[0]) == emb.dim


def test_resolve_embedder_routes_cloud_prefixes(monkeypatch):
    seen: dict[str, str] = {}

    class _FakeVoyage:
        def __init__(self, model: str = "voyage-3", api_key: str | None = None) -> None:
            seen["voyage"] = model

    class _FakeOpenAI:
        def __init__(self, model: str = "openai/text-embedding-3-small", api_key: str | None = None) -> None:
            seen["openai"] = model

    monkeypatch.setattr(recall.embeddings, "VoyageEmbedder", _FakeVoyage)
    monkeypatch.setattr(recall.embeddings, "OpenAICompatEmbedder", _FakeOpenAI)

    resolve_embedder("voyage:voyage-4-large")
    resolve_embedder("openai:text-embedding-3-small")

    assert seen == {
        "voyage": "voyage-4-large",
        "openai": "text-embedding-3-small",
    }
