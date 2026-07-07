import os

import pytest

from recall.embeddings import Embedder, OpenAIEmbedder, VoyageEmbedder

requires_voyage = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"), reason="no VOYAGE_API_KEY"
)
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY"
)


def test_voyage_requires_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        VoyageEmbedder()


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        OpenAIEmbedder()


@requires_voyage
def test_voyage_roundtrip():
    emb = VoyageEmbedder()
    vecs = emb.embed(["hello world"])
    assert isinstance(emb, Embedder)
    assert emb.dim > 0 and len(vecs) == 1 and len(vecs[0]) == emb.dim


@requires_openai
def test_openai_roundtrip():
    emb = OpenAIEmbedder()
    vecs = emb.embed(["hello world"])
    assert isinstance(emb, Embedder)
    assert emb.dim > 0 and len(vecs) == 1 and len(vecs[0]) == emb.dim
