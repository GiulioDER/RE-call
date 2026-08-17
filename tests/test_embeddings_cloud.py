import os
import importlib.util

import pytest

import recall.embeddings
from recall_mcp.service import make_embedder
from recall.embeddings import Embedder, VoyageEmbedder, resolve_embedder

requires_voyage = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY") or importlib.util.find_spec("voyageai") is None,
    reason="no VOYAGE_API_KEY or voyageai SDK",
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
        def __init__(
            self,
            model: str = "openai/text-embedding-3-small",
            api_key: str | None = None,
            dimensions: int | None = None,
            name_prefix: str = "openai",
        ) -> None:
            seen["openai"] = model
            seen["dimensions"] = str(dimensions)
            seen["prefix"] = name_prefix

    monkeypatch.setattr(recall.embeddings, "VoyageEmbedder", _FakeVoyage)
    monkeypatch.setattr(recall.embeddings, "OpenAICompatEmbedder", _FakeOpenAI)

    resolve_embedder("voyage:voyage-4-large")
    resolve_embedder("openai:text-embedding-3-small", env={})

    assert seen == {
        "voyage": "voyage-4-large",
        "openai": "text-embedding-3-small",
        "dimensions": "None",
        "prefix": "openai",
    }


def test_resolve_embedder_routes_openrouter_gemini_with_dimensions(monkeypatch):
    seen: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(
            self,
            model: str = "openai/text-embedding-3-small",
            api_key: str | None = None,
            dimensions: int | None = None,
            name_prefix: str = "openai",
        ) -> None:
            seen["model"] = model
            seen["api_key"] = api_key
            seen["dimensions"] = dimensions
            seen["name_prefix"] = name_prefix

    monkeypatch.setattr(recall.embeddings, "OpenAICompatEmbedder", _FakeOpenAI)

    resolve_embedder(
        "gemini-embedding-2",
        env={"OPENROUTER_API_KEY": "or-key", "RECALL_EMBED_DIMENSIONS": "1536"},
    )

    assert seen == {
        "model": "google/gemini-embedding-2",
        "api_key": "or-key",
        "dimensions": 1536,
        "name_prefix": "openrouter",
    }


def test_sfr_code_alias_requires_explicit_research_model_opt_in(monkeypatch):
    with pytest.raises(ValueError, match="RECALL_ACCEPT_RESEARCH_MODEL_LICENSE"):
        resolve_embedder("sfr-code", env={})
    with pytest.raises(ValueError, match="RECALL_ACCEPT_REMOTE_MODEL_CODE"):
        resolve_embedder("sfr-code", env={"RECALL_ACCEPT_RESEARCH_MODEL_LICENSE": "1"})

    seen: dict[str, object] = {}

    class _FakeSentenceTransformer:
        def __init__(
            self,
            model: str,
            batch_size: int = 64,
            *,
            trust_remote_code: bool = False,
            revision: str | None = None,
            name: str | None = None,
        ) -> None:
            seen["model"] = model
            seen["trust_remote_code"] = trust_remote_code
            seen["revision"] = revision
            seen["name"] = name

    monkeypatch.setattr(recall.embeddings, "SentenceTransformerEmbedder", _FakeSentenceTransformer)
    resolve_embedder(
        "sfr-code",
        env={
            "RECALL_ACCEPT_RESEARCH_MODEL_LICENSE": "1",
            "RECALL_ACCEPT_REMOTE_MODEL_CODE": "1",
        },
    )

    assert seen == {
        "model": "Salesforce/SFR-Embedding-Code-2B_R",
        "trust_remote_code": True,
        "revision": "c73d8631a005876ed5abde34db514b1fb6566973",
        "name": "sfr-code:Salesforce/SFR-Embedding-Code-2B_R",
    }


def test_mcp_embedder_keeps_sfr_code_license_error():
    with pytest.raises(ValueError, match="RECALL_ACCEPT_RESEARCH_MODEL_LICENSE"):
        make_embedder("sfr-code", env={})
