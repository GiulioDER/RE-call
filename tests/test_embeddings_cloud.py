import base64
import json
import os
import importlib.util
import struct
import sys
import types

import pytest
import requests
from requests.structures import CaseInsensitiveDict

import recall.embeddings
from recall._voyage_http import DEFAULT_BASE_URL
from recall_mcp.service import make_embedder
from recall.embeddings import Embedder, VoyageEmbedder, resolve_embedder

#: The same opt-in `tests/test_bench_systems.py` uses for its OpenRouter smoke test. Gating on the
#: key alone meant every run with VOYAGE_API_KEY exported (the VPS3 testbench's `env.sh`, and any
#: user of the hosted embedder) spent Voyage credit on `pytest tests/`. Found 2026-09-26 by a suite
#: pass with fake keys and a resolver that refused and logged every paid-API host: this was one of
#: exactly two tests that reached one, and the only one that authenticates.
PAID_TESTS_OPT_IN = "RECALL_RUN_PAID_TESTS"

#: Placeholder, never a credential: the transport is replaced wherever it is used.
PLACEHOLDER_KEY = "pa-CHANGEME-placeholder"

requires_voyage = pytest.mark.skipif(
    not (
        os.environ.get(PAID_TESTS_OPT_IN)
        and os.environ.get("VOYAGE_API_KEY")
        and importlib.util.find_spec("voyageai") is not None
    ),
    reason=f"spends Voyage credit; set {PAID_TESTS_OPT_IN}=1 with VOYAGE_API_KEY and voyageai",
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


def test_voyage_roundtrip_without_the_network(monkeypatch):
    """`test_voyage_roundtrip`'s assertions, with only the HTTP transport replaced. No credit.

    Invariant: `VoyageEmbedder` sizes itself from the provider's own probe response and returns
    each input's decoded vector at that width, through the real `recall._voyage_http.Client`. What
    the paid test proves beyond this is only that the key authenticates and the account can pay.

    Failure mode caught: an embedder whose declared `dim` and returned vectors disagree, which a
    store only reports at insert time, after the corpus has been embedded and paid for.

    Red proof, 2026-09-26: `VoyageEmbedder.__init__` in `recall/embeddings.py` mutated to take
    `len(...embeddings)` instead of `len(...embeddings[0])` for the dim probe; this test failed at
    `assert emb.dim == len(vector)` (`1 == 3`). Restored, it passes.
    """
    vector = [0.25, -1.5, 3.0]
    body = json.dumps(
        {
            "data": [{"embedding": base64.b64encode(struct.pack("<3f", *vector)).decode()}],
            "usage": {"total_tokens": 2},
        }
    ).encode()
    seen: list[tuple[str, str, list[str]]] = []

    class _Response:
        status_code = 200
        content = body
        headers = CaseInsensitiveDict({"Content-Type": "application/json"})

    def request(self, method, url, headers=None, data=None, timeout=None, **_kwargs):
        seen.append((method.lower(), url, json.loads(data)["input"]))
        return _Response()

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    # CI installs `dev` only, without the SDK. `_voyage_client_class` checks it is INSTALLED (its
    # version is part of the profile identity) and never imports it, so a spec-less module in
    # `sys.modules` is the double that check documents accepting.
    monkeypatch.setitem(sys.modules, "voyageai", types.ModuleType("voyageai"))

    emb = VoyageEmbedder(api_key=PLACEHOLDER_KEY)
    vecs = emb.embed(["hello world"])

    assert isinstance(emb, Embedder)
    assert emb.dim == len(vector)
    assert vecs == [vector]
    assert seen == [
        ("post", f"{DEFAULT_BASE_URL}/embeddings", ["probe"]),
        ("post", f"{DEFAULT_BASE_URL}/embeddings", ["hello world"]),
    ]


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
