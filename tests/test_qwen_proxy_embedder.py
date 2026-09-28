"""The W0 proxy: Qwen3-Embedding-8B through OpenRouter, standing in for `text-embedding-v4`.

A stub `openai` module records every request, so nothing needs a key or the network. Each test
names the mutation of the production code it was watched to fail on (the red proof).
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import pytest

from recall.dashscope import MEMORY_RETRIEVAL_INSTRUCTION
from recall.embedding_registry import registered_profile

PROXY = "qwen3-embedding-8b-mrl1024-openrouter-deepinfra-memory-instruct-v1"
PLAIN_OPENAI = "openai-text-embedding-3-small-v1"
#: A placeholder, never a real credential.
PLACEHOLDER_KEY = "CHANGEME-placeholder-not-a-key"


def _vector(seed: int, width: int) -> list[float]:
    return [math.cos(seed + 0.37 * i) + 1.2 for i in range(width)]


class _StubOpenAI:
    width = 4096
    requests: list[dict[str, Any]] = []

    def __init__(self, **_kwargs: Any) -> None:
        self.embeddings = self

    def create(self, **request: Any) -> Any:
        type(self).requests.append(request)
        return types.SimpleNamespace(
            data=[
                types.SimpleNamespace(embedding=_vector(len(text), type(self).width))
                for text in request["input"]
            ]
        )


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch) -> type[_StubOpenAI]:
    _StubOpenAI.requests = []
    _StubOpenAI.width = 4096
    module = types.ModuleType("openai")
    module.OpenAI = _StubOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return _StubOpenAI


def test_queries_carry_the_instruction_in_qwen_form_and_passages_do_not(stub_openai: Any) -> None:
    """Invariant: a proxy query is sent as ``Instruct: <instruction>`` newline ``Query:<text>``;
    passages go bare; a profile without an instruction sends its query bare.

    Red proof: making `OpenAICompatEmbedder.embed_query` return ``self.embed([text])[0]``
    unconditionally fails the first assertion.
    """
    proxy = registered_profile(PROXY).build(api_key=PLACEHOLDER_KEY)
    proxy.embed_query("when did we talk?")
    assert stub_openai.requests[-1]["input"] == [
        f"Instruct: {MEMORY_RETRIEVAL_INSTRUCTION}\nQuery:when did we talk?"
    ]
    proxy.embed_passages(["a window"])
    assert stub_openai.requests[-1]["input"] == ["a window"]

    stub_openai.width = 1536
    plain = registered_profile(PLAIN_OPENAI).build(api_key=PLACEHOLDER_KEY)
    plain.embed_query("when did we talk?")
    assert stub_openai.requests[-1]["input"] == ["when did we talk?"]


def test_vectors_keep_the_leading_1024_dimensions_renormalised(stub_openai: Any) -> None:
    """Invariant: a 4,096-dimension provider vector becomes its first 1,024 values at unit length.

    Red proof: returning ``head`` without dividing by ``norm`` in `_mrl_truncate` fails the norm
    assertion.
    """
    proxy = registered_profile(PROXY).build(api_key=PLACEHOLDER_KEY)
    vector = proxy.embed_passages(["abc"])[0]
    head = _vector(3, 4096)[:1024]
    norm = math.sqrt(sum(v * v for v in head))
    assert len(vector) == 1024 == proxy.dim
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)
    assert vector == pytest.approx([v / norm for v in head])


def test_the_proxy_is_pinned_to_one_provider_and_other_profiles_are_not(stub_openai: Any) -> None:
    """Invariant: the proxy asks OpenRouter for DeepInfra with no fallback; other OpenRouter
    profiles send no provider preference.

    Red proof: setting ``allow_fallbacks`` to True in `_embed_one_batch` fails the first
    assertion.
    """
    registered_profile(PROXY).build(api_key=PLACEHOLDER_KEY).embed_passages(["x"])
    assert stub_openai.requests[-1]["extra_body"] == {
        "provider": {"order": ["deepinfra"], "allow_fallbacks": False}
    }
    stub_openai.width = 1536
    registered_profile(PLAIN_OPENAI).build(api_key=PLACEHOLDER_KEY).embed_passages(["x"])
    assert "extra_body" not in stub_openai.requests[-1]


def test_truncation_and_provider_are_part_of_the_identity() -> None:
    """Invariant: the truncation width and the pinned provider are key material, so a cache or
    generation cannot reuse vectors made another way.

    Red proof: removing the ``provider_order`` dependency in `RegisteredProfile.identity` fails
    the assertion.
    """
    dependencies = dict(registered_profile(PROXY).identity().dependencies)
    assert dependencies.get("mrl_dimensions") == "1024"
    assert dependencies.get("provider_order") == "deepinfra"


def test_the_no_instruction_pass_also_covers_the_proxy(stub_openai: Any) -> None:
    """Invariant: the W0 second pass drops the instruction for the proxy too.

    Red proof: patching only `DashScopeEmbedder` in `without_query_instruction` leaves the
    proxy's query instructed and fails the assertion.
    """
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import aml_w0_embedding_compare as w0

    proxy = registered_profile(PROXY).build(api_key=PLACEHOLDER_KEY)
    with w0.without_query_instruction():
        proxy.embed_query("when did we talk?")
    assert stub_openai.requests[-1]["input"] == ["when did we talk?"]
