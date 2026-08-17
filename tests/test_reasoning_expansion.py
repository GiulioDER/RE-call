from __future__ import annotations

from types import SimpleNamespace

import pytest

from recall.reasoning_expansion import (
    ExpansionRequest,
    OpenAIExpansionProvider,
    resolve_expansion_provider,
)


class _Client:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.payload))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )


def _request() -> ExpansionRequest:
    return ExpansionRequest(
        query="who owns the project?",
        tenant_id="acme",
        generation_id="gen_1",
        evidence=({"chunk_id": "c1", "text": "project description"},),
        gap_reason="missing_owner",
    )


def test_openai_expansion_provider_parses_strict_proposals_and_metadata() -> None:
    client = _Client(
        '{"proposals":[{"mode":"rewrite","query":"project owner","rationale":"owner missing",'
        '"parent_chunk_ids":["c1"]}]}'
    )
    provider = OpenAIExpansionProvider(
        client,
        model_id="cheap/test-model",
        revision="test-rev",
        cost_per_1k_tokens=0.001,
    )

    report = provider(_request())

    assert report.proposals[0].query == "project owner"
    assert report.proposals[0].parent_chunk_ids == ("c1",)
    assert report.provider_metadata[0].model_id == "cheap/test-model"
    assert report.provider_metadata[0].total_tokens == 14
    assert report.provider_metadata[0].monetary_cost_usd == pytest.approx(0.000014)
    assert report.provider_metadata[0].prompt_digest
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["reasoning_effort"] == "minimal"


def test_openai_expansion_provider_rejects_invalid_mode() -> None:
    client = _Client('{"proposals":[{"mode":"answer","query":"wrong"}]}')
    provider = OpenAIExpansionProvider(client, model_id="cheap/test-model")

    with pytest.raises(ValueError, match="invalid mode"):
        provider(_request())


def test_expansion_provider_is_off_by_default_and_requires_cheap_model_settings() -> None:
    assert resolve_expansion_provider({}) is None

    with pytest.raises(ValueError, match="MODEL is required"):
        resolve_expansion_provider(
            {"RECALL_REASONING_EXPANSION": "1", "RECALL_REASONING_API_KEY": "key"}
        )


def test_expansion_provider_rejects_invalid_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        OpenAIExpansionProvider(_Client('{"proposals":[]}'), model_id="cheap/test-model", reasoning_effort="max")
