from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

import recall_mcp.translation as translation
from recall_mcp.service import EvidenceItemModel, EvidenceResult, SearchHit, SearchResult
from recall_mcp.translation import (
    NullTranslationProvider,
    TranslateJsProvider,
    TranslationError,
    render_search_response,
    render_evidence_response,
    translate_for_display,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        del limit
        return self._body


def _search_result() -> SearchResult:
    hit = SearchHit.model_construct(chunk_id="chunk-1", text="Bonjour", source="memo.md")
    return SearchResult.model_construct(
        query="hello",
        advice="Use only trusted evidence.",
        hits=[hit],
    )


def test_translation_is_disabled_by_default() -> None:
    provider = TranslateJsProvider.from_env({})
    assert isinstance(provider, NullTranslationProvider)


def test_enabled_translation_requires_an_explicit_endpoint() -> None:
    with pytest.raises(ValueError, match="ENDPOINT is required"):
        TranslateJsProvider.from_env({"RECALL_TRANSLATION_ENABLED": "1"})


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_translation_timeout_must_be_finite(timeout: str) -> None:
    with pytest.raises(ValueError, match="positive number"):
        TranslateJsProvider.from_env(
            {
                "RECALL_TRANSLATION_ENABLED": "1",
                "RECALL_TRANSLATION_ENDPOINT": "https://translate.example.test/translate.json",
                "RECALL_TRANSLATION_TIMEOUT_SECONDS": timeout,
            }
        )


def test_endpoint_requires_https_except_explicit_loopback() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        TranslateJsProvider("http://translate.example.test/translate.json")
    assert TranslateJsProvider("http://127.0.0.1:8080/translate.json").endpoint.startswith("http://")


def test_provider_posts_documented_form_and_validates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> _Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"result": 1, "text": ["Ciao", "Mondo"]})

    monkeypatch.setattr(translation._NO_REDIRECT_OPENER, "open", fake_urlopen)
    provider = TranslateJsProvider("https://translate.example.test/translate.json")
    assert provider.translate_batch(["Hello", "World"], locale="italian") == ["Ciao", "Mondo"]
    request = captured["request"]
    assert request is not None
    assert request.full_url == "https://translate.example.test/translate.json"
    assert request.get_method() == "POST"
    assert request.data is not None
    assert b"to=italian" in request.data
    assert b"Hello" in request.data
    assert captured["timeout"] == 5.0


def test_provider_rejects_wrong_item_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        translation._NO_REDIRECT_OPENER,
        "open",
        lambda request, timeout: _Response({"result": 1, "text": ["only one"]}),
    )
    provider = TranslateJsProvider("https://translate.example.test/translate.json")
    with pytest.raises(TranslationError, match="invalid text list"):
        provider.translate_batch(["one", "two"], locale="italian")


def test_display_translation_splits_large_results() -> None:
    batches: list[list[str]] = []

    class _Provider:
        name = "test"
        max_batch = 2

        def translate_batch(self, texts: Sequence[str], *, locale: str) -> list[str]:
            del locale
            batches.append(list(texts))
            return [f"translated:{text}" for text in texts]

    values, translated, warning = translate_for_display(["one", "two", "three"], "italian", _Provider())
    assert values == ["translated:one", "translated:two", "translated:three"]
    assert translated is True
    assert warning is None
    assert batches == [["one", "two"], ["three"]]


def test_localized_response_is_additive_and_fails_back_to_canonical() -> None:
    result = _search_result()
    rendered = render_search_response(result, "italian", NullTranslationProvider())
    payload = json.loads(rendered)
    assert payload["hits"][0]["text"] == "Bonjour"
    assert payload["localized"]["hits"] == [{"chunk_id": "chunk-1", "text": "Bonjour"}]
    assert payload["localized"]["fallback"] is True
    assert payload["localized"]["canonical_unchanged"] is True


def test_invalid_locale_is_rejected() -> None:
    with pytest.raises(ValueError, match="locale"):
        render_search_response(_search_result(), "italian;drop", NullTranslationProvider())


def test_evidence_prompts_are_not_translated() -> None:
    item = EvidenceItemModel.model_construct(chunk_id="chunk-1", text="Bonjour")
    result = EvidenceResult.model_construct(
        query="hello",
        system_prompt="Answer only from the evidence.",
        user_message="[chunk-1] Bonjour",
        items=[item],
        advice="Use the cited evidence.",
    )

    class _Provider:
        name = "test"

        def translate_batch(self, texts: list[str], *, locale: str) -> list[str]:
            del locale
            return [f"translated:{text}" for text in texts]

    payload = json.loads(render_evidence_response(result, "italian", _Provider()))
    assert payload["system_prompt"] == "Answer only from the evidence."
    assert payload["user_message"] == "[chunk-1] Bonjour"
    assert payload["localized"]["items"] == [
        {"chunk_id": "chunk-1", "text": "translated:Bonjour"}
    ]
