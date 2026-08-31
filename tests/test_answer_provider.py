"""Environment resolution for the optional local answer provider.

`resolve_answer_provider` reads its whole configuration from the environment, so the parsing
contract is the surface an operator actually touches: empty means unset (the .env template ships
keys valueless), padded values read as their stripped value, a typo in a boolean raises rather
than silently reading False, and a malformed numeric raises an error naming the variable.
"""

from __future__ import annotations

import pytest

from recall.answer_provider import (
    OpenAICompatibleAnswerProvider,
    resolve_answer_provider,
)


def _enabled(**extra: str) -> dict[str, str]:
    # The model is supplied here because enabling the provider REQUIRES an explicit model,
    # matching the expansion resolver; there is no default model any more.
    return {
        "RECALL_REASONING_ANSWER_ENABLED": "1",
        "RECALL_REASONING_ANSWER_MODEL": "qwen3:4b",
        **extra,
    }


def test_answer_provider_is_off_by_default() -> None:
    assert resolve_answer_provider({}) is None


def test_enabling_without_a_model_raises_naming_the_variable() -> None:
    """No silent default: the expansion resolver requires its model explicitly, and an answer
    model nobody chose is not a safer thing to fall back to."""
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MODEL"):
        resolve_answer_provider({"RECALL_REASONING_ANSWER_ENABLED": "1"})


def test_a_whitespace_only_model_is_not_a_model() -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MODEL"):
        resolve_answer_provider(
            {
                "RECALL_REASONING_ANSWER_ENABLED": "1",
                "RECALL_REASONING_ANSWER_MODEL": "   ",
            }
        )


def test_a_typo_in_the_thinking_flag_raises_instead_of_reading_false() -> None:
    """One sided leniency reads a typo as False while the sibling ENABLED flag raises; the two
    resolvers must agree, and a silent False on a typo is the worse failure."""
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_THINKING"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_THINKING="ture"))


def test_the_thinking_flag_tolerates_surrounding_whitespace() -> None:
    provider = resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_THINKING=" 1 "))
    assert provider is not None
    assert provider.thinking is True


def test_an_empty_timeout_falls_back_to_the_default() -> None:
    """Empty means unset, because .env templates ship keys valueless (recall/profiles.py)."""
    provider = resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_TIMEOUT=""))
    assert provider is not None
    assert provider.client.timeout == 60.0


def test_a_malformed_timeout_names_the_variable() -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_TIMEOUT"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_TIMEOUT="soon"))


def test_empty_max_tokens_falls_back_to_the_default() -> None:
    provider = resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_MAX_TOKENS="   "))
    assert provider is not None
    assert provider.max_tokens == 512


def test_malformed_max_tokens_names_the_variable() -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MAX_TOKENS"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_MAX_TOKENS="many"))


def test_the_reasoning_tool_supplies_the_resolved_provider_rather_than_nothing() -> None:
    """Resolving a provider is useless if the call site still omits it.

    This is the defect the module spent its whole life in: `recall/answer_provider.py` shipped
    complete and tested on 2026-08-15, `ReasoningProviderPorts.answer_provider` existed, and
    `recall_mcp/server.py` never passed one, so `recall_reasoning_query` returned
    `refusal_reason="no_answer_provider"` in every shipped configuration. A provider that is
    resolvable but unreachable looks identical from a client to one that does not exist.

    Asserted against the source, matching
    `tests/test_mcp_trust_mode_env.py::test_the_search_tool_passes_the_policy_rather_than_defaulting`:
    reaching the tool body needs authentication and a live database, and the property under test
    is about the CALL, not the result.

    The keyword's value is checked to be the name bound from `resolve_answer_provider()`, not
    merely present. `answer_provider=None` is a syntactically valid way to satisfy a weaker guard
    while restoring the exact bug.
    """
    import ast
    import pathlib

    import recall_mcp.server as server

    tree = ast.parse(pathlib.Path(server.__file__).read_text(encoding="utf-8"))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reasoning_query"
    ]
    assert len(calls) == 1, "expected exactly one reasoning_query call site in recall_mcp/server.py"

    keyword = next((kw for kw in calls[0].keywords if kw.arg == "answer_provider"), None)
    assert keyword is not None, (
        f"reasoning_query is called at recall_mcp/server.py:{calls[0].lineno} without "
        "answer_provider=, so the tool abstains with no_answer_provider whatever the operator "
        "configured."
    )
    assert isinstance(keyword.value, ast.Name), (
        "answer_provider= must be the name bound from resolve_answer_provider(), not a literal"
    )

    bound = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "resolve_answer_provider"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert keyword.value.id in bound, (
        f"answer_provider={keyword.value.id} is not bound from resolve_answer_provider(); the "
        "tool would pass something the operator's environment does not control."
    )


# --- the OpenAI compatible backend -------------------------------------------------------
#
# `OllamaAnswerProvider` cannot reach a hosted endpoint whatever its base URL says: it rewrites
# the path to `<base>/api/chat` and sends no Authorization header. These cover the second
# backend and the selector that chooses between them.


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.completions = _FakeCompletions(response)
        self.chat = type("_Chat", (), {"completions": self.completions})()


def _response(content: str | None, **usage: object) -> object:
    message = type("_Message", (), {"content": content})()
    choice = type("_Choice", (), {"message": message})()
    return type(
        "_Response",
        (),
        {"choices": [choice], "usage": type("_Usage", (), usage)() if usage else None},
    )()


def _openai_enabled(**extra: str) -> dict[str, str]:
    return _enabled(
        RECALL_REASONING_ANSWER_PROVIDER="openai",
        RECALL_REASONING_ANSWER_API_KEY="sk-test",
        **extra,
    )


def test_the_openai_backend_posts_a_deterministic_json_request() -> None:
    client = _FakeClient(_response('{"answer": "yes", "citations": ["c1"]}'))
    provider = OpenAICompatibleAnswerProvider(client, model_id="deepseek/deepseek-chat")

    assert provider("sys", "usr") == '{"answer": "yes", "citations": ["c1"]}'
    sent = client.completions.calls[0]
    assert sent["model"] == "deepseek/deepseek-chat"
    assert sent["temperature"] == 0
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    # `reasoning_effort` is OpenAI-specific and a non-OpenAI model behind an OpenAI-compatible
    # gateway can reject an unknown field, which would fail every call to a deepseek model.
    assert "reasoning_effort" not in sent


def test_an_unpriced_hosted_call_records_a_null_cost_not_zero() -> None:
    """Null cost is a missing measurement; 0.0 is a claim that the call was free.

    `OllamaAnswerProvider` records 0.0 honestly, because local inference costs nothing. Copying
    that into a hosted provider would make every OpenRouter call assert it was free, and the
    benchmark cost gate only rejects unsupported money claims if missing is recorded as missing.
    """
    client = _FakeClient(_response('{"answer": null}', prompt_tokens=10, completion_tokens=5))
    provider = OpenAICompatibleAnswerProvider(client, model_id="deepseek/deepseek-chat")
    provider("sys", "usr")

    metadata = provider.provider_metadata()
    assert metadata.monetary_cost_usd is None
    assert metadata.total_tokens == 15
    assert metadata.provider_id == "recall.reasoning.answer.openai"


def test_a_configured_price_is_applied_per_thousand_tokens() -> None:
    client = _FakeClient(_response('{"answer": null}', prompt_tokens=600, completion_tokens=400))
    provider = OpenAICompatibleAnswerProvider(
        client, model_id="deepseek/deepseek-chat", cost_per_1k_tokens=0.5
    )
    provider("sys", "usr")

    assert provider.provider_metadata().monetary_cost_usd == 0.5


def test_metadata_is_recorded_even_when_the_call_fails() -> None:
    client = _FakeClient(_response(None))
    provider = OpenAICompatibleAnswerProvider(client, model_id="deepseek/deepseek-chat")

    with pytest.raises(ValueError, match="empty content"):
        provider("sys", "usr")
    assert provider.provider_metadata().latency_ms is not None


def test_the_resolver_selects_the_openai_backend_and_defaults_to_openrouter() -> None:
    provider = resolve_answer_provider(_openai_enabled())
    assert isinstance(provider, OpenAICompatibleAnswerProvider)
    assert str(provider.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_the_ollama_default_endpoint_is_not_reused_for_the_hosted_backend() -> None:
    """One shared default would silently point a hosted backend at loopback."""
    ollama = resolve_answer_provider(_enabled())
    hosted = resolve_answer_provider(_openai_enabled())
    assert ollama is not None and hosted is not None
    assert "127.0.0.1" in ollama.client.endpoint
    assert "127.0.0.1" not in str(hosted.client.base_url)


def test_an_unknown_backend_names_the_variable_and_the_accepted_values() -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_PROVIDER"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_PROVIDER="openrouter"))


def test_the_hosted_backend_requires_a_key() -> None:
    env = _enabled(RECALL_REASONING_ANSWER_PROVIDER="openai")
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_API_KEY"):
        resolve_answer_provider(env)


def test_the_wizards_legacy_key_spelling_is_accepted() -> None:
    """`recall setup` writes RECALL_REASONING_API_KEY, which nothing read for four releases."""
    provider = resolve_answer_provider(
        _enabled(
            RECALL_REASONING_ANSWER_PROVIDER="openai",
            RECALL_REASONING_API_KEY="sk-legacy",
        )
    )
    assert isinstance(provider, OpenAICompatibleAnswerProvider)


def test_the_ollama_only_thinking_switch_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="Ollama-only"):
        resolve_answer_provider(_openai_enabled(RECALL_REASONING_ANSWER_THINKING="1"))


def test_configuration_is_validated_before_the_optional_import(monkeypatch) -> None:
    """A bad timeout must surface as a bad timeout, not as a missing dependency.

    This is the ordering that turned the `floor` CI job red on PR #366 for the expansion
    resolver: it imported the optional extra first, so an invalid configuration on a floor
    install reported the wrong cause.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "openai":
            raise AssertionError("openai was imported before the configuration was validated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_TIMEOUT"):
        resolve_answer_provider(_openai_enabled(RECALL_REASONING_ANSWER_TIMEOUT="soon"))


@pytest.mark.parametrize("value", ["-1", "free", "nan", "1.0.0"])
def test_a_malformed_price_names_the_variable(value: str) -> None:
    """Both rejection paths name the variable, not just the range check.

    The first draft covered only "-1", which parses and reaches the finite/non-negative
    check. A non-numeric value took a bare `float()` and raised "could not convert string to
    float: 'free'" instead, naming nothing an operator could act on. The test asserted the
    property in its name and did not execute the path that breaks it.
    """
    with pytest.raises(ValueError, match="COST_PER_1K_TOKENS"):
        resolve_answer_provider(
            _openai_enabled(RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS=value)
        )
