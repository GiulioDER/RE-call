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


@pytest.mark.parametrize(
    "prompt_tokens,completion_tokens,expected",
    [(1000, 500, 0.75), (200, 100, 0.15), (4000, 0, 2.0)],
    ids=["1500-tokens", "300-tokens", "4000-tokens"],
)
def test_a_configured_price_is_applied_per_thousand_tokens(
    prompt_tokens: int, completion_tokens: int, expected: float
) -> None:
    """Never a 1000-token total, which is the whole point of the parametrisation.

    The first version used prompt=600 completion=400, i.e. exactly 1000 tokens at 0.5/1k, so
    the assertion `== 0.5` held for an implementation that ignored the token count entirely
    and returned the raw configured price. Deleting `total * ... / 1000` left it green — the
    per-thousand property named in the title was the one thing it could not detect. Confirmed
    by mutation during the audit that found it.
    """
    client = _FakeClient(
        _response(
            '{"answer": null}',
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    )
    provider = OpenAICompatibleAnswerProvider(
        client, model_id="deepseek/deepseek-chat", cost_per_1k_tokens=0.5
    )
    provider("sys", "usr")

    assert provider.provider_metadata().monetary_cost_usd == expected


def test_metadata_is_recorded_even_when_the_call_fails() -> None:
    client = _FakeClient(_response(None))
    provider = OpenAICompatibleAnswerProvider(client, model_id="deepseek/deepseek-chat")

    with pytest.raises(ValueError, match="empty content"):
        provider("sys", "usr")
    assert provider.provider_metadata().latency_ms is not None


def test_the_resolver_selects_the_openai_backend_and_defaults_to_openrouter() -> None:
    # Resolving the openai backend constructs the OpenAI client, so this needs the
    # optional extra; the `test` and `floor` CI jobs both install without it. Same
    # idiom as tests/test_reasoning_expansion.py for the sibling resolver.
    pytest.importorskip("openai")
    provider = resolve_answer_provider(_openai_enabled())
    assert isinstance(provider, OpenAICompatibleAnswerProvider)
    assert str(provider.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_the_ollama_default_endpoint_is_not_reused_for_the_hosted_backend() -> None:
    """One shared default would silently point a hosted backend at loopback."""
    # Resolving the openai backend constructs the OpenAI client, so this needs the
    # optional extra; the `test` and `floor` CI jobs both install without it. Same
    # idiom as tests/test_reasoning_expansion.py for the sibling resolver.
    pytest.importorskip("openai")
    ollama = resolve_answer_provider(_enabled())
    hosted = resolve_answer_provider(_openai_enabled())
    assert ollama is not None and hosted is not None
    assert "127.0.0.1" in ollama.client.endpoint
    assert "127.0.0.1" not in str(hosted.client.base_url)


def test_an_unknown_backend_names_the_variable_and_the_accepted_values() -> None:
    """Both halves of the title, since only the first was ever checked.

    Deleting the `'ollama' or 'openai'` clause from the message left this green, so the
    "accepted values" half was decoration. An operator who typed a wrong backend needs to be
    told which ones exist, not merely which variable is wrong.
    """
    with pytest.raises(ValueError) as excinfo:
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_PROVIDER="openrouter"))

    message = str(excinfo.value)
    assert "RECALL_REASONING_ANSWER_PROVIDER" in message
    assert "ollama" in message and "openai" in message
    assert "openrouter" in message, "the refusal should echo what the operator actually set"


def test_the_hosted_backend_requires_a_key() -> None:
    env = _enabled(RECALL_REASONING_ANSWER_PROVIDER="openai")
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_API_KEY"):
        resolve_answer_provider(env)


def test_the_legacy_bare_key_spelling_is_accepted() -> None:
    """The bare RECALL_REASONING_API_KEY is a LEGACY fallback, not what `recall setup` writes.

    Renamed from `test_the_wizards_legacy_key_spelling_is_accepted`: the old name and docstring
    were a fifth site of the same false claim the audit found in the comment and three docs.
    `recall/setup.py` returns only the `RECALL_REASONING_EXPANSION_*` spellings, on purpose,
    because the bare pair is shared between reasoning arms. The FALLBACK is real and worth
    keeping for hand-written and pre-0.11 files; only the story about where the key comes from
    was wrong.
    """
    # Resolving the openai backend constructs the OpenAI client, so this needs the
    # optional extra; the `test` and `floor` CI jobs both install without it. Same
    # idiom as tests/test_reasoning_expansion.py for the sibling resolver.
    pytest.importorskip("openai")
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


def test_without_the_extra_the_resolver_names_the_extra(monkeypatch) -> None:
    """The floor install must get "install the extra", not an ImportError from the SDK.

    Written after CI caught the inverse of this: three tests above resolve a real client, and
    the machine they were written on has `openai` installed, so they were green locally and red
    on both the `test` and `floor` jobs, which install without it. This one runs either way,
    because it forces the ImportError rather than depending on the environment for it, and it
    is the only test that covers the branch a floor install actually takes.
    """
    import builtins

    real_import = builtins.__import__

    def without_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_openai)
    with pytest.raises(ValueError, match=r'pip install "recall-rag\[openai\]"'):
        resolve_answer_provider(_openai_enabled())


def test_the_ollama_backend_needs_no_extra(monkeypatch) -> None:
    """The default backend must stay usable on a floor install, which is the whole point of it."""
    import builtins

    real_import = builtins.__import__

    def without_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_openai)
    provider = resolve_answer_provider(_enabled())
    assert provider is not None
    assert "127.0.0.1" in provider.client.endpoint


def test_the_cli_reasoning_query_supplies_the_resolved_provider_too() -> None:
    """A facility reachable from one of two front doors is the defect this PR exists to remove.

    `recall_mcp/server.py` was wired first and the CLI was not, so `recall reasoning query`
    still returned `abstained / no_answer_provider`. A peer session hit exactly the trap that
    predicts: verifying the integration the obvious way, from the command line, and reading a
    working server as broken. Found by them, not by the suite, because the suite only ever
    asserted the MCP call site.

    Asserted against the source for the same reason as the server guard above: reaching this
    body needs a live database, and the property is about the CALL.

    `audit` is checked to be EXCLUDED in the same pass. Wiring the CLI by pointing every
    reasoning subcommand at a provider would put a model on the diagnostic that exists to
    report what the deterministic layer refuses, and no test would notice.
    """
    import ast
    import pathlib

    import recall.cli_commands.reasoning_cmd as reasoning_cmd

    tree = ast.parse(pathlib.Path(reasoning_cmd.__file__).read_text(encoding="utf-8"))
    calls: dict[str, list[ast.Call]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node)

    # A LIST per name, and an exact count, matching the server guard above. The first version
    # built `{name: node}`, which keeps only the LAST call per name: a second
    # `reasoning_query(` call site added for `trace` with no provider would have gone
    # unchecked while this test stayed green.
    query_calls = calls.get("reasoning_query", [])
    assert len(query_calls) == 1, (
        f"expected exactly one reasoning_query call site in the CLI, found {len(query_calls)}"
    )
    query = query_calls[0]
    keyword = next((kw for kw in query.keywords if kw.arg == "answer_provider"), None)
    assert keyword is not None, (
        f"reasoning_query is called at recall/cli_commands/reasoning_cmd.py:{query.lineno} "
        "without answer_provider=, so `recall reasoning query` abstains with "
        "no_answer_provider whatever the operator configured."
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
    assert isinstance(keyword.value, ast.Name) and keyword.value.id in bound, (
        "answer_provider= must be the name bound from resolve_answer_provider(), not a literal"
    )

    # The resolver raises ValueError on a misconfigured RECALL_REASONING_ANSWER_*, and this
    # CLI's convention is one line on stderr and exit 2, stated in `_stored_extracted_proposals`
    # and applied by the `--include-extracted` handler. An unwrapped call would hand the
    # operator a traceback through the whole resolver instead, defeating the point of messages
    # that go to the trouble of naming the variable.
    guarded = any(
        isinstance(node, ast.Try)
        and any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "resolve_answer_provider"
            for stmt in node.body
            for sub in ast.walk(stmt)
        )
        and any(
            isinstance(h.type, ast.Name) and h.type.id == "ValueError" for h in node.handlers
        )
        for node in ast.walk(tree)
    )
    assert guarded, (
        "resolve_answer_provider() must be wrapped in try/except ValueError in the CLI, so a "
        "misconfiguration prints one line and exits 2 like every other refusal here."
    )

    audit_calls = calls.get("reasoning_audit", [])
    assert len(audit_calls) == 1, "expected exactly one reasoning_audit call site in the CLI"
    audit = audit_calls[0]
    assert not any(kw.arg == "answer_provider" for kw in audit.keywords), (
        "reasoning_audit must not be given a provider: it reports what the deterministic "
        "layer refuses and must not spend a model call to do it."
    )


def test_the_reasoning_tool_description_does_not_route_agents_away() -> None:
    """The description is the only thing that decides whether an agent ever calls this tool.

    Until 2026-09-01 it opened with "Existing retrieval clients should keep using
    `recall_search` or `recall_evidence`", an UNCONDITIONAL redirect. That was honest while
    the tool could not answer, and it survived the wiring that made it able to. A tool that
    can answer and is never called is indistinguishable, in every metric, from one that was
    never built, so this is pinned rather than left to the next person editing a docstring.

    Read from the PUBLISHED description that build_server() hands a client, not from the
    source, because the client's copy is the one that decides anything.
    """
    from recall_mcp.server import build_server

    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    description = tools["recall_reasoning_query"].description or ""
    assert description.strip(), "recall_reasoning_query publishes no description at all"

    lowered = description.lower()
    for redirect in ("should keep using", "should use `recall_search`", "use recall_search instead"):
        assert redirect not in lowered, (
            f"the description tells clients {redirect!r}, which sends every agent elsewhere "
            "regardless of the question. Say WHEN each tool is the right one instead."
        )

    assert "prefer this over" in lowered, (
        "the description must say when to prefer this tool, or an agent has no basis to "
        "choose it over recall_search."
    )
    # The misreading that matters most in practice, and the one the session-start guidance
    # also warns about: an abstention is 'no supported answer', never 'nothing is there'.
    assert "abstained" in lowered and "empty" in lowered, (
        "the description must explain that `abstained` does not mean the corpus is empty; "
        "an agent that reads it as absence concludes the fact does not exist."
    )


def test_a_disagreeing_usage_triple_does_not_destroy_a_paid_answer() -> None:
    """FIX-001. A gateway's own token counts must never cost the caller the answer.

    `_record_metadata` runs in a `finally`, and `ProviderMetadata` REFUSES a triple where
    total != prompt + completion. A gateway may count reasoning or cached-prefill tokens in
    the total and not in the parts (and `_usage_int` floors all three independently), so the
    raise replaced an answer that had already been returned and paid for, surfacing to the
    caller as `abstained / provider_error`.

    Red before the fix: `ValueError: total_tokens must equal prompt_tokens plus completion_tokens`.
    """
    client = _FakeClient(
        _response(
            '{"answer": "yes"}', prompt_tokens=1000, completion_tokens=200, total_tokens=1500
        )
    )
    provider = OpenAICompatibleAnswerProvider(
        client, model_id="deepseek/deepseek-chat", cost_per_1k_tokens=0.5
    )

    assert provider("sys", "usr") == '{"answer": "yes"}'

    metadata = provider.provider_metadata()
    assert metadata.total_tokens == 1500, "the gateway's total is what billing follows"
    assert metadata.prompt_tokens is None and metadata.completion_tokens is None, (
        "parts that disagree with the total are dropped, not silently reconciled"
    )
    assert metadata.monetary_cost_usd == 0.75


def test_a_metadata_failure_never_reports_the_previous_calls_numbers() -> None:
    """FIX-001, second half: the stale-metadata leak, which would double-count real money.

    Falling back to identity-only metadata is what CLEARS the earlier call. Leaving
    `_last_metadata` untouched on failure re-reported the first call's tokens and cost against
    the second response, which is a money claim about a call that did not make it.
    """
    provider = OpenAICompatibleAnswerProvider(
        _FakeClient(_response('{"answer": "one"}', prompt_tokens=100, completion_tokens=50)),
        model_id="deepseek/deepseek-chat",
        cost_per_1k_tokens=1.0,
    )
    provider("sys", "usr")
    assert provider.provider_metadata().monetary_cost_usd == 0.15

    # A response whose usage cannot be represented at all: a negative count.
    provider.client = _FakeClient(  # type: ignore[attr-defined]
        _response('{"answer": "two"}', prompt_tokens=-1, completion_tokens=-1)
    )
    provider("sys", "usr")

    metadata = provider.provider_metadata()
    assert metadata.monetary_cost_usd != 0.15, "call one's cost must not be re-reported"
    assert metadata.total_tokens is None and metadata.prompt_tokens is None
    assert metadata.model_id == "deepseek/deepseek-chat", "identity survives"


def test_an_out_of_range_max_tokens_names_its_variable_before_the_optional_import() -> None:
    """FIX-002. The one numeric the resolver did NOT validate before importing `openai`.

    On a floor install an out-of-range value reported `needs the openai extra`, which is the
    exact PR #366 ordering defect this resolver's own docstring claims to have eliminated —
    and the existing ordering test only covered TIMEOUT, so the uncovered numeric was the one
    that broke the property. The constructor's bound stays as the library-level invariant.
    """
    import builtins

    real_import = builtins.__import__

    def without_openai(name, *args, **kwargs):
        if name == "openai":
            raise AssertionError("openai imported before max_tokens was validated")
        return real_import(name, *args, **kwargs)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", without_openai)
        with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MAX_TOKENS"):
            resolve_answer_provider(_openai_enabled(RECALL_REASONING_ANSWER_MAX_TOKENS="99999"))


def test_a_signed_zero_price_records_a_positive_zero_cost() -> None:
    """FIX-003. `-0.0 < 0` is False, so a signed zero passed every guard and reached the artifact."""
    # Resolving the openai backend constructs the OpenAI client, so this needs the optional
    # extra; the `test` and `floor` CI jobs both install without it. Omitting this is the exact
    # defect commit 67c1de66 fixed, and the anti-regression gate caught it recurring here.
    pytest.importorskip("openai")
    import math as _math

    provider = resolve_answer_provider(
        _openai_enabled(RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS="-0.0")
    )
    assert provider is not None
    assert not _math.copysign(1.0, provider.cost_per_1k_tokens) < 0

    # Assert on the field the finding is ABOUT, not only on its source. The first version
    # checked `cost_per_1k_tokens` alone, which proves where the sign came from rather than
    # that it stayed out of the artifact; the architect gate named that gap.
    provider.client = _FakeClient(  # type: ignore[attr-defined]
        _response('{"answer": null}', prompt_tokens=10, completion_tokens=5)
    )
    provider("sys", "usr")
    recorded = provider.provider_metadata().monetary_cost_usd
    assert recorded == 0.0 and not _math.copysign(1.0, recorded) < 0, (
        f"monetary_cost_usd must be +0.0, got {recorded!r}"
    )

    # The constructor is the library-level invariant for a caller who bypasses the resolver.
    direct = OpenAICompatibleAnswerProvider(object(), model_id="m", cost_per_1k_tokens=-0.0)
    assert not _math.copysign(1.0, direct.cost_per_1k_tokens) < 0


def test_an_empty_revision_means_unset_not_empty() -> None:
    """FIX-004. A valueless .env line recorded model_revision='', which fails the cost gate.

    Exercised on the OLLAMA backend deliberately. The fix applies to both, and ollama needs no
    optional extra, so this stays LIVE on the `test` and `floor` CI jobs instead of skipping
    there. A guard that skips on every machine that runs it is not a guard.
    """
    provider = resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_REVISION="   "))
    assert provider is not None
    assert provider.revision == "unpinned"


# --- deferred findings, second pass -------------------------------------------------------


def test_the_legacy_bare_base_url_is_honoured_beside_the_legacy_bare_key() -> None:
    """ENV-001. Taking the legacy KEY while ignoring the legacy BASE URL is the dangerous half.

    A pre-0.11 `.env` naming a private gateway kept its `RECALL_REASONING_API_KEY` and silently
    acquired the OpenRouter default, so the credential AND the retrieved evidence went to a
    third party the operator never named. The sibling `resolve_expansion_provider` reads the
    bare key, base URL and timeout as a matched trio; this one took only the key.
    """
    pytest.importorskip("openai")
    provider = resolve_answer_provider(
        _enabled(
            RECALL_REASONING_ANSWER_PROVIDER="openai",
            RECALL_REASONING_API_KEY="sk-private",
            RECALL_REASONING_BASE_URL="https://gateway.internal.example/v1",
            RECALL_REASONING_TIMEOUT="12",
        )
    )
    assert provider is not None
    assert "gateway.internal.example" in str(provider.client.base_url), (
        "a legacy private gateway must not be silently replaced by the OpenRouter default"
    )
    assert "openrouter" not in str(provider.client.base_url)
    # The TRIO, not just the URL: without this the timeout half of the fix is unguarded, which
    # is how a half-applied legacy fallback would pass review a second time.
    assert provider.client.timeout == 12.0, "the legacy timeout is part of the same trio"


def test_the_infixed_base_url_still_beats_the_legacy_one() -> None:
    """Precedence must stay preferred-over-legacy, matching the key and the sibling resolver."""
    pytest.importorskip("openai")
    provider = resolve_answer_provider(
        _openai_enabled(
            RECALL_REASONING_ANSWER_BASE_URL="https://preferred.example/v1",
            RECALL_REASONING_BASE_URL="https://legacy.example/v1",
        )
    )
    assert provider is not None
    assert "preferred.example" in str(provider.client.base_url)


def test_the_ollama_backend_ignores_the_bare_hosted_base_url() -> None:
    """The bare family belongs to the hosted arm; feeding it to Ollama rebuilds the path bug.

    `_NativeOllamaClient` rewrites the path to `<base>/api/chat`, so honouring a hosted base URL
    here would produce exactly the unauthenticated POST-to-a-missing-path this module warns
    about. Ollama keeps its loopback default.
    """
    provider = resolve_answer_provider(
        _enabled(RECALL_REASONING_BASE_URL="https://openrouter.ai/api/v1")
    )
    assert provider is not None
    assert "127.0.0.1" in provider.client.endpoint


def test_a_hosted_endpoint_refuses_plaintext_http_because_it_carries_the_key() -> None:
    """SEC-002. This backend is the first to attach `Authorization: Bearer` to the base URL.

    The shared validation accepts `http` because the OLLAMA default is loopback. A copy-pasted
    or downgraded `http://` hosted endpoint would put the operator's key on the wire in
    cleartext, and nothing would say so.
    """
    with pytest.raises(ValueError, match="https"):
        resolve_answer_provider(
            _openai_enabled(RECALL_REASONING_ANSWER_BASE_URL="http://api.example.com/v1")
        )


@pytest.mark.parametrize(
    "url", ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1"], ids=["localhost", "loopback"]
)
def test_a_loopback_hosted_endpoint_may_still_use_http(url: str) -> None:
    """Loopback never leaves the machine, and is how a local gateway or a test double is reached.

    Without this the https rule would break every local OpenAI-compatible server, which is a
    real and legitimate configuration.
    """
    pytest.importorskip("openai")
    provider = resolve_answer_provider(
        _openai_enabled(RECALL_REASONING_ANSWER_BASE_URL=url)
    )
    assert provider is not None


def test_an_openai_only_setting_is_refused_under_ollama_rather_than_ignored() -> None:
    """BUG-006/ENV-003. The mirror image of the THINKING refusal, which this file already has.

    The ollama branch returned before the cost was ever read, so an operator who configured a
    per-1k price got `monetary_cost_usd=0.0` and no indication their setting did nothing.
    Silently ignoring configuration is the exact failure mode this module refuses elsewhere.
    """
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS="0.5"))


def test_a_shared_api_key_does_not_block_the_ollama_backend() -> None:
    """The key is deliberately NOT in the refused set: a shared .env carries it for another arm.

    Refusing it would break every installation that runs expansion and answering side by side,
    which is the ordinary case the legacy fallback exists to serve.
    """
    provider = resolve_answer_provider(
        _enabled(RECALL_REASONING_API_KEY="sk-for-the-expansion-arm")
    )
    assert provider is not None
