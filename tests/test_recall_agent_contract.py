"""Contract tests for `recall_agent`: no database, no `claude_agent_sdk`.

Each test names the behaviour it protects, in the style of `test_their_harness_backend.py`.
The SDK is simulated both ways: absent (its import must produce the install hint, never a bare
ImportError from deep inside) and present (a minimal fake, so `options()` merge semantics are
testable in CI, which deliberately installs without the `agent` extra).
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

import recall_agent.memory as memory_mod
from recall.trust_policy import TrustPolicy
from recall_agent import RecallAgentMemory
from recall_agent.memory import resolve_dsn
from recall_mcp.factories import make_embedder
from recall_mcp.service import SearchResult


class _RefusingRetrievalStore:
    """A store that a strict gate must refuse BEFORE any retrieval method runs.

    It deliberately lacks `snapshot`/`generation_binding`/`resolve_calibration`, so the trust
    layer sees an unbound, uncalibrated store; every data-fetching method raises so a gate that
    leaked past the refusal would fail this test loudly rather than return something.
    """

    tenant = "t"
    generation_id = "legacy"
    table = "chunks"
    dim = 64

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("retrieval reached the store before the trust gate refused")

    query_dense = _refuse
    query_sparse = _refuse
    query = _refuse
    cosines_for = _refuse
    newest_indexed_at = _refuse
    count = _refuse


def _memory(**overrides: Any) -> RecallAgentMemory:
    overrides.setdefault("store", _RefusingRetrievalStore())
    overrides.setdefault("embedder", make_embedder("hashing"))
    overrides.setdefault("policy", TrustPolicy())  # strict, explicitly
    return RecallAgentMemory(**overrides)


def test_a_strict_refusal_renders_the_wire_form_with_advice_and_no_hits() -> None:
    result = asyncio.run(_memory()._recall_search({"query": "anything"}))
    payload = json.loads(result["content"][0]["text"])
    assert payload["trust_state"] == "refused"
    assert payload["calibrated"] is False
    assert payload["advice"]
    assert "hits" not in payload
    assert "query" not in payload
    assert result.get("is_error") is not True


def test_a_refusal_is_raised_before_any_retrieval_touches_the_store() -> None:
    # _RefusingRetrievalStore raises AssertionError from every data method; reaching the rendered
    # refusal at all is the proof that nothing was fetched.
    result = asyncio.run(_memory()._recall_search({"query": "anything"}))
    assert json.loads(result["content"][0]["text"])["code"]


def test_an_empty_trusted_result_stays_distinguishable_from_a_refusal() -> None:
    trusted_empty = SearchResult.model_construct(
        query="q",
        abstained=True,
        reason="",
        trust_state="trusted",
        failure_code=None,
        calibrated=True,
        hits=[],
        related_items=[],
        related_diagnostics=[],
        explanation=None,
    )
    from recall_agent.rendering import render_result

    payload = json.loads(render_result(trusted_empty)["content"][0]["text"])
    assert payload["trust_state"] == "trusted"
    assert payload["hits"] == []
    assert "code" not in payload


def test_dsn_resolution_prefers_explicit_then_serving_then_dsn() -> None:
    env = {"RECALL_SERVING_DSN": "postgresql://serving/db", "RECALL_DSN": "postgresql://plain/db"}
    assert resolve_dsn(env, "postgresql://explicit/db") == "postgresql://explicit/db"
    assert resolve_dsn(env) == "postgresql://serving/db"
    assert resolve_dsn({"RECALL_DSN": "postgresql://plain/db"}) == "postgresql://plain/db"
    assert resolve_dsn({}) == memory_mod.DEFAULT_DSN


def test_from_env_reads_the_mapping_it_is_given_not_the_process_env(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_SERVING_DSN", "postgresql://process-env/should-lose")
    made = RecallAgentMemory.from_env(
        env={"RECALL_DSN": "postgresql://mapping/wins"},
        embedder=make_embedder("hashing"),
    )
    assert made._dsn == "postgresql://mapping/wins"
    # And the mapping's absence of RECALL_TRUST_MODE means strict, regardless of the process env.
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")
    strict = RecallAgentMemory.from_env(env={}, embedder=make_embedder("hashing"))
    assert strict._policy == TrustPolicy()


def test_generation_store_flag_selects_the_generation_store_class(monkeypatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    class FakeStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("plain", args, kwargs))

    class FakeGeneration:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("generation", args, kwargs))

    monkeypatch.setattr(memory_mod, "PgVectorStore", FakeStore)
    monkeypatch.setattr(memory_mod, "GenerationStore", FakeGeneration)
    embedder = make_embedder("hashing")

    RecallAgentMemory(
        dsn="postgresql://x/y", embedder=embedder, use_generation_store=True, tenant="ten"
    )._make_store()
    assert calls[-1][0] == "generation"
    assert calls[-1][1] == ("postgresql://x/y", embedder.dim)
    assert calls[-1][2]["tenant"] == "ten"

    RecallAgentMemory(dsn="postgresql://x/y", embedder=embedder, table="tbl")._make_store()
    assert calls[-1][0] == "plain"
    assert calls[-1][2]["table"] == "tbl"
    assert calls[-1][2]["pool_size"] == 2


def test_an_injected_store_excludes_dsn_table_and_generation_flags() -> None:
    with pytest.raises(ValueError, match="injected store"):
        RecallAgentMemory(
            store=_RefusingRetrievalStore(),  # type: ignore[arg-type]
            dsn="postgresql://x/y",
            embedder=make_embedder("hashing"),
        )


def test_session_start_fails_open_when_the_store_is_unreachable() -> None:
    # _RefusingRetrievalStore.count raises; the hook must swallow that and inject nothing.
    result = asyncio.run(_memory()._session_start({}, None, None))
    assert result == {}


def _install_fake_sdk(monkeypatch) -> types.ModuleType:
    mod = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class HookMatcher:
        def __init__(self, matcher: Any = None, hooks: Any = ()) -> None:
            self.matcher = matcher
            self.hooks = list(hooks)

    def tool(name: str, description: str, schema: Any):
        def deco(fn: Any) -> Any:
            return types.SimpleNamespace(
                name=name, description=description, schema=schema, handler=fn
            )

        return deco

    def create_sdk_mcp_server(name: str, version: str = "1.0.0", tools: Any = None) -> Any:
        return types.SimpleNamespace(name=name, version=version, tools=list(tools or []))

    mod.ClaudeAgentOptions = ClaudeAgentOptions  # type: ignore[attr-defined]
    mod.HookMatcher = HookMatcher  # type: ignore[attr-defined]
    mod.tool = tool  # type: ignore[attr-defined]
    mod.create_sdk_mcp_server = create_sdk_mcp_server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


def test_write_tools_are_off_by_default_and_opt_in(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch)
    memory = _memory()
    read_only = memory.sdk_mcp_server()
    assert [t.name for t in read_only.tools] == ["recall_search", "recall_evidence"]
    writable = memory.sdk_mcp_server(write_tools=True)
    assert [t.name for t in writable.tools] == [
        "recall_search",
        "recall_evidence",
        "recall_index",
        "recall_forget",
    ]
    assert memory.allowed_tools() == [
        "mcp__recall__recall_search",
        "mcp__recall__recall_evidence",
    ]


def test_options_merges_servers_tools_and_hooks_never_replacing(monkeypatch) -> None:
    fake = _install_fake_sdk(monkeypatch)
    memory = _memory()
    other_matcher = fake.HookMatcher(hooks=[lambda *a: {}])
    options = memory.options(
        mcp_servers={"other": {"type": "stdio", "command": "x"}},
        allowed_tools=["Read"],
        hooks={"SessionStart": [other_matcher], "PreToolUse": [other_matcher]},
        model="claude-sonnet-5",
    )
    assert set(options.mcp_servers) == {"recall", "other"}
    assert options.allowed_tools[:2] == [
        "mcp__recall__recall_search",
        "mcp__recall__recall_evidence",
    ]
    assert "Read" in options.allowed_tools
    assert len(options.hooks["SessionStart"]) == 2
    assert options.hooks["PreToolUse"] == [other_matcher]
    assert options.model == "claude-sonnet-5"


def test_options_collision_on_our_own_server_name_raises(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch)
    with pytest.raises(ValueError, match="server name"):
        _memory().options(mcp_servers={"recall": {"type": "stdio", "command": "x"}})


def test_a_missing_sdk_names_the_install_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(ImportError, match=r"recall-rag\[agent\]"):
        _memory().sdk_mcp_server()
    with pytest.raises(ImportError, match=r"recall-rag\[agent\]"):
        _memory().session_start_hook()
    with pytest.raises(ImportError, match=r"recall-rag\[agent\]"):
        _memory().options()
