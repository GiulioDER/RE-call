"""Visibility and retry contracts for the release MCP repairs.

The original regressions were proved on ac67de930. The generation, source-read
and alias eligibility cases were proved on 31e6841dc. The append cases target
the write-start notification at each ledger append.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from recall.generation_types import NoActiveGeneration
from recall.security_policy import SourceRule, SourceSecurityPolicy
from recall.types import Chunk
from recall_mcp import server, use_reports_api
from recall_mcp.auth import SCOPE_READ, SCOPE_WRITE
from recall_mcp.use_reports_api import report_use
from tests.test_mcp_auth_hunt_regressions import (
    _IdempotentLimiter, _Registry, _Token, _authenticated, _run,
)
from tests.test_report_tools_database_names import FILES, NEW_QUOTE, OLD_QUOTE, SERVED, GenerationLikeStore
from tests.test_report_tools_database_names import call as call  # noqa: F401


def test_hidden_alias_does_not_change_a_readable_basename_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-002. Red translates the name to the hidden row and loses the readable fallback."""
    visible = "/corpus/agent-b/note-0001.md"
    hidden = "/corpus/agent-a/note-0001.md"

    class Ledger:
        tenant = "memory"

        def append_audit_event(self, *args: Any, **kwargs: Any) -> str:
            return "synthetic-event"

    def authorized(store: Any, names: Any, policy: Any, context: Any, **kwargs: Any) -> dict[str, Any]:
        return {visible: []} if visible in names or "note-0001.md" in names else {}

    monkeypatch.setattr(use_reports_api, "authorized_report_rows", authorized)
    kwargs = {"task": "Identify the current decision.", "effect": "helped",
              "used": ["note-0001.md"], "security_policy": object(), "access_context": object()}
    first = report_use(Ledger(), **kwargs)
    try:
        second = report_use(Ledger(), aliases={"note-0001.md": hidden}, **kwargs)
    except Exception as exc:
        second = {"failed": type(exc).__name__}
    assert second.get("used") == first["used"] == [visible], second


@pytest.mark.parametrize("tool", ["recall_report_use", "recall_report_stale"])
def test_a_failed_alias_lookup_releases_the_unwritten_request(
    monkeypatch: pytest.MonkeyPatch, tool: str,
) -> None:
    """DAT-002. Red retains the reserved key after a pre-write alias lookup raises."""
    store = GenerationLikeStore()
    store.tenant = "acme"
    pending: set[str] = set()

    class Registry(_Registry):
        def get(self, tenant: str) -> Any:
            return store

    class Limiter(_IdempotentLimiter):
        async def check(self, tenant: str, key: str, **kwargs: Any) -> None:
            request = kwargs.get("idempotency_key")
            if request:
                assert request not in pending, "the pre-write reservation survived"
                pending.add(request)

        async def release_idempotency_reservation(self, tenant: str, key: str) -> None:
            pending.discard(key)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("synthetic lookup interruption")

    monkeypatch.setattr(store, "sources_for_identifiers", broken)
    monkeypatch.setattr(store, "served_sources_for_identifiers", broken, raising=False)
    state = _authenticated(Registry(), Limiter())
    token = _Token([SCOPE_WRITE])
    if tool == "recall_report_use":
        arguments = {"task": "Identify the current decision.", "effect": "helped",
                     "used": ["recall/new.md"]}
    else:
        arguments = {"stale_source": "recall/old.md", "replacing_source": "recall/new.md",
                     "stale_quote": "The old decision was recorded here.",
                     "current_quote": "The new decision was recorded here."}
    with pytest.raises(OSError, match="synthetic lookup"):
        _run(tool, state, monkeypatch, token, idempotency_key="same-request", **arguments)
    assert pending == set(), "the pre-write reservation survived"

    def found(names: list[str]) -> dict[str, list[str]]:
        return {name: [SERVED + name] for name in names}

    monkeypatch.setattr(store, "sources_for_identifiers", found)
    monkeypatch.setattr(store, "served_sources_for_identifiers", found)
    operation = "report_use" if tool == "recall_report_use" else "report_stale"
    monkeypatch.setattr(server, operation, lambda target, **kwargs: (
        target.rows.append((operation, {})) or {"recorded": True}
    ))
    result = _run(tool, state, monkeypatch, token, idempotency_key="same-request", **arguments)
    assert json.loads(result) == {"recorded": True}
    assert len(store.rows) == 1


def _report_arguments(tool: str) -> dict[str, Any]:
    if tool == "recall_report_use":
        return {"task": "Identify the current release.", "effect": "helped", "used": ["recall/new.md"]}
    return {"stale_source": "recall/old.md", "replacing_source": "recall/new.md",
            "stale_quote": OLD_QUOTE, "current_quote": NEW_QUOTE}


@pytest.mark.parametrize("tool", ["recall_report_use", "recall_report_stale"])
def test_alias_and_report_reads_share_a_served_generation(
    call: Any, monkeypatch: pytest.MonkeyPatch, tool: str,
) -> None:
    """BUG-001. A promotion after alias resolution cannot split the report's source reads."""
    store = call.store
    active = SERVED
    pinned = None

    @contextmanager
    def snapshot() -> Iterator[None]:
        nonlocal pinned
        pinned = active
        try:
            yield
        finally:
            pinned = None

    def lookup(names: list[str]) -> dict[str, list[str]]:
        nonlocal active
        mappings = {name: [(pinned or active) + name] for name in names}
        active = "s3://example.test/promoted/"
        return mappings

    monkeypatch.setattr("recall.generation_store.GenerationStore", type(store))
    monkeypatch.setattr(store, "snapshot", snapshot, raising=False)
    monkeypatch.setattr(store, "served_sources_for_identifiers", lookup, raising=False)
    monkeypatch.setattr(store, "source_content_hashes", lambda: {(pinned or active) + f: "h" for f in FILES})
    monkeypatch.setattr(store, "source_chunk_texts", lambda source: [FILES[source.removeprefix(pinned or active)]])

    result = call(tool, **_report_arguments(tool))
    assert not result.startswith("RAISED"), f"promotion changed report source validation: {result}"
    ((_, payload),) = store.rows
    assert payload.get("used", [payload.get("replacing_source")]) == [SERVED + "recall/new.md"]


@pytest.mark.parametrize("tool", ["recall_report_use", "recall_report_stale"])
@pytest.mark.parametrize("failure", ["read", "append"])
def test_report_reservation_tracks_the_write_boundary(
    monkeypatch: pytest.MonkeyPatch, tool: str, failure: str,
) -> None:
    """BUG-002. Retry an unwritten report and retain an indeterminate append for reconciliation."""
    store = GenerationLikeStore()
    store.tenant = "acme"
    pending: set[str] = set()

    class Registry(_Registry):
        def get(self, tenant: str) -> Any:
            return store

    class Limiter(_IdempotentLimiter):
        async def check(self, tenant: str, key: str, **kwargs: Any) -> None:
            request = kwargs.get("idempotency_key")
            if request:
                assert request not in pending
                pending.add(request)

        async def release_idempotency_reservation(self, tenant: str, key: str) -> None:
            pending.discard(key)

    def broken_read() -> dict[str, str]:
        raise OSError("synthetic source read interruption")

    append = store.append_audit_event

    def interrupted_append(*args: Any, **kwargs: Any) -> str:
        append(*args, **kwargs)
        raise OSError("synthetic acknowledgement interruption")

    state, token = _authenticated(Registry(), Limiter()), _Token([SCOPE_WRITE])
    args = {"idempotency_key": "same-request", **_report_arguments(tool)}
    with monkeypatch.context() as fault:
        if failure == "read":
            fault.setattr(store, "source_content_hashes", broken_read)
        else:
            fault.setattr(store, "append_audit_event", interrupted_append)
        with pytest.raises(OSError, match="synthetic .* interruption"):
            _run(tool, state, fault, token, **args)

    if failure == "append":
        assert pending == {"same-request"}, "an indeterminate append lost its reservation"
    else:
        assert pending == set(), "the reservation survived a failure before the first write"
        assert not store.rows
        _run(tool, state, monkeypatch, token, **args)
    assert len(store.rows) == 1


@pytest.mark.parametrize("tool", ["recall_report_use", "recall_report_stale"])
def test_a_qualified_alias_resolves_among_eligible_rows(
    monkeypatch: pytest.MonkeyPatch, tool: str,
) -> None:
    """SEC-001. A legacy row outside the current policy cannot invalidate an eligible alias."""
    store = GenerationLikeStore()
    store.tenant = "acme"
    policy = SourceSecurityPolicy((SourceRule("recall"),))
    old_root = "s3://example.test/old-root/"
    chunks = [
        Chunk(id=f"fixture-{index}-{root}", source=root + file, text=FILES[file], metadata={"file": file, **stamp})
        for index, file in enumerate(FILES)
        for root, stamp in [(SERVED, {"security_policy_digest": policy.digest}), (old_root, {})]
    ]

    class Registry(_Registry):
        def get(self, tenant: str) -> Any:
            return store

    def lookup(names: list[str]) -> dict[str, list[str]]:
        return {name: [SERVED + name, old_root + name] for name in names}

    monkeypatch.setattr(store, "served_sources_for_identifiers", lookup, raising=False)
    monkeypatch.setattr(store, "iter_chunks", lambda: iter(chunks), raising=False)
    state = _authenticated(Registry())
    state["source_security_policy"] = policy
    try:
        result = _run(tool, state, monkeypatch, _Token([SCOPE_READ, SCOPE_WRITE]), **_report_arguments(tool))
    except server.ToolError as exc:
        result = f"RAISED {exc}"
    assert not result.startswith("RAISED"), f"the eligible display name was refused: {result}"
    ((_, payload),) = store.rows
    assert payload.get("used", [payload.get("replacing_source")]) == [SERVED + "recall/new.md"]


def test_a_report_without_sources_needs_no_served_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generation pin exists to bind source reads, so a source-free report needs no pin."""
    store = GenerationLikeStore()
    store.tenant = "acme"

    class Registry(_Registry):
        def get(self, tenant: str) -> Any:
            return store

    def no_generation() -> None:
        raise NoActiveGeneration("no served generation")

    monkeypatch.setattr("recall.generation_store.GenerationStore", type(store))
    monkeypatch.setattr(store, "snapshot", no_generation, raising=False)
    state = _authenticated(Registry())
    state["source_security_policy"] = SourceSecurityPolicy((SourceRule("recall"),))
    try:
        result = _run("recall_report_use", state, monkeypatch, _Token([SCOPE_READ, SCOPE_WRITE]),
                      task="Check the release.", effect="no_difference")
    except NoActiveGeneration as exc:
        result = f"RAISED {exc}"
    assert not result.startswith("RAISED"), f"a source-free report required a generation: {result}"
    assert len(store.rows) == 1
