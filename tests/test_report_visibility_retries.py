"""Visibility and retry contracts for the release MCP repairs.

The exact nodes are run red against ac67de930 before the production changes.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from recall_mcp import use_reports_api
from recall_mcp.auth import SCOPE_WRITE
from recall_mcp.use_reports_api import report_use
from tests.test_mcp_auth_hunt_regressions import (
    _IdempotentLimiter, _Registry, _Token, _authenticated, _run,
)
from tests.test_report_tools_database_names import GenerationLikeStore, SERVED
from tests.server_modules import setattr_on_server


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
    setattr_on_server(monkeypatch, operation, lambda target, **kwargs: (
        target.rows.append((operation, {})) or {"recorded": True}
    ))
    result = _run(tool, state, monkeypatch, token, idempotency_key="same-request", **arguments)
    assert json.loads(result) == {"recorded": True}
    assert len(store.rows) == 1
