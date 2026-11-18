"""Release MCP regressions, proved red on ac67de930 before repair.

Each node drives a registered report or stats tool. Synthetic stores provide the
storage boundary without a database, network, embedding model or paid API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall_mcp import server
from tests.test_report_tools_database_names import FILES, NEW_QUOTE, OLD_QUOTE, SERVED
from tests.test_report_tools_database_names import call as call  # noqa: F401
from tests.server_modules import setattr_on_server


@pytest.mark.parametrize("tool", ["recall_report_use", "recall_report_stale"])
def test_reports_bind_aliases_to_the_served_generation(call: Any, monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    """BUG-003/PERF-001. Red refuses a current alias because a retired association survives."""
    def historical(names: list[str]) -> dict[str, list[str]]:
        return {name: [SERVED + name, SERVED + "other.md"] for name in names}

    def served(names: list[str]) -> dict[str, list[str]]:
        return {name: [SERVED + name] for name in names if name in FILES}

    call.store.served[SERVED + "other.md"] = "A source served under another filename."
    monkeypatch.setattr(call.store, "sources_for_identifiers", historical)
    monkeypatch.setattr(call.store, "served_sources_for_identifiers", served, raising=False)
    if tool == "recall_report_use":
        result = call(tool, task="Identify the current release.", effect="helped", used=["recall/new.md"])
    else:
        result = call(tool, stale_source="recall/old.md", replacing_source="recall/new.md",
                      stale_quote=OLD_QUOTE, current_quote=NEW_QUOTE)
    assert not result.startswith("RAISED"), f"the served alias was not resolved: {result}"
    payload = call.store.rows[0][1]
    assert payload.get("used", [payload.get("replacing_source")]) == [SERVED + "recall/new.md"]


def test_local_stale_sidecar_survives_an_alias_lookup_failure(
    call: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """DAT-001. Red aborts at alias lookup before a valid local report is queued."""
    for file, text in FILES.items():
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("synthetic unavailable lookup")

    monkeypatch.setattr(call.store, "sources_for_identifiers", broken)
    monkeypatch.setattr(call.store, "served_sources_for_identifiers", broken, raising=False)
    result = call("recall_report_stale", stale_source="recall/old.md", replacing_source="recall/new.md",
                  stale_quote=OLD_QUOTE, current_quote=NEW_QUOTE)
    assert not result.startswith("RAISED"), f"the local report was lost: {result}"
    payload = json.loads(result)
    assert payload["recorded_in"] == ["sidecar"]
    assert payload["report_count"] == 1


def _stats_caller() -> tuple[Any, Any]:
    """An authenticated tool context whose authorization must fail before opening a store."""
    tools = {tool.name: tool for tool in server.build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={
        "stores": object(), "embedder": None,
    }))
    return tools["recall_stats"].fn, ctx


def test_scope_denials_keep_the_verified_tenant(monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    """SEC-003. Red records [None, None] for distinct verified tenant claims."""
    caplog.set_level(logging.WARNING, logger="recall.mcp.access")
    fn, ctx = _stats_caller()
    for tenant in ("tenant-a", "tenant-b"):
        token = SimpleNamespace(claims={"tenant": tenant, "principal": "synthetic-reader"},
                                scopes=[], client_id="fixture")
        setattr_on_server(monkeypatch, "get_access_token", lambda: token)
        with pytest.raises(PermissionError):
            asyncio.run(fn(ctx=ctx))
    records = [record for record in caplog.records if record.name == "recall.mcp.access"]
    assert [record.tenant for record in records] == ["tenant-a", "tenant-b"]


def test_access_logging_does_not_block_the_shared_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-002. Red completes the denied task before the loop can release its blocked logger."""
    started, release = threading.Event(), threading.Event()

    def blocked(**kwargs: Any) -> None:
        started.set()
        release.wait(1)

    token = SimpleNamespace(claims={"tenant": "tenant-a"}, scopes=[], client_id="fixture")
    setattr_on_server(monkeypatch, "get_access_token", lambda: token)
    setattr_on_server(monkeypatch, "record_access", blocked)
    fn, ctx = _stats_caller()

    async def run() -> None:
        task = asyncio.create_task(fn(ctx=ctx))
        try:
            await asyncio.sleep(0)
            assert await asyncio.to_thread(started.wait, 1), "the log sink was never reached"
            assert not task.done(), "the shared loop was blocked by its access logger"
        finally:
            release.set()
            with pytest.raises(PermissionError):
                await task

    asyncio.run(run())
