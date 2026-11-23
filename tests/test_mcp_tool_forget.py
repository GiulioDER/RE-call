"""End-to-end check that the `recall_forget` MCP tool itself — not just the service function it
wraps — returns valid JSON matching `ForgetResult`.

MCP tool bodies read the lifespan context from the injected request context that is normally only
present while a real client request is in flight. Passing a minimal context object lets the test
call the actual registered coroutine, annotations, JSON serialization and all, without standing up
a full stdio/SSE transport.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from recall.embeddings import HashingEmbedder
from recall.types import Chunk
from recall_mcp.server import build_server

from tests.conftest import requires_db


def _call_tool(server, name: str, lifespan_context: dict, **kwargs):
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    fake_ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=lifespan_context)
    )

    async def run():
        return await tools[name].fn(ctx=fake_ctx, **kwargs)

    return asyncio.run(run())


@requires_db
def test_recall_forget_tool_returns_json_matching_forget_result(make_store):
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    store.upsert([Chunk("a", "f.md", "the caching decision was adopted")], [[1.0] + [0.0] * 63])
    assert store.count() == 1

    server = build_server()
    out = _call_tool(
        server, "recall_forget",
        {"store": store, "embedder": emb, "calibration": None},
        sources=["f.md"],
    )
    payload = json.loads(out)
    # `outbox_events_scrubbed` and `staged_files_removed` are deliberate expectation updates
    # rather than loosenings: the erasure receipt has to name every store that was swept,
    # because "it was clean" and "it was never consulted" were previously indistinguishable
    # to the caller on an irreversible path. Still an EXACT comparison, so a further field
    # cannot appear unnoticed. The zeros are themselves assertions: this store has no
    # control plane (no outbox to scrub), `f.md` was never a staged upload, and no fact was
    # applied from it (`facts_retracted`, the fact ledger was consulted and owed nothing).
    assert payload == {
        "chunks_removed": 1,
        "sources_removed": ["f.md"],
        "sources_not_found": [],
        "message": "Forgot 1 chunk(s) from 1 source(s).",
        "outbox_events_scrubbed": 0,
        "staged_files_removed": 0,
        "facts_retracted": 0,
        "facts_redacted": 0,
    }
    assert store.count() == 0


@requires_db
def test_recall_forget_tool_reports_not_found_without_touching_memory(make_store):
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    store.upsert([Chunk("a", "f.md", "kept")], [[1.0] + [0.0] * 63])

    server = build_server()
    out = _call_tool(
        server, "recall_forget",
        {"store": store, "embedder": emb, "calibration": None},
        sources=["typo.md"],
    )
    payload = json.loads(out)
    assert payload["chunks_removed"] == 0
    assert payload["sources_removed"] == []
    assert payload["sources_not_found"] == ["typo.md"]
    assert store.count() == 1


@requires_db
def test_recall_forget_replays_the_durable_receipt_without_a_second_delete(make_store):
    """A repeated key replays the first receipt instead of deleting again.

    The key is unique per run because `recall_idempotency_receipts` outlives the test: nothing
    clears it, so on a reused database (xdist's `<db>_gwN` left by an earlier run) a fixed key
    turned the FIRST call into a replay of the old receipt, which deletes nothing.

    Red proof (2026-09-23, VPS3, base `a42f035c`), node
    `tests/test_mcp_tool_forget.py::test_recall_forget_replays_the_durable_receipt_without_a_second_delete`:
    with the old fixed key `"forget-replay-1"`, run 1 on a fresh database passed and run 2 on
    the same database failed `assert store.count() == 0` (`assert 1 == 0`). With this key it
    passed twice on that same database. Mutation: making `PgVectorStore.get_operation_receipt`
    always return None makes the second call run the erasure again, and the test fails
    `assert second == first` (the second receipt reports `chunks_removed` 0). Skipping the
    `replay` return in `recall_mcp.server.recall_forget` is NOT valid red proof: it stops at
    the tool's own `assert store is not None` before this test asserts anything.
    """
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    store.upsert([Chunk("a", "f.md", "the caching decision was adopted")], [[1.0] + [0.0] * 63])
    key = "forget-replay-" + uuid.uuid4().hex

    server = build_server()
    first = _call_tool(
        server,
        "recall_forget",
        {"store": store, "embedder": emb, "calibration": None},
        sources=["f.md"],
        idempotency_key=key,
    )
    second = _call_tool(
        server,
        "recall_forget",
        {"store": store, "embedder": emb, "calibration": None},
        sources=["f.md"],
        idempotency_key=key,
    )

    assert second == first
    assert json.loads(second)["chunks_removed"] == 1
    assert store.count() == 0


def test_recall_forget_is_registered_with_honest_destructive_annotations():
    server = build_server()
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    assert "recall_forget" in tools
    tool = tools["recall_forget"]
    ann = tool.annotations
    # This tool deletes data irreversibly — a client that trusts these hints to decide whether
    # to prompt the user must be told the truth, not inherit recall_index's read-only-ish hints.
    assert ann.read_only_hint is False
    assert ann.destructive_hint is True


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_shadow_preflight_releases_reservation(monkeypatch, cleanup_fails):
    """A shadow lookup failure releases the key without entering erasure (STAKES-001)."""
    from recall_mcp import server_memory_admin
    from recall_mcp.server_common import _ToolDeps

    failure = RuntimeError("shadow store unavailable")
    release = AsyncMock(side_effect=RuntimeError("reservation store unavailable") if cleanup_fails else None)
    registry = SimpleNamespace(get_shadow=Mock(side_effect=failure))
    store = SimpleNamespace(tenant="fixture-tenant")
    state = {"stores": registry, "limiter": SimpleNamespace(release_idempotency_reservation=release)}
    eraser = Mock()
    record = AsyncMock()
    tools = {}

    def register(*, name, **_options):
        def add(tool):
            tools[name] = tool
            return tool

        return add

    # Every await in this fixture completes synchronously. No MCP transport or event loop is needed.
    monkeypatch.setattr(server_memory_admin, "_to_thread", AsyncMock(side_effect=lambda task: task()))
    monkeypatch.setattr(server_memory_admin, "forget_memory", eraser)
    server_memory_admin._register_memory_admin_tools(
        Mock(tool=register),
        _ToolDeps(
            require=AsyncMock(),
            require_mutation=AsyncMock(return_value=(store, None)),
            record_mutation_result=record,
            state=lambda _ctx: state,
            current_tenant=lambda _state: store.tenant,
            access_context=Mock(),
            answer_backend_configured=False,
        ),
    )
    request = tools["recall_forget"](sources=["fixture.md"], ctx=None, idempotency_key="fixture-key")
    try:
        with pytest.raises(RuntimeError, match="shadow store unavailable") as raised:
            request.send(None)
    finally:
        request.close()

    assert raised.value is failure
    registry.get_shadow.assert_called_once_with(store.tenant)
    eraser.assert_not_called()
    record.assert_not_called()
    release.assert_awaited_once_with(store.tenant, "fixture-key")
