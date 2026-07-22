"""End-to-end check that the `recall_forget` MCP tool itself — not just the service function it
wraps — returns valid JSON matching `ForgetResult`.

FastMCP tool bodies read the lifespan context via `mcp.get_context()`, which in turn reads a
request-scoped contextvar (`mcp.server.lowlevel.server.request_ctx`) that is normally only set
while a real client request is in flight. Setting it directly lets the test call the actual
registered coroutine — annotations, JSON serialization and all — without standing up a full
stdio/SSE transport.
"""
from __future__ import annotations

import asyncio
import json

from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

from recall.embeddings import HashingEmbedder
from recall.types import Chunk
from recall_mcp.server import build_server

from tests.conftest import requires_db


def _call_tool(server, name: str, lifespan_context: dict, **kwargs):
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    fake_ctx = RequestContext(
        request_id="test", meta=None, session=None, lifespan_context=lifespan_context
    )

    async def run():
        token = request_ctx.set(fake_ctx)
        try:
            return await tools[name].fn(**kwargs)
        finally:
            request_ctx.reset(token)

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
    assert payload == {
        "chunks_removed": 1,
        "sources_removed": ["f.md"],
        "sources_not_found": [],
        "message": "Forgot 1 chunk(s) from 1 source(s).",
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


def test_recall_forget_is_registered_with_honest_destructive_annotations():
    server = build_server()
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    assert "recall_forget" in tools
    tool = tools["recall_forget"]
    ann = tool.annotations
    # This tool deletes data irreversibly — a client that trusts these hints to decide whether
    # to prompt the user must be told the truth, not inherit recall_index's read-only-ish hints.
    assert ann.readOnlyHint is False
    assert ann.destructiveHint is True
