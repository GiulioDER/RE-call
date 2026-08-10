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
from types import SimpleNamespace

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
    # `outbox_events_scrubbed` is new, and the expectation is updated deliberately rather than
    # loosened: the erasure receipt now has to say whether the migration outbox was swept, because
    # "the outbox was clean" and "the outbox was never consulted" were previously indistinguishable
    # to the caller on an irreversible path. Still an EXACT comparison, so a further field cannot
    # appear unnoticed. 0 here is the correct value and is itself the assertion: this store has no
    # control plane, so there is no outbox to scrub.
    assert payload == {
        "chunks_removed": 1,
        "sources_removed": ["f.md"],
        "sources_not_found": [],
        "message": "Forgot 1 chunk(s) from 1 source(s).",
        "outbox_events_scrubbed": 0,
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
    assert ann.read_only_hint is False
    assert ann.destructive_hint is True
