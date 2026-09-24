from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from recall.trust_policy import TrustFailureCode, TrustRefusal
from recall_mcp import server
from recall_mcp.settings import Settings


def test_search_tool_keeps_trust_refusal_payload_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = TrustRefusal(
        code=TrustFailureCode.INDEX_NOT_READY,
        calibration_status="missing",
        tenant_id="default",
        generation_id="generation-1",
    )

    def refuse(*_args, **_kwargs):
        raise refusal

    monkeypatch.setattr(server, "search_memory", refuse)
    mcp = server.build_server()
    tool = {item.name: item for item in mcp._tool_manager.list_tools()}["recall_search"]
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context={
                "store": object(),
                "embedder": object(),
                "settings": Settings.from_env({}),
            }
        )
    )

    with pytest.raises(ToolError) as raised:
        asyncio.run(tool.run({"query": "what is current?"}, context))

    message = str(raised.value)
    assert message.startswith("Error executing tool recall_search: ")
    payload = json.loads(message.split(": ", 1)[1])
    assert payload["error"] == "trust_refusal"
    assert payload["code"] == "INDEX_NOT_READY"
    assert payload["calibration_status"] == "missing"
    assert payload["generation_id"] == "generation-1"


@pytest.mark.parametrize(
    ("tool_name", "service_name", "arguments"),
    [
        ("recall_related", "related_memory", {"seed_chunk_id": "chunk-1"}),
        (
            "recall_query_construction_challenge",
            "query_construction_challenge",
            {"original_prompt": "what is current?", "query": "what is current?"},
        ),
    ],
)
def test_every_trusted_retrieval_tool_keeps_the_refusal_payload_visible(
    monkeypatch: pytest.MonkeyPatch, tool_name: str, service_name: str, arguments: dict
) -> None:
    """A strict refusal reaches the client as the same structured payload on every tool.

    Invariant: every MCP tool whose body runs trusted retrieval (`trusted_related`, or the
    trusted search inside query construction) maps `TrustRefusal` through
    `_tool_error_for_trust_refusal`, as `recall_search` and `recall_evidence` do, so the client
    receives `{"error": "trust_refusal", "code": ...}` rather than a bare exception string.

    Red proof, recorded 2026-09-23 against `origin/master` at `3cc57b81`, where neither tool
    caught the refusal: both parameters failed at
    `assert message.startswith(f"Error executing tool {tool_name}: ")`, because the client
    received the bare `Error executing tool <name>` with no payload at all. Wrapping both tool
    bodies turns it green.
    """
    refusal = TrustRefusal(
        code=TrustFailureCode.CALIBRATION_STALE,
        calibration_status="stale",
        tenant_id="default",
        generation_id="generation-1",
    )

    def refuse(*_args, **_kwargs):
        raise refusal

    monkeypatch.setattr(server, service_name, refuse)
    mcp = server.build_server()
    tool = {item.name: item for item in mcp._tool_manager.list_tools()}[tool_name]
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context={
                "store": object(),
                "embedder": object(),
                "settings": Settings.from_env({}),
            }
        )
    )

    with pytest.raises(ToolError) as raised:
        asyncio.run(tool.run(arguments, context))

    message = str(raised.value)
    assert message.startswith(f"Error executing tool {tool_name}: ")
    payload = json.loads(message.split(": ", 1)[1])
    assert payload["error"] == "trust_refusal"
    assert payload["code"] == "CALIBRATION_STALE"
    assert payload["generation_id"] == "generation-1"
