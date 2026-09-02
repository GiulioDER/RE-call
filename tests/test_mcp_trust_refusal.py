from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from recall.trust_policy import TrustFailureCode, TrustRefusal
from recall_mcp import server


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
            lifespan_context={"store": object(), "embedder": object()}
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
