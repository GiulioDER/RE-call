"""Gated live round trip: a real Claude Agent SDK session calling the in-process tools.

Off by default three times over: it needs the `agent` extra, the Claude Code CLI on PATH, an
account able to answer one query, and `RECALL_AGENT_E2E=1`. It exists so the wiring the fake-SDK
contract tests cannot see (the SDK's real tool registration and hook plumbing) can be exercised
deliberately before a release.
"""
from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from tests.conftest import requires_db

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

pytestmark = [
    requires_db,
    pytest.mark.skipif(
        os.environ.get("RECALL_AGENT_E2E") != "1",
        reason="live SDK round trip; set RECALL_AGENT_E2E=1 to run",
    ),
    pytest.mark.skipif(
        shutil.which("claude") is None, reason="needs the Claude Code CLI on PATH"
    ),
    pytest.mark.timeout(300),
]


def test_one_live_query_reaches_the_in_process_search_tool(make_store, tmp_path, monkeypatch):
    from recall.calibration import Calibration
    from recall.guards import DEFAULT_GAP_THRESHOLD
    from recall.trust_policy import TrustPolicy
    from recall_agent import RecallAgentMemory
    from recall_mcp.factories import make_embedder

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    (tmp_path / "fact.md").write_text(
        "# Fact\n\nThe release train departs on Thursdays.\n", encoding="utf-8"
    )
    memory = RecallAgentMemory(
        store=make_store(64),
        embedder=make_embedder("hashing"),
        policy=TrustPolicy.development(),
        calibration=Calibration(embedder="test-development", threshold=DEFAULT_GAP_THRESHOLD),
    )
    asyncio.run(memory._recall_index({"path": str(tmp_path)}))

    async def run() -> list[object]:
        messages = []
        options = memory.options(max_turns=2, permission_mode="dontAsk")
        async for message in claude_agent_sdk.query(
            prompt="Use the recall_search tool to find when the release train departs, "
            "then answer in one sentence.",
            options=options,
        ):
            messages.append(message)
        return messages

    messages = asyncio.run(run())
    result = messages[-1]
    assert getattr(result, "is_error", False) is False
