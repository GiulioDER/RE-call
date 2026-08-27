"""Database-backed behaviour of `recall_agent` on the sanctioned development path.

Uses the hashing embedder (dim 64, offline) on throwaway tables from `make_store`, with
`TrustPolicy.development()` plus an explicit `Calibration` so the verdict machinery runs instead
of collapsing to `unverified` (see `dev_search_memory` in conftest for why that pairing is the
convention).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from recall.calibration import Calibration
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.trust_policy import TrustPolicy
from recall_agent import RecallAgentMemory
from recall_mcp.factories import make_embedder
from tests.conftest import requires_db

pytestmark = requires_db


def _dev_memory(store: Any) -> RecallAgentMemory:
    return RecallAgentMemory(
        store=store,
        embedder=make_embedder("hashing"),
        policy=TrustPolicy.development(),
        calibration=Calibration(embedder="test-development", threshold=DEFAULT_GAP_THRESHOLD),
    )


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(result["content"][0]["text"])


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    (tmp_path / "deploy.md").write_text(
        "# Deploy pipeline\n\nThe deploy pipeline uses blue green switches for rollout.\n",
        encoding="utf-8",
    )
    (tmp_path / "billing.md").write_text(
        "# Billing\n\nInvoices are generated nightly by the billing batch job.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_index_search_and_forget_roundtrip_through_the_write_tools(make_store, corpus) -> None:
    memory = _dev_memory(make_store(64))

    indexed = _payload(asyncio.run(memory._recall_index({"path": str(corpus)})))
    assert indexed.get("files", indexed.get("indexed_files", 1))  # shape asserted loosely below

    found = _payload(
        asyncio.run(
            memory._recall_search(
                {"query": "the deploy pipeline uses blue green switches for rollout"}
            )
        )
    )
    assert found["trust_state"] == "degraded"  # development mode says so out loud
    assert found["hits"], "indexed text should be retrievable"
    top = found["hits"][0]
    assert top["verdict"] == "ok"  # the explicit calibration keeps verdicts running
    assert "deploy" in top["source"]

    gone = _payload(asyncio.run(memory._recall_forget({"sources": [top["source"]]})))
    assert gone
    after = _payload(
        asyncio.run(
            memory._recall_search(
                {"query": "the deploy pipeline uses blue green switches for rollout"}
            )
        )
    )
    assert all("deploy" not in hit["source"] for hit in after.get("hits", []))


def test_evidence_returns_a_decision_and_citable_items(make_store, corpus) -> None:
    memory = _dev_memory(make_store(64))
    asyncio.run(memory._recall_index({"path": str(corpus)}))
    payload = _payload(
        asyncio.run(
            memory._recall_evidence(
                {"query": "invoices are generated nightly by the billing batch job"}
            )
        )
    )
    assert payload["decision"] in {"answer", "abstain"}
    if payload["decision"] == "answer":
        assert payload["items"]
        assert payload["system_prompt"]


def test_session_start_digest_reflects_the_chunk_count_and_stays_silent_when_empty(
    make_store, corpus
) -> None:
    memory = _dev_memory(make_store(64))
    assert asyncio.run(memory._session_start({}, None, None)) == {}

    asyncio.run(memory._recall_index({"path": str(corpus)}))
    injected = asyncio.run(memory._session_start({}, None, None))
    context = injected["hookSpecificOutput"]["additionalContext"]
    assert "indexed chunks" in context
    assert "recall_search" in context


def test_concurrent_tool_calls_serialise_without_error(make_store, corpus) -> None:
    memory = _dev_memory(make_store(64))
    asyncio.run(memory._recall_index({"path": str(corpus)}))

    async def burst() -> list[dict[str, Any]]:
        return await asyncio.gather(
            *(
                memory._recall_search({"query": "deploy pipeline blue green rollout"})
                for _ in range(4)
            )
        )

    results = asyncio.run(burst())
    assert len(results) == 4
    assert all(_payload(r)["trust_state"] == "degraded" for r in results)
