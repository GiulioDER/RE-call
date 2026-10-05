"""`recall_report_stale` queues a checked report and changes nothing in memory.

The registered coroutine is called with a minimal stdio context, as `test_mcp_tool_forget.py`
does, so annotations, JSON serialisation and the refusal path are the real ones.

Red proof, 2026-10-05, each mutation alone, failing for the stated reason (JUnit XML), then restored:
- T1 the server returns `{"queued": false}` instead of raising on a refusal:
  `test_a_refusal_reaches_the_client_with_its_reason`, DID NOT RAISE ToolError.
- T2 `report_stale` labels every report `mcp:local`: `test_a_checked_report_is_queued_and_memory_is_unchanged`,
  AssertionError 'mcp:local' == 'mcp:acme'.
- T3 the tool added to the `read` preset: `test_the_tool_is_not_in_the_minimal_presets`, AssertionError.
- T4 the tool handing `_state(ctx)["store"]` to the service instead of the store `_require` returned:
  `tests/test_mcp_tool_authorization.py::test_a_tool_uses_the_store_require_handed_back[recall_report_stale]`,
  AssertionError naming the UNAUTHENTICATED-STDIO-FALLBACK store. That harness only sees this tool
  because `stale_reports_api` was added to its approved service modules in the same change.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from recall.stale_reports import StaleReportQueue, default_queue_path
from recall_mcp.server import build_server

OLD = "The release is 0.12.0 and it is still being prepared for PyPI.\n"
NEW = "Release 0.12.0 was published to PyPI on 2 September.\n"


def _call(name: str, **kwargs):
    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"store": SimpleNamespace(tenant="acme"), "embedder": None, "calibration": None})
    )

    async def run():
        return await tools[name].fn(ctx=ctx, **kwargs)

    return asyncio.run(run())


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "old.md").write_text(OLD, encoding="utf-8")
    (root / "new.md").write_text(NEW, encoding="utf-8")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    return root


def test_a_checked_report_is_queued_and_memory_is_unchanged(corpus: Path) -> None:
    before = {p.name: p.read_bytes() for p in corpus.glob("*.md")}
    out = json.loads(
        _call(
            "recall_report_stale",
            stale_source="old.md",
            replacing_source="new.md",
            stale_quote="it is still being prepared for PyPI",
            current_quote="was published to PyPI on 2 September",
            task="which release is current?",
        )
    )
    assert out["queued"] is True and out["status"] == "pending"
    assert (out["stale_source"], out["replacing_source"]) == ("old.md", "new.md")
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        (report,) = queue.list()
    assert report.client == "mcp:acme"
    assert {p.name: p.read_bytes() for p in corpus.glob("*.md")} == before


def test_a_refusal_reaches_the_client_with_its_reason(corpus: Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        _call(
            "recall_report_stale",
            stale_source="old.md",
            replacing_source="new.md",
            stale_quote="it is still being prepared for PyPI",
            current_quote="this sentence is not in the newer memo",
        )
    payload = json.loads(str(excinfo.value))
    assert payload["error"] == "stale_report_refused"
    assert "current_quote is not verbatim" in payload["reason"]
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        assert queue.count("pending") == 0


def test_the_tool_is_not_in_the_minimal_presets() -> None:
    """It costs context on every turn; the curated `search` and `read` surfaces stay as they were."""
    from recall_mcp.tool_surface import ALL_TOOL_NAMES, TOOL_PRESETS

    assert "recall_report_stale" in ALL_TOOL_NAMES
    assert "recall_report_stale" not in TOOL_PRESETS["search"]
    assert "recall_report_stale" not in TOOL_PRESETS["read"]
