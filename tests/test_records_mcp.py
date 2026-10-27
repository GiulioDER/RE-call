"""The `recall_records` MCP tool on a lite server (Task 1.2).

Invariants and the failure each one catches:
- M1 the tool returns the C5 envelope `lookup_records` builds, every golden record exactly as
  authored. Catches a wrapper that reshapes or re-serialises the record.
- M2 an over-limit query is refused with a message naming the limit, never truncated. Catches a
  refusal the MCP SDK redacts to "Error executing tool" (a plain exception is redacted; a
  `ToolError` keeps its text).
- M3 the tool is registered on a lite server and is part of the `all` surface.

Red proof, recorded 2026-10-21 on the deployment host inside the test bounds:
- M1 `tests/test_records_mcp.py::test_the_tool_returns_every_record_as_authored`: the tool
  mutated to pass `kinds=["bug"]` through regardless of its argument: the non-bug records are
  missing and the equality assertion fails.
- M3 `...::test_the_tool_is_served_on_lite`: `recall_records` removed from `ALL_TOOL_NAMES`.
- M4 `...::test_a_malformed_query_is_refused[bad0]`: the "anchor query needs `repo`" check
  removed from `lookup_records`: the query is answered instead of refused.
- M2 `...::test_over_200_anchors_is_refused_with_the_reason`: the `except RecordsQueryError`
  clause removed, so the plain `ValueError` reaches the SDK: the reason no longer reaches the
  caller in a `ToolError`, and the assertion on `ToolError` and "at most 200" fails.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from recall.index import Indexer
from recall.lite import LiteStore
from recall_mcp import server as srv
from recall_mcp.settings import bootstrap_settings
from recall_mcp.tool_surface import ALL_TOOL_NAMES
from tests.audit_memory_support import EXPECTED, MEMOS, REPO, memo_name
from tests.test_lite_calibration import ContentWordEmbedder
from tests.test_lite_mcp import _serve, lite_env  # noqa: F401  # the fixture is used by name


def _index_golden(env: dict[str, str]) -> None:
    root = Path(env["RECALL_INDEX_ROOT"])
    folder = root / "memos"
    folder.mkdir()
    for memo in MEMOS.glob("*.md"):
        shutil.copyfile(memo, folder / memo.name)
    # The embedder the `lite_env` server opens the file with, so the file's width is its width.
    embedder = ContentWordEmbedder()
    store = LiteStore.from_dsn(env["RECALL_DSN"], dim=embedder.dim)
    try:
        Indexer(store, embedder, allow_prune=True, env={}).index_path(folder, key_root=root)
    finally:
        store.close()


def test_the_tool_is_served_on_lite(lite_env: dict[str, str]) -> None:  # noqa: F811  # the imported fixture
    """M3."""
    assert "recall_records" in ALL_TOOL_NAMES
    tools = {t.name for t in srv.build_server(bootstrap_settings(lite_env))._tool_manager.list_tools()}
    assert "recall_records" in tools


def test_the_tool_returns_every_record_as_authored(lite_env: dict[str, str]) -> None:  # noqa: F811  # the imported fixture
    """M1."""
    _index_golden(lite_env)
    ids = [record["id"] for record in EXPECTED.values()]

    async def body(_state: dict[str, Any], call: Any) -> str:
        return await call("recall_records", ids=ids)

    envelope = json.loads(_serve(lite_env, body))
    assert set(envelope) == {"tenant_id", "generation_id", "truncated", "records"}
    got = {memo_name(r["source"]): r["recall_record"] for r in envelope["records"]}
    assert got == EXPECTED

    async def anchored(_state: dict[str, Any], call: Any) -> str:
        anchor = EXPECTED["RE-call__0123456789ab__BUG-001.md"]["anchors"][0]
        return await call("recall_records", repo=REPO, anchors=[anchor], kinds=["false_positive"])

    by_anchor = json.loads(_serve(lite_env, anchored))
    assert [(r["kind"], r["anchor_match"]) for r in by_anchor["records"]] == [("false_positive", "exact")]


def test_over_200_anchors_is_refused_with_the_reason(lite_env: dict[str, str]) -> None:  # noqa: F811  # the imported fixture
    """M2."""
    anchors = [{"file": f"f{i}.py", "symbol": None, "code_sha": None} for i in range(201)]

    async def body(_state: dict[str, Any], call: Any) -> str:
        return await call("recall_records", repo=REPO, anchors=anchors)

    out = _serve(lite_env, body)
    assert out.startswith("RAISED ToolError") and "at most 200" in out, out


@pytest.mark.parametrize("bad", [{"anchors": [{"file": "a.py"}]}, {"ids": ["x"], "anchors": []}])
def test_a_malformed_query_is_refused(lite_env: dict[str, str], bad: dict[str, Any]) -> None:  # noqa: F811  # the imported fixture
    async def body(_state: dict[str, Any], call: Any) -> str:
        return await call("recall_records", **bad)

    out = _serve(lite_env, body)
    assert out.startswith("RAISED ToolError") and "records_query_refused" in out, out
