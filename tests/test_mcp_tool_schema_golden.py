"""The MCP tool surface a client sees is pinned, byte for byte, in a golden file.

Invariant: `build_server()` publishes exactly the tools in `tests/fixtures/mcp_tool_schemas.json`,
each with the same name, title, description, input schema, output schema and annotations. That is
the contract every MCP client codes against, and nothing else in `tests/` pinned it whole: the
surface tests pin the NAMES, the authorization tests the annotations' scopes.

Failure mode caught: moving or rewriting tool registration (the server split this file was written
for) changes a parameter's name, type, default or required flag, a description, or an annotation,
and every in-process test still passes because each calls the tool function directly.

To change the surface on purpose, regenerate the file and review its diff like any API change:

    RECALL_REGENERATE_TOOL_GOLDEN=1 python -m pytest tests/test_mcp_tool_schema_golden.py

Red proof, recorded 2026-10-02 on a Linux test host against `recall_mcp/server.py` at `94ff1bb5`,
one mutation per run on a copy of the tree, each failing in the per-tool assertion ("tool ... no
longer matches its golden schema"):

- `recall_search`'s `k: int = 5` default changed to 6 (an input schema default);
- `recall_stats`'s docstring first line shortened (a description);
- `recall_stats`'s `idempotent_hint=True` flipped to False (an annotation).

Unmodified, it passes. The golden file was generated from that same commit.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from recall_mcp.server import build_server
from recall_mcp.tool_surface import TOOL_SURFACE_ENV

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "mcp_tool_schemas.json"
REGENERATE = "RECALL_REGENERATE_TOOL_GOLDEN"


def _published_tools() -> list[dict[str, Any]]:
    """What `tools/list` returns to a client, as plain JSON, sorted by name."""
    tools = asyncio.run(build_server().list_tools())
    return sorted(
        (tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools),
        key=lambda tool: tool["name"],
    )


def _render(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_the_published_tool_surface_matches_the_golden_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOOL_SURFACE_ENV, raising=False)
    rendered = _render(_published_tools())
    if os.environ.get(REGENERATE) == "1":
        GOLDEN.write_text(rendered, encoding="utf-8", newline="\n")
        pytest.skip(f"regenerated {GOLDEN.name}; review its diff before committing")
    golden = GOLDEN.read_text(encoding="utf-8")
    published = json.loads(rendered)
    expected = json.loads(golden)
    assert [tool["name"] for tool in published] == [tool["name"] for tool in expected], (
        "the set of published tools changed"
    )
    for got, want in zip(published, expected, strict=True):
        assert got == want, f"tool {want['name']!r} no longer matches its golden schema"
    assert rendered == golden, "the golden file is not in canonical form; regenerate it"
