"""`recall_mcp.server`'s code now spans several modules; patch it and read it as one.

The server split moved each tool family, the lifespan and their shared helpers out of
`recall_mcp/server.py` into `recall_mcp/server_<family>.py`. Each module looks a global up in its
own namespace, so `monkeypatch.setattr(server, "search_memory", fake)` would now reach only the
code left in `server.py` and silently miss the tool that calls it. `setattr_on_server` patches the
name in every module that holds what used to be `server.py`, which is exactly where the old patch
reached. `server_source` and `server_tree` are what the source-reading guards used to take from
`server.py` alone.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

#: Every module that holds code which lived in `recall_mcp/server.py` before the split.
SERVER_MODULES = (
    "recall_mcp.server",
    "recall_mcp.server_common",
    "recall_mcp.server_lifespan",
    "recall_mcp.server_search",
    "recall_mcp.server_facts",
    "recall_mcp.server_reasoning",
    "recall_mcp.server_ingest",
    "recall_mcp.server_calibration",
    "recall_mcp.server_memory_admin",
    "recall_mcp.server_assistant",
)

_ROOT = Path(__file__).resolve().parents[1]


def setattr_on_server(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    """Patch `name` wherever the server's code looks it up, as one patch of `server.py` did."""
    patched = [m for m in map(importlib.import_module, SERVER_MODULES) if hasattr(m, name)]
    assert patched, f"no server module binds {name!r}, so the patch would reach nothing"
    for module in patched:
        monkeypatch.setattr(module, name, value)


def server_source_paths() -> list[Path]:
    return [_ROOT / (name.replace(".", "/") + ".py") for name in SERVER_MODULES]


def server_source() -> str:
    """The server's source, every module in `SERVER_MODULES` order, `server.py` first."""
    return "\n".join(path.read_text(encoding="utf-8") for path in server_source_paths())


def server_tree() -> ast.Module:
    """The server's modules parsed one by one and joined into one tree, `server.py` first."""
    body = [node for path in server_source_paths() for node in ast.parse(path.read_text(encoding="utf-8")).body]
    return ast.Module(body=body, type_ignores=[])
