"""Serving fewer tools, and the ways that could go wrong quietly.

Prior work: `tests/test_mcp_tool_authorization.py` covers who may CALL a tool, which is a
different question and stays the authorisation boundary; this file covers which tools are
OFFERED, which is a context-cost decision. The measured cost that motivates it is recorded in
`docs/preregistrations/2026-08-27-tool-definition-context-cost.md`. Nothing in `tests/` pinned
the served tool list before.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from recall_mcp.tool_surface import (
    ALL_TOOL_NAMES,
    TOOL_PRESETS,
    TOOL_SURFACE_ENV,
    FilteredToolRegistrar,
    ToolSurfaceError,
    resolve_tool_surface,
)

SERVER = Path(__file__).resolve().parents[1] / "recall_mcp" / "server.py"


def _tools_declared_in_server() -> set[str]:
    """Every `@mcp.tool(name=...)` in the server, read from source.

    From source rather than by building a server, because building one needs auth, a database and
    a lifespan; the question here is only which names exist.
    """
    return set(re.findall(r'@mcp\.tool\(\s*name="([^"]+)"', SERVER.read_text(encoding="utf-8")))


def test_the_known_tool_list_matches_the_server() -> None:
    """A tool added to the server without a line here would be silently unselectable.

    `resolve_tool_surface` refuses an unknown name, so a tool missing from `ALL_TOOL_NAMES` could
    not be requested by an operator at all: the refusal that protects against typos would fire on
    a real tool. This is the test that keeps the constant honest.
    """
    assert _tools_declared_in_server() == set(ALL_TOOL_NAMES)


def test_unset_serves_every_tool() -> None:
    """Introducing this feature must not change any existing deployment."""
    assert resolve_tool_surface({}) == ALL_TOOL_NAMES
    assert resolve_tool_surface({TOOL_SURFACE_ENV: ""}) == ALL_TOOL_NAMES
    assert resolve_tool_surface({TOOL_SURFACE_ENV: "   "}) == ALL_TOOL_NAMES


def test_presets_and_explicit_names_compose() -> None:
    assert resolve_tool_surface({TOOL_SURFACE_ENV: "search"}) == TOOL_PRESETS["search"]
    assert resolve_tool_surface({TOOL_SURFACE_ENV: "recall_search"}) == {"recall_search"}
    # Commas or spaces, and a preset plus an extra tool.
    assert resolve_tool_surface({TOOL_SURFACE_ENV: "search, recall_stats"}) == (
        TOOL_PRESETS["search"] | {"recall_stats"}
    )
    assert resolve_tool_surface({TOOL_SURFACE_ENV: "recall_search recall_forget"}) == {
        "recall_search",
        "recall_forget",
    }


def test_a_typo_refuses_rather_than_serving_a_smaller_surface() -> None:
    """The failure this exists to prevent: a server that starts and is quietly missing a tool.

    An operator who writes `recall_serch` and gets a running server with no search tool cannot
    tell that from an agent that chose not to search.
    """
    with pytest.raises(ToolSurfaceError, match="recall_serch"):
        resolve_tool_surface({TOOL_SURFACE_ENV: "recall_serch"})
    with pytest.raises(ToolSurfaceError) as excinfo:
        resolve_tool_surface({TOOL_SURFACE_ENV: "search, recall_reasoning"})
    # The message has to name what IS valid, or the operator is left guessing.
    assert "recall_reasoning_query" in str(excinfo.value)


def test_resolving_to_nothing_refuses() -> None:
    with pytest.raises(ToolSurfaceError, match="no tools at all"):
        resolve_tool_surface({TOOL_SURFACE_ENV: ","})


def test_every_preset_names_only_real_tools() -> None:
    for preset, names in TOOL_PRESETS.items():
        assert names, f"preset {preset!r} is empty"
        assert names <= ALL_TOOL_NAMES, f"preset {preset!r} names a tool that does not exist"


class _FakeMCP:
    def __init__(self) -> None:
        self.registered: list[str] = []
        self.other_attribute_reached = False

    def tool(self, **kwargs):
        self.registered.append(kwargs["name"])

        def deco(fn):
            return fn

        return deco

    def something_else(self) -> str:
        self.other_attribute_reached = True
        return "passed through"


def test_the_registrar_registers_only_the_served_tools() -> None:
    fake = _FakeMCP()
    registrar = FilteredToolRegistrar(fake, frozenset({"recall_search"}))

    @registrar.tool(name="recall_search")
    def served():
        return "served"

    @registrar.tool(name="recall_forget")
    def skipped():
        return "skipped"

    assert fake.registered == ["recall_search"]
    assert registrar.registered == ["recall_search"]
    assert registrar.skipped == ["recall_forget"]
    # A skipped tool's function is still returned unchanged, so the module imports and the
    # function stays callable; it simply never reaches the server's registry.
    assert skipped() == "skipped"
    assert served() == "served"


def test_the_registrar_passes_everything_else_through() -> None:
    """It wraps one method; the rest of the server object must be untouched."""
    fake = _FakeMCP()
    registrar = FilteredToolRegistrar(fake, ALL_TOOL_NAMES)
    assert registrar.something_else() == "passed through"
    assert fake.other_attribute_reached is True


def test_narrowing_the_surface_is_not_an_authorisation_boundary() -> None:
    """Pinned as documentation, because the tempting misuse is to treat it as one.

    Scopes decide who may call a tool. This decides what is offered. A deployment that serves
    `recall_forget` to save nobody any context has not thereby authorised anyone to call it, and
    a deployment that hides it has not thereby secured anything.
    """
    module = (
        Path(__file__).resolve().parents[1] / "recall_mcp" / "tool_surface.py"
    ).read_text(encoding="utf-8")
    assert "not an authorisation boundary" in module
