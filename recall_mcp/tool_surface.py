"""Which of the server's tools a deployment actually serves.

Every tool definition is injected into a session's context and re-sent on every turn, whether or
not the tool is ever called. Measured 2026-08-27 on `anthropic/claude-haiku-4.5` through Claude
Code, on a one-turn session that called nothing: **about 153 input tokens per tool per turn**,
with a fixed ~270 for serving any tools at all. Serving all 18 costs 5,727 input tokens where
serving two costs 3,324.

That is not an abstract cost. Across 112 agent sessions with memory available, measured in the
same work (`docs/preregistrations/2026-08-27-tool-definition-context-cost.md`), the agents called
exactly ONE tool, `recall_search`, 139 times; the other 17 were never invoked. At the rate above,
over a 15-turn session, those unused definitions cost roughly 39,000 input tokens per session.

The same measurement showed the TRANSPORT is worth about 24 tokens per tool, which is to say
nothing. So the lever is the tool list, not stdio versus in-process versus HTTP.

⚠️ This narrows what a server OFFERS. It is not an authorisation boundary and must never be used
as one: scopes decide who may call a tool (`recall_mcp.scopes`), and a caller who should not be
able to erase memory is stopped by not holding `recall:forget`, not by the tool being absent from
this list. Narrowing here saves context; it does not grant safety.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from recall.errors import RecallError

#: Every tool the server can register, as the single source of truth for validation. Pinned
#: against the live registry by `tests/test_tool_surface.py`, so a tool added to `server.py`
#: without a line here fails a test rather than becoming quietly unselectable.
ALL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "recall_search",
        "recall_evidence",
        "recall_related",
        "recall_current_state",
        "recall_current_facts",
        "recall_apply_fact",
        "recall_reasoning_query",
        # Added when #516 landed. Absent from this list the tool is not merely
        # unselectable, it is never registered at all, so every per-tool test in
        # tests/test_mcp_tool_authorization.py fails with a KeyError rather than an
        # authorization message. It is deliberately NOT in the `read` preset: that preset
        # is a curated minimum, and `all` is the default, which is what serves it today.
        "recall_query_construction_challenge",
        "recall_reasoning_projection",
        "recall_reasoning_proposals",
        "recall_rewrite_plan",
        "recall_reasoning_audit",
        "recall_index",
        "recall_tenants",
        "recall_ingest",
        "recall_job_status",
        "recall_calibration_status",
        "recall_calibration_run",
        "recall_calibration_publish",
        "recall_forget",
        "recall_inventory",
        "recall_stats",
    }
)

#: Named surfaces, so the common cases need no list. `search` is the surface the measured agent
#: sessions actually used; `all` is the default and preserves the historical behaviour.
TOOL_PRESETS: Mapping[str, frozenset[str]] = {
    "all": ALL_TOOL_NAMES,
    "search": frozenset({"recall_search", "recall_evidence"}),
    "read": frozenset(
        {
            "recall_search",
            "recall_evidence",
            "recall_related",
            "recall_current_state",
            "recall_stats",
        }
    ),
}

#: The variable a deployment sets. Unset means every tool, so this cannot change an existing
#: install's behaviour by being introduced.
TOOL_SURFACE_ENV = "RECALL_MCP_TOOLS"


class ToolRegistrar(Protocol):
    """What the `_register_*_tools` functions actually need from a server.

    They require exactly one attribute, `tool`, which is why a wrapper can stand in for the
    server without impersonating the rest of its interface. Typing them against this rather than
    against `MCPServer` is what lets `FilteredToolRegistrar` be passed without a cast.

    One OPTIONAL attribute is also read, via `getattr` rather than declared here: `serves(name)`,
    which lets a tool tailor its OUTPUT to the surface (see `FilteredToolRegistrar.serves`). It
    stays off this Protocol on purpose, so a plain `MCPServer` remains a valid registrar; a
    registrar without it means nothing was filtered, i.e. every tool is served.
    """

    def tool(self, *args: Any, **kwargs: Any) -> Callable[[Any], Any]: ...


class ToolSurfaceError(ValueError, RecallError):
    """Raised when the configured tool surface cannot be honoured exactly as written.

    `ValueError` stays the FIRST base, per the convention in `recall.errors`: a caller already
    catching it keeps working, and `except RecallError` starts working too.
    """


def resolve_tool_surface(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Resolve `RECALL_MCP_TOOLS` to the exact set of tools to register.

    Unset or empty resolves to every tool, which is the historical behaviour: introducing this
    module changes nothing for a deployment that does not opt in.

    A typo REFUSES rather than serving a smaller surface. Silently dropping an unrecognised name
    is the worst available failure here: the server would start, the tool would be missing, and
    the agent would simply never use a capability the operator believed was configured, which is
    indistinguishable from the agent choosing not to.
    """
    values = os.environ if env is None else env
    raw = values.get(TOOL_SURFACE_ENV, "").strip()
    if not raw:
        return ALL_TOOL_NAMES

    requested: set[str] = set()
    unknown: list[str] = []
    for part in raw.replace(",", " ").split():
        name = part.strip()
        if not name:
            continue
        if name in TOOL_PRESETS:
            requested |= TOOL_PRESETS[name]
        elif name in ALL_TOOL_NAMES:
            requested.add(name)
        else:
            unknown.append(name)

    if unknown:
        raise ToolSurfaceError(
            f"{TOOL_SURFACE_ENV} names {sorted(unknown)}, which are neither tools nor presets. "
            f"Presets: {sorted(TOOL_PRESETS)}. Tools: {sorted(ALL_TOOL_NAMES)}"
        )
    if not requested:
        raise ToolSurfaceError(
            f"{TOOL_SURFACE_ENV} resolved to no tools at all. Unset it to serve every tool; a "
            f"server with no tools can do nothing and is never what an operator meant."
        )
    return frozenset(requested)


class FilteredToolRegistrar:
    """Wraps the MCP server so `@mcp.tool(...)` registers only the selected tools.

    A wrapper rather than a condition at each of the eighteen registration sites: the tools are
    declared once, with their docstrings and annotations, and this decides which declarations
    reach the server. A new tool is therefore filtered correctly without its author knowing this
    file exists.
    """

    def __init__(self, mcp: ToolRegistrar, served: frozenset[str]) -> None:
        self._mcp = mcp
        self._served = served
        self.registered: list[str] = []
        self.skipped: list[str] = []

    def serves(self, name: str) -> bool:
        """Whether `name` reaches the server, asked at registration time.

        Exists so a tool can tailor its OUTPUT to the surface, not just its registration: search
        advice points at `recall_reasoning_query`, and naming a tool the deployment does not
        serve is worse than saying nothing, because the agent spends a turn discovering it is
        absent. A plain unfiltered server has no `serves`, and callers read that as "everything",
        which is correct: without this wrapper every tool is registered.
        """
        return name in self._served

    def tool(self, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        name = kwargs.get("name")
        if isinstance(name, str) and name not in self._served:
            self.skipped.append(name)

            def _do_not_register(fn: Any) -> Any:
                # Returned unchanged so the module still imports and the function stays callable
                # in tests; it simply never reaches the server's tool registry.
                return fn

            return _do_not_register
        if isinstance(name, str):
            self.registered.append(name)
        return self._mcp.tool(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        # `__getattr__` runs only for attributes normal lookup missed, so reaching it for `_mcp`
        # itself means `__init__` never completed. Deferring to `self._mcp` there would recurse
        # forever and surface as a RecursionError naming this line rather than the real fault.
        if item == "_mcp":
            raise AttributeError(
                "FilteredToolRegistrar._mcp is unset; __init__ did not complete"
            )
        return getattr(self._mcp, item)
