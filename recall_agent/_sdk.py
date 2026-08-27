"""The only module that imports `claude_agent_sdk`.

Everything the SDK's pre-1.0 churn can break is confined here: the tool decorator, the in-process
server constructor, `HookMatcher`, and `ClaudeAgentOptions` assembly. `recall_agent.memory` holds
the behaviour (plain async callables) and stays importable and testable without the extra.
"""
from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any

from recall_agent._descriptions import (
    RECALL_EVIDENCE_DESCRIPTION,
    RECALL_FORGET_DESCRIPTION,
    RECALL_INDEX_DESCRIPTION,
    RECALL_SEARCH_DESCRIPTION,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, McpSdkServerConfig

    from recall_agent.memory import RecallAgentMemory

_RELATED_PROPERTIES: dict[str, Any] = {
    "explain": {
        "type": "boolean",
        "description": "Include the optional machine readable retrieval explanation.",
    },
    "include_related": {
        "type": "boolean",
        "description": "Opt into independently trusted related evidence expansion.",
    },
    "related_relation": {
        "type": "string",
        "enum": ["source", "ordinal", "supersession"],
        "description": "Relation to expand along when expansion is on.",
    },
    "related_max_items": {
        "type": "integer",
        "description": "Maximum related candidates, bounded by the serving contract.",
    },
}

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to recall (natural language)."},
        "source": {
            "type": "string",
            "description": "Optional source filter (only search one file/source).",
        },
        "k": {"type": "integer", "description": "Max hits to return (default 5)."},
        **_RELATED_PROPERTIES,
    },
    "required": ["query"],
}

EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to recall (natural language)."},
        "source": {
            "type": "string",
            "description": "Optional source filter (only search one file/source).",
        },
        "k": {"type": "integer", "description": "Max hits to retrieve (default 5)."},
        "max_items": {
            "type": "integer",
            "description": "Max passages admitted to the bundle; clamped to the effective k.",
        },
        **_RELATED_PROPERTIES,
    },
    "required": ["query"],
}

#: ⛔ No `glob`, deliberately, and this is a security boundary rather than a parity nicety.
#:
#: `recall.index._safe_default_file` returns True unconditionally for any glob other than the
#: default, which switches off the exclusion list keeping `tokens.json`, `credentials.json` and
#: `secrets.json` out of a searchable corpus, and `candidate_files` drops the document-extension
#: gate and the dot-directory prune with it. The MCP server's `recall_index` takes `path` alone
#: for that reason, so a client can never widen what the corpus admits.
#:
#: This schema shipped a `glob` and an audit caught it: the same `tokens.json` is refused under
#: the default and admitted under a model-chosen glob, then read back verbatim through
#: `recall_search`. An operator choosing `recall index --glob` is a human scoping a scan; a tool
#: argument is reachable by any text that reaches the model. A host that genuinely needs a
#: different glob should take it as a constructor argument, where the HOST chooses it.
#: Pinned by `tests/test_recall_agent_tool_surface.py`.
INDEX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Markdown file or directory to index, inside RECALL_INDEX_ROOT.",
        },
    },
    "required": ["path"],
}

FORGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Sources to erase permanently.",
        },
    },
    "required": ["sources"],
}


def _import_sdk() -> Any:
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise ImportError(
            "recall_agent needs the Claude Agent SDK for this call: "
            'pip install "recall-rag[agent]"'
        ) from exc
    return claude_agent_sdk


def _package_version() -> str:
    try:
        return metadata.version("recall-rag")
    except metadata.PackageNotFoundError:
        return "0"


def build_sdk_mcp_server(
    memory: "RecallAgentMemory", *, write_tools: bool = False
) -> "McpSdkServerConfig":
    sdk = _import_sdk()
    specs = [
        ("recall_search", RECALL_SEARCH_DESCRIPTION, SEARCH_SCHEMA, memory._recall_search),
        ("recall_evidence", RECALL_EVIDENCE_DESCRIPTION, EVIDENCE_SCHEMA, memory._recall_evidence),
    ]
    if write_tools:
        specs += [
            ("recall_index", RECALL_INDEX_DESCRIPTION, INDEX_SCHEMA, memory._recall_index),
            ("recall_forget", RECALL_FORGET_DESCRIPTION, FORGET_SCHEMA, memory._recall_forget),
        ]
    tools = [
        sdk.tool(name, description, schema)(handler)
        for name, description, schema, handler in specs
    ]
    return sdk.create_sdk_mcp_server(
        name=memory.server_name, version=_package_version(), tools=tools
    )


def build_session_start_matcher(memory: "RecallAgentMemory") -> "HookMatcher":
    sdk = _import_sdk()
    return sdk.HookMatcher(hooks=[memory._session_start])


def build_options(
    memory: "RecallAgentMemory", *, write_tools: bool, overrides: dict[str, Any]
) -> "ClaudeAgentOptions":
    sdk = _import_sdk()
    overrides = dict(overrides)

    mcp_servers: dict[str, Any] = {
        memory.server_name: build_sdk_mcp_server(memory, write_tools=write_tools)
    }
    extra_servers = dict(overrides.pop("mcp_servers", {}) or {})
    if memory.server_name in extra_servers:
        raise ValueError(
            f"mcp_servers override collides with this memory's own server name "
            f"{memory.server_name!r}; pick a different server_name at construction instead"
        )
    mcp_servers.update(extra_servers)

    allowed_tools = [
        *memory.allowed_tools(write_tools=write_tools),
        *(overrides.pop("allowed_tools", []) or []),
    ]

    hooks: dict[str, list[Any]] = {"SessionStart": [build_session_start_matcher(memory)]}
    for event, matchers in dict(overrides.pop("hooks", {}) or {}).items():
        hooks.setdefault(event, []).extend(matchers)

    return sdk.ClaudeAgentOptions(
        mcp_servers=mcp_servers, allowed_tools=allowed_tools, hooks=hooks, **overrides
    )
