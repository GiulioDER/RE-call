"""docs/API.md must name exactly the registered CLI commands and MCP tools.

The page drifted in both directions before this existed: 10 of 16 MCP tools documented, and
the CLI table missing the command the README leads with (`quickstart`). The same failure mode
as a stale deployed hook, and the same fix shape as the hook-drift assertions in scripts/:
diff the document against the registration, so a surface cannot ship undocumented or linger
in the supported list after removal.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import recall_mcp.server as server_module
from recall.cli import build_parser

API_MD = Path(__file__).resolve().parent.parent / "docs" / "API.md"

#: Cell parser tolerant of trailing HTML-comment markers, the same shape
#: tests/test_h2h_artifact_backs_findings.py uses for its tables.
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")


def _table_commands(section: str) -> list[str]:
    """First-column backticked names from the table under `## <section>`."""
    text = API_MD.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(section)}$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"docs/API.md has no '## {section}' section"
    return [m.group(1) for line in match.group(1).splitlines() if (m := _ROW.match(line))]


def _registered_mcp_tools() -> list[str]:
    """Tool names in registration (tools/list) order, from the module AST.

    AST rather than build_server(): enumerating must not require auth config or a database,
    and the decorator's `name=` constant is the registration.
    """
    tree = ast.parse(Path(server_module.__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "attr", None) == "tool":
                    names.extend(
                        k.value.value
                        for k in dec.keywords
                        if k.arg == "name"
                        and isinstance(k.value, ast.Constant)
                        and isinstance(k.value.value, str)
                    )
    return names


def _registered_cli_commands() -> list[str]:
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return list(sub.choices.keys())


def test_the_mcp_table_names_exactly_the_registered_tools() -> None:
    documented = [name for name in _table_commands("MCP") if name.startswith("recall_")]
    registered = _registered_mcp_tools()
    missing = sorted(set(registered) - set(documented))
    stale = sorted(set(documented) - set(registered))
    assert not missing and not stale, (
        f"docs/API.md's MCP table is out of date. Undocumented tools: {missing or 'none'}; "
        f"documented but not registered: {stale or 'none'}. Add or remove the rows."
    )
    assert documented == registered, (
        "docs/API.md's MCP table is complete but out of order; it documents tools/list order, "
        f"which is {registered}"
    )


def test_the_cli_table_names_exactly_the_registered_commands() -> None:
    documented = [
        name.removeprefix("recall ")
        for name in _table_commands("Command Line")
        if name.startswith("recall ")  # `recall-enterprise` is its own binary, checked below
    ]
    registered = _registered_cli_commands()
    missing = sorted(set(registered) - set(documented))
    stale = sorted(set(documented) - set(registered))
    assert not missing and not stale, (
        f"docs/API.md's Command Line table is out of date. Undocumented commands: "
        f"{missing or 'none'}; documented but not registered: {stale or 'none'}."
    )


def test_the_enterprise_binary_stays_documented() -> None:
    assert "recall-enterprise" in _table_commands("Command Line"), (
        "docs/API.md dropped the recall-enterprise row; it is a shipped console script"
    )


def test_the_cli_table_names_every_nested_subcommand() -> None:
    """A grouped command documents its leaves in its Purpose cell, so `recall generation gc`
    cannot exist undocumented behind a documented `recall generation`."""
    text = API_MD.read_text(encoding="utf-8")
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    undocumented: list[str] = []
    for name, p in sub.choices.items():
        nested = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        if not nested:
            continue
        row = re.search(rf"^\|\s*`recall {re.escape(name)}`\s*\|(.*)\|$", text, re.M)
        assert row, f"no table row for grouped command `recall {name}`"
        undocumented += [
            f"recall {name} {leaf}" for leaf in nested[0].choices if f"`{leaf}`" not in row.group(1)
        ]
    assert not undocumented, (
        f"nested subcommands missing from their Purpose cells: {undocumented}"
    )
