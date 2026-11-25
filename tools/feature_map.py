"""Generate docs/FEATURE_MAP.md from docs/feature_map.toml, and refuse a map that has drifted.

The map tells a reader, human or agent, where each feature lives: its entry points (console
scripts, ``recall`` commands, MCP tools), its modules, tests and docs. Its value depends on being
true, so this tool checks it against the code rather than trusting it:

* every console script in ``pyproject.toml``, every ``recall`` command in
  ``recall.cli._COMMAND_REGISTRATIONS`` and every MCP tool in
  ``recall_mcp.tool_surface.ALL_TOOL_NAMES`` is named by some feature;
* every tracked module of a package in ``PACKAGES`` is matched by some feature's ``modules``;
* every name a feature lists exists, and every glob matches at least one tracked file.

Commands and tools are read from the source with ``ast``, never imported, so this runs with the
standard library alone. The page renders the globs as written, so a new module under an existing
glob needs no regeneration; only a change to the TOML, or a surface no feature claims, does.

    python tools/feature_map.py                 # check, then write docs/FEATURE_MAP.md
    python tools/feature_map.py --check         # check, and compare with the committed page
    python tools/feature_map.py --which PATH    # the features a file belongs to
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "docs/feature_map.toml"
OUTPUT = "docs/FEATURE_MAP.md"
PACKAGES = (
    "recall", "recall_mcp", "recall_agent", "recall_hooks", "recall_interop", "recall_consistency",
    "recall_aml",
)
KEYS = ("scripts", "cli", "mcp", "modules", "tests", "docs")
REQUIRED = ("id", "area", "name", "summary", "modules")


class MapError(Exception):
    pass


def tracked(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [os.fsdecode(name) for name in result.stdout.split(b"\0") if name]


def _strings(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _assigned(path: Path, name: str) -> ast.AST:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if node.value is not None and any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value
    raise MapError(f"{path.as_posix()}: no top-level {name}")


def cli_commands(root: Path) -> set[str]:
    registrations = _assigned(root / "recall" / "cli.py", "_COMMAND_REGISTRATIONS")
    commands: set[str] = set()
    for entry in getattr(registrations, "elts", []):
        if isinstance(entry, ast.Tuple) and entry.elts:
            commands |= _strings(entry.elts[0])
    return commands


def mcp_tools(root: Path) -> set[str]:
    return _strings(_assigned(root / "recall_mcp" / "tool_surface.py", "ALL_TOOL_NAMES"))


def console_scripts(root: Path) -> set[str]:
    return set(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"])


def load(root: Path) -> list[dict[str, Any]]:
    features = tomllib.loads((root / SOURCE).read_text(encoding="utf-8")).get("feature", [])
    if not isinstance(features, list) or not features:
        raise MapError(f"{SOURCE}: no [[feature]] entries")
    return features


def problems(features: list[dict[str, Any]], root: Path, files: list[str]) -> list[str]:
    """Everything that makes the map untrue, as one line each."""
    out: list[str] = []
    surfaces = {"scripts": console_scripts(root), "cli": cli_commands(root), "mcp": mcp_tools(root)}
    claimed: dict[str, set[str]] = {key: set() for key in surfaces}
    modules = [f for f in files if f.endswith(".py") and f.split("/", 1)[0] in PACKAGES]
    covered: set[str] = set()
    seen: set[str] = set()
    for feature in features:
        fid = str(feature.get("id", "?"))
        out += [f"{fid}: missing `{key}`" for key in REQUIRED if not feature.get(key)]
        out += [f"{fid}: unknown key `{key}`" for key in feature if key not in REQUIRED + KEYS]
        if fid in seen:
            out.append(f"{fid}: duplicate id")
        seen.add(fid)
        for key, known in surfaces.items():
            for name in feature.get(key, []):
                claimed[key].add(name)
                if name not in known:
                    out.append(f"{fid}: {key} `{name}` does not exist")
        for key in ("modules", "tests", "docs"):
            for pattern in feature.get(key, []):
                matched = fnmatch.filter(files, pattern)
                if not matched:
                    out.append(f"{fid}: {key} `{pattern}` matches no tracked file")
                if key == "modules":
                    covered.update(matched)
    for key, known in surfaces.items():
        out += [f"{key} `{name}` is in no feature" for name in sorted(known - claimed[key])]
    out += [f"module `{path}` is in no feature" for path in sorted(set(modules) - covered)]
    return out


def _entry_points(feature: dict[str, Any]) -> list[str]:
    labels = {"scripts": "script", "cli": "recall", "mcp": "MCP"}
    return [f"`{name}`" if key == "mcp" else f"`{labels[key]} {name}`" if key == "cli" else f"`{name}` ({labels[key]})"
            for key in labels for name in feature.get(key, [])]


def _node(fid: str) -> str:
    return "f_" + fid.replace("-", "_")


def render(features: list[dict[str, Any]]) -> str:
    areas: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        areas.setdefault(str(feature["area"]), []).append(feature)
    lines = [
        "# Feature map",
        "",
        "Generated by `python tools/feature_map.py` from [`feature_map.toml`](feature_map.toml); edit that",
        "file, never this one. CI refuses a command, MCP tool, console script or module that no feature",
        "claims, a path that does not exist, and a page that differs from its source.",
        "",
        "To find where something lives: look its entry point up in the index below, or run",
        "`python tools/feature_map.py --which <path>` for the features a file belongs to. A `*` in a",
        "path matches across directories. For which module imports which, see",
        "[`ARCHITECTURE_MAP_GENERATED.md`](ARCHITECTURE_MAP_GENERATED.md).",
        "",
        "```mermaid",
        "flowchart LR",
        '  s_cli["recall CLI"]',
        '  s_mcp["MCP server"]',
        '  s_scripts["other console scripts"]',
    ]
    for number, (area, members) in enumerate(areas.items(), start=1):
        lines.append(f'  subgraph a_{number}["{area}"]')
        # Mermaid reads a backtick inside a label as the start of a Markdown string.
        lines += [f'    {_node(str(f["id"]))}["{str(f["name"]).replace("`", "")}"]' for f in members]
        lines.append("  end")
    for feature in features:
        for key, node in (("cli", "s_cli"), ("mcp", "s_mcp")):
            if feature.get(key):
                lines.append(f"  {node} --> {_node(str(feature['id']))}")
        if any(s != "recall" for s in feature.get("scripts", [])):
            lines.append(f"  s_scripts --> {_node(str(feature['id']))}")
    lines += ["```", "", "## Index", "", "| Feature | Area | Entry points |", "|---|---|---|"]
    lines += [
        f"| [{f['name']}](#{f['id']}) | {f['area']} | {', '.join(_entry_points(f)) or 'library only'} |"
        for f in features
    ]
    for feature in features:
        lines += ["", f'<a id="{feature["id"]}"></a>', f"## {feature['name']}", "", str(feature["summary"]), ""]
        if entry := _entry_points(feature):
            lines.append(f"- **Entry points:** {', '.join(entry)}")
        for key, label in (("modules", "Modules"), ("tests", "Tests"), ("docs", "Docs")):
            if values := feature.get(key):
                lines.append(f"- **{label}:** " + ", ".join(f"`{v}`" for v in values))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify the map and the committed page")
    group.add_argument("--which", metavar="PATH", help="print the features whose globs match PATH")
    args = parser.parse_args(argv)
    try:
        features = load(root)
        if args.which:
            path = Path(args.which).as_posix()
            hits = [f for f in features if any(fnmatch.fnmatchcase(path, p) for k in ("modules", "tests", "docs") for p in f.get(k, []))]
            for f in hits:
                print(f"{f['id']}: {f['name']}")
            return 0 if hits else 1
        found = problems(features, root, tracked(root))
    except MapError as error:
        print(error, file=sys.stderr)
        return 2
    if found:
        print("\n".join(found), file=sys.stderr)
        print(f"\n{len(found)} problem(s) in {SOURCE}. Fix it, then run `python tools/feature_map.py`.", file=sys.stderr)
        return 1
    rendered = render(features)
    output = root / OUTPUT
    if args.check:
        committed = output.read_text(encoding="utf-8") if output.is_file() else None
        if committed != rendered:
            print(f"{OUTPUT} is stale: regenerate with `python tools/feature_map.py`", file=sys.stderr)
            return 1
        print(f"{OUTPUT} is current: {len(features)} features")
        return 0
    output.write_bytes(rendered.encode("utf-8"))
    print(f"wrote {OUTPUT}: {len(features)} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
