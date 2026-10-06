"""The memory graph behind the dashboard: every memo a node, every link an edge, from the files alone.

Filesystem only, like the review queue, so the picture is of the memos a person edits, not of a
database generation. Three kinds of edge, each from something a memo actually says:

* `supersedes`: the frontmatter declaration, resolved the way `recall rewrite` resolves it (a name
  that matches no memo, or more than one, draws nothing);
* `link`: a `[[wiki link]]` or a relative markdown link to another memo in the corpus;
* `pending`: a claim waiting in the review queue (an agent report or a cached arbiter proposal).

A node's state: `superseded` when a memo declares it replaced, `expired` when its `valid_until`
has passed, `pending` when a queued claim would supersede it, else `current`. Titles and
descriptions are read for display only and never decide anything.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall.dashboard.review import build_queue
from recall.document import parse_document
from recall.frontmatter import supersedes_key, supersedes_targets, validity_bounds
from recall.lint import lint_corpus

#: A memo linking to at least this many others is an index page. It is kept, but flagged so the
#: page can hide it: one hub joined to every memo pulls the whole picture into a single knot.
HUB_OUT_LINKS = 40
MAX_DESCRIPTION = 280

_WIKI = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
_MD_LINK = re.compile(r"\]\(([^)\s]+?\.md)(?:#[^)]*)?\)")
_HEADING = re.compile(r"^#\s+(.+?)\s*$", re.M)
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_DESCRIPTION = re.compile(r"^description:\s*(.+?)\s*$", re.M)
_MODIFIED = re.compile(r"^\s*modified:\s*['\"]?(\d{4}-\d{2}-\d{2})", re.M)
_FILE_DAY = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2})")
MAX_ISSUE_MESSAGE = 240


def _born(text: str, name: str, path: Path) -> str:
    """The day a memo is dated, for time travel: its `modified:` stamp, else a date in its file name,
    else the file's own modification day. Display and filtering only; nothing is decided by it."""
    front = _FRONT.match(text.replace("\r\n", "\n"))
    if front:
        stamp = _MODIFIED.search(front.group(1))
        if stamp:
            return stamp.group(1)
    day = _FILE_DAY.search(name)
    if day:
        return day.group(1)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date().isoformat()


def _memo_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def _display(text: str, stem: str) -> tuple[str, str]:
    """Title and description for display only: a top-level heading, else the file stem."""
    front = _FRONT.match(text)
    description = ""
    if front:
        found = _DESCRIPTION.search(front.group(1))
        if found:
            description = found.group(1).strip().strip("\"'")
    body = text[front.end() :] if front else text
    heading = _HEADING.search(body)
    title = heading.group(1).strip() if heading else stem
    return title[:160], description[:MAX_DESCRIPTION]


def build_graph(root: Path, *, today: datetime | None = None, include_queue: bool = True) -> dict[str, Any]:
    """Nodes, edges and counts for the page, as plain JSON-ready data."""
    root = root.resolve()
    now = today or datetime.now(UTC)
    files = _memo_files(root)
    names = [path.relative_to(root).as_posix() for path in files]

    by_key: dict[str, list[str]] = {}
    for name in names:
        by_key.setdefault(supersedes_key(Path(name).name), []).append(name)

    def resolve(reference: str) -> str | None:
        matches = by_key.get(supersedes_key(reference.strip()), [])
        return matches[0] if len(matches) == 1 else None

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_edge(source: str, target: str, kind: str) -> None:
        key = (source, target, kind)
        if source != target and key not in seen:
            seen.add(key)
            edges.append({"source": source, "target": target, "kind": kind})

    for path, name in zip(files, names, strict=True):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        meta = parse_document(text).meta
        title, description = _display(text, path.stem)
        expired = False
        start = end = None
        try:
            start, end = validity_bounds(meta)
            expired = end is not None and end < now
        except ValueError:
            pass
        nodes[name] = {
            "id": name,
            "title": title,
            "description": description,
            "type": str(meta.get("type") or ""),
            "state": "expired" if expired else "current",
            "born": _born(text, name, path),
            "valid_from": start.date().isoformat() if start else None,
            "valid_until": end.date().isoformat() if end else None,
            "folder": name.split("/", 1)[0] if "/" in name else "",
            "issues": [],
        }
        for target in supersedes_targets(meta.get("supersedes")):
            resolved = resolve(target)
            if resolved is not None:
                add_edge(name, resolved, "supersedes")
        for match in _WIKI.finditer(text):
            resolved = resolve(match.group(1))
            if resolved is not None:
                add_edge(name, resolved, "link")
        for match in _MD_LINK.finditer(text):
            candidate = (path.parent / match.group(1)).resolve()
            if candidate.is_relative_to(root):
                relative = candidate.relative_to(root).as_posix()
                if relative in nodes or relative in names:
                    add_edge(name, relative, "link")

    queue_notes: tuple[str, ...] = ()
    pending: list[dict[str, str]] = []
    if include_queue:
        queue = build_queue(root)
        queue_notes = queue.notes
        for item in queue.items:
            if item.stale in nodes and item.replacing in nodes:
                add_edge(item.replacing, item.stale, "pending")
                pending.append({"stale": item.stale, "replacing": item.replacing, "claim": item.claim})

    for edge in edges:
        if edge["kind"] == "supersedes":
            nodes[edge["target"]]["state"] = "superseded"
    for claim in pending:
        if nodes[claim["stale"]]["state"] == "current":
            nodes[claim["stale"]]["state"] = "pending"
            nodes[claim["stale"]]["claim"] = claim["claim"]

    out_links: dict[str, int] = {}
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1
        if edge["kind"] == "link":
            out_links[edge["source"]] = out_links.get(edge["source"], 0) + 1
    for name, node in nodes.items():
        node["degree"] = degree.get(name, 0)
        node["hub"] = out_links.get(name, 0) >= HUB_OUT_LINKS
    # Isolated: nothing but index pages connects to it. A memo only an index lists is one nobody
    # links to while working, which is the orphan worth showing.
    connected: set[str] = set()
    for edge in edges:
        if not nodes[edge["source"]]["hub"] and not nodes[edge["target"]]["hub"]:
            connected.update((edge["source"], edge["target"]))
    for name, node in nodes.items():
        node["isolated"] = not node["hub"] and name not in connected

    for issue in lint_corpus(root):
        flagged = nodes.get(issue.file)
        if flagged is not None:
            flagged["issues"].append(
                {"code": issue.code, "level": issue.level, "message": issue.message[:MAX_ISSUE_MESSAGE]}
            )

    counts: dict[str, int] = {"memos": len(nodes), "edges": len(edges)}
    counts["issues"] = sum(len(node["issues"]) for node in nodes.values())
    counts["isolated"] = sum(1 for node in nodes.values() if node["isolated"])
    for state in ("current", "superseded", "expired", "pending"):
        counts[state] = sum(1 for node in nodes.values() if node["state"] == state)
    for kind in ("supersedes", "link", "pending"):
        counts[kind] = sum(1 for edge in edges if edge["kind"] == kind)
    counts["hubs"] = sum(1 for node in nodes.values() if node["hub"])
    return {"nodes": list(nodes.values()), "edges": edges, "counts": counts, "notes": list(queue_notes)}
