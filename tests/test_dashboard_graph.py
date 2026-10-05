"""The memory graph: built from what memos say, served only to this session, drawn without markup.

Red proof, 2026-10-05, each mutation alone, failing for the stated reason (JUnit XML), then restored;
all green after. Against `recall/dashboard/graph.py`:
- G1 a declared supersession does not mark the older memo: assert 'current' == 'superseded'.
- G2 wiki links ignored: the link edges are missing.
- G3 the `.recall` sidecar read as memos: an extra node.
- G4 `valid_until` ignored: assert 'current' == 'expired'.
- G5 the hub threshold made exclusive: assert set() == {'index.md'}.
- G6 a queued claim does not mark its memo pending: ('current', ...) == ('pending', ...).
Against `recall/dashboard/server.py`:
- G7 the graph data served without a session: 200 instead of 403.
- G8 any path under /static/ served from disk: `/static/../server.py` returned 200 instead of 404,
  i.e. the server's own source was readable.
- G9 `'unsafe-inline'` added to script-src: AssertionError on the policy.
Against `recall/dashboard/static/graph.js` (the static guard, labelled as such in its test):
- G10 one `textContent` assignment turned into a markup assignment: the guard names it.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard.graph import HUB_OUT_LINKS, build_graph
from recall.dashboard.server import STATIC, DashboardApp, security_headers
from recall.stale_reports import StaleReportQueue, default_queue_path

TODAY = datetime(2026, 10, 5, tzinfo=UTC)
HOST = {"Host": "127.0.0.1:8765"}


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _graph(root: Path) -> dict:
    return build_graph(root, today=TODAY)


def _edges(graph: dict) -> set[tuple[str, str, str]]:
    return {(e["source"], e["target"], e["kind"]) for e in graph["edges"]}


def _state(graph: dict, name: str) -> str:
    return next(n["state"] for n in graph["nodes"] if n["id"] == name)


def test_a_declared_supersession_is_an_edge_and_marks_the_older_memo(tmp_path: Path) -> None:
    _write(tmp_path, "old.md", "# Old\nThe port is 8080.\n")
    _write(tmp_path, "new.md", "---\nsupersedes: old.md\n---\n# New\nThe port is 9090.\n")
    _write(tmp_path, "dangling.md", "---\nsupersedes: nowhere.md\n---\n# Dangling\n")
    graph = _graph(tmp_path)
    assert ("new.md", "old.md", "supersedes") in _edges(graph)
    assert _state(graph, "old.md") == "superseded"
    assert _state(graph, "new.md") == "current"
    assert not any(e[0] == "dangling.md" for e in _edges(graph))


def test_wiki_and_markdown_links_are_edges_and_the_sidecar_is_not_a_memo(tmp_path: Path) -> None:
    _write(tmp_path, "a.md", "# A\nSee [[b]] and [c](notes/c.md) and [[missing]].\n")
    _write(tmp_path, "b.md", "# B\n")
    _write(tmp_path, "notes/c.md", "# C\n")
    _write(tmp_path, ".recall/hidden.md", "# Not a memo\n")
    graph = _graph(tmp_path)
    assert {("a.md", "b.md", "link"), ("a.md", "notes/c.md", "link")} <= _edges(graph)
    assert {n["id"] for n in graph["nodes"]} == {"a.md", "b.md", "notes/c.md"}


def test_a_memo_past_its_validity_is_expired(tmp_path: Path) -> None:
    _write(tmp_path, "ended.md", "---\nvalid_until: 2026-01-01\n---\n# Ended\n")
    _write(tmp_path, "open.md", "---\nvalid_until: 2027-01-01\n---\n# Open\n")
    graph = _graph(tmp_path)
    assert _state(graph, "ended.md") == "expired"
    assert _state(graph, "open.md") == "current"


def test_an_index_page_is_flagged_as_a_hub(tmp_path: Path) -> None:
    links = " ".join(f"[[m{i}]]" for i in range(HUB_OUT_LINKS))
    _write(tmp_path, "index.md", f"# Index\n{links}\n")
    for i in range(HUB_OUT_LINKS):
        _write(tmp_path, f"m{i}.md", f"# M{i}\n")
    _write(tmp_path, "small.md", "# Small\n[[m1]]\n")
    hubs = {n["id"] for n in _graph(tmp_path)["nodes"] if n["hub"]}
    assert hubs == {"index.md"}


def test_a_queued_claim_is_a_pending_edge_with_its_claim(tmp_path: Path) -> None:
    _write(tmp_path, "old.md", "# Old\nThe release is still being prepared for PyPI.\n")
    _write(tmp_path, "new.md", "# New\nThe release was published to PyPI on 2 September.\n")
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        report = queue.submit(
            tmp_path, stale_source="old.md", replacing_source="new.md",
            stale_quote="is still being prepared for PyPI", current_quote="was published to PyPI on 2 September",
            client="mcp:test", task=None, reported_at=TODAY,
        )
    graph = _graph(tmp_path)
    assert ("new.md", "old.md", "pending") in _edges(graph)
    old = next(n for n in graph["nodes"] if n["id"] == "old.md")
    assert (old["state"], old["claim"]) == ("pending", report.claim_key)


@pytest.fixture
def app(tmp_path: Path, monkeypatch) -> DashboardApp:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    _write(tmp_path, "a.md", "# A <b>bold</b>\n[[b]]\n")
    _write(tmp_path, "b.md", "# B\n")
    return DashboardApp(tmp_path, port=8765, token="tok")


def test_the_graph_data_needs_the_session(app: DashboardApp) -> None:
    assert app.handle("GET", "/api/graph.json", HOST).status == 403
    response = app.handle("GET", "/api/graph.json", {**HOST, "Cookie": "recall_dashboard=tok"})
    assert response.status == 200
    payload = json.loads(response.body)
    assert {n["id"] for n in payload["nodes"]} == {"a.md", "b.md"}


def test_the_page_loads_only_the_packaged_script(app: DashboardApp) -> None:
    signed = {**HOST, "Cookie": "recall_dashboard=tok"}
    page = app.handle("GET", "/graph", signed).body.decode()
    assert re.findall(r"<script[^>]*>", page) == ["<script src='/static/graph.js' defer>"]
    assert "<script src='/static/graph.js' defer></script>" in page
    script = app.handle("GET", "/static/graph.js", signed)
    assert script.status == 200 and dict(script.headers)["Content-Type"].startswith("text/javascript")
    assert app.handle("GET", "/static/graph.js", HOST).status == 403
    assert app.handle("GET", "/static/../server.py", signed).status == 404
    policy = dict(security_headers())["Content-Security-Policy"]
    assert "default-src 'none'" in policy and "script-src 'self';" in policy
    assert "unsafe-inline" not in policy.split("script-src", 1)[1].split(";", 1)[0]


#: APIs that parse a string as markup or as code. Spelled in parts so this guard does not itself
#: read as a use of them to a scanner.
_MARKUP_OR_CODE_SINKS = (
    "inner" + "HTML",
    "outer" + "HTML",
    "insertAdjacent" + "HTML",
    "document." + "write",
    "ev" + "al(",
    "new " + "Function",
)


def test_the_script_never_writes_markup() -> None:
    """A static guard, not a behaviour test: memo text reaches the page only through textContent.

    The script cannot be executed here, so this pins the property at the source: none of the APIs
    that parse a string as markup or code appear in it.
    """
    source = (STATIC / "graph.js").read_text(encoding="utf-8")
    for api in _MARKUP_OR_CODE_SINKS:
        assert api not in source, api
    assert "textContent" in source
