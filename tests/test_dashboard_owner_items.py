"""The dashboard's default tenant, and how the graph page compares dates when it travels in time.

Red proof, 2026-11-04, each mutation alone, failing in its own assertion, then restored green:
- ENV-010: `DEFAULT_TENANT = "memory"` again in `recall/dashboard/server.py` fails
  `test_the_dashboard_opens_the_tenant_every_surface_opens` ("opened memory, not default").
- Time travel: removing the `invalid_metadata` line from `stateAt` in
  `recall/dashboard/static/graph.js` fails
  `test_time_travel_keeps_dates_search_cannot_read_unreadable` (drawn "current").
The other `stateAt` cases pin what was verified by reading: node dates are `YYYY-MM-DD`, so a
string comparison orders them as dates, and `valid_until` holds through its own day, as
`recall.frontmatter.validity_bounds` reads it. They are a nonbehavioural check of code that was
already correct, so red proof does not apply to them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from recall import db_constants
from recall.cli_commands import dashboard_cmd
from recall.dashboard import corpus, server
from recall.supersession import SupersessionGraph

GRAPH_JS = Path(server.__file__).parent / "static" / "graph.js"


class _Stop(Exception):
    pass


def test_the_dashboard_opens_the_tenant_every_surface_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-010: on PostgreSQL the dashboard opened `memory` while every other surface opens `default`."""
    built: dict[str, Any] = {}

    def app(root: Path, **kwargs: Any) -> None:
        built.update(kwargs)
        raise _Stop

    monkeypatch.setattr(server, "DashboardApp", app)
    monkeypatch.setattr(dashboard_cmd, "_database", lambda path: None)
    args = argparse.Namespace(
        root=str(tmp_path), port=8765, reports_tenant=None, db_dsn_file=None, tunnel=None,
        tunnel_ports="55433:55432", config_host="", config_command="",
    )
    with pytest.raises(_Stop):
        dashboard_cmd._cmd_dashboard(args)
    opened = built["default_tenant"]
    assert opened == db_constants.DEFAULT_TENANT == "default", f"ENV-010: opened {opened}, not default"
    assert built["reports_tenants"] == ("default",)


def _state_at(nodes: list[dict[str, Any]], as_of: str | None) -> dict[str, str]:
    """Run `stateAt` from graph.js, as the page defines it, over `nodes` on the day `as_of`."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed: graph.js cannot be run here")
    source = GRAPH_JS.read_text(encoding="utf-8")
    found = re.search(r"\n  function stateAt\(n\) \{\n.*?\n  \}\n", source, re.S)
    assert found, "stateAt is no longer defined in graph.js as this test reads it"
    script = (
        f"const nodes = {json.dumps(nodes)};\n"
        f"let asOf = {json.dumps(as_of)};\n"
        "const byId = new Map(nodes.map((n) => [n.id, n]));\n"
        "const supersededBy = new Map();\n"
        "for (const n of nodes) for (const t of n.supersedes || []) "
        "supersededBy.set(t, [...(supersededBy.get(t) || []), n.id]);\n"
        f"{found.group(0)}\n"
        "console.log(JSON.stringify(Object.fromEntries(nodes.map((n) => [n.id, stateAt(n)]))));\n"
    )
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def _node(name: str, **fields: Any) -> dict[str, Any]:
    return {"id": name, "state": "current", "born": "2026-01-01", "valid_from": None, "valid_until": None, **fields}


def test_time_travel_keeps_dates_search_cannot_read_unreadable() -> None:
    """BUG-006 on the graph page: on a past day, a file with unreadable dates was drawn current."""
    states = _state_at([_node("bad.md", state="invalid_metadata", valid_until="soon")], "2026-03-01")
    assert states["bad.md"] == "invalid_metadata", f"time travel drew unreadable dates {states['bad.md']}"


def test_time_travel_compares_validity_dates_as_days() -> None:
    nodes = [
        _node("ends-that-day.md", valid_until="2026-03-01"),
        _node("ended.md", valid_until="2026-02-28"),
        _node("later.md", valid_from="2026-03-02"),
        _node("old.md"),
        _node("new.md", born="2026-02-01", supersedes=["old.md"]),
        _node("old2.md"),
        _node("future.md", born="2026-04-01", supersedes=["old2.md"]),
    ]
    assert _state_at(nodes, "2026-03-01") == {
        "ends-that-day.md": "current",  # validity_bounds: valid_until holds through its own day
        "ended.md": "expired",
        "later.md": "expired",
        "old.md": "superseded",
        "new.md": "current",
        "old2.md": "current",  # its successor did not exist yet on that day
        "future.md": "current",
    }


def test_the_corpus_graph_sends_dates_the_page_can_compare() -> None:
    """`2026-9-30` reaches the page as `2026-09-30`: as text it would sort after `2026-10-01`."""
    files = [("a.md", 1, None, None, None, "2026-9-30", [], "s3://x/a.md")]
    built = corpus.assemble_graph(files, SupersessionGraph.from_parts({}), [], [], [], today="2026-11-04", tenant="t", generation="g")
    (node,) = built["nodes"]
    assert node["valid_until"] == "2026-09-30"
    assert _state_at([node | {"state": "current"}], "2026-10-01")[node["id"]] == "expired"
