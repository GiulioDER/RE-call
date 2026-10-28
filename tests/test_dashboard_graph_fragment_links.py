"""BUG-003: every original dashboard graph-link producer encodes identity once.

Red proof: all nodes fail the graph-fragment assertion on baseline970dbeb7b.
Health exercises both triage and ordinary lint links. Database transport is
stubbed; the real public page routes and local memo lookup remain in use.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest

from recall.dashboard import db as dbq
from recall.dashboard.server import DashboardApp
from tests.test_dashboard_db_pages import FakeCorpus

SIGNED = {"Host": "127.0.0.1:8765", "Cookie": "recall_dashboard=tok"}


@pytest.mark.parametrize("name", ["literal%20name.md", "100%.md"])
@pytest.mark.parametrize("page", ["memo", "find", "health", "control"])
def test_original_pages_preserve_graph_fragment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, page: str,
) -> None:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "old.md").write_text("# Old\nOld text\n", encoding="utf-8")
    (tmp_path / name).write_text("# Memo\nneedle This replaces [[old]] entirely.\n", encoding="utf-8")
    FakeCorpus().install(monkeypatch)
    source = "file:///memory/" + name
    row = {key: 0 for key in ("retrieved", "first", "ok", "superseded", "low_confidence", "used", "wrong", "helped", "misled", "succeeded", "failed")}
    row.update(source=source, last_retrieved=None, last_reported=None)
    monkeypatch.setattr(dbq, "control", lambda *args, **kwargs: {"summary": {}, "memos": [row]})
    monkeypatch.setattr(dbq, "use_reports", lambda *args, **kwargs: [])
    app = DashboardApp(tmp_path, port=8765, token="tok", db=dbq.DashboardDB("postgresql://fake"))
    path = {
        "memo": "/memo?" + urlencode({"path": name}),
        "find": "/find?q=needle",
        "health": "/health",
        "control": "/control/memo?" + urlencode({"source": source}),
    }[page]
    response = app.handle("GET", path, SIGNED)
    assert response.status == 200
    fragments = re.findall(r"href='/graph#memo=([^']*)'", response.body.decode())
    assert fragments, "Fixture must reach a graph-link producer"
    expected = {quote(name, safe=""), "old.md"}
    assert set(fragments) <= expected, "BUG-003: a graph fragment lost literal filename identity"
