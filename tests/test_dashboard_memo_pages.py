"""The editing pages: a session and a form token to change anything, a preview that writes nothing,
a write only of what was previewed, an undo, and every person-supplied string escaped.

Red proof, 2026-10-06, each mutation alone against `recall/dashboard/server.py`, failing for the stated
reason (JUnit XML), then restored; all four green after:
- P1 the memo page served without a session: 200 instead of 403.
- P2 the memo forms accepted without the form token: 200 instead of 403.
- P3 the preview writes the planned bytes: the memo changed.
- P4 the history note rendered unescaped: `<script>` in the page.
- P5 apply re-reads the file instead of trusting the digest the person was shown: 303 (written)
  instead of 409.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlencode

import pytest

from recall.dashboard import edit
from recall.dashboard.server import DashboardApp
from recall.rewrite import _derived_value

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "old.md").write_text("# Old\nThe port is 8080.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text("# New\nThe port is 9090.\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def app(corpus: Path) -> DashboardApp:
    return DashboardApp(corpus, port=8765, token="tok")


def _post(app: DashboardApp, path: str, fields: list[tuple[str, str]], headers=SIGNED):
    return app.handle("POST", path, headers, urlencode(fields).encode())


def test_the_memo_page_needs_the_session_and_shows_the_values(app: DashboardApp) -> None:
    assert app.handle("GET", "/memo?path=new.md", HOST).status == 403
    page = app.handle("GET", "/memo?path=new.md", SIGNED)
    assert page.status == 200
    assert "It supersedes nothing yet." in page.body.decode()
    assert app.handle("GET", "/memo?path=.recall/edits.sqlite3", SIGNED).status == 404


def test_a_preview_writes_nothing_and_a_post_without_the_token_is_refused(app: DashboardApp, corpus: Path) -> None:
    before = (corpus / "new.md").read_bytes()
    fields = [("path", "new.md"), ("add", "old.md"), ("status", "active")]
    assert _post(app, "/memo/preview", fields).status == 403
    page = _post(app, "/memo/preview", [("csrf", "tok"), *fields])
    assert page.status == 200
    text = page.body.decode()
    assert "+supersedes: old.md" in text and "status active" in text
    assert (corpus / "new.md").read_bytes() == before


def test_apply_writes_what_was_previewed_and_undo_puts_it_back(app: DashboardApp, corpus: Path) -> None:
    before = (corpus / "new.md").read_bytes()
    fields = [("path", "new.md"), ("add", "old.md"), ("status", "active")]
    preview = _post(app, "/memo/preview", [("csrf", "tok"), *fields]).body.decode()
    shown = re.search(r"name='shown_sha' value='([0-9a-f]+)'", preview).group(1)
    saved = _post(app, "/memo/apply", [("csrf", "tok"), ("shown_sha", shown), ("editor", "giulio"), ("note", "<script>x</script>"), *fields])
    assert saved.status == 303
    assert _derived_value((corpus / "new.md").read_bytes(), "status") == "active"
    page = app.handle("GET", "/memo?path=new.md", SIGNED).body.decode()
    assert "<script>x</script>" not in page and "&lt;script&gt;x&lt;/script&gt;" in page
    (record,) = edit.history(corpus, "new.md")
    undone = _post(app, "/memo/undo", [("csrf", "tok"), ("path", "new.md"), ("edit_id", str(record.edit_id)), ("editor", "giulio"), ("note", "wrong")])
    assert undone.status == 303
    assert (corpus / "new.md").read_bytes() == before


def test_apply_with_a_stale_preview_writes_nothing(app: DashboardApp, corpus: Path) -> None:
    fields = [("path", "new.md"), ("status", "draft")]
    preview = _post(app, "/memo/preview", [("csrf", "tok"), *fields]).body.decode()
    shown = re.search(r"name='shown_sha' value='([0-9a-f]+)'", preview).group(1)
    (corpus / "new.md").write_text("# New\nedited meanwhile\n", encoding="utf-8")
    response = _post(app, "/memo/apply", [("csrf", "tok"), ("shown_sha", shown), ("editor", "giulio"), ("note", "x"), *fields])
    assert response.status == 409
    assert _derived_value((corpus / "new.md").read_bytes(), "status") is None
