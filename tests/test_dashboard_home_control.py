"""The home page, the review queue's new address, memo search, and the Control pages.

The database is replaced by fakes of `recall.dashboard.db`'s query functions (the queries
themselves are tested against Postgres in `tests/test_dashboard_control_db.py`).

Invariants and the failure each one catches:
- H1 home names a pending review and links to the queue; with nothing pending, no problems and
  no database it says nothing needs attention rather than showing an empty list.
- H2 the queue lives at /queue, and a decision returns there, not to the home page.
- H3 the Control filters show what they say: "reported wrong" only memories agents called wrong,
  "retrieved, never used" only memories retrieved and never used.
- H4 what an agent wrote in a report (task, query, note) is escaped on the memory's page.
- H5 memo search puts name matches first and escapes the text around a match.
- H6 every new page needs the session.

Red proof, 2026-10-07, each mutation alone against `recall/dashboard/server.py`, failing in the
named assertion (JUnit XML), then restored and green:
- S1 (H1) `if pending:` in `_home_page` made `if False:`: the review card was missing.
- S2 (H2) the decision redirect sent back to "/": Location was not /queue.
- S3 (H3) the `wrong` filter in `_control_page` removed: an unreported memory was listed.
- S4 (H4) `_e(r['task'])` on the memory page rendered raw: `<script>` reached the page.
- S5 (H5) the name-first sort key made constant: a text match came before the name match.
- S6 (H5) the find snippet rendered raw: `<b>` reached the page.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from recall.dashboard import db as dbq
from recall.dashboard import review
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp
from recall.stale_reports import StaleReportQueue, default_queue_path

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
OLD_QUOTE = "the dashboard listens on port 8080"
NEW_QUOTE = "the dashboard listens on port 8765"
WHEN = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def _memo(source: str, **values: Any) -> dict[str, Any]:
    row = {"source": source, "retrieved": 0, "first": 0, "last_retrieved": WHEN, "ok": 0, "superseded": 0,
           "low_confidence": 0, "used": 0, "wrong": 0, "helped": 0, "no_difference": 0, "misled": 0,
           "succeeded": 0, "failed": 0, "last_reported": None}
    row.update(values)
    return row


CONTROL = {
    "summary": {"searches": 5, "answered": 4, "reports": 2, "helped": 1, "no_difference": 0, "misled": 1,
                "succeeded": 1, "failed": 1, "memos_retrieved": 3},
    "memos": [
        _memo("file:///m/port-new.md", retrieved=4, first=3, used=2, helped=1),
        _memo("file:///m/port-old.md", retrieved=3, wrong=1, misled=1),
        _memo("file:///m/never-used.md", retrieved=2),
    ],
}
REPORTS = [{
    "event_id": "e1", "actor": "agent-report", "created_at": WHEN, "task": "<script>alert(1)</script> fix the port",
    "effect": "misled", "used": [], "wrong": ["file:///m/port-old.md"], "task_succeeded": False,
    "query": "<img src=x onerror=1>", "note": "<b>stale</b>",
}]


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "old.md").write_text(f"# Old\nIn the past {OLD_QUOTE}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"# New\nToday {NEW_QUOTE}. The <b>bold</b> port memo.\n", encoding="utf-8")
    (tmp_path / "port-notes.md").write_text("# Notes\nNothing about it in the text.\n", encoding="utf-8")
    return tmp_path


def _report(corpus: Path) -> None:
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        queue.submit(corpus, stale_source="old.md", replacing_source="new.md", stale_quote=OLD_QUOTE,
                     current_quote=NEW_QUOTE, client="mcp:memory", task="port", reported_at=WHEN)


@pytest.fixture
def connected(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> DashboardApp:
    monkeypatch.setattr(dbq, "control", lambda db, tenant: CONTROL)
    monkeypatch.setattr(dbq, "use_reports", lambda db, tenant, limit=50, source="": [r for r in REPORTS if not source or source in r["used"] + r["wrong"]])
    monkeypatch.setattr(dbq, "tenants", lambda db: [])
    return DashboardApp(corpus, port=8765, token="tok", db=DashboardDB("postgresql://fake"))


def test_home_names_a_pending_review_and_says_when_nothing_needs_attention(corpus: Path) -> None:
    app = DashboardApp(corpus, port=8765, token="tok")
    quiet = app.handle("GET", "/", SIGNED).body.decode()
    assert "waiting for your review" not in quiet
    _report(corpus)
    page = app.handle("GET", "/", SIGNED).body.decode()
    assert "1 change waiting for your review" in page, "a pending review was not named on the home page"
    assert "href='/queue'" in page


def test_the_queue_moved_and_a_decision_returns_to_it(corpus: Path) -> None:
    _report(corpus)
    app = DashboardApp(corpus, port=8765, token="tok")
    assert "Review queue" in app.handle("GET", "/queue", SIGNED).body.decode()
    (item,) = review.build_queue(corpus).items
    form = urlencode({"csrf": "tok", "claim": item.claim, "reviewer": "giulio", "note": "no", "digest": review.memo_digest(corpus, "new.md")})
    response = app.handle("POST", "/reject", SIGNED, form.encode())
    assert response.status == 303
    assert dict(response.headers)["Location"].startswith("/queue?"), "a decision did not return to the queue"


def test_the_control_filters_show_what_they_say(connected: DashboardApp) -> None:
    every = connected.handle("GET", "/control", SIGNED).body.decode()
    assert all(name in every for name in ("port-new.md", "port-old.md", "never-used.md"))
    wrong = connected.handle("GET", "/control?sort=wrong", SIGNED).body.decode()
    assert "port-old.md" in wrong
    assert "port-new.md" not in wrong and "never-used.md" not in wrong, "the 'reported wrong' view listed a memory nobody called wrong"
    unused = connected.handle("GET", "/control?sort=unused", SIGNED).body.decode()
    assert "never-used.md" in unused and "port-new.md" not in unused


def test_an_agents_words_are_escaped_on_the_memory_page(connected: DashboardApp) -> None:
    page = connected.handle("GET", "/control/memo?" + urlencode({"source": "file:///m/port-old.md"}), SIGNED)
    assert page.status == 200
    text = page.body.decode()
    assert "<script>alert(1)</script>" not in text and "&lt;script&gt;alert(1)&lt;/script&gt; fix the port" in text
    assert "<img src=x" not in text and "<b>stale</b>" not in text
    assert connected.handle("GET", "/control/memo?" + urlencode({"source": "file:///m/none.md"}), SIGNED).status == 404


def test_memo_search_puts_names_first_and_escapes_the_text(corpus: Path) -> None:
    app = DashboardApp(corpus, port=8765, token="tok")
    page = app.handle("GET", "/find?q=port", SIGNED).body.decode()
    assert page.index("port-notes.md") < page.index("new.md"), "a text match was listed before a name match"
    bold = app.handle("GET", "/find?q=bold", SIGNED).body.decode()
    assert "<b>bold</b>" not in bold and "&lt;b&gt;bold&lt;/b&gt;" in bold


def test_the_new_pages_need_the_session(connected: DashboardApp) -> None:
    for path in ("/", "/queue", "/control", "/control/memo?source=x", "/find?q=port"):
        assert connected.handle("GET", path, HOST).status == 403, path
