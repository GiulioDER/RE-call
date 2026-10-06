"""The activity feed and the health page: every file-backed record, newest first, once each, and
every string from a memo or a person escaped.

Red proof, 2026-10-06, each mutation alone, failing for the stated reason (JUnit XML), then restored;
both green after:
- A1 a reported claim's rejection also read from the ledger: three `rejected` events instead of two.
- A2 events oldest first: the order assertion.
- A3 an activity note rendered unescaped: `<b>` in the page.
- A4 the health page without its lint issues: `self-supersedes` missing.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard import edit
from recall.dashboard.activity import recent_activity
from recall.dashboard.server import DashboardApp
from recall.rewrite import RejectionLedger, claim_key, default_ledger_path
from recall.stale_reports import StaleReportQueue, default_queue_path

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "old.md").write_text("# Old\nThe release is still being prepared for PyPI.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text("# New\nThe release was published to PyPI on 2 September.\n", encoding="utf-8")
    (tmp_path / "self.md").write_text("---\nsupersedes: self.md\n---\n# Self\n", encoding="utf-8")
    old_time = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    for name in ("old.md", "new.md", "self.md"):
        os.utime(tmp_path / name, (old_time, old_time))
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        report = queue.submit(
            tmp_path, stale_source="old.md", replacing_source="new.md",
            stale_quote="is still being prepared for PyPI", current_quote="was published to PyPI on 2 September",
            client="mcp:acme", task=None, reported_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
        queue.mark_reviewed(report.claim_key, status="rejected", reviewer_id="giulio", note="not the same release",
                            reviewed_at=datetime(2026, 10, 2, tzinfo=UTC))
    with RejectionLedger(default_ledger_path(tmp_path)) as ledger:
        # The same claim also sits in the ledger, as the dashboard records it, plus one arbiter claim.
        ledger.reject(report.claim_key, reviewer_id="giulio", reason="not the same release", rejected_at=datetime(2026, 10, 2, tzinfo=UTC))
        ledger.reject(claim_key("supersedes", "self.md", "new.md"), reviewer_id="giulio", reason="unrelated",
                      rejected_at=datetime(2026, 10, 3, tzinfo=UTC))
    plan = edit.plan_edit(tmp_path, "new.md", status="active")
    edit.apply_edit(tmp_path, "new.md", editor="giulio", note="<b>checked</b>", shown_sha=plan.before_sha,
                    now=datetime(2026, 10, 4, tzinfo=UTC), status="active")
    os.utime(tmp_path / "new.md", (old_time, old_time))
    return tmp_path


def test_every_source_appears_once_newest_first(corpus: Path) -> None:
    events = recent_activity(corpus)
    kinds = [event.kind for event in events if event.kind != "changed"]
    assert kinds == ["edit", "rejected", "rejected", "report"]
    assert [e.detail for e in events if e.kind == "rejected"] == ["unrelated", "not the same release"]
    assert sum(1 for e in events if e.kind == "changed") == 3


def test_the_pages_need_the_session_and_escape_what_people_wrote(corpus: Path) -> None:
    app = DashboardApp(corpus, port=8765, token="tok")
    assert app.handle("GET", "/activity", HOST).status == 403
    assert app.handle("GET", "/health", HOST).status == 403
    activity = app.handle("GET", "/activity", SIGNED).body.decode()
    assert "&lt;b&gt;checked&lt;/b&gt;" in activity and "<b>checked</b>" not in activity
    health = app.handle("GET", "/health", SIGNED)
    assert health.status == 200
    page = health.body.decode()
    assert "self-supersedes" in page and "/read?path=self.md" in page
