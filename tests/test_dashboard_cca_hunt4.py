"""Regressions for the two P3 findings of the first 2026-11-04 dashboard hunt that later passes left open.

Run 20261104T131553Z deferred 34 P3 findings unverified. A fourth pass (run 20261104T213645Z)
verified them on b95f37a91: 27 were already fixed by PR 1183 or PR 1190, and two were still
present, fixed here.

- CODE-006: the Activity page decided "undo" by matching the literal "undo of edit", a copy of
  `edit.UNDO_PREFIX`; reworded, every undo would show as an edit and nothing would fail.
- PERF-009: the page for one memory ran the whole-tenant control aggregate and kept one entry.

Red proof, each test run against the pre-fix files on the test host and failing in its own
assertion, then green on the fixed files; the receipts are in the pull request.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from recall.dashboard import db as dbq
from recall.dashboard import edit
from recall.dashboard.activity import recent_activity
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Authorization": "Bearer tok"}


def test_code006_an_undo_is_labelled_by_the_writer_s_own_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CODE-006. Red proof: pre-fix `activity.py` keeps its literal, so the undo reads "edit"."""
    monkeypatch.setattr(edit, "UNDO_PREFIX", "reverted edit")
    (tmp_path / "a.md").write_text("# A\n\nnote\n", encoding="utf-8")
    plan = edit.plan_edit(tmp_path, "a.md", status="active")
    when = datetime(2026, 11, 4, 20, 0, tzinfo=UTC)
    edit.apply_edit(tmp_path, "a.md", editor="giulio", note="mark it", shown_sha=plan.before_sha, now=when, status="active")
    (record,) = edit.history(tmp_path, name="a.md")
    edit.undo_edit(tmp_path, record.edit_id, editor="giulio", note="put it back", now=when)
    kinds = [event.kind for event in recent_activity(tmp_path) if event.kind in ("edit", "undo")]
    assert kinds.count("undo") == 1, f"CODE-006: an undo written with the writer's prefix was listed as {kinds}"


def test_perf009_the_page_for_one_memory_asks_for_that_memory_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-009. Red proof: pre-fix `_control_memo_page` calls `control(db, tenant)` with no `source`."""
    asked: list[dict[str, Any]] = []
    found = {
        "source": "a.md", "retrieved": 1, "first": 1, "last_retrieved": None, "ok": 1, "superseded": 0,
        "low_confidence": 0, "used": 0, "wrong": 0, "helped": 0, "no_difference": 0, "misled": 0,
        "succeeded": 0, "failed": 0, "last_reported": None,
    }

    def control(db: DashboardDB, tenant: str, **kwargs: Any) -> dict[str, Any]:
        asked.append(kwargs)
        return {"summary": {}, "memos": [found]}

    monkeypatch.setattr(dbq, "control", control)
    monkeypatch.setattr(dbq, "use_reports", lambda *args, **kwargs: [])
    monkeypatch.setattr(dbq, "tenants", lambda db: [])
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    app = DashboardApp(tmp_path, port=8765, token="tok", db=DashboardDB("postgresql://fake"))
    app.handle("GET", "/control/memo?" + urlencode({"source": "a.md"}), SIGNED)
    assert asked == [{"source": "a.md"}], f"PERF-009: the one-memory page asked control() for {asked}"


def test_perf009_lite_control_for_one_memory_counts_only_it(tmp_path: Path) -> None:
    """PERF-009, the SQLite twin of K6 in test_dashboard_control_db.py.

    Red proof by mutation (the pre-fix `lite_db.control` has no `source` parameter, which would be
    no proof): with the per-hit `source` check in `lite_db.control` removed, b.md is returned too.
    """
    from recall.dashboard import lite_db
    from recall.lite import LiteStore

    with LiteStore(tmp_path / "m.db", dim=4) as store:
        hits = [{"source": "a.md", "verdict": "ok"}, {"source": "b.md", "verdict": "ok"}, {"source": "a.md", "verdict": "superseded"}]
        store.append_audit_event("search_decision", {"query": "q", "hits": hits})
        store.append_audit_event("use_report", {"task": "t", "effect": "helped", "used": ["a.md", "b.md"], "wrong": []})
    dsn = "sqlite:///" + (tmp_path / "m.db").as_posix()
    everything = {m["source"]: m for m in lite_db.control(dsn, "default")["memos"]}
    only = lite_db.control(dsn, "default", source="a.md")["memos"]
    assert [m["source"] for m in only] == ["a.md"], f"control(source='a.md') returned {[m['source'] for m in only]}"
    assert only[0] == everything["a.md"]
