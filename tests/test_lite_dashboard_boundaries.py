"""PR899 CCA regressions through real SQLite ledger readers and dashboard requests.

Red proof: all 17 parametrized nodes failed in their intended assertions on
the original phase 6 PR head before repairs, 2026-10-12 on the test host.
Startup's four nodes reported "fallback tenant" at DashboardApp.handle; three verdict nodes
reported "unsupported verdict JSON" at lite_db.control; four source nodes reported
"unsupported report container" at control/use_reports; six bounded readers reported
"historical ledger payloads" at searches/search/use_reports/stale_reports. No setup errors or skips.
BUG-001: missing store at CLI startup freezes memory instead of the later file tenant.
DAT-001: nested verdict values break Control rather than preserving the retrieval count.
DAT-002: unsupported source containers break or invent report attribution.
PERF-001: bounded readers materialize the entire event history.
The cursor probe measures rows consumed, while outputs and tenant/type boundaries stay asserted.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from recall.cli_commands import dashboard_cmd
from recall.dashboard import db, lite_db
from recall.dashboard.server import DashboardApp
from recall.lite import LiteStore
from recall.rewrite import claim_key
from recall.stale_reports import quote_fingerprint


class _Launched(Exception):
    """The real command reached its socket boundary, replaced to avoid a server."""


def _launch(monkeypatch: pytest.MonkeyPatch, root: Path, path: Path, reports: str | None) -> DashboardApp:
    monkeypatch.setenv("RECALL_DSN", "sqlite:///" + path.as_posix())
    for name in ("RECALL_SERVING_DSN", "RECALL_DASHBOARD_DSN_FILE", "RECALL_DASHBOARD_TUNNEL"):
        monkeypatch.delenv(name, raising=False)
    apps: list[DashboardApp] = []

    def serve(app: DashboardApp) -> None:
        apps.append(app)
        raise _Launched

    monkeypatch.setattr("recall.dashboard.server.serve", serve)
    parser = argparse.ArgumentParser()
    dashboard_cmd.register(parser.add_subparsers())
    argv = ["dashboard", "--root", str(root), "--no-browser"]
    if reports is not None:
        argv.extend(["--reports-tenant", reports])
    args = parser.parse_args(argv)
    with pytest.raises(_Launched):
        args.func(args)
    return apps[0]


@pytest.mark.parametrize("tenant", ["default", "_team"])
@pytest.mark.parametrize("reports", [None, "other"])
def test_missing_store_recovers_its_tenant_without_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tenant: str, reports: str | None,
) -> None:
    """BUG-001: the ordinary retrieval request must find the newly created file's ledger."""
    path = tmp_path / "memory.db"
    app = _launch(monkeypatch, tmp_path, path, reports)
    assert not path.exists(), "dashboard startup created the corpus file"
    old, new = "The release is still being prepared for distribution.", "The release has been published for general distribution."
    (tmp_path / "old.md").write_text(old, encoding="utf-8")
    (tmp_path / "new.md").write_text(new, encoding="utf-8")
    old_sha, old_chars = quote_fingerprint(old)
    new_sha, new_chars = quote_fingerprint(new)
    with LiteStore(path, dim=4, tenant=tenant) as store:
        store.append_audit_event("search_decision", {"query": "the recovered query", "hits": []})
        store.append_audit_event("stale_report", {
            "stale_source": "old.md", "replacing_source": "new.md",
            "claim_key": claim_key("supersedes", "old.md", "new.md"),
            "stale_quote_sha256": old_sha, "stale_quote_chars": old_chars,
            "current_quote_sha256": new_sha, "current_quote_chars": new_chars,
        })
    signed = {"Host": "127.0.0.1:8765", "Cookie": f"recall_dashboard={app.token}; recall_tenant=memory"}
    page = app.handle("GET", "/retrieval", signed)
    assert page.status == 200
    assert b"the recovered query" in page.body, "the dashboard retained the unavailable store's fallback tenant"
    queue = app.handle("GET", "/queue", signed)
    assert queue.status == 200
    has_claim = b"new.md <em>replaces</em> old.md" in queue.body
    assert has_claim is (reports is None), "implicit recovery or the explicit report tenant boundary changed"


@pytest.mark.parametrize("verdict", [[], {}, ["ok"]])
def test_unsupported_verdicts_do_not_break_control(tmp_path: Path, verdict: Any) -> None:
    """DAT-001: valid JSON in one historical row cannot hide legitimate retrievals."""
    path = tmp_path / "m.db"
    with LiteStore(path, dim=4) as store:
        store.append_audit_event("search_decision", {"hits": [
            {"source": "a.md", "verdict": verdict}, {"source": "b.md", "verdict": "ok"},
        ]})
    try:
        result = db.control(db.DashboardDB("sqlite:///" + path.as_posix()), "default")
    except TypeError:
        pytest.fail("unsupported verdict JSON broke the Control page")
    memos = {m["source"]: m for m in result["memos"]}
    assert memos["a.md"]["retrieved"] == 1 and memos["a.md"]["ok"] == 0
    assert memos["b.md"]["ok"] == 1


@pytest.mark.parametrize("sources", [1, True, "memo.md", {"memo.md": True}])
def test_unsupported_report_containers_do_not_create_memo_counts(tmp_path: Path, sources: Any) -> None:
    """DAT-002: reject invalid containers while retaining valid report rows and attribution."""
    path = tmp_path / "m.db"
    with LiteStore(path, dim=4) as store:
        store.append_audit_event("use_report", {"used": sources, "wrong": sources})
        store.append_audit_event("use_report", {"used": ["real.md"], "effect": "helped"})
    handle = db.DashboardDB("sqlite:///" + path.as_posix())
    try:
        control = db.control(handle, "default")
        reports = db.use_reports(handle, "default")
    except TypeError:
        pytest.fail("unsupported report container broke dashboard reads")
    assert [(m["source"], m["used"]) for m in control["memos"]] == [("real.md", 1)], "unsupported report container fabricated memo sources"
    assert len(reports) == 2 and reports[1]["used"] == [] and reports[1]["wrong"] == []
    assert db.use_reports(handle, "default", source="real.md") == [reports[0]]


class _Cursor:
    def __init__(self, cursor: sqlite3.Cursor, read: list[int], audit: bool) -> None:
        self.cursor, self.read, self.audit = cursor, read, audit

    def fetchone(self) -> Any:
        row = self.cursor.fetchone()
        self.read[0] += int(self.audit and row is not None)
        return row

    def fetchall(self) -> list[Any]:
        rows = self.cursor.fetchall()
        self.read[0] += len(rows) if self.audit else 0
        return rows

    def __iter__(self) -> Iterator[Any]:
        for row in self.cursor:
            self.read[0] += int(self.audit)
            yield row


class _Connection:
    def __init__(self, conn: sqlite3.Connection, read: list[int]) -> None:
        self.conn, self.read = conn, read

    def execute(self, sql: str, args: Any = ()) -> _Cursor:
        return _Cursor(self.conn.execute(sql, args), self.read, "from audit_events" in sql.lower())


@pytest.mark.parametrize("reader", ["searches", "search", "reports", "stale", "filtered-searches", "filtered-reports"])
def test_bounded_reads_do_not_materialize_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str,
) -> None:
    """PERF-001: select correct rows while consuming only the needed historical payloads."""
    path = tmp_path / "m.db"
    with LiteStore(path, dim=4) as store:
        for i in range(30):
            store.append_audit_event("search_decision", {"query": f"Straße {i}", "hits": []}, event_id=f"s-{i}")
            store.append_audit_event("use_report", {"used": ["wanted.md"], "task": str(i)}, event_id=f"u-{i}")
            store.append_audit_event("stale_report", {"task": str(i)}, event_id=f"r-{i}")
    original, read = lite_db.connect, [0]

    @contextmanager
    def probed(dsn: str) -> Iterator[Any]:
        with original(dsn) as connection:
            yield _Connection(connection, read)

    monkeypatch.setattr(lite_db, "connect", probed)
    handle = db.DashboardDB("sqlite:///" + path.as_posix())
    if reader == "search":
        found = db.search(handle, "default", "s-0")
        assert found is not None and found["query"] == "Straße 0"
        bound = 1
    elif reader in ("searches", "filtered-searches"):
        rows = db.searches(handle, "default", limit=2, contains="strasse" if reader.startswith("filtered") else "")
        assert [r["event_id"] for r in rows] == ["s-29", "s-28"]
        bound = 2
    elif reader in ("reports", "filtered-reports"):
        rows = db.use_reports(handle, "default", limit=2, source="wanted.md" if reader.startswith("filtered") else "")
        assert [r["event_id"] for r in rows] == ["u-29", "u-28"]
        bound = 2
    else:
        rows = db.stale_reports(handle, ("default",), limit=2)
        assert [r["event_id"] for r in rows] == ["r-29", "r-28"]
        bound = 2
    assert read[0] <= bound, f"bounded {reader} materialized {read[0]} historical ledger payloads"
    assert db.search(handle, "other", "s-0") is None
    assert db.search(handle, "default", "u-0") is None
