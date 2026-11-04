"""Regressions from the 2026-11-04 DEEP CCA hunt of the dashboard (server, edit, review, corpus).

One test per confirmed finding. Each was run against the pre-fix files (the run's snapshot) and
failed in its own assertion or with the defect's own exception, then passed on the fixed files; the
proof receipts are in the pull request. No test imports a symbol the fixes introduced at module top,
so a pre-fix run reaches the behaviour instead of failing to import.
"""

from __future__ import annotations

import http.client
import pathlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pytest

from recall.dashboard import corpus
from recall.dashboard import db as dbq
from recall.dashboard import edit, review
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp, serve
from recall.rewrite import RewriteRefused
from recall.stale_reports import StaleReportQueue, StaleReportRefused, default_queue_path
from recall.supersession import SupersessionGraph
from tests.test_dashboard_corpus import _detail
from tests.test_dashboard_db_pages import FakeCorpus

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "a.md").write_text("# A\n\nnote\n", encoding="utf-8")
    return tmp_path


def _app(root: Path, **kwargs: Any) -> DashboardApp:
    return DashboardApp(root, port=8765, token="tok", **kwargs)


# ------------------------------------------------------------------ security


def test_sec002_a_unc_link_in_a_memo_never_reaches_the_filesystem(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-002 (P1): a UNC target in memo prose is refused before `resolve()` can open it.

    On Windows `Path.resolve()` opens `\\\\host\\share\\x`, which contacts that host and sends the
    user's credentials, and the containment check ran after it. Two producers on one /read request:
    `_read_page` (wiki and Markdown link targets, the `path` value) and `graph.build_graph`
    (Markdown links of every memo). Baseline: the pre-fix files resolve all three UNC paths.
    Mutation: removing only the `anchored` guard in `build_graph` fails the same assertion with
    the backslash target listed.
    """
    (root / "m.md").write_text(
        "# M\n\nSee [[//attacker.example/share/a]] and [x](\\\\attacker.example\\share\\b.md) and [[a]].\n",
        encoding="utf-8",
    )
    resolved: list[str] = []
    original = pathlib.Path.resolve

    def spy(self: pathlib.Path, strict: bool = False) -> pathlib.Path:
        resolved.append(str(self))  # every resolve in the request: the reader's and build_graph's
        return original(self, strict=strict)

    monkeypatch.setattr(pathlib.Path, "resolve", spy)
    page = _app(root).handle("GET", "/read?path=m.md", SIGNED)
    direct = _app(root).handle("GET", "/read?" + urlencode({"path": "//attacker.example/share/m.md"}), SIGNED)
    assert page.status == 200 and direct.status == 404
    touched = [p for p in resolved if "attacker.example" in p]
    assert touched == [], f"SEC-002: a UNC path reached Path.resolve(): {touched}"


def test_sec001_a_refusal_before_the_session_shows_no_root_and_no_tenants(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-001: the wrong-Host and no-session 403s are readable by a DNS-rebinding page.

    Pre-fix they render the full shell: the absolute memo root and a tenant list queried from the
    corpus database. The assertion on the body (and on the tenant query) fails on the baseline.
    """
    asked: list[str] = []

    def tenants(db: DashboardDB) -> list[dict[str, Any]]:
        asked.append("tenants")
        return [{"tenant": "secret-tenant", "generation": "g"}]

    monkeypatch.setattr(dbq, "tenants", tenants)
    app = _app(root, db=DashboardDB("postgresql://fake"))
    for headers in ({"Host": "evil.example:8765", "Cookie": "recall_dashboard=tok"}, HOST, {**HOST, "Cookie": "recall_dashboard=wrong"}):
        response = app.handle("GET", "/", headers)
        body = response.body.decode()
        assert response.status == 403
        assert str(root) not in body and "secret-tenant" not in body, "SEC-001: a pre-session refusal leaked the shell"
    wrong_link = app.handle("GET", "/?token=nope", HOST)
    assert wrong_link.status == 403 and str(root) not in wrong_link.body.decode()
    assert asked == [], "SEC-001: an unauthenticated request queried the corpus database"


def test_sec004_a_non_ascii_token_is_refused_not_raised(root: Path) -> None:
    """SEC-004: `compare_digest` raises TypeError on a non-ASCII str, so the request got no answer.

    Pre-fix the first call raises TypeError out of `handle` (the defect itself).
    """
    app = _app(root)
    assert app.handle("GET", "/?token=%C3%A9", HOST).status == 403
    assert app.handle("GET", "/", {**HOST, "Cookie": "recall_dashboard=é"}).status == 403
    form = urlencode({"csrf": "é", "file": "a.md", "sha": "x"}).encode()
    assert app.handle("POST", "/lint/restore", SIGNED, form).status == 403


def test_env001_a_dashboard_on_another_port_cannot_log_this_one_out(root: Path) -> None:
    """ENV-001: cookies ignore the port, so a second dashboard's session cookie replaced this one's.

    Pre-fix the session cookie is the fixed `recall_dashboard`: the Set-Cookie name assertion fails,
    and a request carrying another port's token under that name is refused.
    """
    app = _app(root)
    issued = app.handle("GET", "/?token=tok", HOST)
    cookies = [value for name, value in issued.headers if name == "Set-Cookie"]
    assert cookies and cookies[0].startswith("recall_dashboard_8765=tok"), f"ENV-001: session cookie not port-scoped: {cookies}"
    both = {**HOST, "Cookie": "recall_dashboard=other-dashboards-token; recall_dashboard_8765=tok"}
    assert app.handle("GET", "/", both).status == 200, "ENV-001: another port's cookie shadowed this dashboard's"


# ------------------------------------------------------------------ edits and their ledger


def _stick(root: Path, name: str) -> None:
    """Leave an interrupted edit of `name` whose memo has changed since, as a crash would."""
    with edit.EditLedger(root.resolve()):
        pass  # creates the ledger's tables
    with sqlite3.connect(root / ".recall" / "edits.sqlite3") as conn:
        conn.execute(
            "INSERT INTO edit_recovery (operation_id, file, before, after, summary, editor, note, edited_at) "
            "VALUES ('op', ?, ?, ?, '[]', 'p', 'n', ?)",
            (name, b"neither this\n", b"nor this\n", NOW.isoformat()),
        )


def test_stakes001_one_unreconcilable_interrupted_edit_refuses_only_its_own_memo(root: Path) -> None:
    """STAKES-001: an interrupted edit whose memo changed since refused every memo's edits.

    Pre-fix `EditLedger.__init__` raises EditRefused for any memo, so the edit of `b.md` raises and
    the test fails there; after the fix only `a.md` is refused, with the retained-bytes message.
    """
    (root / "b.md").write_text("# B\n\nnote\n", encoding="utf-8")
    _stick(root, "a.md")
    plan = edit.plan_edit(root, "b.md", status="draft")
    done = edit.apply_edit(root, "b.md", editor="p", note="n", shown_sha=plan.before_sha, now=NOW, status="draft")
    assert done.name == "b.md" and (root / "b.md").read_bytes() == plan.after
    assert [r.name for r in edit.history(root)] == ["b.md"]
    stuck = edit.plan_edit(root, "a.md", status="draft")
    with pytest.raises(edit.EditRefused, match="interrupted edit"):
        edit.apply_edit(root, "a.md", editor="p", note="n", shown_sha=stuck.before_sha, now=NOW, status="draft")


def test_bug001_the_memo_page_answers_when_the_edit_history_cannot_be_read(root: Path) -> None:
    """BUG-001: `edit.history` raising EditRefused escaped `_memo_page`, so no response was sent.

    The ledger path is made a directory, so opening it fails. Pre-fix the EditRefused propagates
    out of `handle` (the defect); after the fix the page renders with the reason in an error card.
    """
    (root / ".recall" / "edits.sqlite3").mkdir(parents=True)
    page = _app(root).handle("GET", "/memo?path=a.md", SIGNED)
    assert page.status == 200
    assert "edit history storage is unavailable" in page.body.decode()
    assert _app(root).handle("GET", "/activity", SIGNED).status == 200


# ------------------------------------------------------------------ review decisions


@pytest.fixture
def reported(root: Path) -> Path:
    (root / "old.md").write_text(f"The release is 0.12.0 and {OLD_QUOTE}.\n", encoding="utf-8")
    (root / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}.\n", encoding="utf-8")
    with StaleReportQueue(default_queue_path(root)) as queue:
        queue.submit(
            root, stale_source="old.md", replacing_source="new.md", stale_quote=OLD_QUOTE,
            current_quote=NEW_QUOTE, client="mcp:acme", task="which release?", reported_at=NOW,
        )
    return root


def _no_arbiter(_root: Path):
    raise RewriteRefused("the supersession arbiter is off")


def _accept_form(app: DashboardApp, root: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    monkeypatch.setattr(review, "corpus_proposals", lambda *a, **k: _no_arbiter(root))
    (item,) = app._queue().items
    shown = review.memo_digest(root, review.preview(root, item, NOW).edit_file)
    return urlencode({"csrf": "tok", "claim": item.claim, "reviewer": "p", "note": "n", "digest": shown}).encode()


def test_bug002_an_accept_that_wrote_is_never_reported_as_nothing_written(reported: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-002: a report-queue failure AFTER the memo was written was shown as "Nothing was written".

    `mark_reviewed` refusing (another dashboard closed the report first) raised StaleReportRefused,
    which nothing caught: pre-fix the POST raises out of `handle`. After the fix the accept redirects
    with "Declared", and the memo holds the declaration.
    """
    app = _app(reported)
    form = _accept_form(app, reported, monkeypatch)

    def closed_elsewhere(self: StaleReportQueue, *args: object, **kwargs: object) -> None:
        raise StaleReportRefused("that report was already closed")

    monkeypatch.setattr(StaleReportQueue, "mark_reviewed", closed_elsewhere)
    response = app.handle("POST", "/accept", SIGNED, form)
    location = dict(response.headers).get("Location", "")
    assert response.status == 303 and "Declared" in location and "Nothing+was+written" not in location
    assert "supersedes" in (reported / "new.md").read_text(encoding="utf-8")


def test_stakes002_an_accept_holds_the_cross_process_memo_write_lock(reported: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STAKES-002: an accept wrote the memo holding only the in-process lock.

    While `apply_rewrite` writes, a second process must be unable to take the reservation every
    dashboard edit takes (edits-lock.sqlite3). Pre-fix the probe takes it, and the assertion fails.
    """
    app = _app(reported)
    form = _accept_form(app, reported, monkeypatch)
    held: list[bool] = []
    original = review.apply_rewrite

    def probing(*args: Any, **kwargs: Any) -> Any:
        lock = reported / ".recall" / "edits-lock.sqlite3"
        probe = sqlite3.connect(str(lock), timeout=0)
        try:
            probe.execute("BEGIN IMMEDIATE")
            held.append(False)
        except sqlite3.OperationalError:
            held.append(True)
        finally:
            probe.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(review, "apply_rewrite", probing)
    response = app.handle("POST", "/accept", SIGNED, form)
    assert response.status == 303
    assert held == [True], "STAKES-002: the accept's write was not serialized with dashboard edits in other processes"


# ------------------------------------------------------------------ the request handler


def test_bug003_a_page_that_raises_still_gets_an_answer(root: Path) -> None:
    """BUG-003: an exception from a page dropped the connection with no response at all.

    Pre-fix the client sees RemoteDisconnected (the defect); after the fix a 500 with a short reason.
    """
    app = DashboardApp(root, port=0, token="tok")

    def broken(*args: object, **kwargs: object) -> None:
        raise PermissionError("the memo is locked by another program")

    app.handle = broken  # type: ignore[method-assign]
    server = serve(app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/")
        answer = conn.getresponse()
        assert answer.status == 500 and b"PermissionError" in answer.read()
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------ links between corpora and memos


def test_dat001_a_served_source_links_the_memo_it_names_and_only_for_this_folder(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DAT-001: sources were mapped to a root-level memo by bare file name, for any tenant.

    View 2 for a memory-tenant source under `sub/` must offer `sub/y.md`, and a docs-tenant source
    named like a local memo must offer no memo at all. Pre-fix the first link is missing (no
    `<root>/y.md`) and the assertion fails.
    """
    (root / "sub").mkdir()
    (root / "sub" / "y.md").write_text("# Y\n", encoding="utf-8")
    (root / "y.md").write_text("# other Y\n", encoding="utf-8")
    monkeypatch.setattr(corpus, "source_detail", lambda db, tenant, file: _detail(
        tenant=tenant, file=file, source_uri="file:///srv/memory/" + file, state="current", superseded_by=None,
    ))
    app = _app(root, db=DashboardDB("postgresql://fake"))
    mine = app.handle("GET", "/corpus/source?tenant=memory&file=sub/y.md", SIGNED).body.decode()
    assert "/read?path=sub%2Fy.md" in mine, "DAT-001: the memo the source names was not offered"
    theirs = app.handle("GET", "/corpus/source?tenant=re-call-docs&file=a.md", SIGNED).body.decode()
    assert "read the memo file" not in theirs, "DAT-001: another corpus's file was linked to a local memo"


def test_code001_view_two_encodes_the_graph_fragment(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CODE-001: View 2's link back to the corpus graph left `%` unencoded in `#memo=`.

    graph.js decodes the fragment with decodeURIComponent, so `100%.md` threw. Pre-fix the raw
    name is in the fragment and the assertion fails.
    """
    monkeypatch.setattr(corpus, "source_detail", lambda db, tenant, file: _detail(file=file, state="current", superseded_by=None))
    app = _app(root, db=DashboardDB("postgresql://fake"))
    body = app.handle("GET", "/corpus/source?tenant=re-call-docs&file=100%25.md", SIGNED).body.decode()
    assert f"#memo={quote('100%.md', safe='')}'" in body, "CODE-001: the graph fragment is not percent-encoded"


def test_dat002_a_date_without_zero_padding_is_compared_as_a_date() -> None:
    """DAT-002: `valid_until: 2026-9-30` sorted after `2026-11-04` as a string, so it was current.

    The trust gate parses it as 30 September. Pre-fix `_state` returns 'current' and this fails.
    """
    graph = SupersessionGraph.from_parts({})
    assert corpus._state("a.md", graph, None, "2026-9-30", "2026-11-04") == "expired"
    assert corpus._state("a.md", graph, "2026-9-1", None, "2026-11-04") == "current"
    assert corpus._state("a.md", graph, "2026-12-1", None, "2026-11-04") == "not_yet_valid"


# ------------------------------------------------------------------ database pages


def test_num001_the_overview_says_when_recorded_searches_pass_the_cap(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NUM-001: the overview showed the length of a list capped at 500 as the total.

    Pre-fix a tenant with more than 500 recorded searches shows `<b>500</b>`, and this fails.
    """
    fake = FakeCorpus()
    fake.searches["memory"] = [{"event_id": f"e{i}"} for i in range(800)]
    fake.install(monkeypatch)
    real = dbq.searches
    monkeypatch.setattr(dbq, "searches", lambda db, tenant, limit=100, contains="": real(db, tenant)[:limit])
    body = _app(root, db=DashboardDB("postgresql://fake")).handle("GET", "/overview?tenant=memory", SIGNED).body.decode()
    assert "<b>500+</b>" in body, "NUM-001: a capped count was shown as the total"


def test_perf001_a_failing_tenant_listing_is_not_retried_on_every_page(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-001: a role that can never list tenants reconnected and failed on every page.

    Pre-fix two pages make two `dbq.tenants` calls and the count assertion fails.
    """
    calls: list[int] = []

    def tenants(db: DashboardDB) -> list[dict[str, Any]]:
        calls.append(1)
        raise dbq.DatabaseUnavailable("permission denied for the tenant listing")

    monkeypatch.setattr(dbq, "tenants", tenants)
    app = _app(root, db=DashboardDB("postgresql://fake"))
    assert app.handle("GET", "/memo?path=a.md", SIGNED).status == 200
    assert app.handle("GET", "/memo?path=a.md", SIGNED).status == 200
    assert len(calls) == 1, f"PERF-001: the failing tenant listing ran {len(calls)} times in its window"
