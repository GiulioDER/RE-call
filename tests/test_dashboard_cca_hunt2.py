"""Regressions from the second 2026-11-04 DEEP CCA hunt of the dashboard (the deferred pool, re-verified).

One test per behavioural fix, named after its finding. Each was run against the pre-fix files (the
run's snapshot) and failed in its own assertion or with the defect's own exception, then passed on
the fixed files; the receipts are in the pull request. Nothing the fixes introduced is imported at
module top, so a pre-fix run reaches the behaviour: the form secret is read as
`getattr(app, "csrf", app.token)`, which is the token itself before the fix.

The documentation and refactoring fixes of the same run (DOC-001 to DOC-006, CODE-003, CODE-004,
CODE-005, CODE-006, CODE-012, CODE-013, CODE-015, CODE-017) change no behaviour and are covered by
the existing suites; red proof does not apply to them.
"""

from __future__ import annotations

import os
import pathlib
import re
import sqlite3
import threading
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from recall.dashboard import corpus, edit, graph, review
from recall.dashboard import db as dbq
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp
from recall.rewrite import RewriteRefused
from recall.stale_reports import StaleReportQueue, default_queue_path
from tests.conftest import TEST_DSN, requires_db
from tests.test_dashboard_corpus import TODAY, _detail, built_copies  # noqa: F401
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


@pytest.fixture
def reported(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def no_arbiter(*args: object, **kwargs: object) -> tuple[()]:
        raise RewriteRefused("the supersession arbiter is off")

    monkeypatch.setattr(review, "corpus_proposals", no_arbiter)
    (root / "old.md").write_text(f"The release is 0.12.0 and {OLD_QUOTE}.\n", encoding="utf-8")
    (root / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}.\n", encoding="utf-8")
    with StaleReportQueue(default_queue_path(root)) as queue:
        queue.submit(
            root, stale_source="old.md", replacing_source="new.md", stale_quote=OLD_QUOTE,
            current_quote=NEW_QUOTE, client="mcp:acme", task="which release?", reported_at=NOW,
        )
    return root


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


def _accept_form(app: DashboardApp, root: Path) -> bytes:
    (item,) = app._queue().items
    shown = review.memo_digest(root, review.preview(root, item, NOW).edit_file)
    return urlencode({"csrf": _secret(app), "claim": item.claim, "reviewer": "p", "note": "n", "digest": shown}).encode()


def _app(root: Path, **kwargs: Any) -> DashboardApp:
    return DashboardApp(root, port=8765, token="tok", **kwargs)


def _secret(app: DashboardApp) -> str:
    return str(getattr(app, "csrf", app.token))


def _preview(app: DashboardApp, fields: list[tuple[str, str]]) -> str:
    return app.handle("POST", "/memo/preview", SIGNED, urlencode([("csrf", _secret(app)), *fields]).encode()).body.decode()


def _hidden(page: str, name: str) -> str:
    found = re.search(rf"name='{name}' value='([^']*)'", page)
    return found.group(1) if found else ""


# ------------------------------------------------------------------ cookies, tokens and hosts


def test_bug001_a_foreign_cookie_on_localhost_does_not_lock_the_person_out(root: Path) -> None:
    """BUG-001: SimpleCookie raised on `x@y` and dropped the whole header at a key-only cookie."""
    app = _app(root)
    for header in ("x@y=1; recall_dashboard_8765=tok", "flag; recall_dashboard_8765=tok", "a=hello world; recall_dashboard_8765=tok"):
        assert app.handle("GET", "/", {**HOST, "Cookie": header}).status == 200, header


def test_env011_a_name_outside_latin1_never_breaks_the_reply_after_a_write(root: Path) -> None:
    """ENV-011: the reviewer cookie carried 'Łukasz' raw, which http.server cannot send."""
    (root / "b.md").write_text("# B\n\nnote\n", encoding="utf-8")
    app = _app(root)
    fields = [("path", "b.md"), ("status", "draft")]
    page = _preview(app, fields)
    form = [("csrf", _secret(app)), ("shown_sha", _hidden(page, "shown_sha")), ("shown_after", _hidden(page, "shown_after")),
            ("editor", "Łukasz"), ("note", "n"), *fields]
    response = app.handle("POST", "/memo/apply", SIGNED, urlencode(form).encode())
    assert response.status == 303
    for _name, value in response.headers:
        value.encode("latin-1")  # what http.server's send_header does; raises on the defect


def test_env014_a_tenant_cookie_from_another_dashboard_is_not_applied(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-014: the unscoped `recall_tenant` from an older dashboard still chose this one's tenant."""
    fake = FakeCorpus()
    fake.install(monkeypatch)
    app = _app(root, db=DashboardDB("postgresql://fake"))
    app.handle("GET", "/retrieval", {**HOST, "Cookie": "recall_dashboard=tok; recall_tenant=re-call-code-gen"})
    assert ("searches", "memory") in fake.asked and ("searches", "re-call-code-gen") not in fake.asked


def test_env012_port_80_answers_a_host_without_a_port(root: Path) -> None:
    """ENV-012: browsers send `Host: 127.0.0.1` for port 80, and the allowlist refused it."""
    app = DashboardApp(root, port=80, token="tok")
    assert app.handle("GET", "/?token=tok", {"Host": "127.0.0.1"}).status == 303


def test_env016_a_tenant_the_page_would_ignore_gets_no_link(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-016: `team/alpha` was linked, and picking it silently showed the default tenant."""
    monkeypatch.setattr(dbq, "tenants", lambda db: [{"tenant": "team/alpha", "generation": "g"}, {"tenant": "memory", "generation": "g"}])
    body = _app(root, db=DashboardDB("postgresql://fake")).handle("GET", "/memo?path=a.md", SIGNED).body.decode()
    assert "team%2Falpha" not in body


def test_sec001_the_session_token_alone_does_not_pass_the_form_check(root: Path) -> None:
    """SEC-001: the cookie value was the form secret, and other 127.0.0.1 services receive the cookie.

    The second half pins that the secret is independent of the token, not computed from it: two
    dashboards with the same token must have different form secrets (a function of the token alone,
    such as an HMAC under a public constant, would give both the same one).
    """
    app = _app(root)
    form = urlencode({"csrf": app.token, "file": "a.md", "sha": "x"}).encode()
    assert app.handle("POST", "/lint/restore", SIGNED, form).status == 403
    assert _secret(_app(root)) != _secret(app), "SEC-001: the form secret is a function of the session token"


def test_control_an_error_inside_an_accept_is_not_reported_as_a_lock_failure(reported: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control for the accept's reservation (not a red proof: the pre-fix code also propagated this).

    An OSError raised while the accept writes must reach the caller as itself, never be relabelled as
    "the memo write lock cannot be taken" (a 409 that invites a blind retry).
    """
    (item,) = review.build_queue(reported).items
    shown = review.memo_digest(reported, review.preview(reported, item, NOW).edit_file)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(review, "apply_rewrite", broken)
    with pytest.raises(OSError, match="disk full"):
        review.accept(reported, item, reviewer="p", note="n", shown_digest=shown, now=NOW)


def test_bug009_a_cached_tenant_failure_is_raised_fresh(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-009: one stored exception was re-raised each time, its traceback growing per request."""

    def down(db: DashboardDB) -> list[dict[str, Any]]:
        raise dbq.DatabaseUnavailable("permission denied")

    monkeypatch.setattr(dbq, "tenants", down)
    app = _app(root, db=DashboardDB("postgresql://fake"))
    raised = []
    for _ in range(2):
        with pytest.raises(dbq.DatabaseUnavailable) as caught:
            app._tenants()
        raised.append(caught.value)
    assert raised[0] is not raised[1], "BUG-009: the same exception instance was re-raised"


def test_deploy001_a_database_page_says_what_the_database_said(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DEPLOY-001: a missing table or grant was reported only as 'did not answer'."""

    def broken(db: DashboardDB) -> dict[str, Any]:
        raise dbq.DatabaseUnavailable("UndefinedTable: relation recall_graph_relations_v1 does not exist")

    monkeypatch.setattr(dbq, "availability", broken)
    body = _app(root, db=DashboardDB("postgresql://fake")).handle("GET", "/overview", SIGNED).body.decode()
    assert "UndefinedTable" in body


# ------------------------------------------------------------------ memo writes


def test_stakes001_a_write_that_may_have_landed_is_never_called_nothing_written(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STAKES-001: the memo held the new bytes after a failed restore, and the page said nothing was written."""
    (root / "b.md").write_text("# B\n\nnote\n", encoding="utf-8")
    app = _app(root)
    fields = [("path", "b.md"), ("status", "draft")]
    page = _preview(app, fields)
    writes: list[int] = []
    real_write = edit.atomic_write_bytes

    def write_once(path: Path, data: bytes) -> None:
        writes.append(1)
        if len(writes) > 1:
            raise OSError("disk full")
        real_write(path, data)

    def fail(self: Any, operation_id: str) -> int:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(edit, "atomic_write_bytes", write_once)
    monkeypatch.setattr(edit.EditLedger, "_complete", fail)
    form = [("csrf", _secret(app)), ("shown_sha", _hidden(page, "shown_sha")), ("shown_after", _hidden(page, "shown_after")),
            ("editor", "p"), ("note", "n"), *fields]
    body = app.handle("POST", "/memo/apply", SIGNED, urlencode(form).encode()).body.decode()
    assert "status: draft" in (root / "b.md").read_text(encoding="utf-8")
    assert "Nothing was written" not in body, "STAKES-001: a landed write was reported as nothing written"


def test_stakes002_an_accept_refuses_a_memo_with_an_unreconcilable_interrupted_edit(reported: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STAKES-002: an accept wrote the memo without reconciling the edit ledger, leaving the row stuck for good."""
    app = _app(reported)
    form = _accept_form(app, reported)
    _stick(reported, "new.md")
    before = (reported / "new.md").read_bytes()
    response = app.handle("POST", "/accept", SIGNED, form)
    assert response.status == 409 and (reported / "new.md").read_bytes() == before


def test_stakes006_a_reject_after_the_claim_was_declared_is_refused(reported: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STAKES-006: a reject racing an accept recorded a decline beside a written declaration."""
    (item,) = review.build_queue(reported).items
    shown = review.memo_digest(reported, review.preview(reported, item, NOW).edit_file)
    review.accept(reported, item, reviewer="p", note="n", shown_digest=shown, now=NOW)
    with pytest.raises(review.ReviewRefused):
        review.reject(reported, item, reviewer="q", note="m", now=NOW)
    assert item.claim not in review.rejected_claims(reported)


def test_stakes004_the_editor_refuses_a_file_that_is_not_a_memo(root: Path) -> None:
    """STAKES-004: /memo opened, and /memo/apply wrote front matter into, any file under the root."""
    (root / "notes.txt").write_text("plain text\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config.md").write_text("[core]\n", encoding="utf-8")
    app = _app(root)
    assert app.handle("GET", "/memo?path=notes.txt", SIGNED).status == 404
    assert app.handle("GET", "/memo?path=.git/config.md", SIGNED).status == 404


def test_stakes005_apply_refuses_bytes_the_diff_did_not_show(root: Path) -> None:
    """STAKES-005: apply checked only the before bytes, so a reference re-resolved meanwhile was written unseen."""
    (root / "b.md").write_text("# B\n\nnote\n", encoding="utf-8")
    app = _app(root)
    fields = [("path", "b.md"), ("status", "draft")]
    page = _preview(app, fields)
    form = [("csrf", _secret(app)), ("shown_sha", _hidden(page, "shown_sha")), ("shown_after", "0" * 64),
            ("editor", "p"), ("note", "n"), *fields]
    assert app.handle("POST", "/memo/apply", SIGNED, urlencode(form).encode()).status == 409
    assert "status" not in (root / "b.md").read_text(encoding="utf-8")


def test_stakes007_the_memo_page_says_its_edits_are_refused(root: Path) -> None:
    """STAKES-007: a stuck interrupted edit was invisible until an apply failed."""
    _stick(root, "a.md")
    body = _app(root).handle("GET", "/memo?path=a.md", SIGNED).body.decode()
    assert "interrupted edit" in body, "STAKES-007: the refusal was not shown on the memo page"


def test_dat001_a_folder_qualified_reference_names_one_memo(root: Path) -> None:
    """DAT-001: `b/notes` was refused as ambiguous when a/notes.md existed, and a missing path was rewritten."""
    for folder in ("a", "b"):
        (root / folder).mkdir()
        (root / folder / "notes.md").write_text("# n\n", encoding="utf-8")
    plan = edit.plan_edit(root, "a.md", add_supersedes=("b/notes",))
    assert "b/notes.md" in plan.after.decode("utf-8")
    with pytest.raises(edit.EditRefused):
        edit.plan_edit(root, "a.md", add_supersedes=("c/notes",))


def test_bug003_a_refused_withdrawal_is_not_called_a_bad_edit_id(root: Path) -> None:
    """BUG-003: a non-UTF-8 memo made 'stop superseding' answer 'That edit id is not a number.'"""
    (root / "old.md").write_text("# Old\n", encoding="utf-8")
    (root / "b.md").write_bytes(b"---\nsupersedes: old\n---\n# B caf\xe9\n")
    app = _app(root)
    response = app.handle("POST", "/memo/preview", SIGNED, urlencode([("csrf", _secret(app)), ("path", "b.md"), ("remove", "old")]).encode())
    assert "edit id" not in response.body.decode()


def test_perf005_history_does_not_wait_behind_a_decision(root: Path) -> None:
    """PERF-005: reading history took the decision lock, so GET /memo waited behind every accept."""
    (root / "b.md").write_text("# B\n\nnote\n", encoding="utf-8")
    plan = edit.plan_edit(root, "b.md", status="draft")
    edit.apply_edit(root, "b.md", editor="p", note="n", shown_sha=plan.before_sha, now=NOW, status="draft")
    done: list[int] = []
    with review._DECISION_LOCK:
        reader = threading.Thread(target=lambda: done.append(len(edit.history(root))), daemon=True)
        reader.start()
        reader.join(3)
    assert done == [1], "PERF-005: the history read blocked on the decision lock"


# ------------------------------------------------------------------ the review queue


def test_sec003_a_synced_report_naming_an_absolute_path_is_not_queued(reported: Path) -> None:
    """SEC-003: names read back from the sidecar reached filesystem calls with no anchored check."""
    with sqlite3.connect(default_queue_path(reported)) as conn:
        conn.execute("UPDATE stale_reports SET replacing_source = '//attacker.example/share/new.md'")
    queue = review.build_queue(reported, arbiter_proposals=lambda r: ())
    assert queue.items == (), "SEC-003: an absolute-path report was queued"


def test_bug004_a_name_stand_in_with_a_nul_does_not_break_the_queue(root: Path) -> None:
    """BUG-004: opening a stand-in name (it holds a NUL) raised ValueError out of build_queue."""
    assert review.already_declared(root, "a.md", "\x00name:x.md") is False


def test_num007_a_row_naming_one_memo_twice_is_not_called_elsewhere(root: Path) -> None:
    """NUM-007: a row whose two sources map to one local memo was counted as naming a memo not here."""
    rows = [{"stale_source": "recall/a.md", "replacing_source": "other/a.md"}]
    _items, notes = review.database_items(root, rows)
    assert not any("does not hold" in n for n in notes), notes


def test_perf002_a_decided_claim_is_checked_once_however_many_rows_repeat_it(root: Path) -> None:
    """PERF-002: `closed` (a memo read and parse) ran once per row; rows repeat daily."""
    (root / "b.md").write_text("# B\n", encoding="utf-8")
    calls: list[str] = []
    rows = [{"stale_source": "a.md", "replacing_source": "b.md"}] * 5
    review.database_items(root, rows, closed=lambda key, s, r: calls.append(key) is None)
    assert len(calls) == 1


# ------------------------------------------------------------------ pages and graphs


def test_bug002_one_unreadable_memo_does_not_break_the_graph(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-002: build_graph read every memo with no OSError handling."""
    (root / "b.md").write_text("# B\n", encoding="utf-8")
    real = pathlib.Path.read_text

    def locked(self: pathlib.Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "b.md":
            raise PermissionError("locked")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", locked)
    ids = {n["id"] for n in graph.build_graph(root, include_queue=False, lint=False)["nodes"]}
    assert "a.md" in ids


def test_bug007_find_shows_the_match_after_case_expanding_characters(root: Path) -> None:
    """BUG-007: the offset came from text.lower(), which 'İ' lengthens, so the snippet missed the match."""
    (root / "b.md").write_text("İ" * 200 + " needle here\n", encoding="utf-8")
    body = _app(root).handle("GET", "/find?q=needle", SIGNED).body.decode()
    assert "needle here" in body


@pytest.mark.skipif(os.name == "nt", reason="only POSIX file names can be bytes that are not UTF-8")
def test_dat002_a_memo_name_that_is_not_utf8_does_not_break_the_graph(root: Path) -> None:
    """DAT-002: a surrogate-escaped name went raw into the graph JSON, and its UTF-8 encode raised."""
    with open(os.fsencode(str(root)) + b"/caf\xe9.md", "wb") as handle:
        handle.write(b"# cafe\n")
    response = _app(root).handle("GET", "/api/graph.json", SIGNED)
    assert response.status == 200


def test_num001_the_control_page_never_shows_reports_as_a_share_of_searches(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NUM-001: reports per task divided by answered searches read '150% of answered searches'."""
    FakeCorpus().install(monkeypatch)
    summary = {"searches": 2, "answered": 2, "reports": 3, "memos_retrieved": 0, "helped": 0, "no_difference": 0,
               "misled": 0, "succeeded": 0, "failed": 0}
    monkeypatch.setattr(dbq, "control", lambda *a, **k: {"summary": summary, "memos": []})
    body = _app(root, db=DashboardDB("postgresql://fake")).handle("GET", "/control", SIGNED).body.decode()
    assert "150%" not in body


def test_num002_the_review_card_marks_evidence_past_the_display_cut(root: Path) -> None:
    """NUM-002: the card cut the memo at 20,000 characters before looking for the quote."""
    (root / "long.md").write_text("x " * 15_000 + "\nthe evidence sentence\n", encoding="utf-8")
    card = _app(root)._memo_card("older", "long.md", "the evidence sentence")
    assert "<mark>the evidence sentence</mark>" in card


def test_num011_a_memo_not_yet_in_force_is_not_drawn_current(root: Path) -> None:
    """NUM-011: the memo graph ignored valid_from, while search and the corpus graph say not_yet_valid."""
    (root / "future.md").write_text("---\nvalid_from: 2999-01-01\n---\n# F\n", encoding="utf-8")
    nodes = {n["id"]: n for n in graph.build_graph(root, include_queue=False, lint=False)["nodes"]}
    assert nodes["future.md"]["state"] == "not_yet_valid"


def test_perf001_the_home_page_builds_the_review_queue_once(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-001: the home page built the queue itself and again inside build_graph."""
    calls: list[int] = []
    real = review.build_queue

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, arbiter_proposals=lambda r: (), **{k: v for k, v in kwargs.items() if k != "arbiter_proposals"})

    monkeypatch.setattr(review, "build_queue", counted)
    monkeypatch.setattr(graph, "build_queue", counted)
    assert _app(root).handle("GET", "/", SIGNED).status == 200
    assert len(calls) == 1


def test_perf003_the_health_page_lints_the_corpus_once(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-003: /health linted the whole corpus in build_graph and again for triage."""
    import recall.lint
    from recall.dashboard import triage

    calls: list[int] = []
    real = recall.lint.lint_corpus

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    for module in (recall.lint, graph, triage):
        monkeypatch.setattr(module, "lint_corpus", counted)
    monkeypatch.setattr(review, "corpus_proposals", lambda *a, **k: ())
    assert _app(root).handle("GET", "/health", SIGNED).status == 200
    assert len(calls) == 1


def test_perf006_the_corpus_graph_cache_is_bounded(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-006: one cached body per corpus slug, never evicted."""
    monkeypatch.setattr(corpus, "corpus_graph", lambda db, tenant: {"nodes": [], "edges": [], "counts": {}, "notes": [], "tenant": tenant})
    app = _app(root, db=DashboardDB("postgresql://fake"))
    for i in range(12):
        app.handle("GET", f"/api/graph.json?corpus=t{i}", SIGNED)
    assert len(app._corpus_graphs) <= 8


def test_dat007_view_two_offers_no_local_memo_for_a_path_several_roots_hold(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DAT-007: the other root's copy linked 'read the memo file' to this folder's memo."""
    monkeypatch.setattr(corpus, "source_detail", lambda db, tenant, file: _detail(
        tenant=tenant, file="file:///srv/b/a.md", source_uri="file:///srv/b/a.md", state="current", superseded_by=None, collided=True,
    ))
    body = _app(root, db=DashboardDB("postgresql://fake")).handle("GET", "/corpus/source?tenant=memory&file=file:///srv/b/a.md", SIGNED).body.decode()
    assert "read the memo file" not in body


def test_bug008_a_relative_import_climbing_past_the_tree_resolves_to_nothing() -> None:
    """BUG-008: the negative slice bound wrapped, so `from .....x import y` drew a wrong edge."""
    files = {"a/b/c/d.py", "a/b/x.py"}
    assert corpus.resolve_imports([("a/b/c/d.py", ".....x", "y", None)], files) == []


def test_dat003_corpus_nodes_carry_iso_validity_dates() -> None:
    """DAT-003: raw `2026-9-30` reached graph.js, which compares the dates as strings."""
    files = [("a.md", 1, None, None, None, "2026-9-30", None)]
    built = corpus.assemble_graph(files, {}, [], [], [], today="2026-10-05", tenant="memory", generation="g")
    assert built["nodes"][0]["valid_until"] == "2026-09-30"


def test_perf002_matching_is_unchanged_by_the_index() -> None:
    """Control for PERF-002: the indexed matcher returns what the linear one did (longest trailing path)."""
    names = ["x.md", "recall/x.md", "deep/recall/y.md"]
    assert review.local_memo("file:///srv/memory/recall/x.md", names) == "recall/x.md"
    assert review.local_memo("/srv/x.md", names) == "x.md"
    assert review.local_memo("recall/y.md", names) is None
    assert review.local_memo("a/deep/recall/y.md", names) == "deep/recall/y.md"


@requires_db
def test_bug005_one_copy_of_a_shared_path_lists_no_path_keyed_relations(built_copies: Any) -> None:  # noqa: F811
    """BUG-005: View 2 for one copy listed relations and importers keyed by the path both copies hold."""
    tenant, first, _second = built_copies
    detail = corpus.source_detail(DashboardDB(TEST_DSN), tenant, first, today=date.fromisoformat(TODAY))
    assert detail is not None
    assert detail.get("collided") is True, "BUG-005: a copy of a shared path was not marked"
    assert detail["relations"] == [] and detail["imported_by"] == []
