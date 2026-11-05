"""Regressions for the dashboard findings the second 2026-11-04 hunt deferred, fixed in a third run.

Each behavioural test was run against the pre-fix files (the run's snapshot) and failed in its own
assertion or with the defect's own exception, then passed on the fixed files; the receipts are in
the pull request. Nothing the fixes introduced is imported at module top.

CODE-009 (typed edit changes), CODE-014 (public names), CODE-018 (shared constants, one path
split) and CODE-019 (named display caps) change no behaviour: mypy and the existing suites cover
them, and red proof does not apply.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from recall.dashboard import corpus, edit, review, triage
from recall.dashboard import db as dbq
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp
from recall.rewrite import RewriteRefused
from recall.stale_reports import StaleReportQueue, default_queue_path
from recall.supersession import SupersessionGraph

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
NOW = datetime(2026, 11, 4, 20, 0, tzinfo=UTC)
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


def _accept(root: Path) -> None:
    (item,) = review.build_queue(root).items
    shown = review.memo_digest(root, review.preview(root, item, NOW).edit_file)
    review.accept(root, item, reviewer="ana", note="published", shown_digest=shown, now=NOW)


def test_stakes003_an_accept_is_recorded_as_an_edit_and_can_be_undone(reported: Path) -> None:
    """STAKES-003: an accept wrote `supersedes:` with no record of who, why, or the bytes before."""
    before = (reported / "new.md").read_bytes()
    _accept(reported)
    assert b"supersedes" in (reported / "new.md").read_bytes()
    records = edit.history(reported, "new.md")
    assert records and records[0].editor == "ana" and records[0].note == "published", "STAKES-003: the accept left no record"
    edit.undo_edit(reported, records[0].edit_id, editor="ana", note="wrong call", now=NOW)
    assert (reported / "new.md").read_bytes() == before


def test_an_accept_writes_only_over_the_bytes_the_reviewer_saw(reported: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The digest check ran before the plan and the dry run; the bytes written over were read after.

    A hand edit landing between the two was written over unseen. Here the check is made to pass
    for the version the reviewer saw while the file already holds a later one. Red proof,
    2026-11-05: `recall.dashboard.review._record_accept` without its `sha256(before)` comparison
    wrote the accept (DID NOT RAISE ReviewRefused).
    """
    (item,) = review.build_queue(reported).items
    name = review.preview(reported, item, NOW).edit_file
    shown = review.memo_digest(reported, name)
    later = (reported / name).read_bytes() + b"\nA line written by hand after the review.\n"
    (reported / name).write_bytes(later)
    monkeypatch.setattr(review, "memo_digest", lambda root, memo: shown)  # the check saw the old version
    with pytest.raises(review.ReviewRefused, match="changed since you reviewed it"):
        review.accept(reported, item, reviewer="ana", note="published", shown_digest=shown, now=NOW)
    assert (reported / name).read_bytes() == later, "the accept wrote over bytes the reviewer never saw"
    assert not edit.history(reported, name)


def test_stakes008_showing_a_warning_again_is_a_recorded_named_decision(tmp_path: Path) -> None:
    """STAKES-008: restore deleted a recorded dismissal with no name and no trace, and a no-op said done."""
    ledger = triage.DismissalLedger(triage.default_dismissal_path(tmp_path))
    finding = triage.Finding(file="a.md", sentence="This replaces b.", marker="replaces", sentence_sha="s1",
                             candidates=("b.md",), passive=False, index_page=False, dismissed=False)
    ledger.record(finding, reviewer="ana", note="not a replacement", at=NOW)
    with pytest.raises(triage.TriageRefused):
        triage.restore(tmp_path, "a.md", "s1", reviewer=" ", note="", now=NOW)
    assert triage.restore(tmp_path, "a.md", "s1", reviewer="bo", note="it is one", now=NOW) is True
    with sqlite3.connect(triage.default_dismissal_path(tmp_path)) as conn:
        kept = conn.execute("SELECT dismissed_by, restored_by, restored_note FROM restorations").fetchall()
    assert kept == [("ana", "bo", "it is one")]
    assert triage.restore(tmp_path, "a.md", "s1", reviewer="bo", note="", now=NOW) is False


def test_perf007_the_home_page_does_not_ask_for_retrieval_counts(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-007: every GET / ran the whole-ledger retrieval expansion and discarded it."""
    asked: list[dict[str, Any]] = []

    def control(db: DashboardDB, tenant: str, **kwargs: Any) -> dict[str, Any]:
        asked.append(kwargs)
        return {"summary": {"searches": 0, "answered": 0, "reports": 0, "helped": 0}, "memos": []}

    monkeypatch.setattr(dbq, "control", control)
    monkeypatch.setattr(dbq, "tenants", lambda db: [])
    DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake")).handle("GET", "/", SIGNED)
    assert asked == [{"retrievals": False}], f"PERF-007: the home page asked for {asked}"


def test_bug006_dates_search_cannot_read_are_shown_as_such() -> None:
    """BUG-006: a malformed date or an empty window was drawn current or expired; search says invalid."""
    graph = SupersessionGraph.from_parts({})
    assert corpus._state("a.md", graph, "soon", "2026-9-30", "2026-11-04") == "invalid_metadata"
    assert corpus._state("a.md", graph, "2026-12-01", "2026-01-01", "2026-11-04") == "invalid_metadata"
    assert corpus._state("a.md", graph, None, "2026-9-30", "2026-11-04") == "expired"


def test_code008_a_declaration_is_read_only_inside_the_folder(tmp_path: Path) -> None:
    """CODE-008: already_declared read `root / name` with no containment check, unlike the other readers."""
    root = tmp_path / "memory"
    root.mkdir()
    (tmp_path / "outside.md").write_text("---\nsupersedes: old\n---\n# Outside\n", encoding="utf-8")
    assert review.already_declared(root, "old.md", "../outside.md") is False


def test_code008_the_readers_take_only_names_the_memo_list_has(tmp_path: Path) -> None:
    """CODE-008: `.recall/x.md` or a `.txt` passed memo_digest and already_declared, not memo_files."""
    (tmp_path / ".recall").mkdir()
    declares = "---\nsupersedes: old\n---\n# X\n"
    (tmp_path / ".recall" / "x.md").write_text(declares, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(declares, encoding="utf-8")
    for name in (".recall/x.md", "notes.txt"):
        assert review.already_declared(tmp_path, "old.md", name) is False, f"CODE-008: {name} read as a memo"
        with pytest.raises(review.ReviewRefused):
            review.memo_digest(tmp_path, name)


def test_stakes008_the_page_needs_a_name_to_show_a_warning_again(root: Path) -> None:
    """STAKES-008 at the page: a restore without a name is refused, not silently applied.

    A dismissal is recorded first, so the 409 can only come from the missing name and not from
    there being nothing to restore (the review of the fix found the unseeded form could not fail).
    """
    finding = triage.Finding(file="a.md", sentence="This replaces b.", marker="replaces", sentence_sha="s1",
                             candidates=("b.md",), passive=False, index_page=False, dismissed=False)
    triage.DismissalLedger(triage.default_dismissal_path(root)).record(finding, reviewer="ana", note="", at=NOW)
    app = DashboardApp(root, port=8765, token="tok")
    secret = str(getattr(app, "csrf", app.token))
    response = app.handle("POST", "/lint/restore", SIGNED, urlencode({"file": "a.md", "sha": "s1", "csrf": secret}).encode())
    assert response.status == 409
