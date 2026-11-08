"""Phase 1 of resolving `supersedes:` references at write time: nothing is lost on the dashboard.

Design: recall-lab `research/designs/2026-11-05-write-time-supersedes-references.md`, owner decision
2026-11-05. Each test failed against the pre-change files in its own assertion, then passed
(recorded in the pull request):
- R1 (DAT-005) `review.already_declared` comparing file names only: the claim about `b/notes.md`
  was hidden ("a claim search leaves open was hidden as already decided").
- R2 `graph.build_graph` keeping `current` for unreadable dates, and overwriting the state with
  `superseded`: "a memo search refuses was drawn as current".
- R3 the corpus graph counting by the chain's last memo: the note said 2 for a -> b -> c with c
  forgotten, where only b has no drawn successor. (The second case, b forgotten, where the old rule
  gave no note, was not seen red separately: the first assertion stopped the test.)
- R4 the Health tile's label: "past their valid_until" counted only memos that are not replaced.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard import graph as graph_mod
from recall.dashboard import review
from recall.dashboard.server import DashboardApp
from recall.rewrite import RewriteRefused
from recall.stale_reports import StaleReportQueue, default_queue_path
from tests.test_dashboard_corpus import _file, _graph

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
OLD = "the deploy host for staging is alpha"
NEW = "the deploy host for staging is now gamma"
SIGNED = {"Host": "127.0.0.1:8765", "Authorization": "Bearer tok"}


@pytest.fixture
def shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`a/notes.md` and `b/notes.md`, and `new.md` declaring `supersedes: a/notes.md`."""

    def no_arbiter(*args: object, **kwargs: object) -> tuple[()]:
        raise RewriteRefused("the supersession arbiter is off")

    monkeypatch.setattr(review, "corpus_proposals", no_arbiter)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.md").write_text(f"A: {OLD}.\n", encoding="utf-8")
    (tmp_path / "b" / "notes.md").write_text(f"B: {OLD}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"---\nsupersedes: a/notes.md\n---\nNew: {NEW}.\n", encoding="utf-8")
    return tmp_path


def _report(root: Path, stale: str) -> None:
    with StaleReportQueue(default_queue_path(root)) as queue:
        queue.submit(root, stale_source=stale, replacing_source="new.md", stale_quote=OLD, current_quote=NEW,
                     client="mcp:test", task=None, reported_at=NOW)


def test_r1_a_claim_search_leaves_open_is_shown_and_can_be_rejected(shared: Path) -> None:
    _report(shared, "b/notes.md")
    (item,) = review.build_queue(shared).items or (None,)
    assert item is not None and item.stale == "b/notes.md", "a claim search leaves open was hidden as already decided"
    assert any(line.startswith("ambiguous name: 2 memos are called notes.md") for line in item.details)
    review.reject(shared, item, reviewer="ana", note="b is a different note", now=NOW)
    assert not review.build_queue(shared).items


def test_r1_a_claim_the_newer_memo_already_declares_stays_hidden(shared: Path) -> None:
    """What was declared still closes its claim (green before and after), and a declaration of
    `a/notes.md` does not declare `b/notes.md` (red before the change: `assert not True`)."""
    _report(shared, "a/notes.md")
    assert not review.build_queue(shared).items
    (shared / "old.md").write_text(f"Old: {OLD}.\n", encoding="utf-8")
    (shared / "newer.md").write_text("---\nsupersedes: old\n---\nNewer.\n", encoding="utf-8")
    assert review.already_declared(shared, "old.md", "newer.md"), "a unique bare name stopped counting as declared"
    assert not review.already_declared(shared, "b/notes.md", "new.md")


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_r2_dates_search_cannot_read_are_shown_as_such_even_when_replaced(tmp_path: Path) -> None:
    _write(tmp_path, "bad.md", "---\nvalid_from: 2026-12-01\nvalid_until: 2026-01-01\n---\n# Bad window\n")
    _write(tmp_path, "worse.md", "---\nvalid_until: someday\n---\n# Unreadable\n")
    _write(tmp_path, "new.md", "---\nsupersedes: worse\n---\n# New\n")
    built = graph_mod.build_graph(tmp_path, today=NOW, include_queue=False, lint=False)
    states = {n["id"]: n["state"] for n in built["nodes"]}
    assert states["bad.md"] == "invalid_metadata", "a memo search refuses was drawn as current"
    assert states["worse.md"] == "invalid_metadata", "a memo search refuses was drawn as superseded"
    assert built["counts"]["invalid_metadata"] == 2


def test_r3_the_note_counts_the_files_with_no_drawn_successor() -> None:
    files = [_file("a.md"), _file("b.md")]
    end_forgotten = _graph(files, winner={"a.md": "b.md", "b.md": "c.md"})
    note = next((n for n in end_forgotten["notes"] if "forgotten" in n), "")
    assert note.startswith("1 "), f"the note counted a file whose successor is drawn: {note}"
    middle_forgotten = _graph([_file("a.md"), _file("c.md")], winner={"a.md": "b.md", "b.md": "c.md"})
    note = next((n for n in middle_forgotten["notes"] if "forgotten" in n), "")
    assert note.startswith("1 "), "a superseded file with no drawn successor went unexplained"


def test_r4_the_expired_tile_says_it_leaves_out_replaced_memos(tmp_path: Path) -> None:
    _write(tmp_path, "a.md", "# A\n")
    body = DashboardApp(tmp_path, port=8765, token="tok").handle("GET", "/health", SIGNED).body.decode()
    assert re.search(r"past their valid_until, not replaced", body), "the tile claimed every expired memo"
