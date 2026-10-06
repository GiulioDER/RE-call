"""Triage of `closure-marker-unlinked` on the Health page: offer the edge, never write it; record "not one".

Invariants and the failure each one catches:
- T1 the suggested edge follows the marker's voice: "X replaces Y" goes on X, "X is replaced by
  Y" goes on Y. Backwards, it would bury the live memo under the one it replaced.
- T2 an index page gets no edge suggestion: its line describes the memo it links to.
- T3 "Not a supersession" silences that SENTENCE: the warning leaves the page and the graph's
  counts, and comes back when the sentence is edited.
- T4 a dismissal of a sentence that changed since it was shown is refused, as is one without a
  reviewer, and nothing is recorded.
- T5 memo prose shown on the page is escaped, and the editor prefill is escaped too.

Red proof, 2026-10-07, each mutation alone, failing in the named assertion (JUnit XML), then
restored and green:
- V1 (T1) `recall/dashboard/server.py` `_triage_section` ignoring `f.passive`: the passive memo's
  suggestion put the edge on the wrong file.
- V2 (T2) `recall/dashboard/triage.py` `index_page=` forced False: the finding was not marked an
  index page (`assert False`), before the page check is reached.
- V3 (T3) `closure_findings` matching a dismissal by file only: the edited sentence stayed hidden.
- V4 (T3) `recall/dashboard/graph.py` not skipping silenced files: the issue count stayed 1.
- V5 (T4) the changed-sentence refusal in `triage.dismiss` removed: `DID NOT RAISE TriageRefused`.
- V6 (T5) the sentence rendered without `highlight`: `<script>` reached the page.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest

from recall.dashboard import triage
from recall.dashboard.graph import build_graph
from recall.dashboard.server import DashboardApp

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "old-port.md").write_text("# Old\nThe port is 8080.\n", encoding="utf-8")
    (tmp_path / "new-port.md").write_text("# New\nThis memo replaces [[old-port]] for the port.\n", encoding="utf-8")
    (tmp_path / "stale-plan.md").write_text("# Plan\nThis plan is replaced by [[new-plan]] since May.\n", encoding="utf-8")
    (tmp_path / "new-plan.md").write_text("# New plan\nThe current plan.\n", encoding="utf-8")
    return tmp_path


def _finding(store: Path, name: str) -> triage.Finding:
    return next(f for f in triage.closure_findings(store) if f.file == name)


def _page(store: Path) -> str:
    return DashboardApp(store, port=8765, token="tok").handle("GET", "/health", SIGNED).body.decode()


def test_the_suggested_edge_follows_the_markers_voice(store: Path) -> None:
    page = _page(store)
    assert "Declare: new-port.md supersedes old-port.md" in page
    assert "Declare: new-plan.md supersedes stale-plan.md" in page, "a passive marker put the edge on the replaced memo"
    assert "Declare: stale-plan.md supersedes new-plan.md" not in page


def test_an_index_page_gets_no_edge_suggestion(store: Path) -> None:
    links = "".join(f"- [Memo {i}](memo-{i}.md)\n" for i in range(triage.INDEX_LINKS))
    for i in range(triage.INDEX_LINKS):
        (store / f"memo-{i}.md").write_text(f"# Memo {i}\nText.\n", encoding="utf-8")
    (store / "index.md").write_text(f"# Index\n- [Policy](new-port.md) supersedes the old port note.\n{links}", encoding="utf-8")
    assert _finding(store, "index.md").index_page
    assert "Declare: index.md supersedes" not in _page(store), "an index page was offered an edge"


def test_not_a_supersession_silences_that_sentence_until_it_changes(store: Path) -> None:
    shown = _finding(store, "new-port.md")
    before = build_graph(store)["counts"]["issues"]
    triage.dismiss(store, "new-port.md", shown.sentence_sha, reviewer="giulio", note="only the wording", now=NOW)
    assert _finding(store, "new-port.md").dismissed
    assert build_graph(store)["counts"]["issues"] == before - 1, "a dismissed warning still counted"
    (store / "new-port.md").write_text("# New\nThis memo replaces [[old-port]] for the port and the host.\n", encoding="utf-8")
    assert not _finding(store, "new-port.md").dismissed, "an edited sentence stayed silenced"


def test_a_stale_or_anonymous_dismissal_is_refused(store: Path) -> None:
    shown = _finding(store, "new-port.md")
    (store / "new-port.md").write_text("# New\nThis memo replaces [[old-port]] entirely.\n", encoding="utf-8")
    with pytest.raises(triage.TriageRefused, match="changed since the page was shown"):
        triage.dismiss(store, "new-port.md", shown.sentence_sha, reviewer="giulio", note="", now=NOW)
    current = _finding(store, "new-port.md")
    with pytest.raises(triage.TriageRefused, match="reviewer"):
        triage.dismiss(store, "new-port.md", current.sentence_sha, reviewer="  ", note="", now=NOW)
    assert not _finding(store, "new-port.md").dismissed


def test_prose_and_the_prefill_are_escaped(store: Path) -> None:
    (store / "x.md").write_text("# X\n<script>alert(1)</script> This replaces [[old-port]].\n", encoding="utf-8")
    page = _page(store)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    app = DashboardApp(store, port=8765, token="tok")
    editor = app.handle("GET", "/memo?" + urlencode({"path": "new-port.md", "add": "'><b>x</b>"}), SIGNED).body.decode()
    assert "'><b>x</b>" not in editor and "&#x27;&gt;&lt;b&gt;x&lt;/b&gt;" in editor


def test_the_dismiss_form_needs_the_token(store: Path) -> None:
    shown = _finding(store, "new-port.md")
    app = DashboardApp(store, port=8765, token="tok")
    fields = {"file": "new-port.md", "sha": shown.sentence_sha, "reviewer": "giulio"}
    assert app.handle("POST", "/lint/dismiss", SIGNED, urlencode(fields).encode()).status == 403
    assert not _finding(store, "new-port.md").dismissed
    done = app.handle("POST", "/lint/dismiss", SIGNED, urlencode({**fields, "csrf": "tok"}).encode())
    assert done.status == 303 and _finding(store, "new-port.md").dismissed
    app.handle("POST", "/lint/restore", SIGNED, urlencode({"file": "new-port.md", "sha": shown.sentence_sha, "csrf": "tok"}).encode())
    assert not _finding(store, "new-port.md").dismissed
