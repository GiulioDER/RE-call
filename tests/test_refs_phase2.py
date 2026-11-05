"""Phase 2 of resolving `supersedes:` references at write time: one resolver for every layer.

Design: recall-lab `research/designs/2026-11-05-write-time-supersedes-references.md` (owner decision
2026-11-05). Search already resolves a reference that names a folder by path, and a bare name only
when no other memo shares it (`recall.supersession.ReferenceIndex.match`). Rewrite, lint, the
review queue, the dashboard editor and `add_supersedes_target`'s dedup compared file names only.

Red proof, 2026-11-05: every test below failed against the pre-change files in its own assertion
(recorded in the pull request). The new `resolve_reference` is tested only through those
consumers, since the pre-change code lacks it. The editor test is a strict expected failure until
the cycle check is path-aware (owner decision 2026-11-05: ship this phase, fix that next).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard import edit, review
from recall.frontmatter import add_supersedes_target
from recall.lint import lint_corpus
from recall.rewrite import RewriteRefused, _resolve, plan_rewrite
from recall.stale_reports import StaleReportQueue, default_queue_path
from tests.test_rewrite_routing_contract import _fact

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
OLD = "the deploy host for staging is alpha"
NEW = "the deploy host for staging is now gamma"


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


def test_a_folder_reference_names_one_memo_for_rewrite(shared: Path) -> None:
    try:
        named = _resolve(shared, "b/notes.md")
    except RewriteRefused as exc:
        named = f"refused: {exc}"
    assert named == "b/notes.md", f"a name with its folder was refused as shared: {named}"
    with pytest.raises(RewriteRefused, match=r"matches 2 files in the corpus, not one \(a/notes.md, b/notes.md\)"):
        _resolve(shared, "notes")


def test_a_bare_reference_that_two_memos_share_is_refused_before_it_is_stored(shared: Path) -> None:
    fact = _fact(relation="supersedes", subject_id="notes", object_id="new.md")
    with pytest.raises(RewriteRefused, match="could name 2 memos"):
        plan_rewrite(shared, fact)
    assert plan_rewrite(shared, _fact(relation="supersedes", subject_id="b/notes.md", object_id="new.md")).value == "b/notes.md"


def test_lint_accepts_a_folder_reference_that_names_one_memo(shared: Path) -> None:
    codes = {(i.file, i.code) for i in lint_corpus(shared)}
    assert ("new.md", "ambiguous-supersedes-target") not in codes, "lint called a reference search resolves ambiguous"
    (shared / "new.md").write_text("---\nsupersedes: notes\n---\nNew.\n", encoding="utf-8")
    assert ("new.md", "ambiguous-supersedes-target") in {(i.file, i.code) for i in lint_corpus(shared)}


def test_a_second_memo_with_the_same_name_can_be_declared() -> None:
    raw = b"---\nsupersedes: a/notes.md\n---\nNew.\n"
    assert add_supersedes_target(raw, "b/notes.md") is not None, "b/notes.md was taken for the a/notes.md already declared"
    assert add_supersedes_target(raw, "a/notes") is None
    assert add_supersedes_target(b"---\nsupersedes: notes\n---\nNew.\n", "a/notes.md") is None


@pytest.mark.xfail(
    strict=True,
    reason="the cycle check (`lint.supersession_graph_of`, `supersession_cycle_closed_by`) keys memos by "
    "file name, so a/notes.md superseding b/notes.md reads as a self-supersession; made path-aware next",
)
def test_the_editor_lets_one_folder_supersede_a_namesake_in_another(shared: Path) -> None:
    try:
        after = edit.plan_edit(shared, "a/notes.md", add_supersedes=("b/notes.md",)).after
    except edit.EditRefused as exc:
        after = f"refused: {exc}".encode()
    assert b"supersedes: b/notes.md" in after, f"a memo was refused as superseding itself for sharing a name: {after!r}"


def test_the_queue_claim_about_a_namesake_can_be_accepted(shared: Path) -> None:
    with StaleReportQueue(default_queue_path(shared)) as queue:
        queue.submit(shared, stale_source="b/notes.md", replacing_source="new.md", stale_quote=OLD, current_quote=NEW,
                     client="mcp:test", task=None, reported_at=NOW)
    (item,) = review.build_queue(shared).items
    shown = review.memo_digest(shared, "new.md")
    try:
        review.accept(shared, item, reviewer="ana", note="b is replaced too", shown_digest=shown, now=NOW)
    except review.ReviewRefused as exc:
        pytest.fail(f"the accept did not declare b/notes.md beside a/notes.md: refused, {exc}")
    declared = (shared / "new.md").read_text(encoding="utf-8")
    assert "a/notes.md" in declared and "b/notes.md" in declared, "the accept did not declare b/notes.md beside a/notes.md"
    assert not review.build_queue(shared).items
