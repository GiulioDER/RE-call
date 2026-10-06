"""A person's edits to a memo's declared values: planned exactly, applied only to the memo they saw,
recorded, and undoable only while the memo is still what the edit wrote.

Red proof, 2026-10-06, each mutation alone against `recall/dashboard/edit.py`, failing for the stated
reason (JUnit XML), then restored; all twelve green after:
- E1 the digest of the memo shown not compared: DID NOT RAISE (the edit lands on a changed memo).
- E2 no named person required: DID NOT RAISE.
- E3 undo over a later hand edit: DID NOT RAISE.
- E4 undo of an undone edit: refused, but for the wrong reason (regex `already undone` did not match).
- E5 an ambiguous target accepted: DID NOT RAISE (the `matches 2 memos` case).
- E6 a memo superseding itself accepted: DID NOT RAISE.
- E7 valid_until before valid_from accepted: DID NOT RAISE.
- E8 an edit not recorded: the history is empty.
- E9 an edit that changes nothing accepted: DID NOT RAISE.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard import edit
from recall.frontmatter import parse_frontmatter, supersedes_targets
from recall.rewrite import _derived_value

NOW = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "old.md").write_text("# Old\nThe port is 8080.\n", encoding="utf-8")
    (tmp_path / "older.md").write_text("# Older\nThe port is 80.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text("---\nsupersedes: older.md\n---\n# New\nThe port is 9090.\n", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "twin.md").write_text("# Twin A\n", encoding="utf-8")
    (tmp_path / "b" / "twin.md").write_text("# Twin B\n", encoding="utf-8")
    return tmp_path


def _meta(path: Path) -> dict:
    return parse_frontmatter(path.read_text(encoding="utf-8"))[0]


def _apply(corpus: Path, **changes: object) -> edit.EditRecord:
    plan = edit.plan_edit(corpus, "new.md", **changes)  # type: ignore[arg-type]
    return edit.apply_edit(corpus, "new.md", editor="giulio", note="tidy", shown_sha=plan.before_sha, now=NOW, **changes)


def test_an_edit_changes_exactly_the_values_asked_for(corpus: Path) -> None:
    record = _apply(
        corpus,
        add_supersedes=("old.md",),
        remove_supersedes=("older.md",),
        valid_until="2027-01-01",
        status="active",
    )
    raw = (corpus / "new.md").read_bytes()
    assert supersedes_targets(_meta(corpus / "new.md")["supersedes"]) == ("old.md",)
    assert _meta(corpus / "new.md")["valid_until"] == "2027-01-01"
    assert _derived_value(raw, "status") == "active"
    assert b"The port is 9090." in raw
    assert record.summary == ("no longer supersedes older.md", "supersedes old.md", "valid_until 2027-01-01", "status active")
    assert [h.edit_id for h in edit.history(corpus, "new.md")] == [record.edit_id]


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"add_supersedes": ("missing.md",)}, "not a memo"),
        ({"add_supersedes": ("twin.md",)}, "matches 2 memos"),
        ({"add_supersedes": ("new.md",)}, "cannot supersede itself"),
        ({"remove_supersedes": ("old.md",)}, "does not declare"),
        ({"valid_until": "next tuesday"}, "not a date"),
        ({"valid_from": "2027-01-01", "valid_until": "2026-01-01"}, "before valid_from"),
        ({"status": "archived"}, "status must be one of"),
        ({}, "nothing to change"),
    ],
)
def test_an_edit_that_cannot_be_right_is_refused_and_writes_nothing(corpus: Path, changes: dict, reason: str) -> None:
    before = (corpus / "new.md").read_bytes()
    with pytest.raises(edit.EditRefused, match=reason):
        edit.plan_edit(corpus, "new.md", **changes)
    assert (corpus / "new.md").read_bytes() == before


def test_an_edit_needs_a_person_and_the_memo_the_person_saw(corpus: Path) -> None:
    plan = edit.plan_edit(corpus, "new.md", status="draft")
    with pytest.raises(edit.EditRefused, match="your name"):
        edit.apply_edit(corpus, "new.md", editor=" ", note="x", shown_sha=plan.before_sha, now=NOW, status="draft")
    (corpus / "new.md").write_text("---\nsupersedes: older.md\n---\n# New\nchanged meanwhile\n", encoding="utf-8")
    with pytest.raises(edit.EditRefused, match="changed since you reviewed"):
        edit.apply_edit(corpus, "new.md", editor="giulio", note="x", shown_sha=plan.before_sha, now=NOW, status="draft")
    assert _derived_value((corpus / "new.md").read_bytes(), "status") is None
    assert edit.history(corpus) == []


def test_undo_restores_the_bytes_and_only_while_nothing_changed_since(corpus: Path) -> None:
    before = (corpus / "new.md").read_bytes()
    record = _apply(corpus, status="deprecated")
    edit.undo_edit(corpus, record.edit_id, editor="giulio", note="wrong memo", now=NOW)
    assert (corpus / "new.md").read_bytes() == before
    with pytest.raises(edit.EditRefused, match="already undone"):
        edit.undo_edit(corpus, record.edit_id, editor="giulio", note="again", now=NOW)

    second = _apply(corpus, status="draft")
    (corpus / "new.md").write_bytes((corpus / "new.md").read_bytes() + b"edited by hand\n")
    edited = (corpus / "new.md").read_bytes()
    with pytest.raises(edit.EditRefused, match="changed after edit"):
        edit.undo_edit(corpus, second.edit_id, editor="giulio", note="undo", now=NOW)
    assert (corpus / "new.md").read_bytes() == edited


def test_the_sidecar_and_paths_outside_the_corpus_cannot_be_edited(corpus: Path) -> None:
    (corpus / ".recall").mkdir(exist_ok=True)
    (corpus / ".recall" / "x.md").write_text("# hidden\n", encoding="utf-8")
    for name in (".recall/x.md", "../outside.md", str(corpus / "new.md")):
        with pytest.raises(edit.EditRefused):
            edit.plan_edit(corpus, name, status="draft")
