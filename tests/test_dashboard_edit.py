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


def test_an_edit_that_closes_a_supersession_cycle_is_refused_in_the_preview(corpus: Path) -> None:
    """`new.md` supersedes `older.md`; letting `older.md` supersede `new.md` would close a cycle.

    The edit was refused for a self-edge only, so the cycle reached disk and `recall lint`
    reported it afterwards, while search abstained on both memos.

    Red: baseline, `tests/test_dashboard_edit.py::test_an_edit_that_closes_a_supersession_cycle_is_refused_in_the_preview`
    failed with `DID NOT RAISE EditRefused`; target `recall.dashboard.edit.plan_edit`, which
    refused self-edges and not cycles.
    """
    before = (corpus / "older.md").read_bytes()
    with pytest.raises(edit.EditRefused, match=r"cycle.*\(new -> older\)"):
        edit.plan_edit(corpus, "older.md", add_supersedes=("new.md",))
    with pytest.raises(edit.EditRefused, match="cycle"):
        edit.apply_edit(
            corpus, "older.md", editor="giulio", note="tidy",
            shown_sha="unused: the plan is refused first", now=NOW, add_supersedes=("new.md",),
        )
    assert (corpus / "older.md").read_bytes() == before


def test_an_edit_that_closes_no_cycle_is_still_planned(corpus: Path) -> None:
    """The cycle check must not refuse an ordinary second edge.

    Passes on master; red by mutation of `recall.dashboard.edit._refuse_cycles` raising for every
    added target: this test then failed with `EditRefused`.
    """
    plan = edit.plan_edit(corpus, "old.md", add_supersedes=("older.md",))
    assert plan.summary == ("supersedes older.md",)


def test_an_undo_that_would_restore_a_supersession_cycle_is_refused(corpus: Path) -> None:
    """Undo writes earlier bytes back, so it can restore an edge that now closes a cycle.

    `new.md` supersedes `older.md`. Edit 1 withdraws it; edit 2 lets `older.md` supersede
    `new.md`; undoing edit 1 would put the first edge back beside the second: the cycle
    `plan_edit` refuses, reached the other way round (audit finding STAKES-001, P1).

    Red: branch head 4b68f972b,
    `tests/test_dashboard_edit.py::test_an_undo_that_would_restore_a_supersession_cycle_is_refused`
    failed with `DID NOT RAISE EditRefused`, the undo having written `supersedes: older.md` back;
    target `recall.dashboard.edit.undo_edit`, which checked only that the memo still held the
    bytes edit 1 wrote.
    """
    from recall.lint import lint_corpus

    first = _apply(corpus, remove_supersedes=("older.md",))
    plan = edit.plan_edit(corpus, "older.md", add_supersedes=("new.md",))
    edit.apply_edit(corpus, "older.md", editor="giulio", note="reverse", shown_sha=plan.before_sha, now=NOW,
                    add_supersedes=("new.md",))
    before = (corpus / "new.md").read_bytes()

    with pytest.raises(edit.EditRefused, match=r"cycle.*\(older -> new\)"):
        edit.undo_edit(corpus, first.edit_id, editor="giulio", note="undo", now=NOW)

    assert (corpus / "new.md").read_bytes() == before
    assert not any(i.code == "supersession-cycle" for i in lint_corpus(corpus))
    assert {record.edit_id: record for record in edit.history(corpus)}[first.edit_id].undone_at is None


def test_an_undo_that_restores_no_cycle_still_runs(corpus: Path) -> None:
    """The undo gate refuses a cycle and nothing else.

    Passes at 4b68f972b. Red by mutation of `recall.dashboard.edit.undo_edit` passing every
    target `before` declares to `_refuse_cycles` with each judged as closing a cycle
    (`_restored_targets` made to return `["new.md"]`): this test failed with `EditRefused`.
    """
    first = _apply(corpus, remove_supersedes=("older.md",))
    assert edit.undo_edit(corpus, first.edit_id, editor="giulio", note="undo", now=NOW) == "new.md"
    assert supersedes_targets(_meta(corpus / "new.md").get("supersedes")) == ("older.md",)


def test_the_edit_cycle_check_reads_hidden_directories_as_lint_does(corpus: Path) -> None:
    """A memo under a dot directory is part of the corpus `recall lint` checks for cycles.

    Red: branch head 4b68f972b,
    `tests/test_dashboard_edit.py::test_the_edit_cycle_check_reads_hidden_directories_as_lint_does`
    failed with `DID NOT RAISE EditRefused`; target `recall.dashboard.edit._refuse_cycles`, which
    built its graph from the memos an edit may name and so skipped `.trash/` (audit finding
    ENV-001).
    """
    (corpus / ".trash").mkdir()
    (corpus / ".trash" / "gone.md").write_text("---\nsupersedes: old.md\n---\n# Gone\n", encoding="utf-8")
    (corpus / "older.md").write_text("---\nsupersedes: gone\n---\n# Older\n", encoding="utf-8")

    with pytest.raises(edit.EditRefused, match=r"cycle.*\(older -> gone -> old\)"):
        edit.plan_edit(corpus, "old.md", add_supersedes=("older.md",))


def test_an_edit_that_adds_no_reference_does_not_list_the_corpus(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A date or status edit names no other memo, so the memo listing is not walked for it.

    Red: baseline 8e10eea6a,
    `tests/test_dashboard_edit.py::test_an_edit_that_adds_no_reference_does_not_list_the_corpus`
    failed with `assert 1 == 0`; target `recall.dashboard.edit.plan_edit`, which listed every memo
    for every edit (CCA deferred finding PERF-007).
    """
    listed = []
    real = edit._corpus_names
    monkeypatch.setattr(edit, "_corpus_names", lambda root: listed.append(root) or real(root))
    plan = edit.plan_edit(corpus, "new.md", valid_until="2027-01-01", status="active")
    assert plan.summary == ("valid_until 2027-01-01", "status active")
    assert len(listed) == 0


def _ignores_case(root: Path) -> bool:
    (root / "case-probe.md").write_text("probe\n", encoding="utf-8")
    try:
        return (root / "CASE-PROBE.md").exists()
    finally:
        (root / "case-probe.md").unlink()


@pytest.mark.parametrize("name, target, reason", [
    ("New.md", "new.md", "cannot supersede itself"),
    ("Older.md", "new.md", r"cycle.*\(new -> older\)"),
])
def test_a_memo_named_in_another_case_is_judged_as_the_file_it_opens(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, name: str, target: str, reason: str
) -> None:
    """On a filesystem that ignores case, ``Older.md`` opens ``older.md`` and is judged as it.

    `Path.resolve` keeps the spelling it was given on macOS, so the memo being edited was keyed
    apart from the corpus entry that is the same file: adding ``new.md`` to ``New.md`` passed the
    self-check, and adding it to ``Older.md`` missed the cycle through ``new.md``. The patched
    `resolve_memo` keeps the caller's spelling as macOS does; the test needs a filesystem that
    ignores case (it runs on Windows and macOS, and skips on Linux).

    Red: baseline 8e10eea6a, on Windows,
    `tests/test_dashboard_edit.py::test_a_memo_named_in_another_case_is_judged_as_the_file_it_opens`
    failed with `DID NOT RAISE EditRefused` in both cases; target `recall.dashboard.edit._memo`,
    which used the caller's spelling (CCA deferred finding BUG-010).
    """
    if not _ignores_case(corpus):
        pytest.skip("needs a filesystem that ignores case")
    monkeypatch.setattr(
        edit, "resolve_memo", lambda root, asked: (asked, root.resolve() / asked)
    )
    before = (corpus / name.lower()).read_bytes()
    with pytest.raises(edit.EditRefused, match=reason):
        edit.plan_edit(corpus, name, add_supersedes=(target,))
    assert (corpus / name.lower()).read_bytes() == before


def test_a_directory_that_cannot_be_listed_keeps_the_spelling_given(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spelling a memo as its directory lists it needs the listing; without it the edit goes on
    under the name `resolve_memo` accepted.

    Red: mutation of `recall.dashboard.edit._as_listed` without its `OSError` fallback:
    `tests/test_dashboard_edit.py::test_a_directory_that_cannot_be_listed_keeps_the_spelling_given`
    failed with `PermissionError` from the listing.
    """
    def unlistable(path: object) -> list[str]:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(edit.os, "listdir", unlistable)
    plan = edit.plan_edit(corpus, "new.md", valid_until="2027-01-01")
    assert plan.name == "new.md"
