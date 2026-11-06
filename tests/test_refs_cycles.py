"""The supersession cycle check, by path where memos share a name (write-time references, after phase 2).

The cycle graph keyed every memo by its stem, so `a/notes.md` and `b/notes.md` were one node: a memo
superseding its namesake read as superseding itself, a real cycle between them was named as a
self-edge, and a chain `a/notes -> c -> b/notes` was reported as a cycle it is not. Memos now keep
their stem as key unless another memo shares it, and each reference is resolved by search's rule.

Red proof, 2026-11-06: each test failed against the pre-change `recall/lint.py`,
`recall/rewrite.py`, `recall/fix.py` and `recall/dashboard/edit.py` in its own assertion, and
`tests/test_refs_phase2.py::test_the_editor_lets_one_folder_supersede_a_namesake_in_another` (a
strict expected failure until now) passes (recorded in the pull request).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.dashboard import edit
from recall.lint import lint_corpus


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _codes(root: Path) -> set[tuple[str, str]]:
    return {(issue.file, issue.code) for issue in lint_corpus(root)}


def test_lint_does_not_call_a_namesake_in_another_folder_the_memo_itself(tmp_path: Path) -> None:
    _write(tmp_path, "a/notes.md", "---\nsupersedes: b/notes.md\n---\n# A\n")
    _write(tmp_path, "b/notes.md", "# B\n")
    codes = {code for _file, code in _codes(tmp_path)}
    assert "self-supersedes" not in codes, "a memo superseding its namesake was called superseding itself"
    assert "supersession-cycle" not in codes


def test_a_real_cycle_between_namesakes_is_named_by_path(tmp_path: Path) -> None:
    _write(tmp_path, "a/notes.md", "---\nsupersedes: b/notes.md\n---\n# A\n")
    _write(tmp_path, "b/notes.md", "# B\n")
    try:
        edit.plan_edit(tmp_path, "b/notes.md", add_supersedes=("a/notes.md",))
        message = "accepted"
    except edit.EditRefused as exc:
        message = str(exc)
    assert "would close a supersession cycle" in message and "a/notes" in message and "b/notes" in message, (
        f"the cycle between namesakes was not named by path: {message}"
    )


def test_a_chain_through_another_memo_is_not_a_cycle(tmp_path: Path) -> None:
    _write(tmp_path, "a/notes.md", "---\nsupersedes: c\n---\n# A\n")
    _write(tmp_path, "c.md", "---\nsupersedes: b/notes.md\n---\n# C\n")
    _write(tmp_path, "b/notes.md", "# B\n")
    codes = {code for _file, code in _codes(tmp_path)}
    assert "supersession-cycle" not in codes, "a chain a/notes -> c -> b/notes was reported as a cycle"


def test_a_flat_corpus_cycle_is_still_found_and_named_by_stem(tmp_path: Path) -> None:
    """A control, green before and after: memos with unique names keep their stems as keys."""
    _write(tmp_path, "old.md", "---\nsupersedes: new\n---\n# Old\n")
    _write(tmp_path, "new.md", "# New\n")
    try:
        edit.plan_edit(tmp_path, "new.md", add_supersedes=("old.md",))
        message = "accepted"
    except edit.EditRefused as exc:
        message = str(exc)
    assert "(old -> new)" in message, message


def test_lint_builds_one_reference_index_for_the_whole_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One index per lint run, not one per reference: per reference made a 1,000-memo lint 16x
    slower (0.44 s to 6.8 s, measured 2026-11-06). Red proof: the first version of this change
    built one per reference ("lint built 29 reference indexes for 30 memos")."""
    from recall import supersession

    root = tmp_path
    for i in range(30):
        _write(root, f"m{i}.md", f"---\nsupersedes: m{i - 1}\n---\n# M{i}\n" if i else "# M0\n")
    built: list[int] = []
    real = supersession.ReferenceIndex

    def counted(*args: object, **kwargs: object) -> supersession.ReferenceIndex:
        built.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(supersession, "ReferenceIndex", counted)
    lint_corpus(root)
    assert len(built) <= 2, f"lint built {len(built)} reference indexes for 30 memos"


def test_a_cycle_through_a_memo_named_with_a_double_extension_is_refused(tmp_path: Path) -> None:
    """`y.md.md` is keyed `y.md`; normalising that key a second time made it `y` and the cycle
    through it went unseen. Red proof: the first version of this change accepted the edit."""
    _write(tmp_path, "other.md", "---\nsupersedes: y.md.md\n---\n# Other\n")
    _write(tmp_path, "y.md.md", "# Y\n")
    try:
        edit.plan_edit(tmp_path, "y.md.md", add_supersedes=("other.md",))
        message = "accepted"
    except edit.EditRefused as exc:
        message = str(exc)
    assert "would close a supersession cycle" in message, f"a cycle through y.md.md was let through: {message}"
