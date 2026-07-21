"""`recall lint`: completeness checks on the supersession graph, before indexing.

The trust layer's residual failure mode is authorial: a new memo that REPLACES an old one but
never declares `supersedes:` leaves an orphan that looks valid forever. That failure is not
detectable at read-time (both memos look fine in isolation) — but it IS lintable at write-time.
"""
from __future__ import annotations

import pytest

from recall.lint import LintIssue, lint_corpus


def _write(tmp_path, name: str, text: str):
    (tmp_path / name).write_text(text, encoding="utf-8")


def _codes(issues: list[LintIssue]) -> set[str]:
    return {i.code for i in issues}


def test_clean_corpus_has_no_issues(tmp_path):
    _write(tmp_path, "a.md", "some settled decision")
    _write(tmp_path, "b_v1.md", "rate limit is 100 rps")
    _write(tmp_path, "b_v2.md", "---\nsupersedes: b_v1.md\n---\nrate limit is 20 rps")
    assert lint_corpus(tmp_path) == []


def test_dangling_supersedes_is_an_error(tmp_path):
    _write(tmp_path, "new.md", "---\nsupersedes: ghost.md\n---\nthe new decision")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"dangling-supersedes"}
    assert issues[0].level == "error"
    assert "ghost.md" in issues[0].message


def test_self_supersedes_is_an_error(tmp_path):
    _write(tmp_path, "loop.md", "---\nsupersedes: loop.md\n---\ntext")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"self-supersedes"}
    assert issues[0].level == "error"


def test_supersession_cycle_is_an_error(tmp_path):
    _write(tmp_path, "a.md", "---\nsupersedes: b.md\n---\nA")
    _write(tmp_path, "b.md", "---\nsupersedes: a.md\n---\nB")
    issues = lint_corpus(tmp_path)
    assert "supersession-cycle" in _codes(issues)
    assert all(i.level == "error" for i in issues if i.code == "supersession-cycle")


def test_malformed_validity_date_is_an_error(tmp_path):
    _write(tmp_path, "bad.md", "---\nvalid_until: June 2026\n---\ntext")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"invalid-date"}
    assert issues[0].level == "error"


def test_version_siblings_without_edge_is_a_warning(tmp_path):
    _write(tmp_path, "policy_v1.md", "old policy")
    _write(tmp_path, "policy_v2.md", "new policy, forgot the frontmatter")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"version-sibling-unlinked"}
    assert issues[0].level == "warning"
    assert "policy_v1.md" in issues[0].message


def test_version_siblings_with_edge_are_clean(tmp_path):
    _write(tmp_path, "policy_v1.md", "old policy")
    _write(tmp_path, "policy_v2.md", "---\nsupersedes: policy_v1.md\n---\nnew policy")
    assert lint_corpus(tmp_path) == []


def test_closure_prose_without_edge_is_a_warning(tmp_path):
    _write(tmp_path, "closed.md", "This lane is now superseded by the new approach.")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"closure-marker-unlinked"}
    assert issues[0].level == "warning"


def test_closure_prose_with_edge_is_clean(tmp_path):
    _write(tmp_path, "old.md", "the old approach")
    _write(tmp_path, "closed.md",
           "---\nsupersedes: old.md\n---\nThis lane is now superseded by the new approach.")
    assert lint_corpus(tmp_path) == []


def test_cycle_still_reported_when_a_member_has_a_second_superseder(tmp_path):
    # regression (BUG-001): a single-valued edge map dropped one of two superseders of the
    # same target, and the declared a<->z cycle silently vanished from the report
    _write(tmp_path, "a.md", "---\nsupersedes: z.md\n---\nA")
    _write(tmp_path, "z.md", "---\nsupersedes: a.md\n---\nZ")
    _write(tmp_path, "m.md", "---\nsupersedes: z.md\n---\nM")
    issues = lint_corpus(tmp_path)
    assert "supersession-cycle" in _codes(issues)


def test_same_filename_in_two_subdirs_does_not_shadow_checks(tmp_path):
    # regression (BUG-004): dicts keyed by bare filename let sub2/x.md overwrite sub1/x.md,
    # losing sub1's closure-marker warning entirely
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub2").mkdir()
    _write(tmp_path / "sub1", "x.md", "This memo is deprecated and replaced by a new one.")
    _write(tmp_path / "sub2", "x.md", "a perfectly healthy unrelated note")
    issues = lint_corpus(tmp_path)
    assert "closure-marker-unlinked" in _codes(issues)


def test_ambiguous_supersedes_target_is_a_warning(tmp_path):
    # a supersedes: edge pointing at a basename that exists in TWO places cannot be resolved
    # unambiguously — surface it instead of silently picking one
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub2").mkdir()
    _write(tmp_path / "sub1", "old.md", "v1")
    _write(tmp_path / "sub2", "old.md", "also v1")
    _write(tmp_path, "new.md", "---\nsupersedes: old.md\n---\nv2")
    issues = lint_corpus(tmp_path)
    assert "ambiguous-supersedes-target" in _codes(issues)


def test_zero_padded_version_siblings_warn(tmp_path):
    # deferred NUM-003: reconstructing "stem_v{n+1}" missed zero-padded series entirely —
    # x_v01/x_v02 never triggered the warning (silent false negative in a completeness lint)
    _write(tmp_path, "policy_v01.md", "old policy")
    _write(tmp_path, "policy_v02.md", "new policy, forgot the frontmatter")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"version-sibling-unlinked"}
    assert "policy_v01.md" in issues[0].message


def test_non_contiguous_version_siblings_warn(tmp_path):
    _write(tmp_path, "policy_v1.md", "old policy")
    _write(tmp_path, "policy_v3.md", "new policy, v2 was deleted")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"version-sibling-unlinked"}


def test_missing_root_raises_cleanly_and_cli_exits_2(tmp_path):
    # deferred BUG-003: a nonexistent path fell into the single-file branch and died with a
    # raw FileNotFoundError traceback from read_text
    import pytest

    from recall.cli import main
    from recall.lint import lint_corpus as lc

    with pytest.raises(FileNotFoundError):
        lc(tmp_path / "does-not-exist")
    with pytest.raises(SystemExit) as exc:
        main(["lint", str(tmp_path / "does-not-exist")])
    assert exc.value.code == 2


def test_unreadable_file_is_reported_and_rest_of_corpus_still_linted(tmp_path):
    # deferred BUG-003: one non-UTF8 file aborted the whole lint with zero issues reported
    (tmp_path / "binary.md").write_bytes(b"\xff\xfe\x00broken")
    _write(tmp_path, "new.md", "---\nsupersedes: ghost.md\n---\ntext")
    issues = lint_corpus(tmp_path)
    codes = _codes(issues)
    assert "unreadable-file" in codes
    assert "dangling-supersedes" in codes  # healthy files still checked


def test_cli_lint_exits_nonzero_on_errors_and_zero_on_clean(tmp_path, capsys):
    from recall.cli import main

    _write(tmp_path, "new.md", "---\nsupersedes: ghost.md\n---\ntext")
    with pytest.raises(SystemExit) as exc:
        main(["lint", str(tmp_path)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "dangling-supersedes" in out

    clean = tmp_path / "clean"
    clean.mkdir()
    _write(clean, "a.md", "fine")
    main(["lint", str(clean)])  # returns without raising
    assert "0 errors" in capsys.readouterr().out


def test_ambiguous_superseder_basename_is_reported(tmp_path):
    """The runtime fails closed when the SUPERSEDING basename is duplicated too.

    Without a lint rule for it the operator gets an abstention at query time and a clean
    `recall lint` — an unfixable-looking failure.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "old.md").write_text("the old note", encoding="utf-8")
    (tmp_path / "a" / "dup.md").write_text(
        "---\nsupersedes: old.md\n---\nthe new note", encoding="utf-8"
    )
    (tmp_path / "b" / "dup.md").write_text("an unrelated note", encoding="utf-8")
    codes = {i.code for i in lint_corpus(tmp_path)}
    assert "ambiguous-supersedes-source" in codes


def test_ambiguous_supersedes_target_is_an_error_not_a_warning(tmp_path):
    # read-time now REFUSES to answer from these documents, so lint must exit non-zero
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.md").write_text("one", encoding="utf-8")
    (tmp_path / "b" / "notes.md").write_text("two", encoding="utf-8")
    (tmp_path / "new.md").write_text(
        "---\nsupersedes: notes.md\n---\nthe replacement", encoding="utf-8"
    )
    issues = [i for i in lint_corpus(tmp_path) if i.code == "ambiguous-supersedes-target"]
    assert issues and issues[0].level == "error"
