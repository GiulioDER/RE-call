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
