"""PRs 39 through 30: staged-hook and supersession regression boundaries.

Red baseline: 76b05b712, unchanged test nodes on the remote host. Hook tests target
hooks/pre-commit; scanner tests target recall.check._ANY_REF; reported-subject
tests target recall.fix._is_reported_speech. Assertion failures and controls
are recorded in the audit's regression-proof receipt.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from recall.check import check_file
from recall.fix import extract_edges

ROOT = Path(__file__).resolve().parents[1]


def _git(repo, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT), GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="Test", GIT_COMMITTER_NAME="Test",
               GIT_AUTHOR_EMAIL="test@example.invalid", GIT_COMMITTER_EMAIL="test@example.invalid")
    # Stats profiling cannot discover its configuration from the temporary Git cwd.
    # Retain real mutant selectors so child behavior remains covered.
    if env.get("MUTANT_UNDER_TEST") == "stats":
        env.pop("MUTANT_UNDER_TEST")
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=.git/hooks", *args],
        cwd=repo, env=env, capture_output=True, text=True, check=False,
    )


def _hook_repo(tmp_path, *, force_executable=True):
    if sys.platform == "win32":
        pytest.skip("POSIX Git hook installation and executable modes")
    assert _git(tmp_path, "init", "-q").returncode == 0
    (tmp_path / "hooks").mkdir()
    target = tmp_path / "hooks" / "pre-commit"
    shutil.copy2(ROOT / "hooks" / "pre-commit", target)
    if force_executable:
        target.chmod(0o755)  # Isolate content behavior from DEPLOY-001's independent mode defect.
    (tmp_path / ".git" / "hooks" / "pre-commit").symlink_to("../../hooks/pre-commit")
    return tmp_path


def test_hook_checks_staged_bytes_even_when_worktree_declares_an_edge(tmp_path):
    repo = _hook_repo(tmp_path)
    memo = repo / "new.md"
    memo.write_text("DEPRECATED.\n", encoding="utf-8")
    assert _git(repo, "add", "--", memo.name).returncode == 0
    memo.write_text("---\nvalid_until: 2030-01-01\n---\nDEPRECATED.\n", encoding="utf-8")
    result = _git(repo, "commit", "-qm", "memo")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "state a closure in prose only" in result.stdout + result.stderr


def test_hook_ignores_unstaged_closure_when_index_is_valid(tmp_path):
    repo = _hook_repo(tmp_path)
    memo = repo / "new.md"
    memo.write_text("An ordinary memo.\n", encoding="utf-8")
    assert _git(repo, "add", "--", memo.name).returncode == 0
    memo.write_text("DEPRECATED.\n", encoding="utf-8")
    result = _git(repo, "commit", "-qm", "memo")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["decision notes.md", "decision\tmemo.md", "decision\nmemo.md",
                                 "décision.md", "-decision.md", "0:decision.md"])
def test_hook_preserves_staged_filenames(tmp_path, name):
    repo = _hook_repo(tmp_path)
    (repo / name).write_text("DEPRECATED.\n", encoding="utf-8")
    assert _git(repo, "add", "--", name).returncode == 0
    blocked = _git(repo, "commit", "-qm", "closure")
    assert blocked.returncode == 1, blocked.stdout + blocked.stderr
    assert "state a closure in prose only" in blocked.stdout + blocked.stderr
    (repo / name).write_text("An ordinary memo.\n", encoding="utf-8")
    assert _git(repo, "add", "--", name).returncode == 0
    result = _git(repo, "commit", "-qm", "memo")
    assert result.returncode == 0, result.stdout + result.stderr


def test_hook_checks_renamed_markdown_destinations(tmp_path):
    repo = _hook_repo(tmp_path)
    (repo / "draft.txt").write_text("DEPRECATED.\n", encoding="utf-8")
    assert _git(repo, "add", "draft.txt").returncode == 0
    assert _git(repo, "commit", "-qm", "draft").returncode == 0
    assert _git(repo, "mv", "draft.txt", "new.md").returncode == 0
    assert _git(repo, "diff", "--cached", "--name-status").stdout.startswith("R100")
    result = _git(repo, "commit", "-qm", "rename")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "state a closure in prose only" in result.stdout + result.stderr


def test_documented_hook_installation_is_executable(tmp_path):
    repo = _hook_repo(tmp_path, force_executable=False)
    (repo / "new.md").write_text("DEPRECATED.\n", encoding="utf-8")
    assert _git(repo, "add", "new.md").returncode == 0
    result = _git(repo, "commit", "-qm", "memo")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "state a closure in prose only" in result.stdout + result.stderr


def test_hook_checks_regular_memo_replacing_a_symlink(tmp_path):
    repo = _hook_repo(tmp_path)
    (repo / "plain.txt").write_text("An ordinary memo.\n", encoding="utf-8")
    memo = repo / "note.md"
    memo.symlink_to("plain.txt")
    assert _git(repo, "add", "note.md").returncode == 0
    assert _git(repo, "commit", "-qm", "linked memo").returncode == 0
    memo.unlink()
    memo.write_text("DEPRECATED.\n", encoding="utf-8")
    assert _git(repo, "add", "note.md").returncode == 0
    assert _git(repo, "diff", "--cached", "--name-status").stdout.startswith("T")
    result = _git(repo, "commit", "-qm", "replacement memo")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "state a closure in prose only" in result.stdout + result.stderr


def test_long_nonreference_token_does_not_stall_write_time_check(tmp_path):
    memo = tmp_path / "new.md"
    memo.write_text("a" * 20000 + "\nSee [[old_plan_2026-01-01]].", encoding="utf-8")
    started = time.process_time()
    result = check_file(memo)
    assert time.process_time() - started < 0.25
    assert result.candidates == ["old_plan_2026-01-01"]


@pytest.mark.parametrize("subject", ["release_2026-02-02.md", "[[release_2026-02-02]]",
                                    "[[namespace:release]]", "[[release—plan]]",
                                    pytest.param("[[release-plan-" + "a" * 205 + "]]",
                                                 id="long-wikilink"),
                                    pytest.param("release memo project_2026-01-01",
                                                 id="dated-stem-subject"),
                                    pytest.param("memo-project_2026-01-01",
                                                 id="dated-stem-with-doc-noun")])
def test_reported_document_subject_does_not_become_a_narrator_edge(subject):
    assert extract_edges(f"{subject} supersedes [[old_plan_2026-01-01]].") == ([], [])


@pytest.mark.parametrize("prefix", ["Other memo: This decision", "Other memo; This decision",
                                   "Other memo. This decision", "Other memo - This decision"])
def test_reported_subject_in_an_earlier_clause_does_not_hide_a_real_edge(prefix):
    assert extract_edges(f"{prefix} supersedes [[old_plan_2026-01-01]].") == (
        ["old_plan_2026-01-01"], [],
    )


def test_write_time_reference_case_matching_is_preserved(tmp_path):
    memo = tmp_path / "new.md"
    memo.write_text("See old_plan_2026-01-01.md and IGNORED_PLAN.MD.", encoding="utf-8")
    assert check_file(memo).candidates == ["old_plan_2026-01-01"]
