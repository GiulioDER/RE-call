"""SEC-001/SEC-002: scan the actual paths and bytes selected by Git.

Red proof, 2026-10-28: the first thirteen nodes failed their return-code
assertion against acdd27b05. Targets: tracked(), scan(), main(), and policy loaders
in scripts/check_public_tree.py. The checker is copied unchanged into temporary
repositories so its real CLI, Git selection and policy loaders run.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

CHECKER = Path(__file__).parents[1] / "scripts" / "check_public_tree.py"
PUBLIC = "8.8." + "4.4"


@pytest.fixture
def git_tree(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(CHECKER, tmp_path / "scripts" / CHECKER.name)
    return tmp_path


def stage(root: Path, *names: str) -> None:
    subprocess.run(["git", "-C", str(root), "add", "--", *names], check=True)


def run_check(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("RECALL_PUBLIC_TREE_DENYLIST", None)
    return subprocess.run(
        [sys.executable, str(root / "scripts" / CHECKER.name), *args],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize(
    "name", ["caf\u00e9.md", 'quote"name.md', "tab\tname.md", "line\nname.md"],
    ids=["unicode", "quote", "tab", "newline"],
)
def test_tracked_unusual_names_are_scanned(git_tree: Path, name: str) -> None:
    if sys.platform == "win32" and any(c in name for c in '\"\t\n'):
        pytest.skip("this filename is not representable on Windows")
    (git_tree / name).write_text(PUBLIC, encoding="utf-8")
    stage(git_tree, name)
    result = run_check(git_tree)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ipv4" in result.stdout
    assert PUBLIC not in result.stdout


def test_a_missing_tracked_file_cannot_report_clean(git_tree: Path) -> None:
    note = git_tree / "note.md"
    note.write_text("clean", encoding="utf-8")
    stage(git_tree, note.name)
    note.unlink()
    result = run_check(git_tree)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "not checked" in result.stderr


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize("working", ["clean", "missing"])
def test_staged_leak_is_checked_even_when_working_copy_is_clean_or_missing(
    git_tree: Path, compressed: bool, working: str,
) -> None:
    note = git_tree / ("note.md.gz" if compressed else "note.md")
    raw = PUBLIC.encode()
    note.write_bytes(gzip.compress(raw) if compressed else raw)
    stage(git_tree, note.name)
    if working == "missing":
        note.unlink()
    else:
        note.write_bytes(gzip.compress(b"clean") if compressed else b"clean")
    result = run_check(git_tree, "--staged")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ipv4" in result.stdout


def test_unstaged_leak_does_not_taint_a_clean_index(git_tree: Path) -> None:
    note = git_tree / "note.md"
    note.write_text("clean", encoding="utf-8")
    stage(git_tree, note.name)
    note.write_text(PUBLIC, encoding="utf-8")
    result = run_check(git_tree, "--staged")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("policy", ["allowlist", "baseline"])
def test_unstaged_policy_cannot_authorize_an_indexed_leak(git_tree: Path, policy: str) -> None:
    note = git_tree / "note.md"
    body = PUBLIC if policy == "allowlist" else "/home/" + "privateperson/memo"
    note.write_text(body, encoding="utf-8")
    name = "public_tree_allowlist.txt" if policy == "allowlist" else "public_tree_baseline.json"
    rules = git_tree / "scripts" / name
    rules.write_text("# no exceptions\n" if policy == "allowlist" else "{}\n", encoding="utf-8")
    stage(git_tree, note.name, "scripts/" + name)
    content = PUBLIC + "\tunstaged exception\n" if policy == "allowlist" else (
        '{"note.md": {"home-user": 1}}\n'
    )
    rules.write_text(content, encoding="utf-8")
    result = run_check(git_tree, "--staged")
    assert result.returncode == 1, result.stdout + result.stderr


def test_an_indexed_exception_still_applies(git_tree: Path) -> None:
    """An indexed exception stays effective when the working-tree policy differs."""
    (git_tree / "note.md").write_text(PUBLIC, encoding="utf-8")
    rules = git_tree / "scripts" / "public_tree_allowlist.txt"
    rules.write_text(PUBLIC + "\tsynthetic fixture\n", encoding="utf-8")
    stage(git_tree, "note.md", "scripts/public_tree_allowlist.txt")
    rules.write_text("# no working-tree exceptions\n", encoding="utf-8")
    result = run_check(git_tree, "--staged")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="colon filenames are not representable on Windows")
def test_stage_like_filename_is_read_as_a_literal_path(git_tree: Path) -> None:
    """SEC-002: pin index stage zero before a path resembling Git's stage syntax.

    Red proof uses _index_bytes with ':<path>' instead of ':0:<path>': Git
    then reads the clean note.md blob for the prohibited 0:note.md filename.
    """
    (git_tree / "note.md").write_text("clean", encoding="utf-8")
    (git_tree / "0:note.md").write_text(PUBLIC, encoding="utf-8")
    stage(git_tree, "note.md", "0:note.md")
    result = run_check(git_tree, "--staged")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ipv4" in result.stdout
