"""Build throwaway git repositories for the corpus consistency audit tests.

Separate from `tests/fakes.py` on purpose: that module is documented as minimal duck-typed
fakes with no database and no side effects, and this one shells out to git.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    """Run git in `repo` with signing off, so an ambient global config cannot break a fixture.

    Commit signing keys cannot be registered on this machine, so a globally enabled
    `commit.gpgsign` would fail every fixture repo here with an error that looks nothing like
    the thing under test.
    """
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return done.stdout


def _git_stdin(repo: Path, *args: str, input: str) -> str:
    """Like `git`, but for the one call (`hash-object --stdin`) that needs to pipe input in."""
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=input,
    )
    return done.stdout


def init_repo(tmp_path: Path) -> Path:
    """An empty repo with a committer identity and signing off."""
    repo = tmp_path / "repo"
    repo.mkdir()  # not exist_ok: two builders on one tmp_path would silently merge histories
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    return repo


def two_commit_repo(tmp_path: Path, first: str, second: str) -> Path:
    """A repo whose single `notes.md` is committed twice, with `first` then `second` as its body."""
    repo = init_repo(tmp_path)
    doc = repo / "notes.md"
    doc.write_text(first, encoding="utf-8", newline="\n")
    git(repo, "add", "notes.md")
    git(repo, "commit", "-q", "-m", "first")
    doc.write_text(second, encoding="utf-8", newline="\n")
    git(repo, "add", "notes.md")
    git(repo, "commit", "-q", "-m", "second")
    return repo


def non_ascii_path_repo(tmp_path: Path) -> Path:
    """A repo whose single tracked markdown file has a non-ASCII name, committed twice.

    `core.quotePath` is on by default, so `git ls-files` prints this path quoted and
    octal-escaped unless the caller asks for NUL-delimited output.
    """
    repo = init_repo(tmp_path)
    doc = repo / "café.md"
    doc.write_text("recall@5 is 0.92\n", encoding="utf-8", newline="\n")
    git(repo, "add", "café.md")
    git(repo, "commit", "-q", "-m", "first")
    doc.write_text("recall@5 is 0.945\n", encoding="utf-8", newline="\n")
    git(repo, "add", "café.md")
    git(repo, "commit", "-q", "-m", "second")
    return repo


def directory_then_file_repo(tmp_path: Path) -> Path:
    """A repo where `notes.md` is first a directory, then a regular file, both tracked.

    `git show <sha>:<dir>` exits zero and prints a tree listing. That is not memo text.
    """
    repo = init_repo(tmp_path)
    (repo / "notes.md").mkdir()
    (repo / "notes.md" / "inner.txt").write_text("inner\n", encoding="utf-8", newline="\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "notes.md is a directory here")
    shutil.rmtree(repo / "notes.md")
    (repo / "notes.md").write_text("recall@5 is 0.945\n", encoding="utf-8", newline="\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "notes.md is a file here")
    return repo


def symlink_then_file_repo(tmp_path: Path) -> Path:
    """A repo where `notes.md` is first a symlink, then a regular file, both tracked.

    `cat-file -t` reports `blob` for a symlink same as a real file, so the mode is the only
    thing that separates them.

    Built through the index rather than the filesystem, because Windows checkouts do not
    reliably create real symlinks.

    This machine has `core.symlinks` globally false. With it off, `git add -A` over a path
    already recorded at mode 120000 keeps mode 120000 instead of moving to 100644, so both
    revisions get excluded and the test reads empty, which looks like the guard misfiring when
    in fact the fixture never produced a regular file. The repo-local override below is what
    makes the second commit actually land at 100644.
    """
    repo = init_repo(tmp_path)
    git(repo, "config", "core.symlinks", "true")
    link_sha = _git_stdin(repo, "hash-object", "-w", "--stdin", input="README.md").strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"120000,{link_sha},notes.md")
    git(repo, "commit", "-q", "-m", "notes.md is a symlink here")
    (repo / "notes.md").write_text("recall@5 is 0.945\n", encoding="utf-8", newline="\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "notes.md is a file here")
    return repo
