"""Turn a repository's git history into the corpus shape real agent memory has.

A working tree holds only the correction. Real agent memory is append-only, so the retracted
claim and its replacement both sit in the index, and the retracted one is often the nearer
match. Auditing the tree cannot show that failure. Walking history restores it.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Revision:
    """One historical version of one tracked document."""

    path: str  # repo-relative
    sha: str  # short commit sha
    date: str  # ISO YYYY-MM-DD, commit author date
    body: str  # file content at that revision


def _git(repo: Path, *args: str) -> str:
    """Run one git command, or fail loudly enough that the operator knows what to exclude.

    Failures are fatal by design: an audit that silently drops revisions reports fewer
    contradictions than exist, and under-reporting is the one failure this tool cannot afford.
    Fatal is only useful if it names the document, though, so the path and the git error travel
    with the exception.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        # OSError too: a missing or unspawnable git never raises CalledProcessError, and that is
        # precisely the failure whose bare traceback tells the operator nothing.
        detail = (getattr(exc, "stderr", "") or str(exc)).strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {detail}") from exc
    return done.stdout


def tracked_markdown(repo: Path, glob: str = "**/*.md") -> list[str]:
    """Repo-relative paths of tracked files matching `glob`, sorted.

    The pathspec carries explicit `:(glob)` magic. Under git's default pathspec matching a
    literal `/` in the pattern has to appear in the path too, so a bare `**/*.md` silently
    skips every markdown file at the repository root, which on a small corpus can be all of
    them. A corpus audit that quietly reads nothing is the worst failure this tool has.

    NUL-delimited (`-z`) on purpose. `core.quotePath` is on by default, so a plain newline
    listing prints a non-ASCII path as a quoted, octal-escaped literal (`"docs/caf\\303\\251.md"`)
    rather than the real path, and that literal then matches nothing when handed to `git log`:
    the document is counted but silently contributes zero revisions. `-z` prints the raw bytes.
    """
    out = _git(repo, "ls-files", "-z", "--", f":(glob){glob}")
    return sorted(path for path in out.split("\0") if path)


def revisions(repo: Path, rel_path: str) -> list[Revision]:
    """Every revision of `rel_path` under that exact name, oldest first.

    Deliberately not `--follow`: it reports commits from before a rename, where
    `git show <sha>:<rel_path>` does not resolve. A rename therefore starts a fresh history
    rather than producing a hole, which is the safer failure for an audit.
    """
    log = _git(repo, "log", "--format=%h %ad", "--date=short", "--", rel_path)
    out: list[Revision] = []
    for line in log.splitlines():
        if not line.strip():
            continue
        sha, _, date = line.partition(" ")
        sha = sha.strip()
        entry = _git(repo, "ls-tree", sha, "--", rel_path).strip()
        if not entry:
            continue  # the commit that deleted the file: nothing to read at that revision
        mode = entry.split(" ", 1)[0]
        if not mode.startswith("100"):
            # Only 100644 and 100755 are regular files. 040000 is a directory, 120000 a symlink,
            # 160000 a submodule. `git show` exits zero on all three and prints a tree listing or
            # a link target, which would enter the corpus as fabricated memo text. The mode is
            # the only thing that separates them: `cat-file -t` calls a symlink a blob.
            continue
        # Deliberately outside any `except`. After the mode check a read failure has no benign
        # explanation, and an audit that silently drops revisions undercounts the contradictions
        # it exists to find. Failing loudly is the same choice this library makes at retrieval.
        body = _git(repo, "show", f"{sha}:{rel_path}")
        out.append(Revision(path=rel_path, sha=sha, date=date.strip(), body=body))
    out.reverse()
    return out


def memo_stem(rel_path: str, date: str, sha: str) -> str:
    """Stable unique stem for one revision. Also the target a successor supersedes.

    `recall.frontmatter.supersedes_key` matches on the stem, so this is what a `supersedes:`
    line has to carry.
    """
    flat = rel_path.replace("/", "_").replace("\\", "_")
    if flat.lower().endswith(".md"):
        flat = flat[:-3]
    return f"{flat}__{date}__{sha}"


def write_history_corpus(repo: Path, rel_paths: Iterable[str], out_dir: Path) -> list[Path]:
    """Write one memo per revision. Each carries `valid_from`, and supersedes the one before it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rel_path in rel_paths:
        revs = revisions(repo, rel_path)
        for i, rev in enumerate(revs):
            lines = ["---", f"valid_from: {rev.date}"]
            if i:
                prev = revs[i - 1]
                lines.append(f"supersedes: {memo_stem(prev.path, prev.date, prev.sha)}")
            lines += ["---", "", rev.body]
            target = out_dir / f"{memo_stem(rev.path, rev.date, rev.sha)}.md"
            target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
            written.append(target)
    return written
