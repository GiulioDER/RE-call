"""The ``.recall`` directory RE-call keeps a folder's own state in, and the file that keeps it local.

The lite store, its SQLite sidecars and the review ledgers (edits, rejections, stale reports, lint
dismissals) are written to ``<root>/.recall/``, and ``<root>`` is usually somebody's memo
repository. Those files hold memo text and vectors, so a ``git add .`` there would publish them.
A ``.gitignore`` of ``*`` inside the directory ignores everything in it, itself included, without
touching the repository's own ignore file.
"""

from __future__ import annotations

from pathlib import Path

STATE_DIR = ".recall"
IGNORE_NAME = ".gitignore"
IGNORE_TEXT = (
    "# Written by RE-call: this directory holds memo text, vectors and review ledgers.\n"
    "# Edit or empty this file to track it; RE-call never rewrites one that exists.\n"
    "*\n"
)


def make_parent(path: Path) -> None:
    """Create ``path``'s directory; when that is a ``.recall`` directory, give it a ``.gitignore``.

    The ignore file is written whenever it is absent, so a directory created before this existed
    is covered the next time a writer opens it, and never overwritten once present.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.name != STATE_DIR:
        return
    try:
        with open(parent / IGNORE_NAME, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(IGNORE_TEXT)
    except FileExistsError:
        pass
