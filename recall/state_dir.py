"""The ``.recall`` directory RE-call keeps a folder's own state in, and the file that keeps it local.

The lite store, its SQLite sidecars and the review ledgers (edits, rejections, stale reports, lint
dismissals) are written to ``<root>/.recall/``, and ``<root>`` is usually somebody's memo
repository. Those files hold memo text and vectors, so a ``git add .`` there would publish them.
A ``.gitignore`` of ``*`` inside the directory ignores everything in it, itself included, without
touching the repository's own ignore file.
"""

from __future__ import annotations

import os
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

    On POSIX a new ``.recall`` directory is made 0700: SQLite, its sidecars and the lanes create
    their files at the umask, so at the usual 022 every memo's text would be readable by every
    local account (CCA DEPLOY-003). A directory that exists keeps the mode its owner gave it.
    Windows is left to the folder's inherited ACL, because from Python 3.13 a mode of 0o700 there
    replaces that ACL with one that drops every other account, a sandbox's included.
    """
    parent = path.parent
    if parent.name != STATE_DIR:
        parent.mkdir(parents=True, exist_ok=True)
        return
    parent.mkdir(mode=0o700 if os.name == "posix" else 0o777, parents=True, exist_ok=True)
    write_ignore(parent)


def make_dir(directory: Path) -> None:
    """Create ``directory`` and its parents, making a missing ``.recall`` among them as `make_parent` does.

    An atomic rescue root defaults to ``<project>/.recall/atomic``, and on a PostgreSQL install it
    can be the first thing written there; a plain ``mkdir(parents=True)`` would give ``.recall``
    the umask mode, which every later writer then keeps (CCA ENV-001).
    """
    state = next((p for p in (directory, *directory.parents) if p.name == STATE_DIR), None)
    if state is not None:
        make_parent(state / IGNORE_NAME)
    directory.mkdir(parents=True, exist_ok=True)


def write_ignore(directory: Path) -> None:
    """Give an existing ``directory`` the ``.gitignore`` of ``*`` when it has none; never overwrite one."""
    try:
        with open(directory / IGNORE_NAME, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(IGNORE_TEXT)
    except FileExistsError:
        pass
