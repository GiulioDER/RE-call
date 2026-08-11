"""One audited implementation of "replace a file the user wrote, or leave it alone".

Two modules in this package now edit a memo in place — `recall/fix.py`, which declares a
`supersedes:` edge a memo already states in prose, and `recall/rewrite.py`, which writes back a
reviewed promotion. Both are writing over a document nobody else has a copy of.

The sequence below is four steps and every one of them is load-bearing: `mkstemp` in the target's
own directory so the swap is a same-filesystem rename, `fsync` so the staged bytes are on the
platter before the rename claims they are, `copymode` so the swap does not silently tighten (or,
as root, re-own) the user's file, and `os.replace` so the target is either the old file or the new
one and never a truncation in between. A second copy of that sequence is how one of them ends up
without the `copymode` — so there is one copy, here, and the callers import it.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically: on any failure the original file is left intact.

    `Path.write_bytes` opens the target with mode ``'wb'``, which truncates it to zero bytes at
    open — before a single byte of new content is written. A crash / disk-full / I/O error in that
    window leaves the original truncated or half-written and unrecoverable. Instead, stage the new
    content in a sibling temp file and replace the target in a single ``os.replace``. If anything
    fails, the temp file is removed and the original is never touched.

    Bytes rather than text is the whole point of this being the primitive. Text mode applies the
    platform's newline translation, so an identical `str` becomes different files on Windows and
    Linux, and a caller that has just gone to the trouble of measuring a memo's existing line
    endings would have them overwritten on the way out.
    """
    # Resolved BEFORE staging, so both the temp file's directory and the rename act on the real
    # document. `os.replace` does not follow a symlink: pointed at one it destroys the link and
    # leaves a regular file shadowing the memo, which then silently diverges from the file the
    # user is actually editing. Vaults that link notes in from another tree are common enough.
    path = Path(os.path.realpath(path))
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        try:
            handle = os.fdopen(tmp_fd, "wb")
        except BaseException:
            # `fdopen` only takes ownership of the descriptor once it succeeds, and what makes it
            # fail is running out of descriptors — the condition a leak compounds. On Windows the
            # leaked handle also makes the cleanup `unlink` below fail, so the temp file is left
            # sitting beside the user's memo.
            os.close(tmp_fd)
            raise
        with handle as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            # The temp file is created 0600 by mkstemp; carry the original's permission bits over
            # so the atomic swap does not silently tighten (or, as root, re-own) the user's memo.
            shutil.copymode(path, tmp_name)
        os.replace(tmp_name, path)  # atomic on POSIX and on Windows for a same-directory target
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    _flush_directory(path.parent)


def _flush_directory(directory: Path) -> None:
    """`fsync` the directory so the RENAME is durable, not just the bytes it points at.

    Fsyncing the file puts its contents on the platter; on ext4 and xfs the directory entry
    created by `os.replace` can still be in the page cache, so a power loss reverts the memo to
    its pre-edit content after the caller has already been told the edit was applied.

    Best effort by necessity: a directory cannot be opened for `fsync` on Windows, and a failure
    here means the swap already happened and cannot be undone, so there is nothing to report and
    nothing to roll back.
    """
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, data: str) -> None:
    """`atomic_write_bytes` for a caller holding a `str`, encoded UTF-8 with no translation.

    The line endings in `data` are the line endings written. That is a change in behaviour from
    the text-mode write this replaced, which on Windows turned every ``\\n`` in the string into
    ``\\r\\n`` on disk — so the same input produced different bytes depending on where the tool
    ran, and a caller could not tell what it had written without reading the file back.
    """
    atomic_write_bytes(path, data.encode("utf-8"))


__all__ = ["atomic_write_bytes", "atomic_write_text"]
