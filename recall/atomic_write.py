"""The one audited implementation of "replace a user's own file without risking it".

Two independent code paths now rewrite documents the user wrote by hand — `recall.fix`, which
declares a `supersedes:` edge a memo already states in prose, and `recall.rewrite`, which writes
back a reviewed promotion. A copy-pasted mkstemp/fsync/copymode/os.replace sequence is precisely
how the second one ends up without the `copymode`, and the failure is silent: the file is correct,
its permissions are not, and nothing reads permissions until it matters.

`Path.write_text` is the thing being avoided. It opens the target with mode ``'w'``, which
truncates it to zero bytes *at open* — before a single byte of new content is written. A crash,
a full disk or an I/O error inside that window leaves the original truncated and unrecoverable.
Here the new content is staged in a sibling temp file (same directory, so the swap is a
same-filesystem rename) and committed with a single ``os.replace``. On any failure the temp file
is removed and the original is never touched.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace `path` with `data`, atomically, preserving its permission bits.

    Byte-oriented on purpose. The text wrapper below cannot express "leave this file's line
    endings and byte-order mark exactly as the author wrote them", and both of those are things
    a writer annotating someone's memo has no business changing.
    """
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            # mkstemp creates the staging file 0600. Carrying the original's bits over stops the
            # swap from silently tightening the user's memo — or, running as root, re-owning it.
            shutil.copymode(path, tmp_name)
        os.replace(tmp_name, path)  # atomic on POSIX, and on Windows for a same-directory target
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, data: str) -> None:
    """Atomically write `data` as UTF-8, writing the newlines it actually contains.

    Encoding here rather than deferring to text mode is not incidental. ``open(..., 'w')`` uses
    ``newline=None``, which rewrites every ``\\n`` to ``os.linesep`` on write — so on Windows this
    function's predecessor converted the *entire* file to CRLF as a side effect of inserting one
    line. No test caught it, because `Path.read_text` translates the endings back on the way in,
    so the round trip through the assertion is lossless while the file on disk is not.
    """
    atomic_write_bytes(path, data.encode("utf-8"))


__all__ = ["atomic_write_bytes", "atomic_write_text"]
