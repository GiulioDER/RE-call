"""One audited implementation of "replace a file the user wrote, or leave it alone".

One module in this package edits a memo in place today: `recall/fix.py`, which declares a
`supersedes:` edge a memo already states in prose. It is writing over a document nobody else has a
copy of. This lives in its own module rather than inside that one so the next such writer imports
the sequence instead of reimplementing it.

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
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
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


__all__ = ["atomic_write_bytes"]
