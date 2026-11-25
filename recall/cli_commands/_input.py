"""Read a command's JSON input from a path, or from standard input for ``-``, as bytes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import BinaryIO


def read_input_bytes(spec: str, limit: int | None = None) -> bytes:
    """Return the bytes of ``spec``, at most ``limit + 1`` of them when a limit is given.

    Reading one byte past the limit lets the caller tell an input at the limit from a longer one.
    Standard input is read as bytes, so it decodes like a file whatever the console code page. A
    missing standard input (``sys.stdin`` is None, as when the process starts with it closed) or one
    replaced by a text-only stream raises OSError, so every caller refuses it as it refuses an
    unreadable file.
    """
    size = -1 if limit is None else limit + 1
    if spec == "-":
        stream: BinaryIO | None = getattr(sys.stdin, "buffer", None)
        if stream is None:
            raise OSError("standard input is not readable as bytes")
        return stream.read(size)
    with Path(spec).open("rb") as handle:
        return handle.read(size)
