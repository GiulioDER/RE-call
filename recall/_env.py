"""Minimal, dependency-free `.env` loader for local dev.

Reads ``KEY=VALUE`` lines from a `.env` file into ``os.environ`` WITHOUT overriding variables that
are already set. The `.env` file is gitignored — it is for local secrets (e.g. VOYAGE_API_KEY),
never committed. Entry points call ``load_dotenv()`` so those keys are picked up.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Apply a `.env` ALL-OR-NOTHING: parse the whole file, then set the variables.

    This used to assign into `os.environ` inside the parse loop, one key at a time. A malformed
    line therefore left every earlier key applied and every later one dropped, and because the
    caller treats a failure as "no .env", that half-configured state was silent. Measured
    consequence: a file whose second line held a NUL byte set `VOYAGE_API_KEY` (live, billing)
    while `RECALL_SERVING_DSN` never arrived, so the DSN fell back to the local default, passed
    the insecure-DSN guard because it is local, and indexed into the wrong database.

    A NUL is the realistic trigger, from a `.env` truncated by a crash mid-write: it is valid
    UTF-8, so `read_text` succeeds and `os.environ.__setitem__` is what raises.
    """
    p = Path(path)
    if not p.exists():
        return
    pending: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        # `os.environ` rejects these outright, so catch them during the parse where refusing
        # costs nothing, rather than half way through applying.
        if "\x00" in key or "\x00" in val:
            raise ValueError(f"{p}: embedded null character in {key!r}")
        pending[key] = val
    for key, val in pending.items():
        if key not in os.environ:  # an exported variable always wins over the file
            os.environ[key] = val
