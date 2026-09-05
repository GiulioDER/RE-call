"""Shared storage helpers for staged uploads."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from recall.observability import get_logger

_LOG = get_logger("desktop.uploads")


def _tenant_uploads_root(tenant: str) -> Path:
    """Return the confined staged upload root for one tenant."""
    return Path(os.environ.get("RECALL_INDEX_ROOT", ".")).resolve() / "uploads" / tenant


def delete_staged_sources(tenant: str, sources: Iterable[str]) -> int:
    """Unlink staged upload files whose database rows were just erased."""
    from urllib.parse import urlsplit
    from urllib.request import url2pathname

    uploads_root = _tenant_uploads_root(tenant)
    removed = 0
    touched_dirs: set[Path] = set()
    for source in sources:
        raw = str(source)
        if raw.startswith("file://"):
            raw = url2pathname(urlsplit(raw).path)
        try:
            path = Path(raw).resolve()
        except (OSError, ValueError):
            continue
        if not path.is_relative_to(uploads_root):
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            _LOG.warning("could not remove staged file for a forgotten source: %s", path)
            continue
        removed += 1
        touched_dirs.add(path.parent)
    for directory in touched_dirs:
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed
