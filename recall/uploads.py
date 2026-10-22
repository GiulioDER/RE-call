"""Shared storage helpers for staged uploads."""

from __future__ import annotations

import os
import posixpath
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

from recall.observability import get_logger

_LOG = get_logger("desktop.uploads")


def _location(source: str) -> str:
    """Where a source is, as one comparable string, a local path's `.` and `..` segments folded.

    A `file://` URI becomes the local path it names through `recall.manifest.local_path_for`, the
    one decoder for that question. It keeps a Windows drive letter (`file:///C:/x` is `C:/x`;
    `urlsplit` gave `/C:/x`, so no path prefix ever matched the `as_uri()` sources desktop ingest
    and manifests store) and a UNC host (`file://nas1/share/x` is `//nas1/share/x`, not a local
    `/share/x`). A file URI it refuses stays as written, which matches no path prefix. Any other
    URI is `scheme://netloc/path` with its path unquoted and NOT folded: in an object store `//`
    and `..` are characters of a key, so `s3://b/memory/../x.md` is an object under `memory/`
    (review of PR 935, 2026-10-18; review of PR 946, 2026-10-19).
    """
    if "://" not in source:
        return _fold(Path(source).as_posix())
    parts = urlsplit(source)
    if parts.scheme == "file":
        from recall.manifest import ObjectNotAllowed, local_path_for

        try:
            return _fold(local_path_for(source).as_posix())
        except ObjectNotAllowed:
            return source
    return f"{parts.scheme}://{parts.netloc}{unquote(parts.path)}"


def _fold(path: str) -> str:
    """A local `path` with `.` and `..` folded, so a file cannot sit under an admitted prefix by
    spelling alone: `/root/memory/../docs/x.md` is `/root/docs/x.md`. Never an object key's path."""
    return posixpath.normpath(path) if path else path


def _runs_through_uploads(parts: Sequence[str], tenant: str) -> bool:
    """Whether a path passes through `uploads/<tenant>/`, where every `recall_ingest` upload is
    staged below an index root (`recall.uploads._tenant_uploads_root`).

    Compared without case: on a case-insensitive filesystem staging writes into an `Uploads/`
    that already exists, and the indexed path carries that case. It only ever fails closed.
    """
    folded = tenant.casefold()
    return any(
        parts[i].casefold() == "uploads" and parts[i + 1].casefold() == folded
        for i in range(len(parts) - 1)
    )


def source_location(source: str) -> str:
    """`source` as one comparable local path or URI (`_location`): a `file://` URI decoded, so
    `my%20notes.md` reads `my notes.md` and ends in the file name a chunk records."""
    return _location(source)


def in_upload_staging(source: str, tenant: str) -> bool:
    """Whether `source` lies under `uploads/<tenant>/`, where `recall_ingest` stages every upload,
    read with the one location decoder (`_location`), as `StatingRule.admits` reads it."""
    return _runs_through_uploads(Path(_location(source)).parts, tenant)


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
