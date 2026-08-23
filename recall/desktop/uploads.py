"""Bounded upload staging used by the desktop to hand files to an MCP runtime."""

from __future__ import annotations

import base64
import logging
import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path


_LOG = logging.getLogger("recall.desktop.uploads")


class UploadError(ValueError):
    """Raised when a desktop upload is malformed or exceeds its safety limits."""


def delete_staged_sources(tenant: str, sources: Iterable[str]) -> int:
    """Unlink staged upload files whose DB rows were just erased. Returns files removed.

    Without this, "permanently delete" left the original text on the server filesystem,
    inside the index root, where the next index run over `uploads/` would re-ingest exactly
    the content the caller was told is gone. Best effort by design (the DB delete is already
    committed), and hard-confined to `RECALL_INDEX_ROOT/uploads/<tenant>/`: a forgotten
    source indexed from the user's own directory must NEVER delete the user's file.

    Accepts both source spellings: `file://` URIs (generation-mode manifests) and plain
    absolute paths (the legacy `source` column). Lives HERE beside `stage_uploads` because
    recall_mcp is documented, and AST-checked, as making zero direct file-write calls.
    """
    from urllib.parse import urlsplit
    from urllib.request import url2pathname

    uploads_root = Path(os.environ.get("RECALL_INDEX_ROOT", ".")).resolve() / "uploads" / tenant
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
            directory.rmdir()  # only succeeds when empty, which is the point
        except OSError:
            pass
    return removed


def discard_staging(root: Path) -> None:
    """Remove one job's staging tree, best effort. The failure-path counterpart of
    `stage_uploads`: a refused or failed ingest must not leave content inside the index
    root for a later index run to resurrect."""
    shutil.rmtree(root, ignore_errors=True)


_MAX_TOTAL_BYTES = 50 * 1024 * 1024


def stage_uploads(tenant: str, files: list[dict[str, str]]) -> tuple[str, Path, int]:
    """Decode a bounded upload into the configured server-side staging root.

    Returns (job_id, staging_root, total_bytes). The byte total is what the caller
    debits against the tenant's index-bytes quota; measuring it here keeps the
    figure identical to what was actually written.

    On any refusal the job's staging directory is removed, so a rejected upload
    never leaves partial files behind for a later index run to pick up.
    """
    if not files or len(files) > 500:
        raise UploadError("files must contain between 1 and 500 entries")
    job_id = uuid.uuid4().hex
    root = Path(os.environ.get("RECALL_INDEX_ROOT", ".")).resolve() / "uploads" / tenant / job_id
    total = 0
    seen: set[str] = set()
    root.mkdir(parents=True, exist_ok=True)
    try:
        for item in files:
            name = Path(item.get("name", "")).name
            encoded = item.get("content_b64", "")
            if not name or not encoded:
                raise UploadError("every uploaded file needs name and content_b64")
            if name in seen:
                # Two entries with one name would silently drop the first from the
                # indexed set; refuse rather than guess which one was meant.
                raise UploadError(f"duplicate file name {name!r} in one upload")
            seen.add(name)
            # Refuse an oversized entry before materialising it: base64 encodes 3
            # bytes into 4 characters, so the encoded length bounds the decoded
            # size without decoding. The post-decode check below stays as the
            # exact enforcement.
            if (len(encoded) // 4) * 3 > _MAX_TOTAL_BYTES - total:
                raise UploadError("upload exceeds the 50 MiB request limit")
            try:
                data = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise UploadError(f"invalid base64 content for {name!r}") from exc
            total += len(data)
            if total > _MAX_TOTAL_BYTES:
                raise UploadError("upload exceeds the 50 MiB request limit")
            (root / name).write_bytes(data)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return job_id, root, total
