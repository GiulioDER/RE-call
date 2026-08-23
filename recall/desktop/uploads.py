"""Bounded upload staging used by the desktop to hand files to an MCP runtime."""

from __future__ import annotations

import base64
import os
import shutil
import uuid
from pathlib import Path


class UploadError(ValueError):
    """Raised when a desktop upload is malformed or exceeds its safety limits."""


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
