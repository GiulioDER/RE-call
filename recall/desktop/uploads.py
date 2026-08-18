"""Bounded upload staging used by the desktop to hand files to an MCP runtime."""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path


class UploadError(ValueError):
    """Raised when a desktop upload is malformed or exceeds its safety limits."""


def stage_uploads(tenant: str, files: list[dict[str, str]]) -> tuple[str, Path]:
    """Decode a bounded upload into the configured server-side staging root."""
    if not files or len(files) > 500:
        raise UploadError("files must contain between 1 and 500 entries")
    job_id = uuid.uuid4().hex
    root = Path(os.environ.get("RECALL_INDEX_ROOT", ".")).resolve() / "uploads" / tenant / job_id
    total = 0
    root.mkdir(parents=True, exist_ok=True)
    for item in files:
        name = Path(item.get("name", "")).name
        encoded = item.get("content_b64", "")
        if not name or not encoded:
            raise UploadError("every uploaded file needs name and content_b64")
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise UploadError(f"invalid base64 content for {name!r}") from exc
        total += len(data)
        if total > 50 * 1024 * 1024:
            raise UploadError("upload exceeds the 50 MiB request limit")
        (root / name).write_bytes(data)
    return job_id, root
