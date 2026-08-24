"""Writing an indexed upload, under a root the caller configures."""

import os
from pathlib import Path


def index_root() -> Path:
    """The configured root. It must already exist: creating it is the caller's job."""

    root = Path(os.environ["RECALL_INDEX_ROOT"])
    if not root.is_dir():
        raise RuntimeError(f"RECALL_INDEX_ROOT does not exist: {root}")
    return root


def store_memo(job_id: str, text: str) -> Path:
    """Write one memo for a job and return where it landed."""

    target = index_root() / "uploads" / job_id / "memo.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target
