"""What a result artifact must say about the corpus it measured.

No result JSON in this repo records its corpus row count. On 2026-07-27 two concurrent runs
doubled the LOCOMO corpus (11,764 rows against a correct 5,882); every depth came in ~0.05 low
and nothing errored. #103 later verified 5,882 rows in scripts/run_locomo_arms.sh — and the
number went to the runner's stdout, so its own artifacts cannot show it.

A verification that does not reach the artifact protects only the session that ran it.
"""
from __future__ import annotations

import subprocess
from typing import Any


def _git_sha() -> str | None:
    """Short SHA of the tree that produced this result, or None outside a repo.

    Degrades to None rather than raising or inventing: a result file from a tarball is still a
    result, and a wrong sha is worse than an absent one.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def provenance_block(corpus_rows: int, table: str, tenants: list[str]) -> dict[str, Any]:
    """The block every eval result embeds so a later reader can check it.

    `corpus_rows` is the summed tenant-scoped count actually measured, not a configured or
    expected value — an expectation copied into a result proves nothing about the run.
    """
    return {
        "corpus_rows": corpus_rows,
        "table": table,
        "tenants": sorted(tenants),
        "git_sha": _git_sha(),
    }
