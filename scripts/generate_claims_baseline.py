"""Regenerate `results/CLAIMS_BASELINE.json`.

Run this ONLY when deliberately re-freezing the baseline, and LOOK AT THE SIZE OF THE DIFF before
committing: a regeneration that silently drops entries turns the ratchet into a rubber stamp. The
generator writes the file with an explicit `\n` newline, so it produces an identical file on
Windows and Linux regardless of the platform default line ending or any git checkout filter.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from benchmarks.claim_gate import RESULTS_ROOT, build_baseline


def main() -> int:
    baseline = build_baseline()
    total = sum(sum(counts.values()) for counts in baseline.values())
    payload: dict[str, Any] = {
        "_note": (
            "Frozen multiset of unmarked numbers per gated document. This file may only SHRINK. "
            "Mark a number and remove its row; never add a row."
        )
    }
    payload.update(baseline)
    path = RESULTS_ROOT / "CLAIMS_BASELINE.json"
    # Write to a temp file in the same directory, then atomically replace it — an interrupted run
    # (Ctrl-C, crash, disk full mid-write) must never leave a truncated or half-written ratchet
    # artifact sitting at the real path. `dir=RESULTS_ROOT` keeps the temp file on the same
    # filesystem as the target, which is what makes `os.replace` atomic on both POSIX and Windows.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(RESULTS_ROOT), prefix=".CLAIMS_BASELINE.", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise
    print(f"wrote {path} - {total} unmarked numbers across {len(baseline)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
