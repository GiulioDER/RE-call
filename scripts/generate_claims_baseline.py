"""Regenerate `results/CLAIMS_BASELINE.json`.

Run this ONLY when deliberately re-freezing the baseline, and LOOK AT THE SIZE OF THE DIFF before
committing: a regeneration that silently drops entries turns the ratchet into a rubber stamp. The
generator normalises newlines, so it produces an identical file on Windows and Linux.
"""
from __future__ import annotations

import json
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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} - {total} unmarked numbers across {len(baseline)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
