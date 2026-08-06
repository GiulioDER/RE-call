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
from pathlib import Path
from typing import Any

from benchmarks.claim_gate import RESULTS_ROOT, build_baseline


def _refuse_a_foreign_checkout() -> None:
    """Refuse to re-freeze a checkout other than the one the caller is standing in.

    This script is run as a FILE, so `sys.path[0]` is `scripts/`, not the repository root.
    `benchmarks` therefore resolves through the environment -- and with RE-call pip-installed
    in editable mode, that is whichever checkout the editable install points at, which is
    usually not this one. `RESULTS_ROOT` is derived from the imported module's `__file__`, so
    the ratchet gets rewritten in the OTHER worktree while the diff you are about to review is
    in this one.

    It is a quiet failure in the worst place: the ratchet is the mechanism that stops unbacked
    numbers, both checkouts end up with a baseline nobody inspected, and the run prints
    success. Observed on 2026-08-06, which is why this exists.

    The checkout to compare against is THIS FILE's own, which needs no subprocess and
    cannot fail open. A first version shelled out to `git rev-parse --show-toplevel` and had
    two holes, both of which make a guard pass rather than fire:

    * `git` printing an empty toplevel (bare repo, or run from inside `.git`) left
      `Path("").resolve()`, which is the CWD -- so the guard silently degraded into
      comparing the target against the current directory, something it had never verified
      was a repository at all.
    * `is_relative_to` only asks for containment, so a checkout NESTED under the current one
      passed. That is a normal layout here: this project keeps its worktrees under the
      parent checkout.

    Requiring the two repository roots to be EQUAL closes both, and dropping the subprocess
    removes the swallowed `OSError` that disabled the guard entirely on a box with no `git`
    on PATH -- which is exactly where an editable install is most likely to be the only
    thing resolving `benchmarks`. The check does not involve the CWD, so running from a
    subdirectory still works.
    """
    here = Path(__file__).resolve().parents[1]
    target_root = RESULTS_ROOT.resolve().parent
    if target_root != here:
        raise SystemExit(
            f"refusing to write outside this checkout.\n"
            f"  this script  : {here}\n"
            f"  would write  : {RESULTS_ROOT.resolve() / 'CLAIMS_BASELINE.json'}\n"
            f"  `benchmarks` resolved to: {target_root}\n"
            f"This usually means an editable install points `benchmarks` at another worktree.\n"
            f"Re-run from the repository root with this checkout first on the path:\n"
            f"    PYTHONPATH=. python scripts/generate_claims_baseline.py"
        )


def main() -> int:
    _refuse_a_foreign_checkout()
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
