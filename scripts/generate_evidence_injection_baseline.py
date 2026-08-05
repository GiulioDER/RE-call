"""Regenerate `results/evidence_injection_baseline.json`.

Run from the repository root as a module, so `benchmarks` is importable::

    cd <repo root>
    python -m scripts.generate_evidence_injection_baseline

Run this ONLY when the frozen suite has deliberately changed — a new payload, a new carrier, a new
detector. Regenerating it after a code change instead is how a ratchet becomes a rubber stamp: the
whole point of the artifact is that a later session can show the injection rate did not increase,
and a rate re-recorded from whatever the code currently does proves nothing.

LOOK AT THE DIFF. `escapes` going up, or `positive_control_escapes` going down, means the change
should not be committed. The generator writes with an explicit `\n` newline so it produces an
identical file on Windows and Linux.
"""
from __future__ import annotations

import json
import os
import tempfile

from benchmarks.evidence_injection import BASELINE_PATH, build_baseline


def main() -> int:
    baseline = build_baseline()
    payload: dict[str, object] = {
        "_note": (
            "Frozen adversarial baseline for the evidence prompt boundary. `escapes` may only "
            "SHRINK; `positive_control_escapes` may only GROW. `suite_digest` pins the payload, "
            "carrier and detector lists, so this file cannot be reconciled with an edited suite "
            "by accident. Regenerate with scripts/generate_evidence_injection_baseline.py."
        ),
        "_measures": (
            "Whether corpus-controlled bytes can leave the delimited data region and reach the "
            "model's instruction channel. NOT whether a model is persuaded — that needs a "
            "generator, and none is approved for this program."
        ),
    }
    payload.update(baseline)
    # Same-directory temp file plus os.replace: an interrupted run must never leave a truncated
    # artifact at the real path, and same-filesystem is what makes the replace atomic on Windows.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(BASELINE_PATH.parent), prefix=".evidence_injection_baseline.", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=False)
            handle.write("\n")
        os.replace(tmp_name, BASELINE_PATH)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    print(
        f"wrote {BASELINE_PATH}: {baseline['escapes']}/{baseline['trials']} escapes, "
        f"positive control {baseline['positive_control_escapes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
