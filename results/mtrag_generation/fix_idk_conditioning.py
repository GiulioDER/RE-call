"""Recompute the IDK-conditioned metrics that the official scorer silently failed to compute.

THE BUG, and it invalidates every comparison made before it was found.

`judge_wrapper.get_idk_underspec_score` reads the label as **lower-case** `answerability`:

    answerability_vals = row.get("answerability", [])

The release files (`reference.jsonl`, `RAG.jsonl`) spell it **capitalised** `Answerability`; only
`scripts/evaluation/sample_data/responses-10.jsonl` uses the lower-case form. Our scoring input is
built from the release, so the key was never found: the scorer printed
`Error: answerability is None` 2526 times (842 rows x 3 metrics) and matched a label ZERO times.

With `answerability = None` every label branch is skipped and the function falls through to
`elif idk_eval == 1: return 0`, which PENALISES an abstention instead of rewarding a correct one.
That is why our `_idk_underspecified` values came out BELOW the raw ones, which should have been
the tell.

⚠️ WHY IT MATTERS FOR THE COMPARISON. The published baselines in `RAG.json` carry the CONDITIONED
metric: `rb_agg` is exactly 1.0 on 120/550 UNANSWERABLE and 100/100 CONVERSATIONAL rows and never
on ANSWERABLE or PARTIAL. Comparing our RAW metric against their CONDITIONED one compares two
different quantities, and it understates us on exactly the tasks where a correct abstention is
worth a full point.

This recomputes the conditioning faithfully from fields already present, reading the capitalised
key. No API call, no GPU, no re-generation: `idk_eval`, `underspecified_eval` and the raw metrics
were all computed correctly, and only the final combination step was broken.

Branch order is copied verbatim from the upstream function so behaviour matches for any row where
the label IS found. MTRAG carries no UNDERSPECIFIED bucket, so that branch is unreachable here and
is kept only for fidelity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = ("RL_F", "RB_llm", "RB_agg")


def first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def idk_underspec(answerability, idk_eval, underspec_eval, value):
    """Verbatim port of `judge_wrapper.get_idk_underspec_score`, reading the label that exists."""
    if answerability in ["UNDERSPECIFIED"] and underspec_eval == 1:
        return 1
    elif answerability in ["UNDERSPECIFIED"] and underspec_eval != 1:
        return 0
    elif answerability in ["UNANSWERABLE", "CONVERSATIONAL"] and idk_eval == 1:
        return 1
    elif answerability in ["UNANSWERABLE", "CONVERSATIONAL"] and idk_eval in [0, 0.5]:
        return 0
    elif idk_eval == 1:
        return 0
    else:
        return value


def main(argv):
    src, dst = Path(argv[0]), Path(argv[1])
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]

    missing_label = 0
    changed = {m: 0 for m in METRICS}
    with dst.open("w", encoding="utf-8") as fh:
        for r in rows:
            m = r.get("metrics") or {}
            # The capitalised key, which is what the release actually ships.
            label = first(r.get("Answerability")) or first(r.get("answerability"))
            if label is None:
                missing_label += 1
            idk = first(m.get("idk_eval"))
            und = first(m.get("underspecified_eval"))
            for metric in METRICS:
                raw = first(m.get(metric))
                if raw is None:
                    continue
                fixed = idk_underspec(label, idk, und, raw)
                key = f"{metric}_idk_underspecified"
                before = first(m.get(key))
                if before != fixed:
                    changed[metric] += 1
                m[key] = [fixed]
            r["metrics"] = m
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input": str(src), "output": str(dst), "rows": len(rows),
        "rows_without_label": missing_label,
        "values_corrected": changed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
