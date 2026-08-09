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
    # Per-metric n, because the three metrics do NOT always share a denominator: RAGAS can time out
    # on a row and leave RL_F null while RB_llm and RB_agg are fine. Reporting only len(rows) hid
    # that the Task B abstain harmonic mean averaged RL_F over 832 rows and the other two over 842.
    # 🔑 A rate is named by its denominator, so the denominator has to be in the output.
    defined = {m: 0 for m in METRICS}
    null_raw = {m: 0 for m in METRICS}
    with dst.open("w", encoding="utf-8") as fh:
        for r in rows:
            m = r.get("metrics") or {}
            # Select the key by PRESENCE, not by truthiness of the unwrapped value. `or` would
            # treat a falsy-but-present label ("" or []) as absent and fall through to None, which
            # reinstates the exact broken fall-through this script exists to repair.
            if "Answerability" in r:
                label = first(r["Answerability"])
            elif "answerability" in r:
                label = first(r["answerability"])
            else:
                label = None
            if label is None:
                missing_label += 1
            idk = first(m.get("idk_eval"))
            und = first(m.get("underspecified_eval"))
            for metric in METRICS:
                raw = first(m.get(metric))
                if raw is None:
                    null_raw[metric] += 1
                # Compute unconditionally, exactly as upstream `get_idk_underspec_score` does. An
                # earlier `continue` here left the value the BROKEN scorer wrote: on an
                # UNANSWERABLE row with a null raw metric and idk_eval == 1 that keeps 0 where the
                # conditioned value is 1, which is the very inversion being repaired. It never
                # fired on our data only because all 12 null-RL_F rows are ANSWERABLE.
                fixed = idk_underspec(label, idk, und, raw)
                key = f"{metric}_idk_underspecified"
                before = first(m.get(key))
                if before != fixed:
                    changed[metric] += 1
                if fixed is not None:
                    defined[metric] += 1
                m[key] = [fixed]
            r["metrics"] = m
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input": str(src), "output": str(dst), "rows": len(rows),
        "rows_without_label": missing_label,
        "values_corrected": changed,
        "n_defined_per_metric": defined,
        "raw_metric_null": null_raw,
        "denominators_agree": len(set(defined.values())) == 1,
    }))
    # A missing label silently reinstates the original bug, so it is an error, not a note.
    return 1 if missing_label else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
