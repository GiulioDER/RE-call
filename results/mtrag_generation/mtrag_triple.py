"""RE-call's MTRAG triple and its harmonic mean, computed two ways on purpose.

The MTRAG paper ranks Tasks B and C by the harmonic mean of RL_F, RB_llm and RB_alg. Two choices
in getting there are not obvious, and both are reported rather than picked silently:

1. WHICH VARIANT. The scored file carries `RL_F` and `RL_F_idk_underspecified` (and the same for
   RB_llm / RB_agg). The conditioned variants apply the IDK and underspecified judgements. Which one
   the published table used is not something to assume, so both are printed; whichever is chosen,
   the BASELINES must be recomputed from RAG.json with the SAME field, which is what makes this an
   anchored lift instead of a mismatched comparison.

2. THE MISSING ROWS. 12 of 842 RL_F values are absent, from 12 RAGAS TimeoutErrors during scoring.
   Averaging RL_F over 830 while averaging the other two over 842 mixes populations. So both are
   shown: `all` (each metric over whatever it has) and `complete-case` (only rows where all three
   exist). If they disagree, the missing rows are not missing at random and that matters more than
   the number.

⚠️ The aggregation is a harmonic mean OF THE PER-METRIC MEANS, matching the formula prior work
verified reproduces a self-reported triple to four decimals.
"""

import json
import math
import statistics as st
import sys

TRIPLES = {
    "raw": ("RL_F", "RB_llm", "RB_agg"),
    "idk_conditioned": ("RL_F_idk_underspecified", "RB_llm_idk_underspecified",
                        "RB_agg_idk_underspecified"),
}


def value(row, key):
    v = (row.get("metrics") or {}).get(key)
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v) if isinstance(v, (int, float)) else None


def harmonic(vals):
    if any(v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def report(path, label):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    print(f"\n=== {label}  ({len(rows)} rows) ===")
    for variant, keys in TRIPLES.items():
        per_all = [[value(r, k) for r in rows] for k in keys]
        means_all = [st.mean([v for v in col if v is not None]) for col in per_all]
        ns = [sum(1 for v in col if v is not None) for col in per_all]

        complete = [r for r in rows if all(value(r, k) is not None for k in keys)]
        means_cc = [st.mean([value(r, k) for r in complete]) for k in keys]

        print(f"  {variant}")
        print(f"    {'metric':<34} {'all':>10} {'n':>5}   {'complete-case':>14}")
        for k, ma, n, mc in zip(keys, means_all, ns, means_cc):
            print(f"    {k:<34} {ma:>10.4f} {n:>5}   {mc:>14.4f}")
        print(f"    {'HARMONIC MEAN':<34} {harmonic(means_all):>10.4f} {len(rows):>5}   "
              f"{harmonic(means_cc):>14.4f}   (complete-case n={len(complete)})")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        path, _, label = arg.partition("=")
        report(path, label or path)
