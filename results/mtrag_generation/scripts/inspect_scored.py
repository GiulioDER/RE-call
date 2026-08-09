"""What is actually in a scored MTRAG file, and how much of it is missing.

The judge log for Task B carried 12 TimeoutError and 12 Exception during the RAGAS stage. A metric
that silently comes back NaN for a handful of rows still produces a mean, and that mean is wrong in
a direction nobody can see. So count the missing and the NaN per field BEFORE reading any number.
"""

import collections
import json
import math
import sys


def stats(rows, getter, label):
    vals, missing, nan = [], 0, 0
    for r in rows:
        v = getter(r)
        if isinstance(v, list):
            v = v[0] if v else None
        if v is None:
            missing += 1
        elif isinstance(v, bool):
            vals.append(float(v))
        elif isinstance(v, float) and math.isnan(v):
            nan += 1
        elif isinstance(v, (int, float)):
            vals.append(float(v))
        else:
            missing += 1
    mean = sum(vals) / len(vals) if vals else float("nan")
    print(f"  {label:28} n={len(vals):4} missing={missing:4} nan={nan:4} mean={mean:.4f}")


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"file: {path}")
    print(f"rows: {len(rows)}")

    top = collections.Counter()
    for r in rows:
        top.update(r.keys())
    scoreish = sorted(
        k for k in top
        if any(s in k.lower() for s in ("rl_f", "rb_", "idk", "faith", "underspec", "conditioned"))
    )
    print("score-bearing top-level keys:", scoreish)

    met = collections.Counter()
    for r in rows:
        met.update((r.get("metrics") or {}).keys())
    print("metrics keys:", sorted(met))
    print()

    for k in scoreish:
        stats(rows, lambda r, k=k: r.get(k), k)
    for k in sorted(met):
        stats(rows, lambda r, k=k: (r.get("metrics") or {}).get(k), "metrics." + k)


if __name__ == "__main__":
    main(sys.argv[1])
