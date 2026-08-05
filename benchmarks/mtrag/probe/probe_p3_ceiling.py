"""Recompute the length-calibration ceiling with the answerability conditioning controlled.

PROBE_VERDICT.md quotes +0.068 on RB_alg, derived from band means over ALL instances. The
confound check showed the longest quartile carries 396 UNANSWERABLE instances against 22-39
elsewhere and a 20.7% zeroed-composite rate, so that +0.068 is inflated by conditioning.
This recomputes it on ANSWERABLE tasks with a non-zero composite, where conditioning cannot fire.

Prior work: none found on RB_alg / MTRAGEval generation / length calibration
(docs_search source_type=memory, "MT-RAG generation RB_alg RougeL response length
calibration answer verbosity benchmark", gap_warning false). Two adjacent hits:
[[project-recall-token-f1-harness-offset-2026-07-29]] and
[[project-recall-mtrag-symmetric-baseline-2026-08-04]]. Full search recorded in
benchmarks/PREREGISTRATION-mtrag-rbalg.md.
"""

import json
import re
import string
import sys
from statistics import mean
from typing import Any

PATH = sys.argv[1]


def norm_tokens(s: str) -> list[str]:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return [x for x in re.sub(r"\b(a|an|the)\b", " ", s).split() if x]


def slot(ann: dict[str, Any], key: str) -> float | None:
    node = ann.get(key)
    if not node or "composite" not in node or node["composite"] is None:
        return None
    value = node["composite"].get("value")
    return None if value is None else float(value)


def report(label: str, rows: list[dict[str, Any]]) -> float:
    rows = sorted(rows, key=lambda r: float(r["ratio"]))
    n = len(rows)
    q = n // 4
    qs = [("Q1 shortest", rows[:q]), ("Q2", rows[q:2*q]), ("Q3", rows[2*q:3*q]), ("Q4 longest", rows[3*q:])]
    overall = float(mean(float(r["rb"]) for r in rows))
    print(f"\n--- {label}   n={n}   overall mean rb_agg = {overall:.4f}")
    best: tuple[str, float] = ("", float("-inf"))
    for lab, b in qs:
        m = float(mean(float(r["rb"]) for r in b))
        print(f"    {lab:12} ratio {b[0]['ratio']:5.2f}..{b[-1]['ratio']:6.2f}   mean {m:.4f}   n={len(b)}")
        if m > best[1]:
            best = (lab, m)
    print(f"    best band: {best[0]} at {best[1]:.4f}")
    print(f"    CEILING if every response sat in the best band: {best[1]-overall:+.4f} on RB_alg")
    return best[1] - overall


def main() -> int:
    d = json.load(open(PATH, encoding="utf-8"))
    names = {}
    for m in d["models"]:
        names[(m.get("model_id") or m.get("id") or m.get("name")) if isinstance(m, dict) else m] = (
            (m.get("display_name") or m.get("name")) if isinstance(m, dict) else m
        )
    tasks = {t["task_id"]: t for t in d["tasks"]}

    def cls(tid: str) -> str:
        a = tasks.get(tid, {}).get("Answerability") or []
        return a[0] if isinstance(a, list) and a else "UNKNOWN"

    tgt = {t["task_id"]: len(norm_tokens(t["targets"][0]["text"])) for t in d["tasks"] if t.get("targets")}

    recs: list[dict[str, Any]] = []
    for ev in d["evaluations"]:
        if str(names.get(ev["model_id"], ev["model_id"])).lower() == "target":
            continue
        tl = tgt.get(ev["task_id"])
        pl = len(norm_tokens(ev.get("model_response") or ""))
        rb = slot(ev.get("annotations") or {}, "rb_agg")
        if not tl or not pl or rb is None:
            continue
        recs.append(dict(ratio=pl / tl, rb=rb, cls=cls(ev["task_id"])))

    print("=== length-calibration ceiling, uncontrolled vs controlled ===")
    a = report("(a) ALL instances -- as published in PROBE_VERDICT.md", recs)
    # BUG-003 (bug-auditor, 2026-08-05): (d)'s composite>0 filter deletes exactly the
    # wrongly-abstained-on-answerable cells, which the verdict itself identifies as the short
    # tail -- so the filter is confounded with the effect. (b) is the unfiltered control and
    # was previously computed for the gap but never for the ceiling.
    b = report("(b) ANSWERABLE only -- unfiltered control",
               [r for r in recs if r["cls"] == "ANSWERABLE"])
    dd = report("(d) ANSWERABLE and composite > 0 -- conditioning cannot fire",
                [r for r in recs if r["cls"] == "ANSWERABLE" and r["rb"] > 0])

    print(f"\n  published ceiling      (a) {a:+.4f}")
    print(f"  unfiltered control     (b) {b:+.4f}")
    print(f"  controlled ceiling     (d) {dd:+.4f}")
    print(f"  inflation from conditioning: {a-dd:+.4f}  ({(a-dd)/a:.0%} of the published figure)")

    # BUG-002 (bug-auditor): report() takes the MAX of four band means, which is positive by
    # construction even under the null. Permutation null, so the reader can subtract it.
    import random

    print("\n=== BUG-002 selection-bias null: shuffle the ratio column, keep everything else ===")
    for label, subset in (("(a) all", recs),
                          ("(d) ANSWERABLE & >0", [r for r in recs if r["cls"] == "ANSWERABLE" and r["rb"] > 0])):
        rng = random.Random(20260805)
        nulls = []
        for _ in range(200):
            ratios = [r["ratio"] for r in subset]
            rng.shuffle(ratios)
            shuf = [dict(ratio=x, rb=r["rb"]) for x, r in zip(ratios, subset)]
            shuf.sort(key=lambda r: r["ratio"])
            n = len(shuf)
            q = n // 4
            qs = [shuf[:q], shuf[q:2*q], shuf[2*q:3*q], shuf[3*q:]]
            overall = mean(r["rb"] for r in shuf)
            nulls.append(max(mean(r["rb"] for r in bnd) for bnd in qs) - overall)
        nulls.sort()
        obs = a if label.startswith("(a)") else dd
        p = sum(1 for x in nulls if x >= obs) / len(nulls)
        print(f"  {label:22} null mean {mean(nulls):+.4f}  null p95 {nulls[189]:+.4f}"
              f"  observed {obs:+.4f}  p={p:.3f}")
        print(f"  {'':22} bias-corrected ceiling: {obs-mean(nulls):+.4f}")

    # what the controlled ceiling is worth on the MTRAG-UN gpt-oss-120b row
    def hm(v: list[float]) -> float:
        return 0.0 if any(x <= 0 for x in v) else len(v) / sum(1 / x for x in v)

    base = [0.59, 0.65, 0.37]
    print(f"\n  MTRAG-UN gpt-oss-120b RAG baseline HM = {hm(base):.4f}")
    for lab, lift in (("published ceiling", a), ("controlled ceiling", dd)):
        v = [base[0], base[1], base[2] + lift]
        print(f"    RB_alg +{lift:.4f} ({lab:18}) -> RB_alg {v[2]:.4f}  HM {hm(v):.4f}"
              f"   vs rank-1 0.586: {hm(v)-0.586:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
