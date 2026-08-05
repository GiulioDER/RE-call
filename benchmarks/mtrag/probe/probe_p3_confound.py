"""P3 confound check: is the length gap really length, or is it the answerability conditioning?

PROBE_VERDICT.md reports P3 as +0.1529 using the COMPOSITE rb_agg, which the IDK conditioning
sets to 0 for a wrong answerability call. If long answers are over-represented among
conditioned-out instances, part of that gap is conditioning and not length.

Re-runs P3 three ways:
  (a) as reported            -- composite, all instances
  (b) ANSWERABLE tasks only  -- conditioning cannot fire on a correct non-IDK answer
  (c) composite > 0 only     -- drops every conditioned-out cell
If the gap survives (b) and (c), it is length. If it collapses, PROBE_VERDICT.md needs a fix.

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
from collections import Counter
from statistics import mean

PATH = sys.argv[1]


def norm_tokens(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return [x for x in re.sub(r"\b(a|an|the)\b", " ", s).split() if x]


def slot(ann, key, which="composite"):
    node = ann.get(key)
    if not node or which not in node or node[which] is None:
        return None
    return node[which].get("value")


def bands(rows, key="rb"):
    rows = sorted(rows, key=lambda r: r["ratio"])
    n = len(rows)
    q = n // 4
    near = sorted(rows, key=lambda r: abs(r["ratio"] - 1.0))[:q]
    q4 = rows[3 * q :]
    return mean(r[key] for r in near), mean(r[key] for r in q4), n


def main():
    d = json.load(open(PATH, encoding="utf-8"))
    names = {}
    for m in d["models"]:
        names[(m.get("model_id") or m.get("id") or m.get("name")) if isinstance(m, dict) else m] = (
            (m.get("display_name") or m.get("name")) if isinstance(m, dict) else m
        )
    tasks = {t["task_id"]: t for t in d["tasks"]}

    def cls(tid):
        a = tasks.get(tid, {}).get("Answerability") or []
        return a[0] if isinstance(a, list) and a else "UNKNOWN"

    tgt = {}
    for t in d["tasks"]:
        tg = t.get("targets")
        if tg:
            tgt[t["task_id"]] = len(norm_tokens(tg[0]["text"]))

    recs = []
    for ev in d["evaluations"]:
        mid = names.get(ev["model_id"], ev["model_id"])
        if str(mid).lower() == "target":
            continue
        tl = tgt.get(ev["task_id"])
        pl = len(norm_tokens(ev.get("model_response") or ""))
        rb = slot(ev.get("annotations") or {}, "rb_agg")
        if not tl or not pl or rb is None:
            continue
        recs.append(dict(ratio=pl / tl, rb=rb, cls=cls(ev["task_id"])))

    print(f"records: {len(recs)}")

    # Is length associated with being conditioned out?
    print("\n=== is a zeroed composite associated with length? ===")
    zero = [r for r in recs if r["rb"] == 0]
    nonzero = [r for r in recs if r["rb"] > 0]
    print(f"  composite==0: n={len(zero)} ({len(zero)/len(recs):.1%})  median ratio {sorted(r['ratio'] for r in zero)[len(zero)//2]:.3f}")
    print(f"  composite >0: n={len(nonzero)}  median ratio {sorted(r['ratio'] for r in nonzero)[len(nonzero)//2]:.3f}")
    srt = sorted(recs, key=lambda r: r["ratio"])
    q = len(srt) // 4
    for label, band in (("Q1 shortest", srt[:q]), ("Q2", srt[q:2*q]), ("Q3", srt[2*q:3*q]), ("Q4 longest", srt[3*q:])):
        z = sum(1 for r in band if r["rb"] == 0)
        cc = Counter(r["cls"] for r in band)
        print(f"  {label:12} zero-rate {z/len(band):6.1%}   "
              f"UNANS {cc.get('UNANSWERABLE',0):>4}  PARTIAL {cc.get('PARTIAL',0):>4}  ANS {cc.get('ANSWERABLE',0):>5}")

    print("\n=== P3 re-run three ways (mean rb_agg: nearest-ratio-1.0 band vs longest quartile) ===")
    for label, subset in (
        ("(a) as reported, all instances", recs),
        ("(b) ANSWERABLE tasks only", [r for r in recs if r["cls"] == "ANSWERABLE"]),
        ("(c) composite > 0 only", nonzero),
        ("(d) ANSWERABLE and composite > 0", [r for r in recs if r["cls"] == "ANSWERABLE" and r["rb"] > 0]),
    ):
        if len(subset) < 40:
            print(f"  {label:34} n too small")
            continue
        near, q4, n = bands(subset)
        verdict = "SURVIVES" if (near - q4) >= 0.10 else ("weakened" if (near - q4) >= 0.03 else "COLLAPSES")
        print(f"  {label:34} n={n:>5}  near1.0 {near:.4f}  Q4 {q4:.4f}  gap {near-q4:+.4f}  {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
