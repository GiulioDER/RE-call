"""Verify the bug-auditor's inferred findings against the real RAG.json.

The auditor did not have the file and derived BUG-001/005/006/007 from arithmetic plus a
synthetic fixture. Each is checked here directly.

Prior work: none found on RB_alg / MTRAGEval generation / length calibration
(docs_search source_type=memory, "MT-RAG generation RB_alg RougeL response length
calibration answer verbosity benchmark", gap_warning false). Two adjacent hits:
[[project-recall-token-f1-harness-offset-2026-07-29]] and
[[project-recall-mtrag-symmetric-baseline-2026-08-04]]. Full search recorded in
benchmarks/PREREGISTRATION-mtrag-rbalg.md.
"""

import json
import sys
from collections import Counter
from statistics import mean

PATH = sys.argv[1]
TRIPLE = ("rl_f", "rb_llm", "rb_agg")


def hm(v):
    return 0.0 if any(x is None or x <= 0 for x in v) else len(v) / sum(1 / x for x in v)


def slot(ann, k, which="composite"):
    n = ann.get(k)
    if not n or which not in n or n[which] is None:
        return None
    return n[which].get("value")


d = json.load(open(PATH, encoding="utf-8"))
names = {}
for m in d["models"]:
    names[(m.get("model_id") or m.get("id") or m.get("name")) if isinstance(m, dict) else m] = (
        (m.get("display_name") or m.get("name")) if isinstance(m, dict) else m
    )
tasks = {t["task_id"]: t for t in d["tasks"]}

print("=== BUG-005: multi-label Answerability ===")
lens = Counter(len(t.get("Answerability") or []) for t in d["tasks"])
print("  len(Answerability) distribution:", dict(lens))
print("  tasks with >1 label:", sum(v for k, v in lens.items() if k > 1))

print("\n=== BUG-007: multi-reference targets ===")
tl = Counter(len(t.get("targets") or []) for t in d["tasks"])
print("  len(targets) distribution:", dict(tl))
print("  tasks with >1 target:", sum(v for k, v in tl.items() if k > 1))

print("\n=== BUG-006: does MTRAG carry an UNDERSPECIFIED bucket? ===")
cc = Counter((t.get("Answerability") or ["UNKNOWN"])[0] for t in d["tasks"])
print("  MTRAG task-level classes:", dict(cc))
print("  UNDERSPECIFIED present:", "UNDERSPECIFIED" in cc)

print("\n=== BUG-001: does P4's 120 include the Target pseudo-model? ===")
tot = withtgt = notgt = 0
bad = []
for ev in d["evaluations"]:
    t = tasks.get(ev["task_id"])
    if not t:
        continue
    a = str(t.get("Answerability"))
    if "UNANSWERABLE" not in a.upper():
        continue
    ann = ev.get("annotations") or {}
    if slot(ann, "conditional_idk") != 1:
        continue
    tot += 1
    is_target = str(names.get(ev["model_id"], ev["model_id"])).lower() == "target"
    if is_target:
        withtgt += 1
    else:
        notgt += 1
    trip = [slot(ann, k) for k in TRIPLE]
    if not all(v == 1 for v in trip):
        bad.append((names.get(ev["model_id"]), trip))
print(f"  correct-IDK cells total      : {tot}")
print(f"    of which Target            : {withtgt}")
print(f"    of which real models       : {notgt}")
print(f"  cells NOT scoring exactly 1.0: {len(bad)}")
for b in bad[:5]:
    print("    ", b)
print(f"  -> P4 claim (all score exactly 1.0) holds on models only: "
      f"{notgt}/{notgt} = {'TRUE' if not bad else 'FALSE'}")

print("\n=== BUG-011: does hm(rouge,(brec+1)/2,(bkp+1)/2) reconstruct rb_agg? ===")
ok = miss = 0
errs = []
for ev in d["evaluations"]:
    ann = ev.get("annotations") or {}
    rb = slot(ann, "rb_agg")
    r = slot(ann, "RougeL", "system")
    br = slot(ann, "Bert-Rec", "system")
    bk = slot(ann, "Bert-KPrec", "system")
    if None in (rb, r, br, bk) or rb == 0:
        continue
    recon = hm([r, (br + 1) / 2, (bk + 1) / 2])
    if abs(recon - rb) < 1e-3:
        ok += 1
    else:
        miss += 1
        if len(errs) < 3:
            errs.append((round(rb, 4), round(recon, 4)))
tot2 = ok + miss
print(f"  reconstructed within 1e-3: {ok}/{tot2} = {ok/tot2:.1%}" if tot2 else "  no rows")
if errs:
    print("  sample (exported rb_agg, reconstructed):", errs)
print("  -> 'Bert-Rec is the BertscoreR that rb_agg consumes':",
      "CONFIRMED" if tot2 and ok / tot2 > 0.95 else "NOT CONFIRMED")
