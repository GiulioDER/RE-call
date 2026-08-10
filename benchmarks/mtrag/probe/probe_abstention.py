"""MTRAGEval RB_alg probe -- STEP 3, quantify the lever that P4 exposed.

P1 was FALSIFIED (median length ratio 1.096, not >2.0), so the "models are verbose" premise
is dead and P5 as preregistered is unmotivated. P4 was confirmed at 100%: a correct IDK on an
UNANSWERABLE task scores exactly 1.0 on rl_f, rb_llm AND rb_agg simultaneously.

This script is DIAGNOSTIC, not a new experiment: it measures how often the published baselines
actually abstain correctly, and computes what the ceiling would be if they always did. No new
prediction is being tested here; a new prediction has to be preregistered separately.

Prior work: none found on RB_alg / MTRAGEval generation / length calibration
(docs_search source_type=memory, "MT-RAG generation RB_alg RougeL response length
calibration answer verbosity benchmark", gap_warning false). Two adjacent hits:
[[project-recall-token-f1-harness-offset-2026-07-29]] and
[[project-recall-mtrag-symmetric-baseline-2026-08-04]]. Full search recorded in
benchmarks/archive/preregistrations/PREREGISTRATION-mtrag-rbalg.md.
"""

import json
import sys
from collections import defaultdict
from statistics import mean
from typing import Any

PATH = sys.argv[1]
TRIPLE = ("rl_f", "rb_llm", "rb_agg")


def hm(vals: list[float]) -> float:
    if any(v is None or v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def slot(ann: dict[str, Any], key: str, which: str = "composite") -> float | None:
    node = ann.get(key)
    if not node or which not in node or node[which] is None:
        return None
    value = node[which].get("value")
    return None if value is None else float(value)


def main() -> int:
    d = json.load(open(PATH, encoding="utf-8"))
    names = {}
    for m in d["models"]:
        if isinstance(m, dict):
            names[m.get("model_id") or m.get("id") or m.get("name")] = (
                m.get("display_name") or m.get("name")
            )
        else:
            names[m] = m

    tasks = {t["task_id"]: t for t in d["tasks"]}

    def cls(tid: str) -> str:
        a = tasks.get(tid, {}).get("Answerability") or []
        return (a[0] if isinstance(a, list) and a else str(a)) or "UNKNOWN"

    counts: defaultdict[str, int] = defaultdict(int)
    for t in d["tasks"]:
        a = t.get("Answerability") or []
        counts[(a[0] if isinstance(a, list) and a else "UNKNOWN")] += 1
    print(f"=== TASK-LEVEL answerability (MTRAG, {len(d['tasks'])} tasks) ===")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16} {v:>4}  ({v/len(d['tasks']):.1%})")

    # ---- per model: abstention behaviour on UNANSWERABLE, and metric means -------------
    per: defaultdict[str, defaultdict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ev in d["evaluations"]:
        mid = str(names.get(ev["model_id"], ev["model_id"]))
        if mid.lower() == "target":
            continue
        ann = ev.get("annotations") or {}
        c = cls(ev["task_id"])
        trip = [slot(ann, k) for k in TRIPLE]
        if any(v is None for v in trip):
            continue
        idk = slot(ann, "conditional_idk")
        per[mid][c].append((trip, idk))

    print("\n=== ABSTENTION ON UNANSWERABLE, per model ===")
    hdr = f"{'model':24} {'UNANS n':>8} {'correct IDK':>12} {'rate':>7} {'mean rb_agg':>12}"
    print(hdr)
    print("-" * len(hdr))
    for mid, byc in sorted(per.items()):
        u = byc.get("UNANSWERABLE", [])
        if not u:
            continue
        good = sum(1 for trip, idk in u if idk == 1)
        mrb = mean(t[2] for t, _ in u)
        print(f"{str(mid)[:24]:24} {len(u):>8} {good:>12} {good/len(u):>6.1%} {mrb:>12.4f}")

    # ---- the counterfactual: perfect abstention on UNANSWERABLE only -------------------
    print("\n=== CEILING: what perfect abstention on UNANSWERABLE alone would be worth ===")
    print("   (every UNANSWERABLE cell set to 1.0/1.0/1.0; every other cell left untouched)")
    hdr = (f"{'model':24} {'HM now':>8} {'HM perfect-abstain':>19} {'delta':>8}")
    print(hdr)
    print("-" * len(hdr))
    deltas = []
    for mid, byc in sorted(per.items()):
        allrows = [t for c in byc for t, _ in byc[c]]
        if not allrows:
            continue
        now = hm([mean(r[i] for r in allrows) for i in range(3)])
        cf = []
        for c, rows in byc.items():
            for trip, idk in rows:
                cf.append([1.0, 1.0, 1.0] if c == "UNANSWERABLE" else trip)
        perfect = hm([mean(r[i] for r in cf) for i in range(3)])
        deltas.append(perfect - now)
        print(f"{str(mid)[:24]:24} {now:>8.4f} {perfect:>19.4f} {perfect-now:>+8.4f}")
    print(f"\n  mean delta across models: {mean(deltas):+.4f}")

    # ---- scale to MTRAG-UN, where UNANSWERABLE is a much larger share ------------------
    print("\n=== SCALING TO MTRAG-UN (the SemEval evaluation set) ===")
    # BUG-006 (bug-auditor, 2026-08-05): the two shares were computed over inconsistent
    # denominators. MTRAG carries NO UNDERSPECIFIED bucket (verified: UNANSWERABLE 55,
    # ANSWERABLE 709, PARTIAL 68, CONVERSATIONAL 10), while MTRAG-UN's 507 includes 78
    # UNDERSPECIFIED tasks that are scored by a SEPARATE clarification judge and excluded
    # from these three metrics. Dividing by 507 therefore understates the MTRAG-UN share.
    n_tasks = len(d["tasks"])
    mtrag_unans = counts.get("UNANSWERABLE", 0) / n_tasks
    has_underspec = "UNDERSPECIFIED" in counts
    print(f"  MTRAG task-level classes: {dict(counts)}")
    print(f"  MTRAG carries an UNDERSPECIFIED bucket: {has_underspec}")
    for label, denom in (("over all 507 tasks", 507), ("excluding the 78 UNDERSPECIFIED", 507 - 78)):
        un_unans = 97 / denom
        print(f"  UNANSWERABLE share: MTRAG {mtrag_unans:.1%} ({counts.get('UNANSWERABLE',0)}/{n_tasks})"
              f"  ->  MTRAG-UN {un_unans:.1%} (97/{denom}, {label})"
              f"   ratio {un_unans/mtrag_unans:.2f}x"
              f"   naive scale {mean(deltas)*un_unans/mtrag_unans:+.4f}")
    print(f"  MTRAG-UN also carries PARTIAL {47/507:.1%} and UNDERSPECIFIED {78/507:.1%}")
    print("  NOTE: linear scaling is an approximation, not a measurement, and the two")
    print("        denominators bracket it rather than pinning it. The real number requires")
    print("        running on MTRAG-UN, which is held out.")

    # ---- headline arithmetic on the published MTRAG-UN baseline row --------------------
    print("\n=== WHAT IT WOULD TAKE, on MTRAG-UN's published gpt-oss-120b RAG row ===")
    base = (0.59, 0.65, 0.37)
    print(f"  baseline               RL_F {base[0]:.2f}  RB_llm {base[1]:.2f}  RB_alg {base[2]:.2f}"
          f"   HM {hm(list(base)):.4f}")
    for lift in (0.05, 0.10, 0.13, 0.15):
        lifted: list[float] = [min(1.0, x + lift) for x in base]
        print(f"  +{lift:.2f} to all three    RL_F {lifted[0]:.2f}  RB_llm {lifted[1]:.2f}"
              f"  RB_alg {lifted[2]:.2f}   HM {hm(lifted):.4f}"
              f"   vs rank-1 0.586: {hm(lifted)-0.586:+.4f}")
    print("\n  for contrast, lifting RB_alg ALONE (the original thesis):")
    for rb in (0.44, 0.50, 0.53):
        only_rb: list[float] = [base[0], base[1], rb]
        print(f"    RB_alg -> {rb:.2f}   HM {hm(only_rb):.4f}"
              f"   vs rank-1 0.586: {hm(only_rb)-0.586:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
