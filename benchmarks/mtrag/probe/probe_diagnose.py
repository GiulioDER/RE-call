"""MTRAGEval RB_alg probe -- STEP 2, diagnose the invariant failure, then P1-P4.

Pre-registration: RE-call/benchmarks/PREREGISTRATION-mtrag-rbalg.md
STEP 1 result: HM-of-means over RAG.json does NOT reproduce the published MTRAG table
(systematic +0.018..+0.043 on models, -0.015 on Target). Before any P1-P5 number is
reported, establish WHICH part is wrong: the aggregation formula, or the instance set.

Prior work: none found on RB_alg / MTRAGEval generation / length calibration
(docs_search source_type=memory, "MT-RAG generation RB_alg RougeL response length
calibration answer verbosity benchmark", gap_warning false). Two adjacent hits:
[[project-recall-token-f1-harness-offset-2026-07-29]] and
[[project-recall-mtrag-symmetric-baseline-2026-08-04]]. Full search recorded in
benchmarks/PREREGISTRATION-mtrag-rbalg.md.
"""

import json
import sys
from dataclasses import dataclass
from collections import Counter
from statistics import mean, median
from typing import Any

PATH = sys.argv[1]
TRIPLE = ("rl_f", "rb_llm", "rb_agg")
RAW = ("RougeL", "Bert-Rec", "Bert-KPrec")


@dataclass(frozen=True)
class Rec:
    """One (model, task) evaluation, with the metrics kept optional until narrowed."""

    model: str
    ratio: float
    plen: int
    tlen: int
    rb: float | None
    rouge: float | None
    brec: float | None
    bkp: float | None


def hm(vals: list[float]) -> float:
    if any(v is None or v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def slot(ann: dict[str, Any], key: str, which: str) -> float | None:
    node = ann.get(key)
    if not node or which not in node or node[which] is None:
        return None
    value = node[which].get("value")
    return None if value is None else float(value)


def norm_tokens(s: str) -> list[str]:
    import re
    import string

    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return [x for x in s.split() if x]


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

    tasks = {}
    for t in d["tasks"]:
        tasks[t.get("task_id")] = t
    tk = sorted(d["tasks"][0].keys())
    print("=== TASK SCHEMA ===")
    print(" ", tk)
    sample = d["tasks"][0]
    for k in tk:
        v = sample[k]
        print(f"   {k}: {str(v)[:120]}")

    # ---------------- aggregation cross-check against INDEPENDENT published triples ------
    print("\n=== AGGREGATION FORMULA CROSS-CHECK (independent of RAG.json) ===")
    checks = [
        ("5ting SemEval Task C (self-reported 0.5597)", (0.7692, 0.6784, 0.3867), 0.5597),
        ("MTRAG-UN Table 4 target, RAG (published 0.81)", (0.69, 0.92, 0.88), 0.81),
        ("MTRAG-UN Table 4 gpt-oss-120b, RAG", (0.59, 0.65, 0.37), None),
    ]
    for label, triple, pub in checks:
        got = hm(list(triple))
        note = "" if pub is None else f"   published {pub}   d={got-pub:+.4f}"
        print(f"  HM{triple} = {got:.4f}{note}")

    # ---------------- composite vs system, and the zero/one structure -------------------
    print(f"\n=== COMPOSITE STRUCTURE (all {len(d['evaluations'])} evaluations) ===")
    have: Counter[tuple[str, bool, bool]] = Counter()
    exact: dict[str, Counter[str]] = {k: Counter() for k in TRIPLE}
    for ev in d["evaluations"]:
        ann = ev.get("annotations") or {}
        for k in TRIPLE:
            c = slot(ann, k, "composite")
            s = slot(ann, k, "system")
            have[(k, c is not None, s is not None)] += 1
            if c is not None:
                if c == 0:
                    exact[k]["==0"] += 1
                elif c == 1:
                    exact[k]["==1"] += 1
                else:
                    exact[k]["other"] += 1
    for k in TRIPLE:
        print(f"  {k:8} composite: {dict(exact[k])}")
    print("  (key, has_composite, has_system) ->", dict(have))

    # ---------------- P4: does a correct IDK score exactly 1.0? -------------------------
    print("\n=== P4: correct-IDK composite ===")
    ansfield = None
    for cand in ("answerability", "Answerability", "answerable"):
        if cand in sample:
            ansfield = cand
            break
    print("  answerability field on task:", ansfield)
    if ansfield:
        rows = []
        for ev in d["evaluations"]:
            t = tasks.get(ev["task_id"])
            if not t:
                continue
            a = str(t.get(ansfield))
            ann = ev.get("annotations") or {}
            idk = slot(ann, "conditional_idk", "composite")
            trip = [slot(ann, k, "composite") for k in TRIPLE]
            rows.append((a, idk, trip, names.get(ev["model_id"], ev["model_id"])))
        by = Counter(r[0] for r in rows)
        print("  answerability distribution over evaluations:", dict(by))
        # correct-IDK cells: unanswerable AND conditional_idk == 1
        cells = [r for r in rows if "UNANSWERABLE" in r[0].upper() and r[1] == 1]
        # BUG-001 (bug-auditor, 2026-08-05): this loop originally had no Target guard, so the
        # published "120/120" pooled the reference pseudo-model with real models. Split now.
        # Every other script in this directory excludes Target; this one must too.
        tgt_cells = [r for r in cells if str(r[3]).lower() == "target"]
        mdl_cells = [r for r in cells if str(r[3]).lower() != "target"]
        print(f"  cells with UNANSWERABLE and conditional_idk==1: {len(cells)} total"
              f"  =  {len(mdl_cells)} models + {len(tgt_cells)} Target")
        for label, group in (("MODELS", mdl_cells), ("Target", tgt_cells), ("pooled", cells)):
            if not group:
                continue
            allone = sum(1 for r in group if all(v == 1 for v in r[2]))
            print(f"    {label:7} all three composites == 1.0 exactly: "
                  f"{allone}/{len(group)} ({allone/len(group):.1%})")
            for b in [r for r in group if not all(v == 1 for v in r[2])][:3]:
                print("      counterexample:", b[3], b[2])
        print("    NOTE: the reportable P4 figure is the MODELS row. Target is the reference"
              " answer scoring itself and cannot fail.")

        # BUG-011 (bug-auditor): prove 'Bert-Rec' is the BertscoreR that rb_agg consumes,
        # rather than assuming it from the export's naming.
        ok = miss = 0
        for ev in d["evaluations"]:
            ann = ev.get("annotations") or {}
            rb = slot(ann, "rb_agg", "composite")
            r_ = slot(ann, "RougeL", "system")
            br = slot(ann, "Bert-Rec", "system")
            bk = slot(ann, "Bert-KPrec", "system")
            if rb is None or r_ is None or br is None or bk is None:
                continue
            if rb in (0.0, 1.0):
                continue  # 0 and 1 are the conditioning overriding the raw value
            recon = 0.0
            vals: list[float] = [r_, (br + 1) / 2, (bk + 1) / 2]
            if all(v > 0 for v in vals):
                recon = 3 / sum(1 / v for v in vals)
            ok, miss = (ok + 1, miss) if abs(recon - rb) < 1e-3 else (ok, miss + 1)
        if ok + miss:
            print(f"  rb_agg reconstruction from (RougeL, Bert-Rec, Bert-KPrec): "
                  f"{ok}/{ok+miss} = {ok/(ok+miss):.1%} within 1e-3 on unconditioned rows")

    # ---------------- P1/P2/P3: length and the binding component ------------------------
    print("\n=== P1/P2/P3 ===")
    tgt_len = {}
    for t in d["tasks"]:
        tg = t.get("targets")
        if isinstance(tg, list) and tg:
            txt = tg[0].get("text") if isinstance(tg[0], dict) else str(tg[0])
            if txt:
                tgt_len[t.get("task_id")] = len(norm_tokens(txt))

    print(f"  tasks with a usable target length: {len(tgt_len)}/{len(d['tasks'])}")

    recs: list[Rec] = []
    for ev in d["evaluations"]:
        mid = names.get(ev["model_id"], ev["model_id"])
        if str(mid).lower() == "target":
            continue
        tl = tgt_len.get(ev["task_id"])
        if not tl:
            continue
        resp = ev.get("model_response") or ""
        pl = len(norm_tokens(resp))
        if pl == 0:
            continue
        ann = ev.get("annotations") or {}
        recs.append(
            Rec(
                model=str(mid),
                ratio=pl / tl,
                plen=pl,
                tlen=tl,
                rb=slot(ann, "rb_agg", "composite"),
                rouge=slot(ann, "RougeL", "system"),
                brec=slot(ann, "Bert-Rec", "system"),
                bkp=slot(ann, "Bert-KPrec", "system"),
            )
        )

    print(f"  usable (non-target) records: {len(recs)}")
    ratios = [r.ratio for r in recs]
    print(f"  P1  median length ratio (pred/target) = {median(ratios):.3f}   mean = {mean(ratios):.3f}")
    print(f"      median pred tokens = {median([r.plen for r in recs]):.0f}  "
          f"median target tokens = {median([r.tlen for r in recs]):.0f}")
    print(f"      P1 PREDICTED >2.0, falsified if <=1.5  ->  "
          f"{'CONFIRMED' if median(ratios)>2.0 else ('FALSIFIED' if median(ratios)<=1.5 else 'INCONCLUSIVE')}")

    # P2: which of the three rb_agg components is the minimum
    mins: Counter[str] = Counter()
    n_p2 = 0
    for r in recs:
        if r.rouge is None or r.brec is None or r.bkp is None:
            continue
        comp: dict[str, float] = {
            "rouge": r.rouge,
            "bert_rec": (r.brec + 1) / 2,
            "bert_kprec": (r.bkp + 1) / 2,
        }
        mins[min(comp, key=lambda k: comp[k])] += 1
        n_p2 += 1
    print(f"  P2  argmin over rb_agg components, n={n_p2}: "
          + ", ".join(f"{k}={v} ({v/n_p2:.1%})" for k, v in mins.most_common()))
    share = mins["rouge"] / n_p2 if n_p2 else 0.0
    print(f"      P2 PREDICTED rouge is min >70%, falsified if <=50%  ->  "
          f"{'CONFIRMED' if share>0.70 else ('FALSIFIED' if share<=0.50 else 'INCONCLUSIVE')}")

    # P3: rb_agg by length-ratio quartile. Narrowed to (ratio, rb) pairs so the optional
    # never reaches the arithmetic.
    pairs: list[tuple[float, float]] = sorted(
        ((r.ratio, r.rb) for r in recs if r.rb is not None), key=lambda p: p[0]
    )
    n = len(pairs)
    q = n // 4
    bands: list[tuple[str, list[tuple[float, float]]]] = [
        ("Q1 shortest", pairs[:q]),
        ("Q2", pairs[q : 2 * q]),
        ("Q3", pairs[2 * q : 3 * q]),
        ("Q4 longest", pairs[3 * q :]),
    ]
    print(f"  P3  rb_agg (composite) by length-ratio quartile, n={n}:")
    for label, band in bands:
        if band:
            print(f"      {label:12} ratio {band[0][0]:.2f}..{band[-1][0]:.2f}  "
                  f"mean rb_agg {mean(rb for _, rb in band):.4f}  n={len(band)}")
    # band nearest ratio 1.0
    near = sorted(pairs, key=lambda p: abs(p[0] - 1.0))[:q]
    m_near = mean(rb for _, rb in near)
    m_q4 = mean(rb for _, rb in bands[-1][1])
    gap = m_near - m_q4
    print(f"      nearest-to-1.0 band mean rb_agg = {m_near:.4f}   Q4 = {m_q4:.4f}   gap = {gap:+.4f}")
    print(f"      P3 PREDICTED gap >= +0.10, falsified if < +0.03 or negative  ->  "
          f"{'CONFIRMED' if gap>=0.10 else ('FALSIFIED' if gap<0.03 else 'INCONCLUSIVE')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
