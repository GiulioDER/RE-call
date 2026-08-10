"""Verify the load-bearing premise of PREREGISTRATION-mtrag-abstention.md sec.0 and sec.3.

Two things, both of which a pre-registration then rests on:

1. sec.0 -- on an UNANSWERABLE task the answerability-CONDITIONED composite is binary: exactly
   1.0 on a correct IDK, exactly 0.0 otherwise. It was first inferred from mean(rb_agg) equalling
   each model's correct-IDK rate, which is CONSISTENT with the claim without proving it. Checked
   here on the JOINT distribution of the three metrics, not three independent marginals, because
   the claim says "all three simultaneously" and three marginals of 72/423 are equally consistent
   with three disjoint sets of 72.
2. sec.3 -- the payoff `a`, the mean score on tasks WHERE ANSWERING PAYS, WHEN ANSWERING. The
   qualifier is the whole point: cells where the model abstained are the OTHER branch of the
   decision and scoring them 0.0 into `a` contaminates it.

Prior work: the abstention design is governed by [[project-recall-abstention-bounded-domain-2026-07-24]]
-- six cheap signals already measured and CLOSED, two named dead ends. Full searches recorded in
benchmarks/archive/preregistrations/PREREGISTRATION-mtrag-rbalg.md and benchmarks/archive/preregistrations/PREREGISTRATION-mtrag-abstention.md.

Exits non-zero if either premise fails OR if the population is empty, because a check that cannot
tell "nothing was wrong" from "nothing was examined" is not a check.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

TRIPLE = ("rl_f", "rb_llm", "rb_agg")
# Classes where ANSWERING pays. PARTIAL is here on evidence, not by assumption: abstaining on a
# PARTIAL task scores exactly 0.0 on every one of its cells, exactly as it does on ANSWERABLE.
ANSWER_PAYS = ("ANSWERABLE", "PARTIAL")


def slot(ann: dict[str, Any], key: str) -> float | None:
    node = ann.get(key)
    if not node or "composite" not in node or node["composite"] is None:
        return None
    value = node["composite"].get("value")
    return None if value is None else float(value)


def display_names(models: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in models:
        if isinstance(m, dict):
            out[str(m.get("model_id") or m.get("id") or m.get("name"))] = str(
                m.get("display_name") or m.get("name")
            )
        else:
            out[str(m)] = str(m)
    return out


def answerability(task: dict[str, Any] | None) -> str:
    if not task:
        return "?"
    a = task.get("Answerability") or []
    return str(a[0]) if isinstance(a, list) and a else "?"


def main() -> int:
    path = sys.argv[1]
    d = json.load(open(path, encoding="utf-8"))
    names = display_names(d["models"])
    tasks: dict[str, dict[str, Any]] = {t["task_id"]: t for t in d["tasks"]}

    rows: list[dict[str, Any]] = []
    for ev in d["evaluations"]:
        if names.get(str(ev["model_id"]), str(ev["model_id"])).lower() == "target":
            continue
        ann = ev.get("annotations") or {}
        rows.append(
            {
                "cls": answerability(tasks.get(ev["task_id"])),
                "idk": slot(ann, "conditional_idk"),
                "rb": slot(ann, "rb_agg"),
                "trip": tuple(slot(ann, k) for k in TRIPLE),
            }
        )

    if not rows:
        print("NO CELLS SELECTED -- the run is vacuous, not confirming")
        return 2

    # Full class coverage, so a reader can see that the two populations below do NOT partition
    # the data. 9.3% of cells sit outside {ANSWERABLE, UNANSWERABLE} on the pinned revision.
    coverage = Counter(str(r["cls"]) for r in rows)
    print(f"non-Target cells: {len(rows)}")
    print(f"  class coverage: {dict(coverage)}")
    residual = sum(v for k, v in coverage.items() if k not in (*ANSWER_PAYS, "UNANSWERABLE"))
    print(f"  outside ANSWER_PAYS + UNANSWERABLE: {residual} ({residual/len(rows):.1%})")

    # ---- premise 1: the JOINT distribution on UNANSWERABLE -----------------------------
    unans = [r for r in rows if r["cls"] == "UNANSWERABLE"]
    if not unans:
        print("NO UNANSWERABLE CELLS -- vacuous")
        return 2
    joint = Counter(r["trip"] for r in unans)
    print(f"\nUNANSWERABLE cells: {len(unans)}")
    print(f"  distinct (rl_f, rb_llm, rb_agg) tuples: {dict(joint)}")
    binary = set(joint) <= {(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)}
    print(f"  PREMISE 1 'binary, and all three move together': "
          f"{'CONFIRMED' if binary else 'FALSE -- sec.0 must be rewritten'}")

    # ---- premise 2: the payoff, measured only where answering actually happened ---------
    pays = [r for r in rows if r["cls"] in ANSWER_PAYS]
    answered = [r for r in pays if r["idk"] == 1]
    abstained = [r for r in pays if r["idk"] != 1]
    if not answered:
        print("NO ANSWERING CELLS -- vacuous")
        return 2
    all_zero = all(r["rb"] == 0 for r in abstained)
    print(f"\nANSWER_PAYS cells: {len(pays)}   answered {len(answered)}   abstained {len(abstained)}")
    print(f"  every abstained cell scores rb_agg == 0.0: {all_zero}"
          "   <- why they must not enter `a`")

    def payoff(sample: list[dict[str, Any]]) -> tuple[float, float]:
        a = sum(float(r["rb"]) for r in sample) / len(sample)
        return a, a / (1 + a)

    a_primary, p_primary = payoff(answered)
    print(f"\n  PRIMARY  a = {a_primary:.4f} (n={len(answered)}, answering, "
          f"{'+'.join(ANSWER_PAYS)})   ->  p* = {p_primary:.4f}")
    for label, sample in (
        ("ANSWERABLE only, answering", [r for r in answered if r["cls"] == "ANSWERABLE"]),
        ("ANSWERABLE only, incl. abstentions (WRONG, contaminated)",
         [r for r in pays if r["cls"] == "ANSWERABLE"]),
    ):
        if sample:
            a, p = payoff(sample)
            print(f"  sens.    a = {a:.4f} (n={len(sample)}, {label})   ->  p* = {p:.4f}")

    ok = binary and all_zero
    if not ok:
        print("\nA PREMISE FAILED -- do not copy the threshold above; sec.3 needs rewriting.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
