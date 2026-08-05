"""What Arm R's AUC costs at an operating point, and what the shipped gap_warning actually did.

DIAGNOSTIC over the Arm R artifact, not a new hypothesis test. Arm R's preregistered output is the
regime classification, which already landed at AUC 0.621. This turns that into the error rates a
decision would actually pay, so "weak signal" becomes a number instead of an adjective.

THREE operating points are reported, because they are three different things and an earlier draft
of this file conflated them:

* the library's own `best_threshold` (the q05/q95 midpoint `from_samples` derives),
* the PREREGISTERED break-even, which lives in PROBABILITY space -- sec.3 derives
  p* = a/(1+a) = 0.2957 and is mapped onto cosine here through `Calibration.confidence`, the
  shipped sigmoid, by abstaining when P(unanswerable) > p* i.e. confidence < 1 - p*,
* the IN-SAMPLE optimum over the actual score breakpoints, which is an upper bound and is
  labelled as such: it is chosen on the same samples it is scored on.

Payoffs are PREREGISTERED (benchmarks/PREREGISTRATION-mtrag-abstention.md sec.3) and not
re-derived: correct abstention 1.0, false abstention 0.0, answering an unanswerable 0.0, answering
pays a = 0.4199. That constant is measured over ANSWERABLE + PARTIAL, so the population here MUST
be the same one -- an earlier draft used the constant while dropping the 68 PARTIAL tasks.
CONVERSATIONAL is excluded because every one of its cells scores 1.0 regardless of the response,
making it payoff-neutral.

Prior work: governed by [[project-recall-abstention-bounded-domain-2026-07-24]] (six cheap signals
measured and CLOSED, two named dead ends) and [[project-recall-threshold-embedder-fragile-2026-07-28]]
(the 0.50 floor is inert on bge-small). Full search recorded in
benchmarks/PREREGISTRATION-mtrag-abstention.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from recall.calibration import from_samples

#: Preregistered payoff and its population. Changing either without the other is BUG-001.
PAYOFF_ANSWER = 0.4199
ANSWER_PAYS = ("ANSWERABLE", "PARTIAL")
ABSTAIN_PAYS = "UNANSWERABLE"
EXCLUDED = ("CONVERSATIONAL",)  # payoff-neutral: always scores 1.0
#: sec.3 break-even, in PROBABILITY space. p* = a / (1 + a).
P_STAR = PAYOFF_ANSWER / (1 + PAYOFF_ANSWER)


def mean_score(caught: int, false_abstain: int, n_pay: int, n_unans: int) -> float:
    return (caught * 1.0 + (n_pay - false_abstain) * PAYOFF_ANSWER) / (n_pay + n_unans)


def counts_at(pay: list[float], unans: list[float], thr: float) -> tuple[int, int]:
    """(false abstentions, unanswerable caught) when abstaining below `thr`."""
    return sum(1 for v in pay if v < thr), sum(1 for v in unans if v < thr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    args = ap.parse_args()

    d: dict[str, Any] = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if r["top1_cosine"] is not None]
    if not rows:
        print("NO SCORED ROWS -- vacuous")
        return 2
    pay = [float(r["top1_cosine"]) for r in rows if r["cls"] in ANSWER_PAYS]
    unans = [float(r["top1_cosine"]) for r in rows if r["cls"] == ABSTAIN_PAYS]
    if not pay or not unans:
        print("A CLASS IS EMPTY -- vacuous")
        return 2
    n_pay, n_unans, total = len(pay), len(unans), len(pay) + len(unans)

    lo, hi = d["separability_ci"]
    print(f"separability {d['separability']:.4f}  CI [{lo:.4f}, {hi:.4f}]  "
          f"(ANSWERABLE vs UNANSWERABLE, as preregistered)")
    print(f"cost population: {'+'.join(ANSWER_PAYS)} n={n_pay}  vs  {ABSTAIN_PAYS} n={n_unans}"
          f"   (excluded: {', '.join(EXCLUDED)})")
    print(f"  payoff a = {PAYOFF_ANSWER}   break-even p* = a/(1+a) = {P_STAR:.4f}")

    baseline = mean_score(0, 0, n_pay, n_unans)
    oracle = mean_score(n_unans, 0, n_pay, n_unans)
    bound = oracle - baseline
    print(f"\n  always answer : {baseline:.4f}")
    print(f"  oracle        : {oracle:.4f}   available bound {bound:+.4f}")

    # The calibration is rebuilt from the artifact's own rows so that `confidence` -- which needs
    # both threshold AND scale -- is available; the artifact persists only the threshold.
    ans_only = [float(r["top1_cosine"]) for r in rows if r["cls"] == "ANSWERABLE"]
    cal = from_samples(str(d["embedder"]), ans_only, unans)
    # Abstain when P(unanswerable) > p*  <=>  confidence < 1 - p*. Monotone, so invert by scan.
    target = 1.0 - P_STAR
    breaks = sorted({*pay, *unans})
    prereg_thr = next((c for c in breaks if cal.confidence(c) >= target), breaks[-1])

    print("\n=== operating points ===")
    print(f"  {'rule':38} {'thr':>7} {'false-abst':>11} {'caught':>8} {'mean':>8} {'delta':>9}")
    for label, thr in (
        ("library best_threshold (q05/q95 mid)", float(d["threshold"])),
        (f"PREREGISTERED break-even p*={P_STAR:.4f}", prereg_thr),
    ):
        fa, c = counts_at(pay, unans, thr)
        m = mean_score(c, fa, n_pay, n_unans)
        print(f"  {label:38} {thr:>7.4f} {fa/n_pay:>10.1%} {c/n_unans:>7.1%} {m:>8.4f} {m-baseline:>+9.4f}")

    # Exact optimum over the score breakpoints, not a 0.01 lattice: mean_score is a step function
    # that only changes at observed values, so a lattice maximum is a strict lower bound.
    best = (float("-inf"), 0.0, 0, 0)
    for thr in breaks:
        fa, c = counts_at(pay, unans, thr)
        m = mean_score(c, fa, n_pay, n_unans)
        if m > best[0]:
            best = (m, thr, fa, c)
    m, thr, fa, c = best
    print(f"  {'IN-SAMPLE optimum over breakpoints':38} {thr:>7.4f} {fa/n_pay:>10.1%} "
          f"{c/n_unans:>7.1%} {m:>8.4f} {m-baseline:>+9.4f}")
    share = (m - baseline) / bound if bound > 0 else 0.0
    print(f"\n  the shipped signal recovers {share:.1%} of the available bound,")
    print(f"  and that figure is IN-SAMPLE (chosen and scored on the same {total} tasks),")
    print("  so it is an upper bound on what a held-out threshold would achieve.")
    if m <= baseline:
        print("  NOTE: no threshold beats always-answering. Every operating point loses.")

    print("\n=== what the shipped gap_warning actually did ===")
    gw = Counter((str(r["cls"]), bool(r["gap_warning"])) for r in rows)
    fired_total = sum(v for (_, f), v in gw.items() if f)
    for cls in (*ANSWER_PAYS, ABSTAIN_PAYS, *EXCLUDED):
        fired, quiet = gw[(cls, True)], gw[(cls, False)]
        if fired + quiet:
            print(f"  {cls:15} fired {fired:>4}/{fired+quiet:<4} = {fired/(fired+quiet):.1%}")
    if fired_total == 0:
        print(f"  INERT: fired 0 times in {len(rows)} queries, on every class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
