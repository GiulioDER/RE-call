"""Score a ladder arm on the axis that matters: can the score tell answerable from unanswerable?

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

**Written before the v3 arm finished, deliberately.** The analysis is part of the pre-registration
(`benchmarks/PREREGISTRATION-ladder-v3.md`), not a choice made after seeing numbers. An analysis
selected once the data is visible is a free parameter, and this project has already paid for one
of those.

Why AUC-vs-answerable rather than the within-rung deltas v2 published: v2's paired deltas compare
`r=0.00` to further rungs, and every one of those instances has its gold excised — that is an axis
*inside* the unanswerable set. The claim "answerability is graded" names the answerable/unanswerable
boundary, so it has to be scored against the answerable originals. On the v2 arm that curve runs
AUC 0.567 / 0.784 / 0.841 / 0.921 / 0.968 — graded, and **barely above chance at the near rung**
(`results/ladder/H1_VERDICT_v2.md`, addendum).

Both are reported here: the direct contrast, and v2's deltas for continuity.

Stdlib only, like `score.py` — the arithmetic deciding a pre-registered verdict should be readable
without trusting a library.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from benchmarks.ladder.manifest import RING_ORIGINAL, read_manifest
from benchmarks.ladder.rings import ring_to_fraction

#: The shipped abstention floor both prior arms ran with. Not a proposal — see
#: [[project-recall-threshold-embedder-fragile-2026-07-28]]: it is inert on some embedders and
#: over-eager on others, so its value is reported, never recommended.
SHIPPED_FLOOR = 0.50

#: P3's pre-registered bar for per-question consistency, and P4's inertness bar.
P3_MONOTONE_FLOOR = 0.70
P4_ABSTAIN_CEILING = 0.02


def auc(positive: list[float], negative: list[float]) -> float:
    """P(a random `negative` scores below a random `positive`), ties at 0.5.

    Mann-Whitney, written out rather than imported: it is threshold-free, so unlike an accuracy at
    some cutoff it cannot be inflated by choosing the cutoff after seeing the data.
    """
    if not positive or not negative:
        return float("nan")
    wins = 0.0
    for a in positive:
        for b in negative:
            wins += 1.0 if b < a else (0.5 if b == a else 0.0)
    return wins / (len(positive) * len(negative))


def bootstrap_ci(
    values: list[float], *, seed: int = 0, iterations: int = 10_000
) -> tuple[float, float, float]:
    """Mean with a bootstrap 95 % CI. Raises on empty input rather than returning a fake zero —
    a triple of zeros is indistinguishable from a tightly-measured null (the defect `score.py`
    already carries a guard for)."""
    if not values:
        raise ValueError("no values to bootstrap: absent data must not render as a measured zero")
    n = len(values)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iterations))
    return sum(values) / n, means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def load_cosines(manifest: Path, responses: Path) -> dict[str, dict[int, float]]:
    """pair_id -> ring -> top-1 cosine. Instances with no recorded response are omitted, never
    defaulted — a missing row is missing data."""
    instances, _header = read_manifest(manifest)
    by_id = {i.instance_id: i for i in instances}
    out: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for line in responses.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        inst = by_id.get(row["instance_id"])
        if inst is None or row.get("top_cosine") is None:
            continue
        out[inst.pair_id][inst.ring] = float(row["top_cosine"])
    return dict(out)


def _label(ring: int) -> str:
    return "original" if ring == RING_ORIGINAL else f"r={ring_to_fraction(ring):.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--label", default="arm", help="arm name for the report header")
    args = parser.parse_args(argv)

    cos = load_cosines(args.manifest, args.responses)
    rungs = sorted({r for c in cos.values() for r in c} - {RING_ORIGINAL})
    answerable = [c[RING_ORIGINAL] for c in cos.values() if RING_ORIGINAL in c]

    print(f"=== {args.label} ===")
    print(f"pairs with a scored answerable original: {len(answerable)}")
    if not answerable:
        raise SystemExit("no answerable originals scored - nothing to contrast against")

    print("\nDIRECT CONTRAST - AUC(answerable vs rung), threshold-free:")
    aucs: dict[int, float] = {}
    for r in rungs:
        neg = [c[r] for c in cos.values() if r in c]
        aucs[r] = auc(answerable, neg)
        print(f"  answerable vs {_label(r):<8} n={len(neg):<4} AUC {aucs[r]:.3f}")

    print("\nWITHIN-UNANSWERABLE (v2's axis, for continuity) - paired delta vs r=0.00:")
    base = rungs[0]
    for r in rungs[1:]:
        d = [c[r] - c[base] for c in cos.values() if r in c and base in c]
        m, lo, hi = bootstrap_ci(d)
        print(f"  {_label(r):<8} n={len(d):<4} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    full = [c for c in cos.values() if all(r in c for r in rungs)]
    mono = sum(1 for c in full if all(c[rungs[k]] >= c[rungs[k + 1]] for k in range(len(rungs) - 1)))
    print(f"\nmonotone NON-INCREASING across all rungs: {mono}/{len(full)} = {mono/len(full):.3f}")
    print("per-step (strict decrease / ties / non-increase):")
    for k in range(len(rungs) - 1):
        a, b = rungs[k], rungs[k + 1]
        s = sum(1 for c in full if c[b] < c[a])
        t = sum(1 for c in full if c[b] == c[a])
        print(f"  {_label(a)} -> {_label(b)}: {s} / {t} / {s + t}")

    below = [(r, sum(1 for c in cos.values() if r in c and c[r] < SHIPPED_FLOOR)) for r in rungs]
    total = sum(n for _, n in below)
    print(f"\nshipped floor {SHIPPED_FLOOR}: {total} responses below it across all rungs")

    print("\n--- pre-registered predictions (PREREGISTRATION-ladder-v3.md) ---")
    top = rungs[-1]
    d_top = [c[top] - c[base] for c in cos.values() if top in c and base in c]
    m, lo, hi = bootstrap_ci(d_top)
    p1 = m < 0 and hi < 0
    print(f"P1 (KILL) gradient reproduces: delta={m:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {'PASS' if p1 else 'FAIL'}")
    means = [sum(c[r] for c in full) / len(full) for r in rungs]
    p2 = all(means[k] >= means[k + 1] for k in range(len(means) - 1))
    print(f"P2 monotone rung means: {'PASS' if p2 else 'FAIL'}")
    p3 = (mono / len(full)) >= P3_MONOTONE_FLOOR
    print(f"P3 per-question >= {P3_MONOTONE_FLOOR}: {mono/len(full):.3f} -> {'PASS' if p3 else 'FAIL'}")
    worst = max(n / max(1, sum(1 for c in cos.values() if r in c)) for r, n in below)
    p4 = worst <= P4_ABSTAIN_CEILING
    print(f"P4 floor inert (<= {P4_ABSTAIN_CEILING} at every rung): worst {worst:.3f} -> {'PASS' if p4 else 'FAIL'}")
    return 0 if p1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
