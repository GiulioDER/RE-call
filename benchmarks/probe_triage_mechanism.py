"""What is `ratio_8_over_1` actually reading? Pass A, offline, from a frozen retrieval fixture.

Pre-registered at `results/enterprise_rag/PREREGISTRATION-triage-mechanism.md` (`1a153cd`).

The one surviving number from the triage work is a held-out AUC of 0.642 for a feature whose own
correction says: *"I do not currently have an explanation for why it predicts."* This module tests
a specific explanation rather than searching for more features.

**The hypothesis.** `_rrf` gives each chunk `1 / (60 + rank + 1)` from every leg that returned it,
so with three legs the fused score is dominated by HOW MANY legs found the chunk. The list is
ordered by that fused score, while each hit's stored `score` is the dense cosine. So the dense
score at fused rank 8 is largely a readout of how far down the fused list you get before hitting
chunks the dense leg did not choose, and `ratio_8_over_1` would be a leg-disagreement statistic
wearing a score-curve costume.

Everything Pass A needs is already in a capture-1 fixture, because the dense scores are stored
**in fused order**: the disagreement between the two orderings is fully observable. M4 and M5 need
the fused score and the per-leg ranks, which only a capture-2 fixture carries, and are skipped
with a printed reason on an older one rather than silently omitted.

⛔ Registered look count: three disagreement features (M3, M5, M6). Anything tried after reading
these numbers is exploration and must be labelled so.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.console import use_utf8_output
from benchmarks.analyse_triage import auc, load_question_text
from benchmarks.explore_triage_signal import _labels, build_features

#: Below this, a denominator is not a scale, it is noise, and the ratio it produces would sort
#: among genuine values rather than beside them.
DENOMINATOR_EPSILON = 1e-9


def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional ranks, ties averaged. Shared by `spearman` so ties cannot be handled two ways."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, ties averaged. NaN when either side has no ordering to correlate.

    Rank-based rather than Pearson because the two quantities compared here, a fused position and
    a dense cosine, live on unrelated scales and only their ORDERING is comparable.
    """
    if any(v != v for v in (*xs, *ys)):
        raise ValueError("NaN in input; refusing a correlation that would be row-order noise")
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    covariance = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    spread_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    spread_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if spread_x == 0.0 or spread_y == 0.0:  # an all-tied side has no ordering at all
        return float("nan")
    return covariance / (spread_x * spread_y)


def inversions(scores: Sequence[float]) -> int:
    """Pairs that are out of descending order. The list is best-first, so each one is a place
    where fusion promoted a chunk the dense leg scored lower. Ties are not inversions.

    O(n^2), deliberately: it runs over the top 8, and a merge-sort version would be harder to
    read for no measurable gain.
    """
    return sum(
        1
        for i in range(len(scores))
        for j in range(i + 1, len(scores))
        if scores[j] > scores[i]
    )


def published_ratio(scores: Sequence[float]) -> float:
    """`ratio_8_over_1` exactly as `explore_triage_signal.build_features` computes it.

    ⚠️ Warts included, on purpose. This is the definition behind the published 0.6375 and 0.642,
    so it is the only one against which those numbers can be reproduced (apparatus check A4). Its
    two known hazards, a 0.0 out of range and an unguarded denominator, are addressed in
    `guarded_ratio` rather than here.
    """
    def at(i: int) -> float:
        return scores[i] if 0 <= i < len(scores) else 0.0

    return at(7) / at(0) if at(0) else 0.0


def inverted(scores: Sequence[float]) -> bool:
    """The registered inverted regime: the dense score at rank 8 exceeds the one at rank 1.

    ⚠️ ONE predicate, used by both M2 (which counts it) and M6 (which excludes it). The
    registration glosses this as "`ratio_8_over_1 > 1`", and the two are the same statement only
    while the denominator is positive: dividing by a negative rank-1 cosine flips the inequality,
    so a raw comparison and a ratio comparison disagree on exactly the rows the registration warns
    contain negative scores. Two predicates would make M6's kept set not the complement of M2's
    counted set, and P1 would then be evaluated on a regime nobody defined.

    The raw comparison is the one the registration leads with, so it is the one implemented.
    `ratio_disagrees_with_inverted` reports how often the gloss would have said otherwise.
    """
    return len(scores) >= 8 and scores[7] > scores[0]


def ratio_disagrees_with_inverted(scores: Sequence[float]) -> bool:
    """True where the registration's two phrasings of the inverted regime part company."""
    return inverted(scores) != (published_ratio(scores) > 1.0)


def guarded_ratio(scores: Sequence[float]) -> float | None:
    """`ratio_8_over_1` with its two hazards refused instead of papered over.

    `None` where the ratio has no meaning: a pool shorter than 8 (which the published version
    scored as 0.0, below every genuine value), or a denominator at or below the epsilon, including
    the negative dense cosines that 5 of 500 fixture rows contain.

    A negative NUMERATOR is kept. A dense score that has gone negative by rank 8 is precisely the
    disagreement under investigation, and discarding it would remove the signal being measured.
    """
    if len(scores) < 8:
        return None
    if scores[0] <= DENOMINATOR_EPSILON:
        return None
    return scores[7] / scores[0]


def _auc_over(values: Sequence[float | None], labels: Sequence[int]) -> tuple[float, int, int]:
    """AUC over the rows where `values` is not None, with the n and the positive count it used.

    A feature that refuses some rows must report the denominator it actually scored on. Silently
    dropping rows and printing an AUC beside features scored on all 500 would compare two numbers
    that are not comparable.
    """
    kept = [(v, y) for v, y in zip(values, labels, strict=True) if v is not None]
    if not kept:
        return float("nan"), 0, 0
    positives = sum(y for _, y in kept)
    return auc([v for v, _ in kept], [y for _, y in kept]), len(kept), positives


def _matched(
    values: Sequence[float | None],
    baseline: Sequence[float],
    labels: Sequence[int],
    name: str,
    prediction: str,
) -> str:
    """One line for a restricted feature, with the BASELINE recomputed on the same rows.

    🔑 This exists because P2 is decided by a comparison, not by a level. A feature scored on the
    rows it accepts, printed beside a baseline scored on all 500, is two numbers over two
    denominators, and reading "below 0.6375" off them is not a comparison at all. Where the subsets
    coincide the matched baseline simply equals the published one, and the line says so.
    """
    value, n, positives = _auc_over(values, labels)
    mask = [b if v is not None else None for v, b in zip(values, baseline, strict=True)]
    matched, _, _ = _auc_over(mask, labels)
    return (
        f"  {name} = {value:.4f}   on n={n} ({positives} positive); "
        f"published ratio on those SAME rows = {matched:.4f}   [{prediction}]"
    )


def _digest(evidence: Mapping[str, Any]) -> str:
    """The fixture's own digest recipe, re-derived here so A1 does not need the freeze module."""
    from benchmarks.freeze_supersession_evidence import evidence_digest

    return evidence_digest(dict(evidence))


def main(argv: Sequence[str] | None = None) -> int:
    use_utf8_output()  # argparse prints this module's docstring; cp1252 cannot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--label", default="missed_any",
                        choices=("missed_any", "recoverable", "absent"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expect-digest", default=None,
                        help="apparatus check A1: refuse a fixture that is not this one")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(args.retrieval.read_text(encoding="utf-8"))
    rows = payload["evidence"]
    questions = load_question_text(args.questions, rows)
    digest = _digest(rows)
    print(f"fixture: {args.retrieval.name}  n={len(rows)}  digest={digest[:8]}…")
    # `.` as well as `…`: a prefix copied from the pre-registration carries a Unicode ellipsis,
    # one typed by hand carries three dots, and stripping only the first rejects the right fixture.
    if args.expect_digest and not digest.startswith(args.expect_digest.rstrip("….")):
        raise SystemExit(
            f"A1 FAILED: fixture digest {digest} does not start with {args.expect_digest!r}. "
            "This is not the fixture the pre-registration named; refusing to score it."
        )
    print(f"A1 digest check: {'PASSED' if args.expect_digest else 'not requested'}")

    # Positive evidence on EVERY row, not the absence of counter-evidence: `all()` over an empty
    # generator is True, so a fixture whose rows all have an empty `ranked` list announced itself
    # as capture 2 and then reported M4 and M5 as `nan on n=0` instead of "not scoreable".
    # The hit inspected is the deepest one M4 and M5 actually read, and both fields are required.
    has_fusion = bool(rows) and all(
        row["ranked"]
        and all(key in row["ranked"][min(7, len(row["ranked"]) - 1)]
                for key in ("fused_score", "ranks"))
        for row in rows.values()
    )
    print(f"capture: {'2 (carries fused_score and per-leg ranks)' if has_fusion else '1 (dense score only; M4 and M5 are NOT scoreable)'}")
    print()

    labels: list[int] = []
    rho: list[float] = []
    inverted_top: list[float] = []
    published: list[float] = []
    guarded: list[float | None] = []
    fused_ratio: list[float | None] = []
    legs_at_8: list[float | None] = []
    inverted_flags: list[bool] = []

    disagreeing = 0
    a4_failures: list[str] = []

    for question_id, row in rows.items():
        label = _labels(row, args.top_k)[args.label]
        if label is None:
            continue
        scores = [float(hit["score"]) for hit in row["ranked"]]
        top = scores[: args.top_k]
        labels.append(label)

        # A4, literally as registered: per row, to 1e-9, against the script that produced the
        # 0.642. `published_ratio` is a second copy of that expression, and two copies drift into
        # a plausible number rather than an error, so they are compared rather than assumed equal.
        theirs = build_features(row, questions[question_id], args.top_k)["ratio_8_over_1"]
        if abs(published_ratio(scores) - theirs) > 1e-9:
            a4_failures.append(question_id)

        # M1 is over the WHOLE pool, not the top 8: the hypothesis is about how fusion and the
        # dense leg disagree overall, and a top-8 window would see only the head of that.
        correlation = spearman(list(range(len(scores))), scores)
        if not math.isnan(correlation):
            rho.append(correlation)
        inverted_top.append(float(inversions(top)))
        published.append(published_ratio(scores))
        guarded.append(guarded_ratio(scores))
        inverted_flags.append(inverted(scores))
        if ratio_disagrees_with_inverted(scores):
            disagreeing += 1

        if has_fusion:
            fused = [float(hit["fused_score"]) for hit in row["ranked"]]
            fused_ratio.append(fused[7] / fused[0] if len(fused) >= 8 and fused[0] else None)
            ranks = row["ranked"][7]["ranks"] if len(row["ranked"]) >= 8 else None
            legs_at_8.append(float(sum(r is not None for r in ranks)) if ranks else None)

    n = len(labels)
    if not n:
        raise SystemExit(
            f"no row in this fixture yields a {args.label!r} label; nothing to score. "
            "Every row was skipped, which usually means expected_doc_ids is empty throughout."
        )
    positives = sum(labels)
    print(f"label = {args.label}: {positives}/{n} positive ({positives/n:.1%})")
    print()

    print("=== apparatus, before the registered numbers ===")
    if a4_failures:
        raise SystemExit(
            f"A4 FAILED on {len(a4_failures)} of {n} rows, e.g. {a4_failures[:3]}: this probe's "
            "ratio_8_over_1 disagrees with explore_triage_signal's by more than 1e-9. It is "
            "therefore not the feature the published 0.642 belongs to; refusing to report."
        )
    print(f"  A4 per-row agreement with explore_triage_signal: PASSED on all {n} rows (1e-9)")
    rnd = [random.Random(args.seed + i).random() for i in range(n)]
    print(f"  A2 random feature      AUC = {auc(rnd, labels):.4f}   (expected ~0.50)")
    print(f"  A2 label as a feature  AUC = {auc([float(y) for y in labels], labels):.4f}   (expected 1.00)")
    published_auc = auc(published, labels)
    print(f"  A4 published ratio_8_over_1 AUC = {published_auc:.4f}   "
          "(0.6375 on missed_any reproduces the published number)")
    moved = sum(1 for p, g in zip(published, guarded, strict=True)
                if g is None or abs(g - p) > 1e-12)
    print(f"  A4 rows where the guard changes the value: {moved}/{n}")
    print(f"  rows where the registration's two phrasings of 'inverted' disagree: {disagreeing}/{n}")
    print()

    print("=== registered ===")
    mean_rho = statistics.fmean(rho) if rho else float("nan")
    print(f"  M1 mean per-query Spearman(position, dense score) = {mean_rho:+.4f}   "
          f"over {len(rho)}/{n} queries   [predicted -0.55, -0.30 to -0.75]")
    above_rank1 = sum(inverted_flags)
    print(f"  M2 fraction with dense(rank 8) > dense(rank 1)    = {above_rank1/n:.4f}   "
          f"({above_rank1}/{n})   [predicted 0.12, 0.05 to 0.25]")
    print(f"  M3 AUC of inversions in the top {args.top_k}                = "
          f"{auc(inverted_top, labels):.4f}   [predicted 0.60, 0.53 to 0.68; <=0.53 FALSIFIES]")

    # The complement of M2's counted set, by construction: one predicate, recorded once per row
    # in the loop above and used both ways here.
    flat = [None if inv else p for p, inv in zip(published, inverted_flags, strict=True)]
    # No matched baseline here, unlike M4 and M5: M6 IS the published feature on a subset, so a
    # "published ratio on those same rows" column would print the same number twice and read as
    # agreement between two measurements rather than one number restated.
    m6, m6_n, m6_pos = _auc_over(flat, labels)
    print(f"  M6 AUC of ratio_8_over_1 outside the inverted regime = {m6:.4f}   "
          f"on n={m6_n} ({m6_pos} positive)   [predicted 0.60, 0.52 to 0.68]")

    p3 = spearman(inverted_top, published)
    print(f"  P3 Spearman(inversions, ratio_8_over_1)          = {p3:+.4f}   "
          "[predicted |rho| > 0.5]")
    print()

    if has_fusion:
        # 🔑 P2 is "M4 below the published 0.6375", which is a COMPARISON. `_matched` recomputes
        # the published ratio on whatever rows M4 could score, so the two numbers share a
        # denominator. Where the subsets coincide the matched figure equals the published one.
        print(_matched(fused_ratio, published, labels,
                       "M4 AUC of fused(rank 8)/fused(rank 1)",
                       "predicted 0.55, 0.50 to 0.63; P2 wants it BELOW the matched figure"))
        distinct = len({v for v in fused_ratio if v is not None})
        print(f"     distinct values of the fused ratio: {distinct}   "
              "(confound 4: a low M4 may be ties rather than absence of information)")
        print(_matched(legs_at_8, published, labels,
                       "M5 AUC of legs_hit at rank 8         ",
                       "predicted 0.58, 0.50 to 0.66"))
    else:
        print("  M4, M5: NOT SCOREABLE on this capture. Re-freeze with the capture-2 script,")
        print("          which records fused_score and per-leg ranks at no extra retrieval cost.")

    # Guarded variant reported alongside, never instead: the published number is the one with a
    # held-out result behind it, and replacing it here would quietly restate history.
    print()
    print(_matched(guarded, published, labels,
                   "  (guarded ratio_8_over_1, comparison only)",
                   "not registered; the published definition is the one with a held-out number"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
