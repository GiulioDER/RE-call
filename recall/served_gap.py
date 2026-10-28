"""Does serving threshold the statistic calibration measured? A report, never a gate.

Calibration samples each labelled query's EXACT best cosine over the whole generation
(`recall.eval.calibrate.measure_top_cosines`). Serving thresholds the cosines of the hits it
returns, after fusion, optional reranking and the cut to k (`recall.trust_gate`): a query abstains
when no returned hit clears the threshold. When the exact top chunk is fused out of the returned
hits, serving reads a lower score than the one the threshold was fitted on, and nothing reported it.

This module compares the two for every stored labelled query of a calibration and reports how
often they agree, how large the gaps are, and how many queries the gap moves across the threshold.
Measured on 2026-10-21 against two production tenants through the live MCP server at k=5: 72 of 78
and 47 of 48 queries agreed within 0.001; every query whose exact top chunk was fused out was
answerable (largest drop 0.073); no verdict changed. A fused-out chunk only ever LOWERS the served
score, so this slack can add false abstentions near the threshold, never false confirms.

It changes nothing: no threshold, no gate, no stored row. The served side is whatever search the
caller passes, so a run measures the search path it was given and no other.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: Two maxima closer than this agree. One embedding call apart, a hosted model was measured to
#: move a query's best cosine by up to 0.0051 (1 query of 78); pass a wider tolerance when the
#: served and exact sides embed the query separately on such a model.
DEFAULT_TOLERANCE = 0.001

_CLASSES = ("answerable", "unanswerable")


@dataclass(frozen=True)
class ServedGapRow:
    """One labelled query: its exact best cosine, the best cosine served, and the served decision.

    `served` is None when serving returned no hit carrying a dense cosine; such a query disagrees
    and is served below any threshold.
    """

    answerable: bool
    exact: float
    served: float | None
    abstained: bool


def served_gap_report(
    rows: Iterable[ServedGapRow], threshold: float, tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, Any]:
    """Agreement, gap sizes and verdict flips at `threshold`, overall and per labelled class.

    A served score equal to the threshold clears it, as in `recall.trust_gate`, where a hit is
    `low_confidence` only strictly below the threshold.
    """
    if not (math.isfinite(tolerance) and tolerance >= 0):
        # A negative or NaN tolerance inverts the agreement test: identical maxima would read as
        # served above the exact one.
        raise ValueError(f"tolerance must be a finite number >= 0, got {tolerance!r}")
    items = list(rows)
    per_class: dict[str, dict[str, int]] = {
        name: {"n": 0, "agree": 0, "score_flips": 0, "abstain_disagreements": 0} for name in _CLASSES
    }
    gaps: list[float] = []
    nothing_served = 0
    above = 0
    flips: list[dict[str, Any]] = []
    for row in items:
        served = row.served
        gap = row.exact - served if served is not None else None
        # Two-sided: a served maximum ABOVE the exact one is not a fused-out chunk, it is noise
        # between two embedding calls or a read of another generation, and it must stay visible.
        agrees = gap is not None and abs(gap) <= tolerance
        exact_clears = row.exact >= threshold
        flipped = exact_clears != (served is not None and served >= threshold)
        abstain_disagrees = exact_clears != (not row.abstained)
        bucket = per_class["answerable" if row.answerable else "unanswerable"]
        bucket["n"] += 1
        bucket["agree"] += agrees
        bucket["score_flips"] += flipped
        bucket["abstain_disagreements"] += abstain_disagrees
        if gap is not None and gap < -tolerance:
            above += 1
        elif gap is None:
            nothing_served += 1
        elif not agrees:
            gaps.append(round(gap, 4))
        if flipped:
            flips.append({
                "answerable": row.answerable,
                "exact": round(row.exact, 4),
                "served": None if served is None else round(served, 4),
            })
    finite = sorted(gaps)
    n = len(items)
    agree = sum(bucket["agree"] for bucket in per_class.values())
    return {
        "n": n,
        "threshold": float(threshold),
        "tolerance": float(tolerance),
        "agree": agree,
        "agreement_rate": round(agree / n, 4) if n else None,
        "gaps_above_tolerance": len(gaps) + nothing_served,
        "nothing_served": nothing_served,
        "served_above_exact": above,
        "gap_median": round(statistics.median(finite), 4) if finite else None,
        "gap_max": finite[-1] if finite else None,
        "score_flips": sum(bucket["score_flips"] for bucket in per_class.values()),
        "abstain_disagreements": sum(
            bucket["abstain_disagreements"] for bucket in per_class.values()
        ),
        "by_class": per_class,
        "flips": flips,
    }


def describe(report: Mapping[str, Any]) -> str:
    """One line for the CLI."""
    below = report["gaps_above_tolerance"]
    line = (
        f"served versus exact maximum: {report['agree']} of {report['n']} agree within "
        f"{report['tolerance']}; {report['score_flips']} verdict flip(s) at threshold "
        f"{report['threshold']:.3f}"
    )
    if report["gaps_above_tolerance"]:
        line += (
            f"; {below} served below the exact maximum, "
            f"median gap {report['gap_median']}, largest {report['gap_max']}"
        )
    if report.get("served_above_exact"):
        line += f"; {report['served_above_exact']} served above the exact maximum"
    if report["abstain_disagreements"]:
        line += (
            f"; {report['abstain_disagreements']} abstain disagreement(s) between the served "
            "decision and the exact score"
        )
    return line + " (reported only, never a gate)"
