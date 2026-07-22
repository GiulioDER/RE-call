"""Runtime calibration: a persistable per-embedder abstention threshold + confidence mapping.

The evaluation study (results/FINDINGS.md §2) showed a fixed cosine threshold does not transfer
across embedders — each model's cosines live in a different regime. This module turns that
finding into a runtime artifact: calibrate once against a small labeled answerable/unanswerable
query set, save the result, and every search maps raw cosine to a calibrated confidence.

The confidence is a calibrated *ranking* confidence — a monotone logistic centered on the
calibrated decision boundary (0.5 exactly at the threshold) — not a true posterior probability;
the calibration sets are small.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles

from recall.observability import get_logger

DEFAULT_SCALE = 0.05
DEFAULT_PATH = "calibration.json"
ENV_VAR = "RECALL_CALIBRATION"

_log = get_logger("calibration")


#: Quantiles bounding the gap: the answerable floor and the unanswerable ceiling. Both are
#: deliberately not the extremes — one outlier on either side must not define the boundary.
ANSWERABLE_FLOOR_Q = 0.05
UNANSWERABLE_CEILING_Q = 0.95


def _quantile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def best_threshold(answerable: list[float], unanswerable: list[float]) -> float:
    """Threshold placed in the MIDDLE of the observed gap between the two distributions.

    Specifically the midpoint of ``q05(answerable)`` and ``q95(unanswerable)``.

    The previous rule minimised misclassification on the samples given to it, which sounds
    principled and is not: the cheapest way to keep every answerable sample above the boundary is
    to put the boundary exactly ON the lowest one. That has three measured consequences.

    - **No margin on the answerable side.** Any real answer scoring below the weakest calibration
      sample abstains. Leave-one-out false-abstain was ``1/n`` even on perfectly separable data.
    - **One sample decides everything.** The answerable distribution has a long lower tail
      (measured with bge-small: min 0.601, p25 0.913), so the boundary sat at the bottom of that
      tail and let **20.5%** of genuinely unanswerable queries through.
    - **It inherited ANN noise.** HNSW index builds are nondeterministic, so the identity of the
      worst sample changed on every rebuild and the whole operating point moved with it
      (coverage swung 0.40–0.84 on one host — issue #26).

    Measured on the same data, fitted on half the queries and scored on the other half over four
    index rebuilds, this rule cuts the false-confident rate from **0.205 to 0.045** and costs
    **1%** of answerable queries. Going further is a bad trade: a q20 floor drives false-abstain
    to 0.31 to buy the last few points of FCR.

    ⚠️ **Outlier robustness needs samples.** The floor is a 5th percentile, and a 5% tail is not
    identifiable from a handful of points, so below roughly 20 answerable samples it collapses
    onto the minimum and one bad retrieval moves the boundary again. Bisecting the gap still adds
    margin at any size — that part always holds — but a small calibration set buys margin, not
    stability. Calibrate against a few hundred labelled queries if the threshold matters.

    Degenerate inputs fall back rather than invent a boundary: with no unanswerable samples there
    is no gap to bisect, so the answerable floor is used; with neither class, the module default.
    """
    if not answerable and not unanswerable:
        return 0.5
    a = sorted(answerable)
    u = sorted(unanswerable)
    if not a:  # only negatives: sit just above their ceiling
        return math.floor(_quantile(u, UNANSWERABLE_CEILING_Q) * 1000) / 1000
    floor = _quantile(a, ANSWERABLE_FLOOR_Q)
    if not u:
        return math.floor(floor * 1000) / 1000
    ceiling = _quantile(u, UNANSWERABLE_CEILING_Q)
    # Overlapping distributions still bisect: the midpoint splits the overlap instead of
    # collapsing onto one class, which is the least-bad boundary when no clean gap exists.
    # Round DOWN so rounding can only ever make the guard more permissive, never silently
    # abstain on a calibration sample that sat exactly on the boundary.
    return math.floor((floor + ceiling) / 2 * 1000) / 1000


@dataclass(frozen=True)
class Calibration:
    embedder: str
    threshold: float
    scale: float = DEFAULT_SCALE

    def confidence(self, cosine: float) -> float:
        """Monotone cosine -> [0, 1] mapping; exactly 0.5 at the calibrated threshold."""
        x = (cosine - self.threshold) / self.scale
        x = max(-60.0, min(60.0, x))  # clamp: math.exp overflows past ~709; ±60 already saturates
        return 1.0 / (1.0 + math.exp(-x))


def from_samples(embedder: str, answerable: list[float], unanswerable: list[float]) -> Calibration:
    """Build a calibration from per-query top-cosine samples (see recall.eval.calibrate).

    The logistic scale is derived from the separation between the distributions
    (q25(answerable) - q75(unanswerable)) / 4, floored at 0.01; with fewer than two samples
    on either side there is no spread to measure, so DEFAULT_SCALE is used.
    """
    thr = best_threshold(answerable, unanswerable)
    if len(answerable) >= 2 and len(unanswerable) >= 2:
        # method="inclusive" stays bounded by the observed data; the default exclusive
        # method extrapolates beyond [min, max] for n=2 samples.
        q25_ans = quantiles(answerable, n=4, method="inclusive")[0]
        q75_unans = quantiles(unanswerable, n=4, method="inclusive")[2]
        scale = max((q25_ans - q75_unans) / 4, 0.01)
    else:
        scale = DEFAULT_SCALE
    return Calibration(embedder=embedder, threshold=thr, scale=round(scale, 4))


def _resolve_path(path: str | Path | None) -> Path:
    return Path(path or os.environ.get(ENV_VAR) or DEFAULT_PATH)


def save(cal: Calibration, path: str | Path | None = None) -> Path:
    """Write the calibration JSON; returns the path written."""
    p = _resolve_path(path)
    p.write_text(
        json.dumps(
            {"embedder": cal.embedder, "threshold": cal.threshold, "scale": cal.scale}, indent=2
        ),
        encoding="utf-8",
    )
    return p


def load_for(embedder: str, path: str | Path | None = None) -> Calibration | None:
    """Load the calibration for `embedder`, or None when it cannot be applied safely.

    Returns None (uncalibrated fallback, flagged in every result) when the file is absent,
    unreadable, malformed, calibrated for a DIFFERENT embedder, or carries out-of-range values —
    a threshold calibrated in another model's cosine regime must never be applied, and a
    corrupt file must never be able to disable abstention silently (NaN threshold) or crash
    every search (zero/negative scale).
    """
    p = _resolve_path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("embedder") != embedder:
            return None
        threshold = float(data["threshold"])
        scale = float(data["scale"])
    except (OSError, json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError):
        _log.warning("ignoring unreadable calibration file %s (uncalibrated fallback)", p)
        return None
    if not (math.isfinite(threshold) and -1.0 <= threshold <= 1.0
            and math.isfinite(scale) and scale > 0.0):
        _log.warning(
            "ignoring out-of-range calibration in %s (threshold=%r, scale=%r) — "
            "uncalibrated fallback", p, threshold, scale,
        )
        return None
    return Calibration(embedder=embedder, threshold=threshold, scale=scale)
