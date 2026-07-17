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
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles

DEFAULT_SCALE = 0.05
DEFAULT_PATH = "calibration.json"
ENV_VAR = "RECALL_CALIBRATION"


def best_threshold(answerable: list[float], unanswerable: list[float]) -> float:
    """Threshold minimising misclassification: answerable should score >= it, unanswerable below."""
    candidates = sorted(set(answerable + unanswerable))
    best_thr, best_err = 0.5, len(answerable) + len(unanswerable) + 1
    for c in candidates:
        err = sum(1 for a in answerable if a < c) + sum(1 for u in unanswerable if u >= c)
        if err < best_err:
            best_err, best_thr = err, c
    # Round DOWN: the optimizer guarantees answerable samples score >= the chosen candidate;
    # nearest-rounding could lift the threshold above the candidate and flip a boundary
    # answerable case to low_confidence at runtime.
    return math.floor(best_thr * 1000) / 1000


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
        print(f"recall: ignoring unreadable calibration file {p} (uncalibrated fallback)",
              file=sys.stderr)
        return None
    if not (math.isfinite(threshold) and -1.0 <= threshold <= 1.0
            and math.isfinite(scale) and scale > 0.0):
        print(f"recall: ignoring out-of-range calibration in {p} "
              f"(threshold={threshold!r}, scale={scale!r}) — uncalibrated fallback",
              file=sys.stderr)
        return None
    return Calibration(embedder=embedder, threshold=threshold, scale=scale)
