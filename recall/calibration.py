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

#: Minimum samples per class before a percentile boundary means anything. FINDINGS §6: the q05
#: floor "cannot exclude anything below ~20 answerable samples", so it collapses onto the minimum
#: and one bad retrieval moves the operating point. The q95 ceiling has the same problem from the
#: other side, so the bar applies to both classes.
MIN_CALIBRATION_SAMPLES = 20

#: Separability (AUC) below which a threshold is NOT certified.
#:
#: A judgement anchored on two measured points, not a fitted constant — §2 of FINDINGS is the
#: standing warning against shipping a constant that merely looks principled:
#:
#: - LongMemEval per-question haystacks measured AUC **0.753**. The threshold fitted there had a
#:   best *in-sample* balanced error of 0.285 and a deployed false-abstain of **0.443** — it
#:   refused nearly half the questions retrieval had just answered correctly. Unusable.
#: - On the corpora where abstention works (PEPs abstention accuracy 1.00; the 14-document
#:   corpus) the classes are cleanly disjoint — bge-small answerable 0.70–0.90 against
#:   unanswerable 0.51–0.64 — i.e. AUC 1.00.
#:
#: 0.90 sits between them, nearer the working end, because certifying an unusable threshold means
#: silently refusing real answers. Override it if your own labelled set says otherwise — that is
#: the point of measuring instead of asserting.
MIN_SEPARABILITY = 0.90


def separability(answerable: list[float], unanswerable: list[float]) -> float | None:
    """Probability a random answerable sample outscores a random unanswerable one (AUC).

    Threshold-free on purpose. Every other quality figure in this module depends on where the
    boundary was put, which makes it partly a statement about the fitting rule; this is a property
    of the two distributions alone. It therefore cannot be inflated by fitting and scoring on the
    same samples — the defect FINDINGS §2b retracted a published number for.

    1.0 is perfect ordering, 0.5 is no signal at all, ties score half a win. Returns None when
    either class is empty, because "cannot judge" is not the same as "judged and found wanting".
    """
    if not answerable or not unanswerable:
        return None
    wins = sum(1 for a in answerable for u in unanswerable if a > u)
    ties = sum(1 for a in answerable for u in unanswerable if a == u)
    return (wins + 0.5 * ties) / (len(answerable) * len(unanswerable))


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
    #: Diagnosis of the calibration set this threshold came from. All three default to None so a
    #: calibration built or loaded WITHOUT a diagnosis reports "unknown" rather than "fine" —
    #: treating a missing diagnosis as a pass would let the silent failure survive an upgrade.
    separability: float | None = None
    n_answerable: int | None = None
    n_unanswerable: int | None = None

    @property
    def certified(self) -> bool | None:
        """Is this threshold supportable by the data it was fitted on?

        Tri-state on purpose. ``True`` passed both checks; ``False`` failed one; ``None`` means
        there was nothing to judge (one-class samples, or an artifact written before this check
        existed). ``None`` must never be read as ``True``.

        **This is a diagnosis and changes nothing at runtime.** `threshold`, `scale` and
        `confidence()` are identical whether or not it certifies. A gate that also silently moved
        the boundary would replace one invisible failure with another, and an upgrade would change
        retrieval behaviour without anyone asking for it.
        """
        if self.separability is None:
            return None
        if self.n_answerable is None or self.n_unanswerable is None:
            return None
        if min(self.n_answerable, self.n_unanswerable) < MIN_CALIBRATION_SAMPLES:
            return False
        return self.separability >= MIN_SEPARABILITY

    @property
    def certification_reason(self) -> str:
        """Why `certified` came out the way it did, in one line fit for a log or a CLI."""
        # Spelled out rather than delegated to `certified is None`. It is the same condition, but
        # written this way both a reader and the type checker can see that the three fields are
        # non-None below — `certified` is a property and narrows nothing.
        if self.separability is None or self.n_answerable is None or self.n_unanswerable is None:
            return (
                "not judged: the calibration set had only one class, or this artifact predates "
                "the separability check"
            )
        thin = [
            f"{name}={n}"
            for name, n in (("answerable", self.n_answerable),
                            ("unanswerable", self.n_unanswerable))
            if n is not None and n < MIN_CALIBRATION_SAMPLES
        ]
        if thin:
            return (
                f"too few samples ({', '.join(thin)}; need >= {MIN_CALIBRATION_SAMPLES} of each): "
                "a q05/q95 boundary is not identifiable from a handful of points and collapses "
                "onto the extremes"
            )
        if self.separability < MIN_SEPARABILITY:
            return (
                f"separability {self.separability:.3f} < {MIN_SEPARABILITY}: answerable and "
                "unanswerable scores overlap, so NO threshold separates them — this one will "
                "refuse real answers, reject unanswerable queries, or both, and moving it only "
                "trades one error for the other"
            )
        return (
            f"separability {self.separability:.3f} over {self.n_answerable}/"
            f"{self.n_unanswerable} samples"
        )

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
    cal = Calibration(
        embedder=embedder,
        threshold=thr,
        scale=round(scale, 4),
        separability=separability(answerable, unanswerable),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
    )
    # Warned here, not only in the CLI: most callers build a calibration through the library and
    # never run `recall calibrate`, and a diagnosis only the CLI prints is one a server deployment
    # never receives.
    if cal.certified is False:
        _log.warning(
            "abstention threshold %.3f for %s is NOT certified — %s",
            cal.threshold, embedder, cal.certification_reason,
        )
    return cal


def _resolve_path(path: str | Path | None) -> Path:
    return Path(path or os.environ.get(ENV_VAR) or DEFAULT_PATH)


def save(cal: Calibration, path: str | Path | None = None) -> Path:
    """Write the calibration JSON; returns the path written."""
    p = _resolve_path(path)
    # The diagnosis travels WITH the threshold. A calibration.json that records "separability
    # 0.75, not certified" explains itself to whoever finds it months later; one carrying only a
    # number cannot be told apart from a working one.
    payload = {"embedder": cal.embedder, "threshold": cal.threshold, "scale": cal.scale}
    if cal.separability is not None:
        payload["separability"] = round(cal.separability, 4)
    if cal.n_answerable is not None:
        payload["n_answerable"] = cal.n_answerable
    if cal.n_unanswerable is not None:
        payload["n_unanswerable"] = cal.n_unanswerable
    if cal.certified is not None:
        payload["certified"] = cal.certified
        payload["certification_reason"] = cal.certification_reason
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    # A malformed or absent diagnosis degrades to None ("not judged"), never to a pass. The
    # counts are read back too: `certified` needs them, so dropping them would silently turn a
    # refusal into "unknown" on every reload.
    sep = data.get("separability")
    try:
        sep = float(sep) if sep is not None else None
        if sep is not None and not (math.isfinite(sep) and 0.0 <= sep <= 1.0):
            sep = None
    except (TypeError, ValueError):
        sep = None

    def _count(key: str) -> int | None:
        v = data.get(key)
        return v if isinstance(v, int) and v >= 0 else None

    cal = Calibration(embedder=embedder, threshold=threshold, scale=scale, separability=sep,
                      n_answerable=_count("n_answerable"),
                      n_unanswerable=_count("n_unanswerable"))
    if cal.certified is False:
        _log.warning(
            "loaded calibration %s is NOT certified — %s", p, cal.certification_reason
        )
    return cal
