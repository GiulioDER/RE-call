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

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles
from typing import TYPE_CHECKING

from recall.observability import get_logger

if TYPE_CHECKING:  # `recall.embeddings` imports this module, so the runtime import would cycle
    from recall.embeddings import EmbeddingProfile

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

#: z for the two-sided 95% interval on separability. The bar above is applied to the interval's
#: LOWER bound, so this is the confidence with which a certification claims to clear it.
SEPARABILITY_Z = 1.96


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


def separability_interval(
    auc: float, n_answerable: int, n_unanswerable: int, z: float = SEPARABILITY_Z
) -> tuple[float, float]:
    """Two-sided confidence interval on `separability`, by the Hanley & McNeil (1982) estimator.

    A point AUC is not a measurement without its width, and the width here is dominated by the
    SMALLER class. Abstention sets are lopsided by nature — unanswerable queries are the expensive
    ones to label — so the honest interval is wide exactly where the honest answer matters.

    Two reasons this estimator and not a bootstrap. It needs only ``(auc, n, n)``, so a
    calibration **loaded from disk** — which persists the AUC and the counts, never the raw
    samples — can still be judged; a bootstrap could only ever judge a calibration built in the
    same process, and a rule that silently stops applying after a round-trip is the class of
    silent failure this module exists to remove. And it is deterministic: the same artifact
    certifies the same way on every host, where a resampled bound would jitter across the bar.

    The estimator assumes exponential score distributions. It is standard, published and
    reproducible from the numbers in a calibration file, which is what a claim needs to be
    checkable; where it errs it is mildly conservative, and conservative is the safe direction
    for a gate that decides whether to trust an abstention.

    ⚠️ Do not read the crude ``sqrt(A(1-A)/n_min)`` shortcut as this quantity. It ignores the
    larger class entirely: on the LongMemEval samples (AUC 0.753 over 470/30) it reports SE 0.079
    against this estimator's **0.037** — wide enough to leave 0.90 inside the interval and make an
    unusable threshold look merely unproven. Measured in FINDINGS §10b.
    """
    if n_answerable < 1 or n_unanswerable < 1:
        return (0.0, 1.0)
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    variance = (
        auc * (1 - auc)
        + (n_answerable - 1) * (q1 - auc * auc)
        + (n_unanswerable - 1) * (q2 - auc * auc)
    ) / (n_answerable * n_unanswerable)
    # Clamped: rounding can drive the variance fractionally negative at AUC 1.0, where the true
    # standard error is exactly 0 and the interval is the point itself.
    half = z * math.sqrt(max(variance, 0.0))
    return (max(0.0, auc - half), min(1.0, auc + half))


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
    def separability_ci(self) -> tuple[float, float] | None:
        """95% interval on `separability`, or None when there was nothing to judge."""
        if self.separability is None:
            return None
        if self.n_answerable is None or self.n_unanswerable is None:
            return None
        return separability_interval(self.separability, self.n_answerable, self.n_unanswerable)

    @property
    def certified(self) -> bool | None:
        """Is this threshold supportable by the data it was fitted on?

        Tri-state on purpose. ``True`` passed both checks; ``False`` failed one; ``None`` means
        there was nothing to judge (one-class samples, or an artifact written before this check
        existed). ``None`` must never be read as ``True``.

        The separability bar is applied to the **lower bound** of the interval, not to the point
        estimate. A point AUC above 0.90 on a thin calibration set is a sample that came out
        lucky as often as it is a corpus that separates: at 20 samples per class — the minimum
        this module accepts — a measured 0.95 carries a lower bound of 0.879, so the data has
        not established the bar it appears to clear. Certifying on the point estimate would let
        exactly the failure this check exists to catch back in through small-sample noise, which
        is the same defect as the in-sample fit FINDINGS §2b retracted a number for, wearing a
        different hat. The direction matters more than the width: an over-certified threshold
        refuses real answers silently, while an under-certified one prints how many more samples
        would settle it.

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
        ci = self.separability_ci
        assert ci is not None  # guarded by the None checks above
        return ci[0] >= MIN_SEPARABILITY

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
        lo, hi = separability_interval(
            self.separability, self.n_answerable, self.n_unanswerable
        )
        if self.separability < MIN_SEPARABILITY:
            return (
                f"separability {self.separability:.3f} [{lo:.3f}, {hi:.3f}] < "
                f"{MIN_SEPARABILITY}: answerable and unanswerable scores overlap, so NO threshold "
                "separates them — this one will refuse real answers, reject unanswerable queries, "
                "or both, and moving it only trades one error for the other"
            )
        if lo < MIN_SEPARABILITY:
            # Distinguished from the case above because the remedy is completely different: this
            # corpus may well separate, and the way to find out is more labels, not a new signal.
            return (
                f"separability {self.separability:.3f} clears {MIN_SEPARABILITY} but its 95% "
                f"interval [{lo:.3f}, {hi:.3f}] does not: {self.n_answerable}/"
                f"{self.n_unanswerable} samples cannot establish the bar this estimate appears to "
                "reach. Label more queries — the interval narrows with the smaller class"
            )
        return (
            f"separability {self.separability:.3f} [{lo:.3f}, {hi:.3f}] over "
            f"{self.n_answerable}/{self.n_unanswerable} samples"
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


#: JSON key holding `EmbeddingProfile.fingerprint()` — the SHA256 over the COMPLETE immutable
#: embedding identity, not just its profile id. See `load_for_profile` for why the id alone is
#: not enough to decide that a threshold may be applied.
PROFILE_FINGERPRINT_KEY = "profile_fingerprint"


def save(cal: Calibration, path: str | Path | None = None) -> Path:
    """Write the calibration JSON; returns the path written."""
    p = _resolve_path(path)
    # The diagnosis travels WITH the threshold. A calibration.json that records "separability
    # 0.75, not certified" explains itself to whoever finds it months later; one carrying only a
    # number cannot be told apart from a working one.
    payload = {"embedder": cal.embedder, "threshold": cal.threshold, "scale": cal.scale}
    if cal.separability is not None:
        payload["separability"] = round(cal.separability, 4)
    # Recomputable from (separability, n, n) rather than stored state, but written out because the
    # point estimate is what a reader anchors on and the interval is what the verdict was taken
    # from. A file that records only 0.95 next to `certified: false` reads as a bug.
    if cal.separability_ci is not None:
        payload["separability_ci"] = [round(v, 4) for v in cal.separability_ci]
    if cal.n_answerable is not None:
        payload["n_answerable"] = cal.n_answerable
    if cal.n_unanswerable is not None:
        payload["n_unanswerable"] = cal.n_unanswerable
    if cal.certified is not None:
        payload["certified"] = cal.certified
        payload["certification_reason"] = cal.certification_reason
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


#: (path, embedder) -> (content digest, Calibration|None). See `load_for` for why this exists,
#: and why the digest is of the CONTENT rather than of the file's metadata.
_LOAD_CACHE: dict[tuple[str, str], tuple[bytes, "Calibration | None"]] = {}

#: Cache key for "present but unreadable", which has no content to digest. Cannot be mistaken for
#: a real entry: a blake2b digest below is always exactly 16 bytes, and this is not.
_UNREADABLE = b"<unreadable>"


def load_for(embedder: str, path: str | Path | None = None) -> Calibration | None:
    """Load the calibration for `embedder`, or None when it cannot be applied safely.

    Returns None (uncalibrated fallback, flagged in every result) when the file is absent,
    unreadable, malformed, calibrated for a DIFFERENT embedder, or carries out-of-range values —
    a threshold calibrated in another model's cosine regime must never be applied, and a
    corrupt file must never be able to disable abstention silently (NaN threshold) or crash
    every search (zero/negative scale).
    """
    p = _resolve_path(path)
    key = (str(p), embedder)
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return None  # never calibrated: the ordinary case, not worth a warning
    except OSError:
        # Warned once per failure rather than once per call: this runs on EVERY query, so a file
        # that is present but unreadable would otherwise log at query rate.
        if _LOAD_CACHE.get(key, (b"", None))[0] != _UNREADABLE:
            _log.warning("ignoring unreadable calibration file %s (uncalibrated fallback)", p)
        _LOAD_CACHE[key] = (_UNREADABLE, None)
        return None
    # Cached by (path, embedder) and invalidated on a digest of the file's CONTENT.
    # `trusted_search` calls this on EVERY query, so an uncached load would put a JSON parse and
    # full validation in the hot retrieval path of a library whose measured advantage is latency
    # (median 77 ms against a competitor's 104 ms). The digest skips that while still picking up
    # a re-calibration written by a concurrent `recall calibrate`, which a load-once cache would
    # miss until restart.
    #
    # This used to be invalidated on `st_mtime_ns` and that was WRONG, not merely imprecise: an
    # mtime is a timestamp, not a version. Effective timestamp granularity is ~1 ms — the smallest
    # non-zero delta between consecutive writes measured on ext4 is 1,000,001 ns — so two writes
    # inside one tick carry the SAME mtime and the second was never seen. The stale threshold was
    # then served indefinitely, because nothing re-checks until the mtime changes again, and a
    # threshold gates abstention: retrieval behaviour changed with nothing logged anywhere.
    #
    # Rare it was not. Rewrite-then-reload, which is exactly what `recall calibrate` does, served
    # the STALE threshold on 223 of 300 trials (74 %) on ext4 and 0 of 300 with the digest. The
    # bug reached CI as a flake instead of a red build only because a same-tick collision needs
    # the two writes close together, which depended on which test file warmed the imports first.
    #
    # Neither size nor mtime+size fixes it: a threshold edit (0.42 -> 0.31) is byte-for-byte the
    # same length. Only the content distinguishes the two files.
    #
    # Cost of the correctness, measured on VPS2 ext4 against a 308-byte calibration written by
    # `recall calibrate`: the cached call goes from 13.8 us (stat) to 38.4 us (read + digest),
    # against 62 us for the uncached parse. So the cache still earns its place — it saves ~24 us
    # per query rather than ~48 — and the 24 us it gives up is 0.03 % of a 77 ms query.
    #
    # Races are benign: two threads may both load and both store, and the value is identical
    # because it derives from the same bytes. No lock, therefore no lock in the hot path either.
    digest = hashlib.blake2b(raw, digest_size=16).digest()
    hit = _LOAD_CACHE.get(key)
    if hit is not None and hit[0] == digest:
        return hit[1]
    try:
        data = json.loads(raw.decode("utf-8"))
        if data.get("embedder") != embedder:
            _LOAD_CACHE[key] = (digest, None)
            return None
        threshold = float(data["threshold"])
        scale = float(data["scale"])
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError):
        _log.warning("ignoring unreadable calibration file %s (uncalibrated fallback)", p)
        _LOAD_CACHE[key] = (digest, None)
        return None
    if not (math.isfinite(threshold) and -1.0 <= threshold <= 1.0
            and math.isfinite(scale) and scale > 0.0):
        _log.warning(
            "ignoring out-of-range calibration in %s (threshold=%r, scale=%r) — "
            "uncalibrated fallback", p, threshold, scale,
        )
        # Recorded like every other rejection above. Without this the one branch that already
        # KNOWS the file is broken was the one that re-read and re-parsed it on every query, and
        # repeated its warning each time.
        _LOAD_CACHE[key] = (digest, None)
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
    _LOAD_CACHE[key] = (digest, cal)
    return cal


def save_for_profile(
    cal: Calibration, profile: "EmbeddingProfile", path: str | Path | None = None
) -> Path:
    """`save`, plus the complete embedding identity the threshold was fitted under.

    Refuses a calibration whose `embedder` is not this profile's id, because the file would then
    carry two disagreeing claims about what it applies to and `load_for` and `load_for_profile`
    would resolve it differently.
    """
    if cal.embedder != profile.profile_id:
        raise ValueError(
            f"calibration is for embedder {cal.embedder!r} but the profile is "
            f"{profile.profile_id!r}; the file would claim two different owners"
        )
    written = save(cal, path)
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload[PROFILE_FINGERPRINT_KEY] = profile.fingerprint()
    written.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return written


def load_for_profile(
    profile: "EmbeddingProfile", path: str | Path | None = None
) -> Calibration | None:
    """Load a calibration only when it was fitted under this EXACT embedding identity.

    An EXTENSION of `load_for`, never a relaxation: every check `load_for` makes still runs, and
    this adds one more on top. `load_for` filters on the profile ID string, which is the coarsest
    part of an embedding's identity. Two runs can share `bge-small-symmetric-v1` and differ in
    artifact digest, dimension, encoder modes, normalization, instruction version, chunker version
    or inference-library version — every one of which moves the cosine regime the threshold was
    fitted in, and every one of which is already key material in
    `EmbeddingProfile.fingerprint()`.

    **Absent fingerprint fails CLOSED.** A file predating this key cannot demonstrate which
    identity it belongs to, and serving it because it does not contradict us is precisely
    cross-profile reuse with the evidence missing rather than absent. The caller gets None, which
    every consumer already handles as "uncalibrated", so the failure is a degraded mode and not a
    crash. `load_for` is untouched, so a legacy caller keeps the behaviour it had.

    Reads the file a second time rather than threading the raw payload out of `load_for`'s cached
    path: this runs once per benchmark arm, not once per query, and the alternative was widening
    `load_for`'s return type on the hot retrieval path to serve a caller that is not on it.
    """
    cal = load_for(profile.profile_id, path)
    if cal is None:
        return None
    resolved = _resolve_path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        stored = payload.get(PROFILE_FINGERPRINT_KEY)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(stored, str) or not stored:
        _log.warning(
            "calibration %s records no %s, so it cannot show which embedding identity it was "
            "fitted under — refusing it rather than applying it to %s (uncalibrated fallback)",
            resolved, PROFILE_FINGERPRINT_KEY, profile.profile_id,
        )
        return None
    expected = profile.fingerprint()
    if stored != expected:
        _log.warning(
            "calibration %s was fitted under embedding identity %s, not %s — refusing it "
            "(uncalibrated fallback). The profile ids match; something else in the identity "
            "does not.",
            resolved, stored, expected,
        )
        return None
    return cal
