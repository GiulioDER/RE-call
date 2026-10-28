"""A near-miss certificate for a PostgreSQL calibration: measured and reported, never a gate.

**Which population the existing certificate covers.** A PostgreSQL calibration certifies that the
top cosine of its labelled answerable queries outranks that of its labelled unanswerable ones
(`recall.calibration`, the lower bound of a 95% Hanley-McNeil interval on the AUC reaching
`MIN_SEPARABILITY`). Those unanswerable queries are off-topic by construction:
`recall.wizard.queryset` draws them from subjects the corpus never touches, because perturbed
in-domain negatives were "not separable at all", and the hand-authored sets serving today follow the
same pattern. So the certified separability is that of the far-gap class. It says nothing about a
question that uses the corpus's own vocabulary and still has no answer in it, which is the
question an agent asks. Where that class has been measured it separates far worse (LongMemEval
per-question haystacks: AUC 0.753, `results/FINDINGS.md`).

**What this adds.** A second, reported certificate per calibration, against a held-out set of
in-domain near-miss questions (plausible, about the corpus, unanswerable from it, reviewed by a
person). The questions are scored by the calibration sampling rule
(`CalibrationRepository.score_query_set`: the exact best cosine over the calibration's own
generation) and compared with the answerable scores stored in the artifact, at the artifact's own
threshold. The report keeps the lite store's shape (`recall.near_miss`, validated by
`recall.support.validated_coverage_report`): the held-out questions are its `in_vocabulary`
class and the same `provisional_in_vocabulary` rule applies; `unknown_entity` is unmeasured
because held-out questions are not built around invented codes. The questions are scored by cosine
alone: the unknown-term check is not applied to this measurement (`unknown_term_gate` reads
`not_run`), so `false_confirm_rate` is the score's share and can only overstate what a gated
search confirms. It adds the AUC, its interval, the false-confirm rate at the threshold and the
same certification bars, so the two certificates read side by side.

**It never gates.** Nothing in `resolve`, carry-forward, drift or the trust policy reads it: this
certificate is expected to fail, and a gate on it would make strict mode refuse every tenant. It is
stored as one append-only audit event per measurement, bound to the calibration id, holding counts,
scores and the set's digest and never a question's text: near-miss questions written over a
private corpus carry that corpus's content. A carried calibration has a new id and so inherits no
measurement. The same set must never be used to tune a score and then to certify it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recall.calibration import (
    MIN_CALIBRATION_SAMPLES,
    MIN_SEPARABILITY,
    separability,
    separability_interval,
)
from recall.support import MIN_PROBES, PROVISIONAL_BAR, validated_coverage_report

#: How the report was measured, beside the lite store's `synthetic_corpus_probes`.
METHOD = "held_out_near_miss_questions"

#: The audit event one measurement appends (`recall_audit_events`). Append only: a new measurement
#: is a new row, and the newest row for a calibration id is the one reported.
EVENT_TYPE = "calibration_near_miss_measured"
#: A stored interval agrees with the recomputed one within float noise of a JSON round trip.
_INTERVAL_TOLERANCE = 1e-9

#: The sentence status prints beside every PostgreSQL certificate.
CERTIFICATE_POPULATION = (
    "certified against the labelled set's unanswerable queries, which are off-topic by "
    "construction (far gap); in-domain near-miss questions are reported separately in "
    "near_miss_certificate and never gate"
)


def certificate_report(
    answerable: Sequence[float], near_miss: Sequence[float], threshold: float
) -> dict[str, Any]:
    """The near-miss certificate for one calibration, from scores alone.

    A near-miss question is confirmed when its score is at or above `threshold`, because serving
    marks a hit `low_confidence` only when its cosine is strictly below it
    (`recall.trust_gate`), so a question scoring exactly the threshold gets an `ok` hit.
    """
    scores = [float(score) for score in near_miss]
    n = len(scores)
    rejected = sum(score < threshold for score in scores)
    auc = separability([float(score) for score in answerable], scores)
    interval, certified, reason = _judgement(auc, len(answerable), n)
    return {
        "method": METHOD,
        "unknown_entity": {"n": 0, "rejected": 0, "rate": None, "state": "unmeasured"},
        "in_vocabulary": {
            "n": n,
            "rejected": rejected,
            "rate": round(rejected / n, 3) if n else None,
            "state": "measured" if n >= MIN_PROBES else "unmeasured",
        },
        # The lite rule, unchanged: unmeasured, or fewer than PROVISIONAL_BAR rejected.
        "provisional_in_vocabulary": n < MIN_PROBES or rejected / n < PROVISIONAL_BAR,
        "unknown_term_gate": "not_run",
        "threshold": float(threshold),
        "n_answerable": len(answerable),
        "separability": auc,
        "separability_ci": interval,
        "false_confirm_rate": round((n - rejected) / n, 4) if n else None,
        "certified": certified,
        "certification_reason": reason,
        "near_miss_scores": scores,
    }


def validated_certificate(report: Mapping[str, Any], calibration_id: str) -> dict[str, Any]:
    """The stored report, checked against its own scores; raises ValueError when it disagrees."""
    clean = validated_coverage_report(dict(report), calibration_id)
    if report.get("method") != METHOD:
        raise ValueError("not a near-miss certificate")
    scores = report.get("near_miss_scores")
    raw_threshold = report.get("threshold")
    n_answerable = report.get("n_answerable")
    if (not isinstance(scores, list) or not all(_number(s) for s in scores)
            or not isinstance(raw_threshold, (int, float)) or isinstance(raw_threshold, bool)
            or type(n_answerable) is not int):
        raise ValueError("invalid near-miss certificate fields")
    threshold = float(raw_threshold)
    in_vocab = clean["in_vocabulary"]
    if in_vocab["n"] != len(scores) or in_vocab["rejected"] != sum(s < threshold for s in scores):
        raise ValueError("near-miss counts disagree with the stored scores")
    auc, interval, rate = (report.get(k) for k in ("separability", "separability_ci",
                                                   "false_confirm_rate"))
    n, rejected = in_vocab["n"], in_vocab["rejected"]
    if (not (auc is None or (_number(auc) and 0.0 <= auc <= 1.0))
            or not (interval is None or (isinstance(interval, list) and len(interval) == 2
                                         and all(_number(v) for v in interval)))
            or rate != (round((n - rejected) / n, 4) if n else None)
            or report.get("certified") not in (True, False, None)):
        raise ValueError("near-miss certificate statistics are malformed")
    expected_interval, expected_certified, _reason = _judgement(auc, n_answerable, n)
    if (report.get("certified") is not expected_certified
            or (interval is None) != (expected_interval is None)
            or (interval is not None and expected_interval is not None
                and any(abs(a - b) > _INTERVAL_TOLERANCE
                        for a, b in zip(interval, expected_interval, strict=True)))):
        raise ValueError("near-miss certificate verdict disagrees with its statistics")
    for key in ("separability", "separability_ci", "false_confirm_rate", "certified",
                "certification_reason", "generation_id", "query_set_digest"):
        clean[key] = report.get(key)
    clean.update(method=METHOD, threshold=threshold, n_answerable=n_answerable,
                 near_miss_scores=[float(s) for s in scores])
    return clean


def _judgement(
    auc: float | None, n_answerable: int, n: int
) -> tuple[list[float] | None, bool | None, str]:
    """The interval, verdict and reason for an AUC: the producer's rule, and the validator's."""
    if auc is None:
        return None, None, "nothing to judge: a class is empty"
    low, high = separability_interval(auc, n_answerable, n)
    if min(n_answerable, n) < MIN_CALIBRATION_SAMPLES:
        return [low, high], False, (
            f"too few samples: {n_answerable} answerable and {n} near-miss, "
            f"{MIN_CALIBRATION_SAMPLES} each required"
        )
    certified = low >= MIN_SEPARABILITY
    # Four places: a bound just under the bar must not print as the bar itself ("0.900 is below").
    return [low, high], certified, (
        f"separability lower bound {low:.4f} {'reaches' if certified else 'is below'} "
        f"{MIN_SEPARABILITY}"
    )


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def describe(report: Mapping[str, Any] | None) -> str:
    """One line for a status message or the CLI."""
    if report is None:
        return ("near-miss certificate: unmeasured (no held-out near-miss set has been scored "
                "against this calibration); provisional for in-vocabulary near-misses")
    in_vocab = report["in_vocabulary"]
    n, confirmed = in_vocab["n"], in_vocab["n"] - in_vocab["rejected"]
    ci = report.get("separability_ci") or [None, None]
    auc = report.get("separability")
    verdict = {True: "certified", False: "not certified", None: "not judged"}[report.get("certified")]
    line = (
        f"near-miss certificate: {confirmed} of {n} held-out near-miss question(s) clear threshold "
        f"{report['threshold']:.3f}"
    )
    if auc is not None and ci[0] is not None:
        line += (
            f"; separability {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] against "
            f"{report['n_answerable']} answerable, {verdict}"
        )
    if report.get("provisional_in_vocabulary"):
        line += "; provisional for in-vocabulary near-misses"
    return line + " (reported only, never a gate)"
