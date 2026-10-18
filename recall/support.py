"""Describe retrieval eligibility separately from semantic answerability.

These diagnostics never admit evidence or change a retrieval verdict. Coverage is read inside
trusted_search's snapshot, using only the calibration id that search actually selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from recall.observability import get_logger

_log = get_logger("support")

ELIGIBILITY_NOTE = (
    " Citable passages passed retrieval trust checks; this does not establish that they "
    "answer the question. query_support is not_checked: establish that each cited passage "
    "supports the claim before answering. Retrieval confidence is not answer correctness."
)


@dataclass(frozen=True)
class QuerySupport:
    """No semantic verifier runs in this release, including on empty or certified results."""

    status: Literal["not_checked"] = "not_checked"
    scope: Literal["returned_evidence"] = "returned_evidence"
    method: None = None


@dataclass(frozen=True)
class CalibrationCoverage:
    """Optional synthetic diagnostics, never a certification of this query's answerability.

    Missing or differently bound measurements are unmeasured. An optional reader failure is
    unavailable, distinct from absence. A carried calibration does not inherit its parent's
    near-miss measurements. The report's per-class states remain authoritative.
    """

    calibration_id: str | None = None
    status: Literal["measured", "unmeasured", "unavailable"] = "unmeasured"
    method: Literal["synthetic_corpus_probes"] | None = None
    report: dict[str, Any] | None = None


def calibration_coverage(store: object, calibration_id: str | None) -> CalibrationCoverage:
    """Read only the served fit's report; optional faults cannot change serving behavior."""
    empty = CalibrationCoverage(calibration_id=calibration_id)
    if calibration_id is None:
        return empty
    try:
        reader = getattr(store, "near_miss_coverage", None)
        if not callable(reader):
            return empty
        report = reader(calibration_id)
        if report is None:
            return empty
        if not isinstance(report, dict) or report.get("calibration_id") != calibration_id:
            return empty
        # Only the documented, content-free diagnostic shape crosses this boundary.
        # A damaged optional report must not break CLI formatting or pretend it measured a class.
        clean: dict[str, Any] = {"calibration_id": calibration_id}
        for name in ("unknown_entity", "in_vocabulary"):
            entry = report.get(name)
            if not isinstance(entry, dict):
                raise ValueError("invalid coverage class")
            n, rejected, rate, state = (entry.get(k) for k in ("n", "rejected", "rate", "state"))
            if (type(n) is not int or type(rejected) is not int or not 0 <= rejected <= n
                    or state not in {"measured", "unmeasured"}):
                raise ValueError("invalid coverage counts")
            if (n == 0 and rate is not None) or (n > 0 and (
                    not isinstance(rate, (float, int)) or isinstance(rate, bool) or not 0 <= rate <= 1)):
                raise ValueError("invalid coverage rate")
            clean[name] = {"n": n, "rejected": rejected, "rate": rate, "state": state}
        if not isinstance(report.get("provisional_in_vocabulary"), bool):
            raise ValueError("invalid coverage warning")
        clean["provisional_in_vocabulary"] = report["provisional_in_vocabulary"]
        gate = report.get("unknown_term_gate")
        if gate not in {"checked", "unavailable", "disabled", "not_run"}:
            raise ValueError("invalid coverage gate")
        clean["unknown_term_gate"] = gate
        measured = all(
            isinstance(report.get(name), dict) and report[name].get("state") == "measured"
            for name in ("unknown_entity", "in_vocabulary")
        )
        return CalibrationCoverage(
            calibration_id=calibration_id,
            status="measured" if measured else "unmeasured",
            method="synthetic_corpus_probes",
            report=clean,
        )
    except Exception as exc:  # BROAD-CATCH: fail-open, optional diagnostics never gate retrieval
        _log.warning("Calibration coverage unavailable (%s)", type(exc).__name__)
        return CalibrationCoverage(calibration_id=calibration_id, status="unavailable")
