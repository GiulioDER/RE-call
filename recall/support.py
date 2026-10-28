"""Describe retrieval eligibility separately from semantic answerability.

These diagnostics never admit evidence or change a retrieval verdict. Coverage is read inside
trusted_search's snapshot, using only the calibration id that search actually selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal, Protocol, get_args

from recall.observability import get_logger

_log = get_logger("support")

# Shared by the probe producer and its serving/decoder boundary. Keep this contract data-only.
PROVISIONAL_BAR = 0.90
MIN_PROBES = 10
CLASSES = ("unknown_entity", "in_vocabulary")
#: The vocabularies of the support and coverage diagnostics, read by the dataclasses below, the
#: report validator and the reasoning response decoder, so a new state cannot reach one of them only.
SupportStatus = Literal["not_checked"]
SupportScope = Literal["returned_evidence"]
CoverageStatus = Literal["measured", "unmeasured", "unavailable"]
CoverageMethod = Literal["synthetic_corpus_probes"]
UnknownTermGate = Literal["checked", "unavailable", "disabled", "not_run"]
SYNTHETIC_PROBES: CoverageMethod = "synthetic_corpus_probes"
#: The audit event a PostgreSQL calibration's coverage report is stored as (lite keeps it in meta).
COVERAGE_EVENT = "calibration_near_miss_coverage"


#: The newest audit event of one type bound to one calibration id. Both near-miss reports are read
#: this way (coverage at each fit, the held-out certificate), so a change to the ordering or the
#: filter cannot reach one reader and miss the other.
LATEST_CALIBRATION_EVENT_SQL = (
    "SELECT payload FROM recall_audit_events WHERE tenant_id = %s "
    "AND event_type = %s AND payload->>'calibration_id' = %s "
    "ORDER BY created_at DESC, event_id DESC LIMIT 1"
)


class _Executes(Protocol):
    """The one connection method the reader uses: a psycopg connection, typed without importing it."""

    def execute(self, query: Any, params: Any = ..., /) -> Any: ...


def latest_calibration_event(
    conn: _Executes, tenant: str, event_type: str, calibration_id: str
) -> dict[str, Any] | None:
    """The newest `event_type` payload recorded for `calibration_id`, or None. Unvalidated."""
    row = conn.execute(LATEST_CALIBRATION_EVENT_SQL, (tenant, event_type, calibration_id)).fetchone()
    return dict(row[0]) if row is not None and isinstance(row[0], Mapping) else None

ELIGIBILITY_NOTE = (
    " Citable passages passed retrieval trust checks; this does not establish that they "
    "answer the question. query_support is not_checked: establish that each cited passage "
    "supports the claim before answering. Retrieval confidence is not answer correctness."
)


@dataclass(frozen=True)
class QuerySupport:
    """No semantic verifier runs in this release, including on empty or certified results."""

    status: SupportStatus = "not_checked"
    scope: SupportScope = "returned_evidence"
    method: None = None


@dataclass(frozen=True)
class CalibrationCoverage:
    """Optional synthetic diagnostics, never a certification of this query's answerability.

    Missing or differently bound measurements are unmeasured. An optional reader failure is
    unavailable, distinct from absence. A carried calibration does not inherit its parent's
    near-miss measurements. The report's per-class states remain authoritative.
    """

    calibration_id: str | None = None
    status: CoverageStatus = "unmeasured"
    method: CoverageMethod | None = None
    report: dict[str, Any] | None = None


def merge_calibration_coverage(
    initial: CalibrationCoverage, *expanded: CalibrationCoverage,
) -> CalibrationCoverage:
    """Carry identical diagnostics; mixed fits or measurements have no aggregate report."""
    if all(coverage == initial for coverage in expanded):
        return initial
    calibration_id = initial.calibration_id if all(
        coverage.calibration_id == initial.calibration_id for coverage in expanded
    ) else None
    return CalibrationCoverage(calibration_id=calibration_id)


def validated_coverage_report(report: dict[str, Any], calibration_id: str) -> dict[str, Any]:
    """Copy content-free diagnostics only when they agree with the producer's count rules."""
    if report.get("calibration_id") != calibration_id:
        raise ValueError("coverage calibration mismatch")
    clean: dict[str, Any] = {"calibration_id": calibration_id}
    for name in CLASSES:
        entry = report.get(name)
        if not isinstance(entry, dict):
            raise ValueError("invalid coverage class")
        n, rejected, rate, state = (entry.get(k) for k in ("n", "rejected", "rate", "state"))
        if type(n) is not int or type(rejected) is not int or not 0 <= rejected <= n:
            raise ValueError("invalid coverage counts")
        if state != ("measured" if n >= MIN_PROBES else "unmeasured"):
            raise ValueError("inconsistent coverage state")
        expected_rate = round(rejected / n, 3) if n else None
        if (n == 0 and rate is not None) or (n > 0 and (
                not isinstance(rate, (float, int)) or isinstance(rate, bool)
                or rate != expected_rate)):
            raise ValueError("inconsistent coverage rate")
        clean[name] = {"n": n, "rejected": rejected, "rate": rate, "state": state}
    in_vocab = clean["in_vocabulary"]
    provisional = in_vocab["state"] == "unmeasured" or (
        in_vocab["n"] > 0 and in_vocab["rejected"] / in_vocab["n"] < PROVISIONAL_BAR
    )
    if (not isinstance(report.get("provisional_in_vocabulary"), bool)
            or report["provisional_in_vocabulary"] != provisional):
        raise ValueError("inconsistent coverage warning")
    clean["provisional_in_vocabulary"] = provisional
    gate = report.get("unknown_term_gate")
    if gate not in get_args(UnknownTermGate):
        raise ValueError("invalid coverage gate")
    clean["unknown_term_gate"] = gate
    return clean


def coverage_status(clean: Mapping[str, Any]) -> Literal["measured", "unmeasured"]:
    """`measured` only when every probe class of a validated report is measured."""
    return "measured" if all(clean[name]["state"] == "measured" for name in CLASSES) else "unmeasured"


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
        clean = validated_coverage_report(report, calibration_id)
        return CalibrationCoverage(
            calibration_id=calibration_id,
            status=coverage_status(clean),
            method=SYNTHETIC_PROBES,
            report=clean,
        )
    except Exception as exc:  # BROAD-CATCH: fail-open, optional diagnostics never gate retrieval
        _log.warning("Calibration coverage unavailable (%s)", type(exc).__name__)
        return CalibrationCoverage(calibration_id=calibration_id, status="unavailable")
