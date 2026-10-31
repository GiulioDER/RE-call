"""Tenant-scoped MCP status operations."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from recall.calibration_v2 import CalibrationRepository
from recall.generations import GenerationManager, NoActiveGeneration
from recall.observability import get_logger
from recall.store import PgVectorStore

_log = get_logger("mcp.status")


class JobLedger:
    """Tenant scoped, bounded record of ingest jobs."""

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        ttl_seconds: float = 86400.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, float, dict[str, object]]] = {}

    def put(self, job_id: str, tenant: str, payload: dict[str, object]) -> None:
        now = self._clock()
        with self._lock:
            expired = [
                key
                for key, (_, stamp, _payload) in self._entries.items()
                if now - stamp > self._ttl_seconds
            ]
            for key in expired:
                del self._entries[key]
            while len(self._entries) >= self._max_entries:
                del self._entries[next(iter(self._entries))]
            self._entries[job_id] = (tenant, now, payload)

    def get(self, job_id: str, tenant: str) -> dict[str, object] | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return None
            owner, stamp, payload = entry
            if now - stamp > self._ttl_seconds or owner != tenant:
                if now - stamp > self._ttl_seconds:
                    del self._entries[job_id]
                return None
            return payload


def job_status(
    store: PgVectorStore, job_id: str, jobs: JobLedger | dict[str, object]
) -> dict[str, object]:
    """Return one job record after the caller has been authorized for its tenant."""
    if isinstance(jobs, JobLedger):
        value = jobs.get(job_id, str(store.tenant))
    else:
        candidate = jobs.get(job_id)
        value = (
            candidate
            if isinstance(candidate, dict) and candidate.get("tenant") in (None, str(store.tenant))
            else None
        )
    return value if isinstance(value, dict) else {"job_id": job_id, "state": "unknown"}


def calibration_status(store: PgVectorStore) -> dict[str, object]:
    """Return calibration bound to the generation the tenant currently serves."""
    from recall.lite import LiteStore

    if isinstance(store, LiteStore):
        from recall.lite.calibration import status_report

        return status_report(store)
    repository = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp")
    manager = GenerationManager(store._dsn, store.tenant, actor="recall-mcp")
    try:
        generation_id = manager.active_generation_id()
    except NoActiveGeneration:
        return {
            "tenant": store.tenant,
            "status": "missing",
            "message": "No active generation exists for this tenant.",
        }
    # Only the active generation's rows: the tenant's full history grows with every fit.
    records = repository.list_records(generation_id)
    resolution = manager.calibration_status_for(generation_id)
    matching = [item for item in records if item.get("lifecycle_state") == "published"] or records
    record = repository.show_record(str(matching[0]["calibration_id"])) if matching else {}
    message = str(record.get("certification_reason", ""))
    coverage: dict[str, object] = {}
    if record:
        from recall.near_miss import coverage_summary

        # The same report, key and message as lite status (`recall.lite.calibration.status_report`);
        # a calibration measured at no fit (a carried one) reads as unavailable and provisional.
        report, summary = coverage_summary(repository.near_miss_coverage, str(record["calibration_id"]))
        coverage = {"near_miss_coverage": report}
        message += "; " + summary
    return {
        "tenant": store.tenant,
        "generation_id": generation_id,
        "status": resolution,
        "message": message,
        **record,
        **coverage,
        **(_near_miss_status(repository, str(record["calibration_id"])) if record else {}),
    }


def _near_miss_status(
    repository: CalibrationRepository, calibration_id: str
) -> dict[str, object]:
    """The calibration's near-miss certificate, reported beside it. Read only, fail-open.

    `resolution` above is computed before this runs and nothing here can change it: the
    certificate is a report about the threshold, never a gate on it (`recall.near_miss_certificate`).
    """
    from recall.near_miss_certificate import CERTIFICATE_POPULATION, describe

    try:
        report = repository.near_miss_report(calibration_id)
        summary = describe(report)
    except Exception as exc:  # BROAD-CATCH: fail-open, an optional report never fails status
        _log.warning("Near-miss certificate unavailable (%s)", type(exc).__name__)
        report = None
        summary = "near-miss certificate unavailable; provisional for in-vocabulary near-misses"
    return {
        "certificate_population": CERTIFICATE_POPULATION,
        "near_miss_certificate": report,
        "near_miss_certificate_summary": summary,
    }
