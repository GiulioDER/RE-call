"""MCP job and calibration status boundary."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from recall.calibration_v2 import CalibrationRepository
from recall.generations import GenerationManager, NoActiveGeneration

if TYPE_CHECKING:
    from recall.store import PgVectorStore


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
    repository = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp")
    records = repository.list_records()
    manager = GenerationManager(store._dsn, store.tenant, actor="recall-mcp")
    try:
        generation_id = manager.active_generation_id()
    except NoActiveGeneration:
        return {
            "tenant": store.tenant,
            "status": "missing",
            "message": "No active generation exists for this tenant.",
        }
    resolution = manager.calibration_status_for(generation_id)
    matching = [
        item
        for item in records
        if str(item.get("generation_id")) == generation_id
        and item.get("lifecycle_state") == "published"
    ]
    if not matching:
        matching = [item for item in records if str(item.get("generation_id")) == generation_id]
    record = repository.show_record(str(matching[0]["calibration_id"])) if matching else {}
    return {
        "tenant": store.tenant,
        "generation_id": generation_id,
        "status": resolution,
        "message": str(record.get("certification_reason", "")),
        **record,
    }
