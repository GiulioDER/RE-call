"""MCP tenant lifecycle and bounded administrative operations."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from recall.current_state import MAX_CURRENT_STATE_RECORDS, project_current_state
from recall.guards import staleness
from recall.observability import METRICS, get_logger
from recall.uploads import delete_staged_sources
from recall_mcp.models import (
    CurrentStateRecordModel,
    CurrentStateResult,
    ForgetResult,
    InventoryEntry,
    InventoryResult,
    MemoryStatsResult,
)
from recall.security_policy import AccessContext, SourceSecurityPolicy

if TYPE_CHECKING:
    from recall.control_plane import ControlPlane
    from recall.current_state import CurrentStateProjection
    from recall.store import PgVectorStore

_log = get_logger("mcp.service")
MAX_FORGET_SOURCES = 1000


def _validate_security_context(
    store: PgVectorStore,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
) -> None:
    if security_policy is None:
        return
    if access_context is None:
        raise ValueError("access_context is required when security_policy is configured")
    store_tenant = getattr(store, "tenant", None)
    if isinstance(store_tenant, str) and store_tenant != access_context.tenant:
        raise PermissionError("access context tenant does not match the serving store")


def current_state_memory(
    store: PgVectorStore,
    *,
    as_of: datetime | None = None,
    source: str | None = None,
    max_records: int = MAX_CURRENT_STATE_RECORDS,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> CurrentStateResult:
    """Return a bounded, deterministic authored current state projection."""
    _validate_security_context(store, security_policy, access_context)
    projection: CurrentStateProjection = project_current_state(
        store, as_of=as_of, source=source, max_records=max_records
    )
    if security_policy is not None:
        assert access_context is not None
        projection = replace(
            projection,
            records=tuple(
                replace(
                    record,
                    successor_chain=tuple(
                        successor
                        for successor in record.successor_chain
                        if security_policy.decide(successor, access_context).allowed
                    ),
                )
                for record in projection.records
                if security_policy.decide(record.source, access_context).allowed
            ),
        )
    return CurrentStateResult(
        schema_version=projection.schema_version,
        projection_id=projection.projection_id,
        tenant_id=projection.tenant_id,
        generation_id=projection.generation_id,
        pipeline_fingerprint=projection.pipeline_fingerprint,
        corpus_fingerprint=projection.corpus_fingerprint,
        as_of=projection.as_of.isoformat(),
        records=[
            CurrentStateRecordModel(
                state_id=record.state_id,
                source=record.source,
                state=record.state,
                chunk_ids=list(record.chunk_ids),
                successor_chain=list(record.successor_chain),
                valid_from=record.valid_from.isoformat() if record.valid_from else None,
                valid_until=record.valid_until.isoformat() if record.valid_until else None,
                diagnostics=list(record.diagnostics),
            )
            for record in projection.records
        ],
    )


def forget_memory(
    store: PgVectorStore,
    sources: list[str],
    shadow_store: PgVectorStore | None = None,
    control_plane: ControlPlane | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
) -> ForgetResult:
    """Permanently delete every indexed chunk for the given sources."""
    if not sources:
        raise ValueError("sources must be a non-empty list")
    if len(sources) > MAX_FORGET_SOURCES:
        raise ValueError(
            f"{len(sources)} sources requested, over the {MAX_FORGET_SOURCES} limit for one "
            f"call. Deletion is irreversible; split the request so each one stays reviewable."
        )
    requested = list(dict.fromkeys(sources))
    if security_policy is not None:
        if security_context is None:
            raise ValueError("security_context is required when security_policy is configured")
        if getattr(store, "tenant", None) != security_context.tenant:
            raise PermissionError("source security context tenant does not match the store")
        for source in requested:
            if not security_policy.decide(source, security_context).allowed:
                raise PermissionError(f"source {source!r} is not authorized for erasure")
    resolved = store.sources_for_identifiers(requested)
    if shadow_store is not None:
        shadow_resolved = shadow_store.sources_for_identifiers(requested)
        for identifier, values in shadow_resolved.items():
            bucket = resolved.setdefault(identifier, [])
            bucket.extend(value for value in values if value not in bucket)
    found = [source for source in requested if source in resolved]
    not_found = [source for source in requested if source not in resolved]
    to_delete = sorted({source for identifier in found for source in resolved[identifier]})
    if to_delete and shadow_store is not None:
        chunks_removed = store.delete_sources_across([store.table, shadow_store.table], to_delete)
    else:
        chunks_removed = store.delete_sources(to_delete) if to_delete else 0
    outbox_events_scrubbed = 0
    if control_plane is not None:
        try:
            outbox_events_scrubbed = control_plane.erase_sources_from_pending(
                store.tenant, sorted({*requested, *to_delete})
            )
        except Exception:  # BROAD-CATCH: error-translation
            _log.exception("outbox scrub failed after chunk deletion for tenant %r", store.tenant)
            outbox_events_scrubbed = -1
    staged_files_removed = 0
    try:
        staged_files_removed = delete_staged_sources(store.tenant, to_delete)
    except Exception:  # BROAD-CATCH: error-translation
        _log.exception(
            "staged upload cleanup failed after chunk deletion for tenant %r", store.tenant
        )
        staged_files_removed = -1
    if found and not_found:
        message = (
            f"Forgot {chunks_removed} chunk(s) from {len(found)} source(s); "
            f"{len(not_found)} source(s) not found: {', '.join(not_found)}."
        )
    elif found:
        message = f"Forgot {chunks_removed} chunk(s) from {len(found)} source(s)."
    else:
        message = f"No matching source(s) found — nothing deleted: {', '.join(not_found)}."
    if outbox_events_scrubbed < 0:
        message += (
            " WARNING: the chunk deletion succeeded but scrubbing the migration outbox failed; "
            "re-run this forget before the next replay or the text may be restored."
        )
    elif outbox_events_scrubbed:
        message += f" Scrubbed {outbox_events_scrubbed} pending replay record(s)."
    if staged_files_removed < 0:
        message += (
            " WARNING: the chunk deletion succeeded but staged upload cleanup failed; "
            "re-run this forget before the next index or the text may be restored."
        )
    elif staged_files_removed:
        message += f" Removed {staged_files_removed} staged upload file(s)."
    return ForgetResult(
        chunks_removed=chunks_removed,
        sources_removed=found,
        sources_not_found=not_found,
        message=message,
        outbox_events_scrubbed=outbox_events_scrubbed,
        staged_files_removed=staged_files_removed,
    )


def memory_stats(store: PgVectorStore, max_age: timedelta = timedelta(days=2)) -> MemoryStatsResult:
    """Report memory size and freshness."""
    newest = store.newest_indexed_at()
    stale = staleness(newest, datetime.now(UTC), max_age).stale
    return MemoryStatsResult(
        chunks=store.count(),
        newest_indexed_at=newest.isoformat() if newest else None,
        stale=stale,
        metrics=METRICS.snapshot(),
    )


def memory_inventory(
    store: PgVectorStore,
    *,
    limit: int = 5000,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> InventoryResult:
    """Return a bounded, ordered inventory keyed by raw source content digests."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    ordered = sorted(store.source_raw_hashes().items())
    if security_policy is not None:
        if access_context is None:
            raise ValueError("access_context is required when security_policy is configured")
        if getattr(store, "tenant", None) != access_context.tenant:
            raise PermissionError("access context tenant does not match the store")
        ordered = [
            item for item in ordered if security_policy.decide(item[0], access_context).allowed
        ]
    return InventoryResult(
        entries=[
            InventoryEntry(source=source, sha256=digest) for source, digest in ordered[:limit]
        ],
        truncated=len(ordered) > limit,
    )
