"""MCP tenant lifecycle and bounded administrative operations."""

from __future__ import annotations

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

if TYPE_CHECKING:
    from recall.control_plane import ControlPlane
    from recall.current_state import CurrentStateProjection
    from recall.store import PgVectorStore

_log = get_logger("mcp.service")
# Upper bound on one `recall_forget` call's source list. No legitimate erasure names a thousand
# sources in one call.
MAX_FORGET_SOURCES = 1000


def current_state_memory(
    store: PgVectorStore,
    *,
    as_of: datetime | None = None,
    source: str | None = None,
    max_records: int = MAX_CURRENT_STATE_RECORDS,
) -> CurrentStateResult:
    """Return a bounded, deterministic authored current state projection.

    ``as_of`` fixes the point in time, ``source`` narrows the projection, and ``max_records``
    prevents a serving request from assembling an unbounded response.  The underlying library
    function remains available without a bound for offline projection work.

    Args:
        store: tenant bound read store.
        as_of: optional point in time for authored validity and supersession.
        source: optional canonical source filter.
        max_records: positive serving bound on projected source records.

    Raises:
        ValueError: if the bound is invalid or the projection exceeds it.
    """
    projection: CurrentStateProjection = project_current_state(
        store, as_of=as_of, source=source, max_records=max_records
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
) -> ForgetResult:
    """Permanently delete every indexed chunk for the given sources; return what actually went away.

    This is the right-to-erasure path: irreversible and tenant-scoped (only ever touches the
    calling store's own tenant — see `PgVectorStore.delete_sources`). A source that does not
    exist for this tenant is reported in `sources_not_found`, never silently folded into a "0
    removed, success" result — a typo'd source name must be visibly distinguishable from one
    that was actually forgotten.

    **Erasure reaches the migration outbox too, when a control plane is supplied.** It did not,
    and that was a hole in "permanently delete" rather than a missing nicety: while a shadow
    migration is in flight, `recall_migration_events.payload` holds the full text and vectors of
    every chunk in the batch. Deleting from both chunk tables and stopping there left the erased
    text sitting in the outbox, and a later `replay` would have written it back into both
    generations. The scrub runs AFTER the deletes, so a crash between them leaves the outbox
    entry, which replay converges and the next erasure removes; the reverse order could scrub the
    replay record and then fail to delete, which loses the shadow write with nothing left to
    replay it from.
    """
    if not sources:
        raise ValueError("sources must be a non-empty list")
    # Bounded BEFORE de-duplication: the cost this guards is the list the client sent, and
    # de-duplicating first would let a million-element list of one repeated value through.
    if len(sources) > MAX_FORGET_SOURCES:
        raise ValueError(
            f"{len(sources)} sources requested, over the {MAX_FORGET_SOURCES} limit for one "
            f"call. Deletion is irreversible; split the request so each one stays reviewable."
        )
    requested = list(dict.fromkeys(sources))  # de-dup, preserve order
    # An identifier is whatever recall_search showed the caller: the root-relative `file` for an
    # indexed chunk, or the raw `source` for a legacy row. Resolve each to the absolute `source`
    # value(s) deletion keys on — matching `metadata->>'file'` OR `source`, tenant-scoped by the
    # store — so following the documented erasure contract actually deletes. (Previously forget
    # compared the relative id straight against the absolute `source` column and matched nothing.)
    resolved = store.sources_for_identifiers(requested)  # {identifier: [source, ...]}
    if shadow_store is not None:
        shadow_resolved = shadow_store.sources_for_identifiers(requested)
        for identifier, values in shadow_resolved.items():
            bucket = resolved.setdefault(identifier, [])
            bucket.extend(value for value in values if value not in bucket)
    found = [s for s in requested if s in resolved]
    not_found = [s for s in requested if s not in resolved]
    to_delete = sorted({src for ident in found for src in resolved[ident]})
    if to_delete and shadow_store is not None:
        chunks_removed = store.delete_sources_across([store.table, shadow_store.table], to_delete)
    else:
        chunks_removed = store.delete_sources(to_delete) if to_delete else 0
    outbox_events_scrubbed = 0
    if control_plane is not None:
        # NOT gated on `to_delete`. That gate made the scrub unable to fire in exactly the state
        # it was written for: a crash between `append_event` and the two `replace_sources` calls
        # leaves the batch's full text and vectors in the outbox with ZERO rows in either chunk
        # table, so `sources_for_identifiers` resolves nothing, `to_delete` is empty, and the
        # caller was told "no matching source(s) found" while the text sat waiting for a replay
        # to write it back into both generations. Three auditors found this independently.
        #
        # Keyed on the union of what was requested and what resolved: an identifier the caller
        # supplied may itself be the absolute source the payload records, which is the only
        # handle available when no chunk row survives to resolve it.
        try:
            outbox_events_scrubbed = control_plane.erase_sources_from_pending(
                store.tenant, sorted({*requested, *to_delete})
            )
        except Exception:  # BROAD-CATCH: error-translation
            # The deletes above are committed and irreversible. Losing the ForgetResult to a
            # bookkeeping failure would tell the caller nothing was deleted when everything was,
            # and a retry would then report the sources as not found. Report the shortfall
            # instead, and keep it in the receipt.
            _log.exception("outbox scrub failed after chunk deletion for tenant %r", store.tenant)
            outbox_events_scrubbed = -1
    staged_files_removed = 0
    try:
        staged_files_removed = delete_staged_sources(store.tenant, to_delete)
    except Exception:  # BROAD-CATCH: error-translation
        # Database erasure is already committed and irreversible. Preserve its receipt while
        # making a failed filesystem cleanup explicit so the caller can retry before re-indexing.
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
    """Report memory size and freshness (`stale` is True when the newest chunk is older than `max_age`, default 2 days)."""
    newest = store.newest_indexed_at()
    stale = staleness(newest, datetime.now(UTC), max_age).stale
    return MemoryStatsResult(
        chunks=store.count(),
        newest_indexed_at=newest.isoformat() if newest else None,
        stale=stale,
        metrics=METRICS.snapshot(),
    )


def memory_inventory(store: PgVectorStore, *, limit: int = 5000) -> InventoryResult:
    """Return a bounded, ordered inventory keyed by raw source content digests."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    ordered = sorted(store.source_raw_hashes().items())
    return InventoryResult(
        entries=[
            InventoryEntry(source=source, sha256=digest) for source, digest in ordered[:limit]
        ],
        truncated=len(ordered) > limit,
    )
