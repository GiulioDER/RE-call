"""MCP tenant lifecycle and bounded administrative operations."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from typing import TYPE_CHECKING, Any

from recall.current_state import MAX_CURRENT_STATE_RECORDS, project_current_state
from recall.fact_retraction import FactErasure, fact_ledger_for, prepare_fact_erasure
from recall.guards import staleness
from recall.observability import METRICS, get_logger
from recall.uploads import delete_staged_sources
from recall_mcp.models import (
    CurrentStateRecordModel,
    CurrentStateResult,
    ForgetResult,
    InventoryEntry,
    WithdrawResult,
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


def _fact_ledger_for(store: object) -> Any | None:
    """The fact ledger `recall_apply_fact` writes for `store`, or None for a store that has none.

    Resolved with the server's environment (`RECALL_FACT_WRITE_DSN`), as `recall_apply_fact` is.
    """
    from recall_mcp.settings import runtime_environment

    return fact_ledger_for(store, runtime_environment())


def _forget_retractions(
    store: PgVectorStore,
    shadow_store: PgVectorStore | None,
    requested: list[str],
    to_delete: list[str],
    withdraw: bool = False,
) -> FactErasure | None:
    """Prepare, before any chunk is deleted, the retraction and redaction this forget owes the
    fact ledger (with `withdraw`, a withdrawal in place of the retraction).

    None for a store with no ledger; otherwise `recall.fact_retraction.prepare_fact_erasure`,
    which states the rule, with the redactor `recall.fact_redaction.fact_redactor_for` resolves
    from the same environment.
    """
    from recall.fact_redaction import fact_redactor_for
    from recall_mcp.settings import runtime_environment

    ledger = _fact_ledger_for(store)
    if ledger is None:
        return None
    redactor = fact_redactor_for(store, runtime_environment())
    return prepare_fact_erasure(
        ledger, store, shadow_store, named=requested, erased=to_delete, redactor=redactor,
        withdraw=withdraw,
    )


def forget_memory(
    store: PgVectorStore,
    sources: list[str],
    shadow_store: PgVectorStore | None = None,
    control_plane: ControlPlane | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
    *,
    erase_memo_requests: bool = True,
) -> ForgetResult:
    """Permanently delete every indexed chunk for the given sources, and erase the fact values
    they supplied.

    Every fact the fact ledger holds whose evidence was only in these sources is retracted with
    them (`recall.fact_ledger.plan_retractions`), so `recall_current_facts` stops listing it, and
    then its value is erased in place (`recall.fact_redaction`): from the ledger events, from every
    evidence card citing these sources, and from the replay receipts that echo them. The history
    keeps its shape (event types, times, memo names, one redaction record).

    On a lite store the resolve and the delete hold the file's write lane (`recall.lite.writer_lock`),
    and everything runs inside `LiteStore.erasing()`, so the bytes it frees are zeroed in the file
    and its write-ahead log.

    A source that is an assistant memo (`recall.lite.assistant_memos.forget_memo_requests`) also
    loses its request journal hashes, whether or not its file is still on disk.
    `erase_memo_requests=False` is for `recall_forget_memo`, which erases them itself just before
    it deletes the file, and puts them back if the delete fails.
    """
    from recall.backends import write_session
    from recall.lite import LiteStore

    erasing = store.erasing() if isinstance(store, LiteStore) else nullcontext()
    with write_session(store, "forget"), erasing:
        return _forget_memory_unlocked(
            store, sources, shadow_store, control_plane, security_policy, security_context,
            erase_memo_requests=erase_memo_requests,
        )


def withdraw_memory(
    store: PgVectorStore,
    sources: list[str],
    shadow_store: PgVectorStore | None = None,
    control_plane: ControlPlane | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
) -> WithdrawResult:
    """Withdraw memos that were a mistake: erase them as `forget_memory` does, and give back what
    they superseded (owner decision, 2026-10-22).

    Forgetting keeps retired what a memo retired; withdrawing does not. At the document layer the
    memo's `supersedes:` claims become void and whatever superseded it supersedes what it
    superseded instead (`recall.supersession.SupersessionRecord`); in the fact ledger each fact
    it alone supported gets a `withdrawn` event, which also undoes the supersession it made
    (`recall.fact_ledger.plan_withdrawals`). Withdrawing the head of a line makes its predecessor
    current, withdrawing the middle splices the chain, and a memo forgotten or pruned earlier can
    be withdrawn by the name it had.
    """
    from recall.backends import write_session
    from recall.lite import LiteStore

    erasing = store.erasing() if isinstance(store, LiteStore) else nullcontext()
    with write_session(store, "forget"), erasing:
        result = _forget_memory_unlocked(
            store, sources, shadow_store, control_plane, security_policy, security_context,
            withdraw=True,
        )
    return WithdrawResult(
        chunks_removed=result.chunks_removed,
        sources_withdrawn=result.sources_removed,
        sources_not_found=result.sources_not_found,
        message=result.message,
        outbox_events_scrubbed=result.outbox_events_scrubbed,
        staged_files_removed=result.staged_files_removed,
        facts_withdrawn=result.facts_retracted,
        facts_redacted=result.facts_redacted,
    )


#: The audit event a forget writes, just before deleting them, for the assistant memos it removes.
MEMO_JOURNAL_EVENT = "forget_memo_journal"


def _record_memo_journals(store: PgVectorStore, sources: list[str]) -> None:
    """Record in `store`'s own audit ledger each assistant memo among `sources`, by its source only.

    A store with no audit ledger records nothing, and its re-run then erases only what it deletes.
    An append that fails raises before any row is deleted.
    """
    from recall.lite.assistant_memos import is_assistant_memo

    append = getattr(store, "append_audit_event", None)
    if not callable(append):
        return
    for source in sorted(source for source in set(sources) if is_assistant_memo(source)):
        identity = json.dumps([getattr(store, "tenant", None), source], ensure_ascii=False)
        append(
            MEMO_JOURNAL_EVENT, {}, source_uri=source,
            event_id="forget_memo_journal_" + hashlib.sha256(identity.encode()).hexdigest(),
        )


def _recorded_memo_journals(store: PgVectorStore, names: list[str]) -> set[str]:
    """The `names` this store recorded deleting as assistant memos (`_record_memo_journals`)."""
    recorded = getattr(store, "recorded_memo_journals", None)
    return set(recorded(names)) if callable(recorded) and names else set()


def _authorize_erasure(
    store: PgVectorStore,
    shadow_store: PgVectorStore | None,
    to_delete: list[str],
    security_policy: SourceSecurityPolicy,
    security_context: AccessContext,
    *,
    withdraw: bool,
) -> None:
    """Refuse an erasure unless the policy allows every name the sources about to go are
    authorized by, before anything is deleted.

    Allowing the names a caller asked for is not enough: a name resolves to every source whose
    `file` or `source` it matches, and each of those rows is authorized by its own `file` (or its
    `source` without one), which can be a name the policy denies. A source with no row and no
    record (resolved from a build's manifest) is decided by its own name, which is a requested one.
    """
    subjects: dict[str, set[str]] = {}
    for target in (store, shadow_store):
        if target is None:
            continue
        for source, names in target.erasure_subjects(to_delete, recorded=withdraw).items():
            subjects.setdefault(source, set()).update(names)
    for source in to_delete:
        for name in sorted(subjects.get(source) or {source}):
            if not security_policy.decide(name, security_context).allowed:
                raise PermissionError(f"source {name!r} is not authorized for erasure")


def _forget_memory_unlocked(
    store: PgVectorStore,
    sources: list[str],
    shadow_store: PgVectorStore | None = None,
    control_plane: ControlPlane | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
    *,
    withdraw: bool = False,
    erase_memo_requests: bool = True,
) -> ForgetResult:
    """`forget_memory` (or with `withdraw`, `withdraw_memory`) without the lite store's write
    lane; call one of those."""
    verb, did = ("withdrawal", "Withdrew") if withdraw else ("deletion", "Forgot")
    if not sources:
        raise ValueError("sources must be a non-empty list")
    if len(sources) > MAX_FORGET_SOURCES:
        raise ValueError(
            f"{len(sources)} sources requested, over the {MAX_FORGET_SOURCES} limit for one "
            f"call. A {verb} is irreversible; split the request so each one stays reviewable."
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
    if withdraw:
        # A memo forgotten or pruned earlier has no rows left, only the record that keeps what it
        # retired retired; withdrawing it by the name it had is how that is undone.
        for target in (store, shadow_store):
            recorded = getattr(target, "recorded_sources_for_identifiers", None)
            if target is None or not callable(recorded):
                continue
            for identifier, values in recorded(requested).items():
                bucket = resolved.setdefault(identifier, [])
                bucket.extend(value for value in values if value not in bucket)
    found = [source for source in requested if source in resolved]
    not_found = [source for source in requested if source not in resolved]
    to_delete = sorted({source for identifier in found for source in resolved[identifier]})
    if security_policy is not None:
        assert security_context is not None
        _authorize_erasure(
            store, shadow_store, to_delete, security_policy, security_context, withdraw=withdraw
        )
    facts_retracted = 0
    facts_redacted = 0
    erasure: FactErasure | None = None
    try:
        erasure = _forget_retractions(store, shadow_store, requested, to_delete, withdraw)
    except Exception:  # BROAD-CATCH: error-translation
        _log.exception(
            "fact retraction plan failed before chunk deletion for tenant %r",
            getattr(store, "tenant", None),
        )
        # A deletion without its recovery aliases would make an absolute-source retry
        # irrecoverable. A preparation failure therefore leaves the chunks in place.
        raise
    if erase_memo_requests:
        # Before the rows go, for the same reason: the record is how a re-run, which finds no
        # rows, may still name these memos to the journal erasure below.
        _record_memo_journals(store, to_delete)
    if to_delete and shadow_store is not None:
        chunks_removed = store.delete_sources_across(
            [store.table, shadow_store.table], to_delete, withdraw=withdraw
        )
    else:
        chunks_removed = store.delete_sources(to_delete, withdraw=withdraw) if to_delete else 0
    outbox_events_scrubbed = 0
    if control_plane is not None:
        try:
            outbox_events_scrubbed = control_plane.erase_sources_from_pending(
                store.tenant, sorted({*requested, *to_delete})
            )
        except Exception:  # BROAD-CATCH: error-translation
            _log.exception("outbox scrub failed after chunk deletion for tenant %r", store.tenant)
            outbox_events_scrubbed = -1
    if erasure is not None:
        try:
            facts_retracted = erasure.retract()
        except Exception:  # BROAD-CATCH: error-translation
            _log.exception(
                "fact retraction failed after chunk deletion for tenant %r", store.tenant
            )
            facts_retracted = -1
        if facts_retracted < 0:
            # A value erased before its retraction is appended would leave the retraction nothing
            # to pair with; the re-run does both.
            facts_redacted = -1
        else:
            try:
                facts_redacted = erasure.redact()
            except Exception:  # BROAD-CATCH: error-translation
                _log.exception(
                    "fact value redaction failed after chunk deletion for tenant %r", store.tenant
                )
                facts_redacted = -1
    staged_files_removed = 0
    try:
        staged_files_removed = delete_staged_sources(store.tenant, to_delete)
    except Exception:  # BROAD-CATCH: error-translation
        _log.exception(
            "staged upload cleanup failed after chunk deletion for tenant %r", store.tenant
        )
        staged_files_removed = -1
    memo_requests_erased = 0
    if erase_memo_requests:
        from recall.lite.assistant_memos import forget_memo_requests

        try:
            # Only memos this store held: what it deleted now, and requested names it recorded
            # deleting earlier (a re-run after a failure here finds no rows left).
            memo_requests_erased = forget_memo_requests([*to_delete, *_recorded_memo_journals(store, requested)])
        except Exception:  # BROAD-CATCH: error-translation
            _log.exception(
                "memo request journal erasure failed after chunk deletion for tenant %r", store.tenant
            )
            memo_requests_erased = -1
    if found and not_found:
        message = (
            f"{did} {chunks_removed} chunk(s) from {len(found)} source(s); "
            f"{len(not_found)} source(s) not found: {', '.join(not_found)}."
        )
    elif found:
        message = f"{did} {chunks_removed} chunk(s) from {len(found)} source(s)."
    else:
        message = f"No matching source(s) found — nothing deleted: {', '.join(not_found)}."
    if outbox_events_scrubbed < 0:
        message += (
            " WARNING: the chunk deletion succeeded but scrubbing the migration outbox failed; "
            "re-run this forget before the next replay or the text may be restored."
        )
    elif outbox_events_scrubbed:
        message += f" Scrubbed {outbox_events_scrubbed} pending replay record(s)."
    if facts_retracted < 0 and withdraw:
        message += (
            " WARNING: withdrawing the facts these sources supported failed; re-run this "
            "withdraw with the same source names, or recall_current_facts keeps their state."
        )
    elif facts_retracted < 0:
        message += (
            " WARNING: retracting the facts these sources supported failed; re-run this forget "
            "with the same source names, or recall_current_facts keeps listing them."
        )
    elif facts_retracted and withdraw:
        message += f" Withdrew {facts_retracted} fact(s) that had no other evidence."
    elif facts_retracted:
        message += f" Retracted {facts_retracted} fact(s) that had no other evidence."
    if facts_redacted < 0:
        message += (
            " WARNING: erasing the fact values these sources supplied failed, so they stay in the "
            f"fact ledger; re-run this {'withdraw' if withdraw else 'forget'} with the same "
            "source names to erase them."
        )
    elif facts_redacted:
        message += f" Erased the values of {facts_redacted} fact(s) from the fact ledger."
    if staged_files_removed < 0:
        message += (
            " WARNING: the chunk deletion succeeded but staged upload cleanup failed; "
            "remove the surviving staged files before the next index or the text may be restored."
        )
    elif staged_files_removed:
        message += f" Removed {staged_files_removed} staged upload file(s)."
    if memo_requests_erased < 0:
        message += (
            " WARNING: the chunk deletion succeeded but erasing the memo request journal failed, "
            "so it still holds hashes of the memo; re-run this forget with the same source names."
        )
    return ForgetResult(
        chunks_removed=chunks_removed,
        sources_removed=found,
        sources_not_found=not_found,
        message=message,
        outbox_events_scrubbed=outbox_events_scrubbed,
        staged_files_removed=staged_files_removed,
        facts_retracted=facts_retracted,
        facts_redacted=facts_redacted,
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
