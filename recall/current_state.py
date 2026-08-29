"""Deterministic current state projection for authored mutable memory.

This projection is intentionally separate from retrieval.  It turns authored validity windows and
supersession edges into an immutable, generation bound view without making inferred claims or
changing trust decisions.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from recall.dependency_invalidation import (
    MAX_INVALIDATION_CHAIN,
    DependencyProjection,
    build_dependency_projection,
)
from recall.frontmatter import supersedes_key, validity_bounds
from recall.lineage import canonical_sha256
from recall.store import EdgeCandidates
from recall.types import Chunk

CurrentState = Literal[
    "current",
    "superseded",
    "expired",
    "not_yet_valid",
    "not_yet_known",
    "ambiguous",
    "invalid",
    "dependency_invalidated",
]

CURRENT_STATE_SCHEMA_VERSION = 2
MAX_CURRENT_STATE_RECORDS = 1000


class StateStore(Protocol):
    @property
    def tenant(self) -> str: ...

    def iter_chunks(self, batch_size: int = 1000) -> Iterable[Chunk]: ...

    def supersession_all(self) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]: ...


@dataclass(frozen=True)
class CurrentStateRecord:
    """One canonical authored source and its fail closed point in time state."""

    state_id: str
    source: str
    state: CurrentState
    chunk_ids: tuple[str, ...]
    successor_chain: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    diagnostics: tuple[str, ...] = ()
    base_state: str | None = None
    authority: str = "unknown"
    dependencies: tuple[str, ...] = ()
    invalidation_chain: tuple[str, ...] = ()


@dataclass(frozen=True)
class CurrentStateProjection:
    """Immutable projection identity and records for one tenant generation and as_of instant."""

    schema_version: int
    projection_id: str
    tenant_id: str
    generation_id: str
    pipeline_fingerprint: str | None
    corpus_fingerprint: str | None
    as_of: datetime
    records: tuple[CurrentStateRecord, ...]
    dependency_projection: DependencyProjection | None = None
    known_as_of: datetime | None = None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _file(chunk: Chunk) -> str:
    value = chunk.metadata.get("file")
    return value if isinstance(value, str) and value else chunk.source


def _direct_successors(
    target: str, candidates: EdgeCandidates, as_of: datetime
) -> tuple[str, ...]:
    return tuple(
        sorted(
            successor
            for successor, asserted_at in candidates.get(target, ())
            if asserted_at is None or _utc(asserted_at) <= as_of
        )
    )


def _chain(
    source: str, candidates: EdgeCandidates, as_of: datetime
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    chain: list[str] = []
    seen = {source}
    current = source
    while True:
        successors = _direct_successors(current, candidates, as_of)
        if len(successors) > 1:
            return tuple(chain), ("multiple_live_successors",)
        if not successors:
            return tuple(chain), ()
        successor = successors[0]
        if successor in seen:
            return tuple(chain), ("supersession_cycle",)
        chain.append(successor)
        seen.add(successor)
        current = successor


def _record(
    source: str,
    chunks: list[Chunk],
    candidates: EdgeCandidates,
    unresolved: frozenset[str],
    as_of: datetime,
    tenant_id: str,
    generation_id: str,
    known_as_of: datetime | None = None,
    asserted_at: datetime | None = None,
) -> CurrentStateRecord:
    chunk_ids = tuple(sorted(chunk.id for chunk in chunks))
    diagnostics: list[str] = []
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    chain: tuple[str, ...] = ()
    for chunk in chunks:
        supersedes = chunk.metadata.get("supersedes")
        if supersedes is not None and not isinstance(supersedes, str):
            diagnostics.append("malformed_supersession_metadata")
        try:
            start, end = validity_bounds(chunk.metadata)
        except ValueError:
            diagnostics.append("malformed_validity_metadata")
            continue
        valid_from = start if valid_from is None else min(valid_from, start) if start else valid_from
        valid_until = end if valid_until is None else max(valid_until, end) if end else valid_until

    if diagnostics:
        state: CurrentState = "invalid"
    else:
        chain, chain_diagnostics = _chain(source, candidates, as_of)
        diagnostics.extend(chain_diagnostics)
        if source in unresolved or any(
            supersedes_key(str(chunk.metadata.get("supersedes", ""))) in unresolved
            for chunk in chunks
        ):
            diagnostics.append("unresolved_supersession_reference")
        if diagnostics:
            state = "ambiguous"
        elif (
            known_as_of is not None
            and asserted_at is not None
            and _utc(asserted_at) > known_as_of
        ):
            state = "not_yet_known"
        elif chain:
            state = "superseded"
        elif valid_from is not None and as_of < _utc(valid_from):
            state = "not_yet_valid"
        elif valid_until is not None and as_of > _utc(valid_until):
            state = "expired"
        else:
            state = "current"

    payload = {
        "schema_version": CURRENT_STATE_SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "generation_id": generation_id,
        "source": source,
        "state": state,
        "chunk_ids": chunk_ids,
        "successor_chain": chain,
        "as_of": as_of.isoformat(),
    }
    return CurrentStateRecord(
        state_id=f"state_{canonical_sha256(payload)[:24]}",
        source=source,
        state=state,
        chunk_ids=chunk_ids,
        successor_chain=chain,
        valid_from=valid_from,
        valid_until=valid_until,
        diagnostics=tuple(sorted(set(diagnostics))),
        base_state=state,
    )


def project_current_state(
    store: StateStore,
    as_of: datetime | None = None,
    known_as_of: datetime | None = None,
    source: str | None = None,
    max_records: int | None = None,
) -> CurrentStateProjection:
    """Build a deterministic current state view from one generation snapshot.

    Pass an explicit ``as_of`` instant when the projection must be reproducible across calls.
    ``max_records`` is an optional fail closed bound for serving surfaces.  The complete library
    projection remains available when it is ``None``.
    """
    if max_records is not None and (
        isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1
    ):
        raise ValueError("max_records must be a positive int or None")
    if max_records is not None and max_records > MAX_CURRENT_STATE_RECORDS:
        raise ValueError(f"max_records must be <= {MAX_CURRENT_STATE_RECORDS}")
    instant = _utc(as_of or datetime.now(UTC))
    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        with snapshot() as generation_id:
            return _project(
                store, instant, known_as_of, source, str(generation_id), max_records
            )
    generation_id = str(getattr(store, "generation_id", "legacy"))
    return _project(store, instant, known_as_of, source, generation_id, max_records)


def _project(
    store: StateStore,
    as_of: datetime,
    known_as_of: datetime | None,
    source: str | None,
    generation_id: str,
    max_records: int | None,
) -> CurrentStateProjection:
    chunks_by_source: dict[str, list[Chunk]] = {}
    asserted_at_by_source: dict[str, datetime | None] = {}
    timed_reader = getattr(store, "iter_chunks_with_times", None)
    if callable(timed_reader):
        chunk_rows = timed_reader()
    else:
        chunk_rows = ((chunk, None) for chunk in store.iter_chunks())
    for chunk, asserted_at in chunk_rows:
        key = _file(chunk)
        chunks_by_source.setdefault(key, []).append(chunk)
        previous = asserted_at_by_source.get(key)
        if key not in asserted_at_by_source or (
            asserted_at is not None and (previous is None or asserted_at < previous)
        ):
            asserted_at_by_source[key] = asserted_at
    _edges, unresolved, candidates = store.supersession_all()
    all_records = tuple(
        _record(
            key,
            chunks,
            candidates,
            unresolved,
            as_of,
            str(store.tenant),
            generation_id,
            None if known_as_of is None else _utc(known_as_of),
            asserted_at_by_source.get(key),
        )
        for key, chunks in sorted(chunks_by_source.items())
    )
    binding_reader = getattr(store, "generation_binding", None)
    binding = binding_reader() if callable(binding_reader) else {}
    dependency_projection = build_dependency_projection(
        [chunk for chunks in chunks_by_source.values() for chunk in chunks],
        tenant_id=str(store.tenant),
        generation_id=generation_id,
        base_states={record.source: record.state for record in all_records},
        as_of=as_of,
        known_as_of=known_as_of,
        asserted_at_by_source=asserted_at_by_source,
        corpus_fingerprint=binding.get("corpus_fingerprint"),
    )
    records_list: list[CurrentStateRecord] = []
    for record in all_records:
        if source is not None and record.source != source and not any(
            chunk.source == source for chunk in chunks_by_source[record.source]
        ):
            continue
        if max_records is not None and len(records_list) >= max_records:
            raise ValueError("current state projection exceeds max_records")
        reason = dependency_projection.reason_for(record.source)
        diagnostics = record.diagnostics
        state: CurrentState = record.state
        invalidation_chain: tuple[str, ...] = ()
        if reason is not None:
            state = "dependency_invalidated"
            invalidation_chain = reason.bounded_path(MAX_INVALIDATION_CHAIN)
            diagnostics = tuple(sorted(set(diagnostics + ("dependency_invalidated",))))
        authority = dependency_projection.authorities.get(record.source, "unknown")
        dependencies = dependency_projection.dependencies.get(record.source, ())
        state_id = "state_" + canonical_sha256(
            {
                "schema_version": CURRENT_STATE_SCHEMA_VERSION,
                "tenant_id": str(store.tenant),
                "generation_id": generation_id,
                "source": record.source,
                "state": state,
                "base_state": record.base_state,
                "authority": authority,
                "dependencies": dependencies,
                "invalidation_chain": invalidation_chain,
                "as_of": as_of.isoformat(),
            }
        )[:24]
        records_list.append(
            CurrentStateRecord(
                state_id=state_id,
                source=record.source,
                state=state,
                chunk_ids=record.chunk_ids,
                successor_chain=record.successor_chain,
                valid_from=record.valid_from,
                valid_until=record.valid_until,
                diagnostics=diagnostics,
                base_state=record.base_state,
                authority=authority,
                dependencies=dependencies,
                invalidation_chain=invalidation_chain,
            )
        )
    records = tuple(records_list)
    payload = {
        "schema_version": CURRENT_STATE_SCHEMA_VERSION,
        "tenant_id": str(store.tenant),
        "generation_id": generation_id,
        "as_of": as_of.isoformat(),
        "known_as_of": _utc(known_as_of).isoformat() if known_as_of else None,
        "records": [record.state_id for record in records],
        "dependency_projection_id": dependency_projection.projection_id,
    }
    return CurrentStateProjection(
        schema_version=CURRENT_STATE_SCHEMA_VERSION,
        projection_id=f"csp_{canonical_sha256(payload)[:24]}",
        tenant_id=str(store.tenant),
        generation_id=generation_id,
        pipeline_fingerprint=binding.get("pipeline_fingerprint"),
        corpus_fingerprint=binding.get("corpus_fingerprint"),
        as_of=as_of,
        records=records,
        dependency_projection=dependency_projection,
        known_as_of=_utc(known_as_of) if known_as_of is not None else None,
    )


__all__ = [
    "CURRENT_STATE_SCHEMA_VERSION",
    "MAX_CURRENT_STATE_RECORDS",
    "CurrentStateProjection",
    "CurrentStateRecord",
    "CurrentState",
    "project_current_state",
]
