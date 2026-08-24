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

from recall.frontmatter import supersedes_key, validity_bounds
from recall.lineage import canonical_sha256
from recall.store import EdgeCandidates
from recall.types import Chunk

CurrentState = Literal[
    "current",
    "superseded",
    "expired",
    "not_yet_valid",
    "ambiguous",
    "invalid",
]

CURRENT_STATE_SCHEMA_VERSION = 1
MAX_CURRENT_STATE_RECORDS = 1000


class StateStore(Protocol):
    tenant: str

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
    )


def project_current_state(
    store: StateStore,
    as_of: datetime | None = None,
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
            return _project(store, instant, source, str(generation_id), max_records)
    generation_id = str(getattr(store, "generation_id", "legacy"))
    return _project(store, instant, source, generation_id, max_records)


def _project(
    store: StateStore,
    as_of: datetime,
    source: str | None,
    generation_id: str,
    max_records: int | None,
) -> CurrentStateProjection:
    chunks_by_source: dict[str, list[Chunk]] = {}
    for chunk in store.iter_chunks():
        key = _file(chunk)
        if source is None or key == source or chunk.source == source:
            if max_records is not None and key not in chunks_by_source:
                if len(chunks_by_source) >= max_records:
                    raise ValueError("current state projection exceeds max_records")
            chunks_by_source.setdefault(key, []).append(chunk)
    _edges, unresolved, candidates = store.supersession_all()
    records = tuple(
        _record(
            key,
            chunks,
            candidates,
            unresolved,
            as_of,
            str(store.tenant),
            generation_id,
        )
        for key, chunks in sorted(chunks_by_source.items())
    )
    binding_reader = getattr(store, "generation_binding", None)
    binding = binding_reader() if callable(binding_reader) else {}
    payload = {
        "schema_version": CURRENT_STATE_SCHEMA_VERSION,
        "tenant_id": str(store.tenant),
        "generation_id": generation_id,
        "as_of": as_of.isoformat(),
        "records": [record.state_id for record in records],
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
    )


__all__ = [
    "CURRENT_STATE_SCHEMA_VERSION",
    "MAX_CURRENT_STATE_RECORDS",
    "CurrentStateProjection",
    "CurrentStateRecord",
    "CurrentState",
    "project_current_state",
]
