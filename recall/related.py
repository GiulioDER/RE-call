"""Trusted, generation scoped evidence related to an existing chunk."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from recall.calibration import Calibration
from recall.lineage import canonical_sha256
from recall.store import EdgeCandidates
from recall.trust import evaluate
from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal, TrustState, code_for_status
from recall.types import (
    Chunk,
    RetrievalDiagnostics,
    RetrievalResult,
    ScoredChunk,
    StalenessReport,
    TrustedHit,
)

RelatedRelation = Literal["source", "ordinal", "supersession"]
MAX_RELATED_ITEMS = 50


class RelatedStore(Protocol):
    tenant: str

    def iter_chunks(self, batch_size: int = 1000) -> Iterable[Chunk]: ...

    def related_chunks(
        self, seed_chunk_id: str, relation: RelatedRelation, max_items: int
    ) -> tuple[Chunk, list[Chunk]] | None: ...

    def supersession_all(self) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]: ...


@dataclass(frozen=True)
class RelatedEvidenceResult:
    seed_chunk_id: str
    relation: RelatedRelation
    items: tuple[TrustedHit, ...]
    rejected_count: int
    generation_id: str
    explanation: dict[str, object] | None = None


def _file(chunk: Chunk) -> str:
    value = chunk.metadata.get("file")
    return value if isinstance(value, str) and value else chunk.source


def _ordinal(chunk: Chunk) -> int | None:
    value = chunk.metadata.get("ord")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _generation_binding(store: RelatedStore) -> dict[str, str]:
    reader = getattr(store, "generation_binding", None)
    if callable(reader):
        return {str(key): str(value) for key, value in reader().items() if value is not None}
    return {
        "tenant_id": str(store.tenant),
        "generation_id": str(getattr(store, "generation_id", "legacy")),
    }


def trusted_related(
    store: RelatedStore,
    seed_chunk_id: str,
    *,
    relation: RelatedRelation = "source",
    max_items: int = 5,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    now: datetime | None = None,
    explain: bool = False,
    _generation_snapshot: bool = True,
) -> RelatedEvidenceResult:
    """Find related chunks and independently evaluate every candidate for trust.

    The operation is read only.  Relatedness is structural, not semantic, and a seed verdict is
    never copied to a related chunk.
    """
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        raise ValueError("max_items must be a positive int")
    if max_items > MAX_RELATED_ITEMS:
        raise ValueError(f"max_items must be <= {MAX_RELATED_ITEMS}")
    if relation not in {"source", "ordinal", "supersession"}:
        raise ValueError("relation must be source, ordinal, or supersession")
    snapshot = getattr(store, "snapshot", None)
    if _generation_snapshot and callable(snapshot):
        with snapshot():
            return trusted_related(
                store,
                seed_chunk_id,
                relation=relation,
                max_items=max_items,
                calibration=calibration,
                policy=policy,
                now=now,
                explain=explain,
                _generation_snapshot=False,
            )
    binding = _generation_binding(store)
    # A direct library caller that supplies an explicit calibration retains the legacy offline
    # behavior when no policy was supplied. Serving adapters always pass their resolved policy,
    # so a production request cannot bypass the strict gate by taking this lower level path.
    active_policy = policy or (
        TrustPolicy.development() if calibration is not None else TrustPolicy.from_env()
    )
    calibration_status = "legacy_unbound" if calibration is not None else "missing"
    if calibration is None:
        resolver = getattr(store, "resolve_calibration", None)
        if callable(resolver):
            resolution = resolver()
            calibration_status = str(resolution.status.value)
            artifact = getattr(resolution, "artifact", None)
            if artifact is not None:
                calibration = artifact.runtime
    failure_code = code_for_status(calibration_status)
    if failure_code is not None and not binding.get("generation_id"):
        failure_code = TrustFailureCode.INDEX_NOT_READY
    if failure_code is not None and active_policy.strict:
        raise TrustRefusal(
            code=failure_code,
            calibration_status=calibration_status,
            tenant_id=binding.get("tenant_id") or getattr(store, "tenant", None),
            generation_id=binding.get("generation_id"),
            pipeline_fingerprint=binding.get("pipeline_fingerprint"),
            corpus_fingerprint=binding.get("corpus_fingerprint"),
        )
    store_query = getattr(store, "related_chunks", None)
    if callable(store_query):
        bounded = store_query(seed_chunk_id, relation, max_items)
    else:
        bounded = None
    if bounded is None:
        seed = next((chunk for chunk in store.iter_chunks() if chunk.id == seed_chunk_id), None)
        if seed is None:
            raise ValueError(f"seed chunk not found: {seed_chunk_id!r}")
        candidates_iter: Iterable[Chunk] = store.iter_chunks()
    else:
        seed, candidates_iter = bounded
    seed_file = _file(seed)
    seed_ord = _ordinal(seed)
    if relation == "supersession":
        edges, unresolved, edge_candidates = store.supersession_all()
    else:
        edges, unresolved, edge_candidates = {}, frozenset(), {}
    related: list[ScoredChunk] = []
    for chunk in candidates_iter:
        if chunk.id == seed_chunk_id:
            continue
        file = _file(chunk)
        ordinal = _ordinal(chunk)
        same_source = file == seed_file
        adjacent = (
            seed_ord is not None and ordinal is not None and abs(ordinal - seed_ord) <= 2
        )
        supersession = (
            file == edges.get(seed_file)
            or seed_file == edges.get(file)
            or any(successor == file for successor, _when in edge_candidates.get(seed_file, ()))
        )
        matches = {
            "source": same_source,
            "ordinal": same_source and adjacent,
            "supersession": supersession,
        }
        if matches[relation]:
            related.append(
                ScoredChunk(
                    chunk=chunk,
                    score=0.0,
                    score_kind="structural",
                    indexed_at=chunk.metadata.get("indexed_at"),
                )
            )
    related.sort(key=lambda item: (abs((_ordinal(item.chunk) or 0) - (seed_ord or 0)), item.chunk.id))
    related = related[:max_items]
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    binding = _generation_binding(store)
    trusted = evaluate(
        RetrievalResult(
            query=f"related:{seed_chunk_id}",
            hits=related,
            gap_warning=False,
            staleness=StalenessReport(False, instant, timedelta(0), timedelta(days=2)),
            diagnostics=RetrievalDiagnostics(
                retrieval_profile="related-v1",
                index_generation=binding.get("generation_id", "legacy"),
                candidate_pool_size=len(related),
            ),
        ),
        edges,
        calibration,
        instant,
        unresolved=unresolved,
        edge_candidates=edge_candidates,
        generation_binding=binding,
    )
    if failure_code is not None:
        trusted = replace(
            trusted,
            trust_state=TrustState.DEGRADED.value,
            failure_code=failure_code.value,
        )
        if calibration is None:
            trusted = replace(
                trusted,
                hits=[replace(hit, verdict="unverified") for hit in trusted.hits],
                abstained=False,
                reason="",
            )
    ok = tuple(hit for hit in trusted.hits if hit.verdict == "ok")
    explanation = None
    if explain:
        explanation = {
            "reason_code": "structural_relatedness",
            "seed_chunk_id": seed_chunk_id,
            "relation": relation,
            "candidate_count": len(related),
            "trusted_count": len(ok),
            "rejected_count": len(related) - len(ok),
            "generation_id": binding.get("generation_id", "legacy"),
            "explanation_id": "rx_" + canonical_sha256(
                {"seed": seed_chunk_id, "relation": relation, "generation": binding}
            )[:24],
        }
    return RelatedEvidenceResult(
        seed_chunk_id=seed_chunk_id,
        relation=relation,
        items=ok,
        rejected_count=len(related) - len(ok),
        generation_id=binding.get("generation_id", "legacy"),
        explanation=explanation,
    )


__all__ = ["MAX_RELATED_ITEMS", "RelatedEvidenceResult", "RelatedRelation", "trusted_related"]
