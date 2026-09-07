"""Shared pure helpers for MCP reasoning contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from recall.reasoning import GenerationSelection, ReasoningPolicy
from recall.trust import decision_state_for, is_trusted
from recall.types import TrustedHit, TrustedResult

if TYPE_CHECKING:
    from recall.store import PgVectorStore


def _reasoning_policy(mode: str, graph_expansion: str = "off") -> ReasoningPolicy:
    if mode == "retrieval_only":
        return ReasoningPolicy(name="retrieval_only", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "review_required":
        return ReasoningPolicy(name="review_required", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "proposal_assisted":
        return ReasoningPolicy(name="proposal_assisted", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "evidence_assembly":
        return ReasoningPolicy(name="evidence_assembly", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    raise ValueError("unknown reasoning mode")


def _reasoning_generation(store: PgVectorStore) -> GenerationSelection:
    binding = getattr(store, "generation_binding", None)
    if callable(binding):
        payload = binding()
        return GenerationSelection(
            generation_id=str(payload["generation_id"]),
            pipeline_fingerprint=str(payload["pipeline_fingerprint"]),
            corpus_fingerprint=str(payload["corpus_fingerprint"]),
        )
    generation_id = str(getattr(store, "generation_id", "legacy"))
    return GenerationSelection(generation_id=generation_id if generation_id != "legacy" else None)


def _query_construction_generation(generation: GenerationSelection) -> dict[str, object]:
    return {
        "generation_id": generation.generation_id,
        "pipeline_fingerprint": generation.pipeline_fingerprint,
        "corpus_fingerprint": generation.corpus_fingerprint,
    }


def _query_construction_hit(trusted_hit: TrustedHit) -> dict[str, object]:
    chunk = trusted_hit.chunk
    return {
        "chunk_id": chunk.id,
        "source": chunk.source,
        "text": chunk.text[:2_000],
        "score": trusted_hit.cosine,
        "confidence": trusted_hit.confidence,
        "verdict": trusted_hit.verdict,
        "ordinal": trusted_hit.provenance.ord,
    }


def _query_construction_retrieval(result: TrustedResult) -> dict[str, object]:
    return {
        "query": result.query,
        "decision_state": result.decision_state
        or decision_state_for(result.hits, gap_warning=result.gap_warning),
        "abstained": result.abstained,
        "reason": result.reason,
        "gap_warning": result.gap_warning,
        "trust_state": result.trust_state,
        "calibration_status": result.calibration_status,
        "tenant_id": result.tenant_id,
        "generation_id": result.generation_id,
        "pipeline_fingerprint": result.pipeline_fingerprint,
        "corpus_fingerprint": result.corpus_fingerprint,
        "hits": [_query_construction_hit(hit) for hit in result.hits],
    }


def _query_construction_evidence(result: TrustedResult) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "chunk_id": hit.chunk.id,
            "source": hit.chunk.source,
            "text": hit.chunk.text,
            "verdict": hit.verdict,
        }
        for hit in result.hits[:5]
        if is_trusted(hit)
    )


def _query_construction_anchors(result: TrustedResult) -> tuple[str, ...]:
    """Expose only bounded corpus identifiers as graph anchors, never generated text."""

    anchors: list[str] = []
    for hit in result.hits:
        if hit.verdict != "ok":
            continue
        for value in (hit.chunk.source, hit.chunk.id):
            if value and value not in anchors:
                anchors.append(value)
            if len(anchors) >= 8:
                return tuple(anchors)
    return tuple(anchors)


def _same_generation(expected: GenerationSelection, result: TrustedResult) -> None:
    checks = (
        ("generation_id", expected.generation_id, result.generation_id),
        ("pipeline_fingerprint", expected.pipeline_fingerprint, result.pipeline_fingerprint),
        ("corpus_fingerprint", expected.corpus_fingerprint, result.corpus_fingerprint),
    )
    for name, requested, actual in checks:
        if requested is not None and actual != requested:
            raise ValueError(f"retrieval {name} does not match the construction generation")
