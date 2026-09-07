"""Graph-first retrieval orchestration for MCP."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from recall.calibration import Calibration
from recall.graph_first import (
    GraphFirstCandidate,
    GraphFirstMode,
    MAX_GRAPH_FIRST_CANDIDATES,
    build_graph_first_candidates,
)
from recall.reasoning_expansion import merge_trusted_results
from recall.reasoning_graph import project_store_graph
from recall.semantic_graph import SemanticGraphProjection
from recall.trust import is_trusted
from recall.trust_policy import TrustPolicy
from recall.types import TrustedResult
from recall_mcp.reasoning_common import (
    _query_construction_generation,
    _query_construction_retrieval,
    _reasoning_generation,
    _same_generation,
)
from recall_mcp.retrieval import _retrieve_trusted

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore


def graph_first_retrieval(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    *,
    mode: GraphFirstMode = "hybrid",
    source: str | None = None,
    k: int = 5,
    max_candidates: int = MAX_GRAPH_FIRST_CANDIDATES,
    expected_generation_id: str | None = None,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    _retrieve_trusted_fn=_retrieve_trusted,
    _project_store_graph_fn=project_store_graph,
) -> dict[str, object]:
    """Probe bounded graph-derived query seeds before ordinary trusted retrieval."""
    if mode not in {"entity", "relation", "hybrid"}:
        raise ValueError("mode must be 'entity', 'relation', or 'hybrid'")
    if not 1 <= max_candidates <= MAX_GRAPH_FIRST_CANDIDATES:
        raise ValueError(f"max_candidates must be between 1 and {MAX_GRAPH_FIRST_CANDIDATES}")
    if not query.strip():
        raise ValueError("query must be non-empty")

    generation = _reasoning_generation(store)
    if expected_generation_id is not None and expected_generation_id != generation.generation_id:
        return {
            "status": "refused",
            "mode": mode,
            "refusal_reason": "generation_mismatch",
            "generation": _query_construction_generation(generation),
            "diagnostics": {"retrieval_calls": 0, "graph": {"readiness": "not_checked"}},
        }

    graph_started = time.perf_counter()
    semantic: SemanticGraphProjection | None = None
    graph_reason: str | None = None
    readiness_reader = getattr(store, "graph_readiness", None)
    loader = getattr(store, "load_semantic_graph", None)
    try:
        readiness = readiness_reader() if callable(readiness_reader) else None
        if callable(loader) and generation.generation_id is not None:
            semantic = cast(SemanticGraphProjection | None, loader(generation.generation_id))
        else:
            semantic = _project_store_graph_fn(store, include_text=False).semantic_graph
        if readiness is not None and not readiness.ready:
            graph_reason = "graph_not_ready"
        elif semantic is None:
            graph_reason = "graph_not_ready"
        elif semantic.tenant_id != store.tenant:
            graph_reason = "tenant_mismatch"
        elif generation.generation_id and semantic.generation_id != generation.generation_id:
            graph_reason = "generation_mismatch"
        elif (
            generation.pipeline_fingerprint
            and semantic.pipeline_fingerprint != generation.pipeline_fingerprint
        ):
            graph_reason = "pipeline_mismatch"
        elif (
            generation.corpus_fingerprint
            and semantic.corpus_fingerprint != generation.corpus_fingerprint
        ):
            graph_reason = "corpus_mismatch"
    except Exception as exc:  # BROAD-CATCH: fail-open
        graph_reason = type(exc).__name__
        semantic = None

    graph_candidates: tuple[GraphFirstCandidate, ...] = ()
    if semantic is not None and graph_reason is None:
        graph_candidates = build_graph_first_candidates(
            semantic, query, mode=mode, max_candidates=max_candidates
        )

    baseline = _retrieve_trusted_fn(store, embedder, query, source, k, calibration, policy).result
    baseline = replace(
        baseline,
        tenant_id=baseline.tenant_id or store.tenant,
        generation_id=baseline.generation_id or generation.generation_id,
    )
    _same_generation(generation, baseline)

    candidate_results: list[TrustedResult] = []
    failures: list[str] = []
    for candidate in graph_candidates:
        try:
            result = _retrieve_trusted_fn(
                store, embedder, candidate.query, source, k, calibration, policy
            ).result
            result = replace(
                result,
                tenant_id=result.tenant_id or store.tenant,
                generation_id=result.generation_id or generation.generation_id,
            )
            _same_generation(generation, result)
            candidate_results.append(result)
        except Exception as exc:  # BROAD-CATCH: fail-open
            failures.append(type(exc).__name__)

    merged = merge_trusted_results(baseline, candidate_results, original_query=query)
    merged = replace(
        merged,
        tenant_id=merged.tenant_id or store.tenant,
        generation_id=merged.generation_id or generation.generation_id,
    )
    baseline_ids = {hit.chunk.id for hit in baseline.hits if is_trusted(hit)}
    merged_ids = {hit.chunk.id for hit in merged.hits if is_trusted(hit)}
    return {
        "status": "complete",
        "mode": mode,
        "generation": _query_construction_generation(generation),
        "baseline_retrieval": _query_construction_retrieval(baseline),
        "candidate_queries": [candidate.to_dict() for candidate in graph_candidates],
        "candidate_retrievals": [
            _query_construction_retrieval(result) for result in candidate_results
        ],
        "retrieval": _query_construction_retrieval(merged),
        "new_trusted_chunk_ids": sorted(merged_ids - baseline_ids),
        "diagnostics": {
            "retrieval_calls": 1 + len(candidate_results),
            "model_calls": 0,
            "token_cost": 0,
            "graph": {
                "readiness": "ready"
                if semantic is not None and graph_reason is None
                else "not_ready",
                "reason": graph_reason,
                "entities_inspected": len(semantic.entities) if semantic is not None else 0,
                "mentions_inspected": len(semantic.mentions) if semantic is not None else 0,
                "relations_inspected": len(semantic.relations) if semantic is not None else 0,
                "diagnostics_encountered": len(semantic.diagnostics) if semantic is not None else 0,
                "candidates_discovered": len(graph_candidates),
                "candidates_accepted": len(graph_candidates),
                "candidates_rejected": 0,
                "candidate_retrieval_failures": len(failures),
                "latency_ms": round((time.perf_counter() - graph_started) * 1000.0, 3),
            },
            "new_trusted_items": len(merged_ids - baseline_ids),
            "provider_failures": failures,
        },
    }
