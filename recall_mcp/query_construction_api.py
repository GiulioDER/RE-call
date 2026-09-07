"""Stateless MCP query construction protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from recall.embeddings import Embedder
from recall.query_construction import (
    MAX_QUERY_CANDIDATES,
    MAX_QUERY_CHARS as MAX_QUERY_CONSTRUCTION_QUERY_CHARS,
    MAX_QUERY_CONSTRUCTION_ROUNDS,
    QueryConstructionArm,
    QueryConstructionRequest,
    QueryProposal,
    RetrievalSignal,
    build_control_proposals,
    build_original_model_challenge,
    parse_query_frame,
    should_request_original_model_refinement,
    validate_query_proposals,
)
from recall.reasoning import (
    GenerationSelection,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    SemanticGraphExpansionResult,
)
from recall.reasoning_expansion import merge_trusted_results
from recall.reasoning_planner import ReasoningBudget
from recall.trust import is_trusted
from recall.trust_policy import TrustPolicy
from recall.types import TrustedResult
from recall_mcp.graph_expansion import _expand_semantic_graph
from recall_mcp.reasoning_common import (
    _query_construction_anchors,
    _query_construction_evidence,
    _query_construction_generation,
    _query_construction_retrieval,
    _reasoning_generation,
    _same_generation,
)
from recall_mcp.retrieval import _retrieve_trusted

if TYPE_CHECKING:
    from recall.calibration import Calibration
    from recall.store import PgVectorStore


# Keep these ceilings in the protocol owner. They bound both prompt size and the graph work that
# a client can cause across multiple stateless continuation calls.
MAX_QUERY_CONSTRUCTION_PROMPT_CHARS = 4_000
MAX_QUERY_CONSTRUCTION_GRAPH_NODES = 128


def _query_construction_graph(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    retrieval: TrustedResult,
    generation: GenerationSelection,
    calibration: Calibration | None,
    graph_expansion: str,
    max_graph_nodes: int,
    *,
    _expand_semantic_graph_fn: Callable[..., SemanticGraphExpansionResult] = _expand_semantic_graph,
) -> tuple[TrustedResult, dict[str, object]]:
    if graph_expansion == "off":
        return retrieval, {
            "readiness": "not_requested",
            "entities_inspected": 0,
            "relations_inspected": 0,
            "candidates_discovered": 0,
            "candidates_rejected": 0,
            "diagnostics_encountered": 0,
            "latency_ms": 0.0,
        }
    graph_request = ReasoningRequest(
        query=query,
        tenant_id=store.tenant,
        generation=generation,
        providers=ReasoningProviderPorts(retriever=lambda _request: retrieval),
        policy=ReasoningPolicy(name="retrieval_only", graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=max_graph_nodes, max_graph_hops=1),
    )
    try:
        expanded = _expand_semantic_graph_fn(
            store, graph_request, retrieval, calibration, embedder
        )
    except Exception as exc:  # BROAD-CATCH: fail-open
        return retrieval, {
            "readiness": "GRAPH_PROVIDER_ERROR",
            "error": type(exc).__name__,
            "entities_inspected": 0,
            "relations_inspected": 0,
            "candidates_discovered": 0,
            "candidates_rejected": 0,
            "diagnostics_encountered": 0,
            "latency_ms": 0.0,
        }
    return expanded.retrieval, {
        "readiness": expanded.readiness,
        "entities_inspected": expanded.entities_inspected,
        "relations_inspected": expanded.relations_inspected,
        "candidates_discovered": expanded.candidates_discovered,
        "candidates_rejected": expanded.candidates_rejected,
        "diagnostics_encountered": expanded.diagnostics_encountered,
        "latency_ms": expanded.latency_ms,
    }


def query_construction_challenge(
    store: PgVectorStore,
    embedder: Embedder,
    original_prompt: str,
    query: str,
    *,
    arm: QueryConstructionArm = "original_loop",
    source: str | None = None,
    k: int = 5,
    round_index: int = 0,
    frame: Mapping[str, object] | None = None,
    expected_generation_id: str | None = None,
    graph_expansion: str = "off",
    max_graph_nodes: int = 32,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    _retrieve_trusted_fn: Callable[..., Any] = _retrieve_trusted,
    _query_construction_graph_fn: Callable[..., tuple[TrustedResult, dict[str, object]]] = _query_construction_graph,
) -> dict[str, object]:
    """Run one stateless phase of original model query construction.

    With no frame, this retrieves the original query and returns a challenge prompt. With a frame,
    it validates the model output, executes the selected bounded controller, and returns either a
    final retrieval result or the next challenge. The original model is always outside this
    service, which keeps the MCP tool deterministic and makes the benchmark replayable.
    """

    if arm not in {"original_loop", "pyramid"}:
        raise ValueError("arm must be 'original_loop' or 'pyramid'")
    if graph_expansion not in {"off", "one_hop"}:
        raise ValueError("graph_expansion must be 'off' or 'one_hop'")
    if not 0 <= round_index < MAX_QUERY_CONSTRUCTION_ROUNDS:
        raise ValueError("round_index must be 0 or 1")
    if not original_prompt.strip():
        raise ValueError("original_prompt must be non-empty")
    if len(original_prompt) > MAX_QUERY_CONSTRUCTION_PROMPT_CHARS:
        raise ValueError("original_prompt is too long")
    if not query.strip():
        raise ValueError("query must be non-empty")
    if len(query) > MAX_QUERY_CONSTRUCTION_QUERY_CHARS:
        raise ValueError("query is too long")
    if not 1 <= max_graph_nodes <= MAX_QUERY_CONSTRUCTION_GRAPH_NODES:
        raise ValueError(
            f"max_graph_nodes must be between 1 and {MAX_QUERY_CONSTRUCTION_GRAPH_NODES}"
        )

    generation = _reasoning_generation(store)
    if expected_generation_id is not None and expected_generation_id != generation.generation_id:
        return {
            "status": "refused",
            "arm": arm,
            "round_index": round_index,
            "refusal_reason": "generation_mismatch",
            "generation": _query_construction_generation(generation),
            "diagnostics": {"retrieval_calls": 0, "challenge_issued": False},
        }

    baseline = _retrieve_trusted_fn(
        store, embedder, query, source, k, calibration, policy
    ).result
    baseline = replace(
        baseline,
        tenant_id=baseline.tenant_id or store.tenant,
        generation_id=baseline.generation_id or generation.generation_id,
    )
    _same_generation(generation, baseline)
    baseline_evidence = _query_construction_evidence(baseline)
    request = QueryConstructionRequest(
        original_prompt=original_prompt,
        original_query=query,
        trusted_evidence=baseline_evidence,
        graph_anchors=_query_construction_anchors(baseline),
        gap_reason=baseline.reason or "retrieval_gap",
        round_index=round_index,
    )

    if frame is None:
        challenge = build_original_model_challenge(request)
        return {
            "status": "challenge",
            "arm": arm,
            "round_index": round_index,
            "challenge_prompt": challenge.prompt,
            "frame_schema": [
                "task_object",
                "intended_action",
                "failure_or_risk",
                "memory_need",
                "artifacts",
                "query",
                "need_more",
            ],
            "generation": _query_construction_generation(generation),
            "retrieval": _query_construction_retrieval(baseline),
            "diagnostics": {
                "retrieval_calls": 1,
                "challenge_issued": True,
                "candidate_count": 0,
                "accepted_candidate_count": 0,
                "rejected_candidate_count": 0,
                "original_model_calls": 1,
                "graph": {"readiness": "deferred_until_trusted_seed"},
            },
        }

    try:
        parsed_frame = parse_query_frame(frame)
    except (TypeError, ValueError) as exc:
        return {
            "status": "fallback",
            "arm": arm,
            "round_index": round_index,
            "refusal_reason": "invalid_frame",
            "error": str(exc),
            "generation": _query_construction_generation(generation),
            "retrieval": _query_construction_retrieval(baseline),
            "diagnostics": {
                "retrieval_calls": 1,
                "challenge_issued": False,
                "original_model_calls": 1,
            },
        }

    proposals: tuple[QueryProposal, ...]
    if arm == "original_loop":
        proposals = (
            QueryProposal(
                parsed_frame.query,
                "literal",
                "original model refinement",
                tuple(
                    str(item["chunk_id"])
                    for item in baseline_evidence
                    if item.get("verdict") == "ok"
                ),
            ),
        )
    else:
        proposals = build_control_proposals(
            parsed_frame,
            original_query=query,
            trusted_evidence=baseline_evidence,
        )
    validation = validate_query_proposals(
        QueryConstructionRequest(
            original_prompt=original_prompt,
            original_query=query,
            trusted_evidence=baseline_evidence,
            graph_anchors=_query_construction_anchors(baseline),
            gap_reason=baseline.reason or "retrieval_gap",
            round_index=round_index,
            max_candidates=MAX_QUERY_CANDIDATES,
        ),
        proposals,
    )

    expanded_results: list[TrustedResult] = []
    failures: list[str] = []
    for proposal in validation.accepted:
        try:
            candidate = _retrieve_trusted_fn(
                store, embedder, proposal.query, source, k, calibration, policy
            ).result
            candidate = replace(
                candidate,
                tenant_id=candidate.tenant_id or store.tenant,
                generation_id=candidate.generation_id or generation.generation_id,
            )
            _same_generation(generation, candidate)
            expanded_results.append(candidate)
        except Exception as exc:  # BROAD-CATCH: fail-open
            failures.append(type(exc).__name__)

    merged = merge_trusted_results(baseline, expanded_results, original_query=query)
    merged = replace(
        merged,
        tenant_id=merged.tenant_id or store.tenant,
        generation_id=merged.generation_id or generation.generation_id,
    )
    baseline_ids = {hit.chunk.id for hit in baseline.hits if is_trusted(hit)}
    merged_ids = {hit.chunk.id for hit in merged.hits if is_trusted(hit)}
    new_ids = tuple(sorted(merged_ids - baseline_ids))
    if new_ids:
        graph_result, graph_diagnostics = _query_construction_graph_fn(
            store,
            embedder,
            parsed_frame.query,
            merged,
            generation,
            calibration,
            graph_expansion,
            max_graph_nodes,
        )
    else:
        graph_result = merged
        graph_diagnostics = {
            "readiness": "deferred_until_trusted_seed",
            "entities_inspected": 0,
            "relations_inspected": 0,
            "candidates_discovered": 0,
            "candidates_rejected": 0,
            "diagnostics_encountered": 0,
            "latency_ms": 0.0,
        }
    signal = RetrievalSignal(
        trusted_items=len([hit for hit in graph_result.hits if is_trusted(hit)]),
        new_trusted_items=len(new_ids),
        gap_warning=graph_result.gap_warning or graph_result.abstained,
        agent_says_need_more=parsed_frame.need_more,
    )
    needs_followup = should_request_original_model_refinement(
        signal, round_index=round_index
    )
    response: dict[str, object] = {
        "status": "challenge" if needs_followup else "complete",
        "arm": arm,
        "round_index": round_index,
        "frame": {
            "task_object": parsed_frame.task_object,
            "intended_action": parsed_frame.intended_action,
            "failure_or_risk": parsed_frame.failure_or_risk,
            "memory_need": parsed_frame.memory_need,
            "artifacts": list(parsed_frame.artifacts),
            "query": parsed_frame.query,
            "need_more": parsed_frame.need_more,
        },
        "generation": _query_construction_generation(generation),
        "retrieval": _query_construction_retrieval(graph_result),
        "new_trusted_chunk_ids": list(new_ids),
        "accepted_candidates": [
            {
                "query": proposal.query,
                "kind": proposal.kind,
                "rationale": proposal.rationale,
                "parent_chunk_ids": list(proposal.parent_chunk_ids),
            }
            for proposal in validation.accepted
        ],
        "rejected_candidates": [
            {"query": proposal.query, "kind": proposal.kind, "reason": reason}
            for proposal, reason in validation.rejected
        ],
        "diagnostics": {
            "retrieval_calls": 1 + len(expanded_results),
            "challenge_issued": needs_followup,
            "candidate_count": len(proposals),
            "accepted_candidate_count": len(validation.accepted),
            "rejected_candidate_count": len(validation.rejected),
            "new_trusted_items": len(new_ids),
            "original_model_calls": 1 + (1 if needs_followup else 0),
            "provider_failures": failures,
            "graph": graph_diagnostics,
        },
    }
    if needs_followup:
        followup_request = QueryConstructionRequest(
            original_prompt=original_prompt,
            original_query=parsed_frame.query,
            trusted_evidence=_query_construction_evidence(graph_result),
            graph_anchors=_query_construction_anchors(graph_result),
            gap_reason=graph_result.reason or "retrieval_gap",
            round_index=round_index + 1,
        )
        response["next_challenge_prompt"] = build_original_model_challenge(
            followup_request
        ).prompt
        response["next_round_index"] = round_index + 1
    return response


__all__ = [
    "MAX_QUERY_CANDIDATES",
    "MAX_QUERY_CONSTRUCTION_GRAPH_NODES",
    "MAX_QUERY_CONSTRUCTION_PROMPT_CHARS",
    "MAX_QUERY_CONSTRUCTION_QUERY_CHARS",
    "MAX_QUERY_CONSTRUCTION_ROUNDS",
    "QueryConstructionArm",
    "QueryConstructionRequest",
    "QueryProposal",
    "RetrievalSignal",
    "build_control_proposals",
    "build_original_model_challenge",
    "parse_query_frame",
    "query_construction_challenge",
    "should_request_original_model_refinement",
    "validate_query_proposals",
]
