"""Execution boundary for bounded model assisted retrieval expansion.

This module owns expansion provider calls and their fail open result shaping.  It deliberately
does not import the public reasoning module at runtime.  The request type is a protocol input to
the expansion ports, while request validation remains injected by the reasoning entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from recall.evidence import EvidenceBundle, build_evidence_bundle
from recall.reasoning_expansion import (
    ExpansionProposal,
    ExpansionReport,
    ExpansionRequest,
    ReasoningExpansionProvider,
    ReasoningExpansionRetriever,
    RetrievalExpansionTrace,
    evidence_payload,
    merge_trusted_results,
)
from recall.reasoning_proposals import ProviderFailure, ProviderFailureKind
from recall.types import TrustedResult

if TYPE_CHECKING:
    from recall.reasoning import ReasoningRequest


def expansion_failure(
    message: str, kind: ProviderFailureKind = "provider_error"
) -> ProviderFailure:
    """Build the expansion phase's in band failure record."""
    return ProviderFailure(
        kind=kind,
        provider_id="recall.reasoning",
        model_id="unknown",
        provider_revision="unknown",
        message=message,
    )


def expansion_unavailable(
    retrieval: TrustedResult, bundle: EvidenceBundle, reason: str
) -> tuple[
    TrustedResult,
    EvidenceBundle,
    RetrievalExpansionTrace,
    tuple[ProviderFailure, ...],
    int,
]:
    """Return the untouched baseline when an expansion port is missing."""
    return (
        retrieval,
        bundle,
        RetrievalExpansionTrace(attempted=False, rounds=0, fallback_reason=reason),
        (expansion_failure(reason),),
        0,
    )


def run_model_expansion(
    request: ReasoningRequest,
    retrieval: TrustedResult,
    bundle: EvidenceBundle,
    *,
    retriever: ReasoningExpansionRetriever,
    provider: ReasoningExpansionProvider,
    initial_retrieval: TrustedResult,
    initial_bundle: EvidenceBundle,
    initial_ids: set[str],
    depth_result: TrustedResult | None,
    depth_proposal: ExpansionProposal | None,
    depth_accepted: tuple[str, ...],
    executed_queries: list[str],
    rounds_executed: int,
    validate_retrieval_binding: Callable[[ReasoningRequest, TrustedResult], None],
    validation_error: Callable[[str], Exception],
) -> tuple[
    TrustedResult,
    EvidenceBundle,
    RetrievalExpansionTrace,
    tuple[ProviderFailure, ...],
    int,
]:
    """Run model expansion after deterministic depth expansion.

    Returns ``(retrieval, evidence_bundle, trace, provider_failures, calls)``. On provider,
    cardinality, validation, or binding failure, the first two elements remain the initial baseline.
    ``calls`` includes a failed provider call.
    """
    gap_reason = (
        (retrieval.reason or "retrieval_gap")
        if retrieval.gap_warning or retrieval.abstained
        else "assess_evidence_completeness"
    )
    expansion_request = ExpansionRequest(
        query=request.query,
        tenant_id=request.tenant_id,
        generation_id=request.generation.generation_id,
        evidence=evidence_payload(retrieval),
        gap_reason=gap_reason,
    )
    try:
        report = provider(expansion_request)
        if not isinstance(report, ExpansionReport):
            raise TypeError("expansion provider returned a non ExpansionReport value")
    except Exception as exc:  # BROAD-CATCH: fail-open
        return (
            initial_retrieval,
            initial_bundle,
            RetrievalExpansionTrace(
                attempted=True,
                rounds=rounds_executed,
                executed_queries=tuple(executed_queries),
                fallback_reason="provider_failure",
            ),
            (expansion_failure(type(exc).__name__),),
            1,
        )

    failures = tuple(report.provider_failures)
    if failures:
        return (
            initial_retrieval,
            initial_bundle,
            RetrievalExpansionTrace(
                attempted=True,
                rounds=rounds_executed,
                proposals=report.proposals,
                executed_queries=tuple(executed_queries),
                fallback_reason="provider_failure",
            ),
            failures,
            1,
        )
    if len(report.proposals) > expansion_request.max_queries:
        return (
            initial_retrieval,
            initial_bundle,
            RetrievalExpansionTrace(
                attempted=True,
                rounds=rounds_executed,
                fallback_reason="wrong_cardinality",
            ),
            (
                expansion_failure(
                    "too_many_expansion_queries", kind="wrong_cardinality"
                ),
            ),
            1,
        )

    proposals = tuple(
        proposal
        for proposal in report.proposals
        if not (depth_result is not None and proposal.mode == "depth")
    )
    expanded_results: list[TrustedResult] = []
    if proposals:
        rounds_executed += 1
    for proposal in proposals:
        try:
            expanded = retriever(request, proposal, retrieval)
            validate_retrieval_binding(request, expanded)
            if request.policy.require_certified_evidence and expanded.trust_state != "trusted":
                raise validation_error("expanded retrieval is not certified")
        except Exception as exc:  # BROAD-CATCH: fail-open
            return (
                initial_retrieval,
                initial_bundle,
                RetrievalExpansionTrace(
                    attempted=True,
                    rounds=rounds_executed,
                    proposals=report.proposals,
                    executed_queries=tuple(executed_queries),
                    fallback_reason="expanded_retrieval_failure",
                ),
                (expansion_failure(type(exc).__name__),),
                1,
            )
        expanded_results.append(expanded)
        executed_queries.append(proposal.query)

    if not expanded_results and depth_result is not None:
        return (
            retrieval,
            bundle,
            RetrievalExpansionTrace(
                attempted=True,
                rounds=rounds_executed,
                proposals=(depth_proposal,) + report.proposals
                if depth_proposal is not None
                else report.proposals,
                executed_queries=tuple(executed_queries),
                accepted_chunk_ids=depth_accepted,
                fallback_reason="no_expansion_proposal",
            ),
            (),
            1,
        )
    if not expanded_results:
        return (
            initial_retrieval,
            initial_bundle,
            RetrievalExpansionTrace(
                attempted=True,
                rounds=rounds_executed,
                proposals=report.proposals,
                executed_queries=tuple(executed_queries),
                fallback_reason="no_expansion_proposal",
            ),
            (),
            1,
        )

    merged = merge_trusted_results(retrieval, expanded_results, original_query=request.query)
    merged_bundle = build_evidence_bundle(merged, request.evidence_policy)
    accepted = tuple(
        hit.chunk.id
        for result in expanded_results
        for hit in result.hits
        if hit.verdict == "ok" and hit.chunk.id not in initial_ids
    )
    return (
        merged,
        merged_bundle,
        RetrievalExpansionTrace(
            attempted=True,
            rounds=rounds_executed,
            proposals=(depth_proposal,) + report.proposals
            if depth_proposal is not None
            else report.proposals,
            executed_queries=tuple(executed_queries),
            accepted_chunk_ids=accepted,
        ),
        (),
        1,
    )


__all__ = ["expansion_failure", "expansion_unavailable", "run_model_expansion"]
