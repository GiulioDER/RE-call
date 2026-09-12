"""Execution boundary for optional semantic graph expansion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING

from recall.reasoning_proposals import ProviderFailure
from recall.types import ScoredChunk, TrustedResult

if TYPE_CHECKING:
    from recall.reasoning import ReasoningRequest


@dataclass(frozen=True)
class SemanticGraphExpansionResult:
    """The bounded result of one semantic graph expansion.

    ``scored_candidates`` is populated only by the graph first serving adapter. Those candidates
    are deliberately still untrusted: the adapter protects the direct prefix and assembles the
    final context before the single trust evaluation at the retrieval boundary.
    """

    retrieval: TrustedResult
    readiness: str
    entities_inspected: int = 0
    relations_inspected: int = 0
    candidates_discovered: int = 0
    candidates_rejected: int = 0
    relation_seed_activations: Mapping[str, int] = dataclass_field(default_factory=dict)
    relation_candidates_accepted: Mapping[str, int] = dataclass_field(default_factory=dict)
    relation_new_trusted_evidence: Mapping[str, int] = dataclass_field(default_factory=dict)
    diagnostics_encountered: int = 0
    latency_ms: float = 0.0
    #: Pairs of (rejection reason, count) for individual CANDIDATES the expansion's admission
    #: policy refused. The reason vocabulary belongs to the graph expansion provider's admission
    #: policy (for example "hub_entity" or "cosine_admission"); the pairs are surfaced
    #: verbatim as :attr:`ReasoningDiagnostics.graph_admission_rejections`.
    admission_rejections: tuple[tuple[str, int], ...] = ()
    #: Pairs of (reason, count) for a refusal of the WHOLE expansion, before any candidate is
    #: discovered. Counted rather than made a bare flag so the shape matches
    #: :attr:`admission_rejections` and a sweep can aggregate both the same way.
    expansion_refusals: tuple[tuple[str, int], ...] = ()
    gate_reason: str | None = None
    policy_fingerprint: str | None = None
    scored_candidates: tuple[ScoredChunk, ...] = ()


def run_graph_expansion(
    request: ReasoningRequest,
    retrieval: TrustedResult,
    *,
    provider: Callable[[ReasoningRequest, TrustedResult], SemanticGraphExpansionResult] | None,
    validate_retrieval_binding: Callable[[ReasoningRequest, TrustedResult], None],
) -> tuple[
    TrustedResult,
    SemanticGraphExpansionResult | None,
    ProviderFailure | None,
    str | None,
]:
    """Execute one optional graph expansion and preserve the validated seed on refusal.

    Returns ``(retrieval, expansion, provider_failure, refusal_reason)``. Provider exceptions,
    invalid return values, and binding failures become explicit provider failures instead of
    escaping the reasoning boundary.
    """
    if request.policy.graph_expansion != "one_hop":
        return retrieval, None, None, None
    if provider is None:
        return (
            retrieval,
            SemanticGraphExpansionResult(retrieval=retrieval, readiness="GRAPH_NOT_READY"),
            None,
            "GRAPH_NOT_READY",
        )

    graph_failure: ProviderFailure | None = None
    try:
        graph_expansion = provider(request, retrieval)
    except TimeoutError as exc:
        graph_expansion = SemanticGraphExpansionResult(
            retrieval=retrieval,
            readiness="GRAPH_PROVIDER_TIMEOUT",
        )
        graph_failure = ProviderFailure(
            kind="timeout",
            provider_id="semantic-graph",
            model_id="deterministic",
            provider_revision="v1",
            message=type(exc).__name__,
        )
    except Exception as exc:  # BROAD-CATCH: fail-open
        graph_expansion = SemanticGraphExpansionResult(
            retrieval=retrieval,
            readiness="GRAPH_PROVIDER_ERROR",
        )
        graph_failure = ProviderFailure(
            kind="provider_error",
            provider_id="semantic-graph",
            model_id="deterministic",
            provider_revision="v1",
            message=type(exc).__name__,
        )

    if not isinstance(graph_expansion, SemanticGraphExpansionResult):
        graph_expansion = SemanticGraphExpansionResult(
            retrieval=retrieval,
            readiness="GRAPH_PROVIDER_ERROR",
        )
        graph_failure = ProviderFailure(
            kind="provider_error",
            provider_id="semantic-graph",
            model_id="deterministic",
            provider_revision="v1",
            message="TypeError",
        )
    if graph_expansion.readiness == "ready":
        try:
            validate_retrieval_binding(request, graph_expansion.retrieval)
        except ValueError as exc:
            # A provider returning evidence bound to another tenant, generation, or corpus is a
            # misbehaving provider, not a caller error. Keep the validated seed retrieval.
            graph_expansion = SemanticGraphExpansionResult(
                retrieval=retrieval,
                readiness="GRAPH_PROVIDER_ERROR",
            )
            graph_failure = ProviderFailure(
                kind="provider_error",
                provider_id="semantic-graph",
                model_id="deterministic",
                provider_revision="v1",
                message=type(exc).__name__,
            )
    if graph_expansion.readiness != "ready":
        return retrieval, graph_expansion, graph_failure, graph_expansion.readiness
    return graph_expansion.retrieval, graph_expansion, graph_failure, None


__all__ = ["SemanticGraphExpansionResult", "run_graph_expansion"]
