"""Public provider neutral reasoning API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
import math
from typing import Any, Literal, Protocol, cast

from recall.evidence import (
    AnswerEnvelope,
    EvidenceBundle,
    EvidencePolicy,
    EvidenceValidationError,
    build_evidence_bundle,
    normalize_citations,
    parse_answer_envelope,
    render_evidence_prompt,
    validate_answer,
)
from recall.reasoning_graph import ReasoningGraphProjection
from recall.reasoning_planner import (
    EvidenceDecision,
    ExpansionStep,
    InferenceProposalTrace,
    PlannerInitialRetrieval,
    ReasoningBudget,
    ReasoningBudgetUsage,
    ReasoningPlan,
    ReasoningTrace,
    UnresolvedGap,
    plan_multi_hop_evidence,
)
from recall.reasoning_proposals import InferenceProposal, ProposalProtocolReport
from recall.types import TrustedResult
from recall.trust import is_trusted

REASONING_API_VERSION = 1

ReasoningPolicyName = Literal[
    "retrieval_only",
    "evidence_assembly",
    "proposal_assisted",
    "review_required",
]
ReasoningOutcome = Literal["answered", "abstained", "needs_clarification", "needs_review"]


class ReasoningValidationError(ValueError):
    """The reasoning request crossed a trust, tenant, generation, or provider boundary."""


class ReasoningRetriever(Protocol):
    def __call__(self, request: "ReasoningRequest") -> TrustedResult:
        ...


class ReasoningGraphProvider(Protocol):
    def __call__(
        self, request: "ReasoningRequest", retrieval: TrustedResult
    ) -> ReasoningGraphProjection:
        ...


class ReasoningProposalProvider(Protocol):
    def __call__(
        self,
        request: "ReasoningRequest",
        graph: ReasoningGraphProjection,
        retrieval: TrustedResult,
    ) -> Sequence[InferenceProposal] | ProposalProtocolReport:
        ...


ReasoningAnswerProvider = Callable[[str, str], str | dict[str, object] | AnswerEnvelope]


@dataclass(frozen=True)
class GenerationSelection:
    """Generation identity the reasoning run is allowed to use."""

    generation_id: str | None = None
    pipeline_fingerprint: str | None = None
    corpus_fingerprint: str | None = None


@dataclass(frozen=True)
class ReasoningPolicy:
    """Explicit policy for how far the public reasoning API may go."""

    name: ReasoningPolicyName = "evidence_assembly"
    require_certified_evidence: bool = True
    allow_proposal_guided_expansion: bool = False
    require_human_review_on_proposals: bool = False

    def __post_init__(self) -> None:
        if self.name == "proposal_assisted" and not self.allow_proposal_guided_expansion:
            object.__setattr__(self, "allow_proposal_guided_expansion", True)
        if self.name == "review_required":
            object.__setattr__(self, "allow_proposal_guided_expansion", True)
            object.__setattr__(self, "require_human_review_on_proposals", True)


@dataclass(frozen=True)
class ReasoningProviderPorts:
    """Provider ports consumed by :func:`reason`."""

    retriever: ReasoningRetriever
    graph_provider: ReasoningGraphProvider | None = None
    proposal_provider: ReasoningProposalProvider | None = None
    answer_provider: ReasoningAnswerProvider | None = None


@dataclass(frozen=True)
class ReasoningRequest:
    """Typed public request for one reasoning run."""

    query: str
    tenant_id: str
    generation: GenerationSelection
    providers: ReasoningProviderPorts
    policy: ReasoningPolicy = ReasoningPolicy()
    budget: ReasoningBudget = ReasoningBudget()
    evidence_policy: EvidencePolicy = EvidencePolicy()
    known_as_of: datetime | None = None

    @property
    def generation_id(self) -> str | None:
        return self.generation.generation_id


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source: str
    ordinal: int | None = None


@dataclass(frozen=True)
class Contradiction:
    proposal_id: str
    subject_id: str
    object_id: str
    evidence_ids: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class ReasoningDiagnostics:
    latency_ms: int
    budget: ReasoningBudget
    budget_used: ReasoningBudgetUsage | None
    retrieval_stage_ms: Mapping[str, float]
    generator_invoked: bool
    citations_normalized: bool


@dataclass(frozen=True)
class ReasoningResponse:
    """Typed public response for one reasoning run."""

    schema_version: int
    outcome: ReasoningOutcome
    answer: str | None
    clarification_request: str | None
    trusted_evidence: EvidenceBundle
    inference_proposals: tuple[InferenceProposal, ...]
    reasoning_trace: ReasoningTrace | None
    contradictions: tuple[Contradiction, ...]
    unsupported_gaps: tuple[UnresolvedGap, ...]
    citations: tuple[Citation, ...]
    calibration_id: str | None
    calibration_status: str
    tenant_id: str | None
    generation_id: str | None
    pipeline_fingerprint: str | None
    corpus_fingerprint: str | None
    query_set_digest: str | None
    trust_state: str
    refusal_reason: str | None
    diagnostics: ReasoningDiagnostics

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _to_json_value(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReasoningResponse":
        return reasoning_response_from_dict(payload)


def reason(request: ReasoningRequest) -> ReasoningResponse:
    """Run retrieval, optional graph planning, and optional answer generation.

    Existing retrieval callers are not part of this path and see no signature or result change.
    The entry point consumes provider ports so a CLI, MCP server, SDK, or test harness can supply
    its own retrieval, graph, proposal, and answer implementations while sharing the same trust
    and citation checks.
    """
    started = datetime.now().timestamp()
    if not request.query.strip():
        empty_bundle = _empty_bundle(request)
        return _response(
            request=request,
            retrieval=None,
            bundle=empty_bundle,
            outcome="needs_clarification",
            answer=None,
            clarification_request="Please provide a non-empty query.",
            proposals=(),
            plan=None,
            refusal_reason="empty_query",
            generator_invoked=False,
            citations_normalized=False,
            started=started,
        )

    retrieval = request.providers.retriever(request)
    _validate_retrieval_binding(request, retrieval)
    bundle = build_evidence_bundle(retrieval, request.evidence_policy)

    if request.policy.require_certified_evidence and bundle.trust_state != "trusted":
        return _response(
            request=request,
            retrieval=retrieval,
            bundle=bundle,
            outcome="abstained",
            answer=None,
            proposals=(),
            plan=None,
            refusal_reason="uncertified_evidence",
            generator_invoked=False,
            citations_normalized=False,
            started=started,
        )

    if request.policy.name == "retrieval_only":
        return _response(
            request=request,
            retrieval=retrieval,
            bundle=bundle,
            outcome="abstained",
            answer=None,
            proposals=(),
            plan=None,
            refusal_reason="retrieval_only_policy",
            generator_invoked=False,
            citations_normalized=False,
            started=started,
        )

    graph: ReasoningGraphProjection | None = None
    proposals: tuple[InferenceProposal, ...] = ()
    plan: ReasoningPlan | None = None
    if request.policy.allow_proposal_guided_expansion:
        if request.providers.graph_provider is None:
            raise ReasoningValidationError("proposal assisted reasoning requires graph_provider")
        graph = request.providers.graph_provider(request, retrieval)
        _validate_graph_binding(request, retrieval, graph)
        proposals = _proposal_tuple(request, graph, retrieval)
        _validate_proposals(graph, proposals)
        plan = plan_multi_hop_evidence(
            retrieval,
            graph,
            proposals=proposals,
            budget=request.budget,
        )
        if plan.outcome == "failed_closed":
            outcome: ReasoningOutcome = (
                "needs_review" if plan.stop_reason == "ambiguous_evidence" else "abstained"
            )
            return _response(
                request=request,
                retrieval=retrieval,
                bundle=bundle,
                outcome=outcome,
                answer=None,
                proposals=proposals,
                plan=plan,
                refusal_reason=plan.stop_reason,
                generator_invoked=False,
                citations_normalized=False,
                started=started,
            )
        if request.policy.require_human_review_on_proposals and proposals:
            return _response(
                request=request,
                retrieval=retrieval,
                bundle=bundle,
                outcome="needs_review",
                answer=None,
                proposals=proposals,
                plan=plan,
                refusal_reason="review_required_policy",
                generator_invoked=False,
                citations_normalized=False,
                started=started,
            )

    if bundle.decision == "abstain":
        return _response(
            request=request,
            retrieval=retrieval,
            bundle=bundle,
            outcome="abstained",
            answer=None,
            proposals=proposals,
            plan=plan,
            refusal_reason=bundle.reason_code,
            generator_invoked=False,
            citations_normalized=False,
            started=started,
        )
    if request.providers.answer_provider is None:
        return _response(
            request=request,
            retrieval=retrieval,
            bundle=bundle,
            outcome="abstained",
            answer=None,
            proposals=proposals,
            plan=plan,
            refusal_reason="no_answer_provider",
            generator_invoked=False,
            citations_normalized=False,
            started=started,
        )

    system, user = render_evidence_prompt(bundle)
    raw = parse_answer_envelope(request.providers.answer_provider(system, user))
    envelope = normalize_citations(raw)
    validation = validate_answer(envelope, bundle)
    if not validation.valid:
        raise EvidenceValidationError("; ".join(validation.errors))
    citations_normalized = envelope.citations != raw.citations
    if envelope.insufficient_evidence:
        return _response(
            request=request,
            retrieval=retrieval,
            bundle=bundle,
            outcome="abstained",
            answer=None,
            proposals=proposals,
            plan=plan,
            refusal_reason="provider_abstained",
            generator_invoked=True,
            citations_normalized=citations_normalized,
            started=started,
        )
    return _response(
        request=request,
        retrieval=retrieval,
        bundle=bundle,
        outcome="answered",
        answer=envelope.answer,
        proposals=proposals,
        plan=plan,
        citations=envelope.citations,
        generator_invoked=True,
        citations_normalized=citations_normalized,
        started=started,
    )


def reasoning_response_from_dict(payload: Mapping[str, object]) -> ReasoningResponse:
    """Deserialize the JSON form emitted by :meth:`ReasoningResponse.to_dict`."""
    bundle = _evidence_bundle_from_dict(_mapping(payload["trusted_evidence"]))
    trust_state = _trust_state(payload["trust_state"])
    if bundle.trust_state != trust_state:
        raise EvidenceValidationError("trust_state mismatch between response and trusted_evidence")
    proposals = tuple(
        _proposal_from_dict(_mapping(item)) for item in _sequence(payload["inference_proposals"])
    )
    contradictions = tuple(
        Contradiction(
            proposal_id=str(item["proposal_id"]),
            subject_id=str(item["subject_id"]),
            object_id=str(item["object_id"]),
            evidence_ids=tuple(str(value) for value in _sequence(item["evidence_ids"])),
            explanation=str(item["explanation"]),
        )
        for item in (_mapping(value) for value in _sequence(payload["contradictions"]))
    )
    citations = tuple(
        Citation(
            chunk_id=str(item["chunk_id"]),
            source=str(item["source"]),
            ordinal=_optional_int(item.get("ordinal")),
        )
        for item in (_mapping(value) for value in _sequence(payload["citations"]))
    )
    diagnostics_payload = _mapping(payload["diagnostics"])
    diagnostics = ReasoningDiagnostics(
        latency_ms=_required_int(diagnostics_payload["latency_ms"]),
        budget=_budget_from_dict(_mapping(diagnostics_payload["budget"])),
        budget_used=_optional_budget_usage_from_dict(diagnostics_payload.get("budget_used")),
        retrieval_stage_ms={
            key: _required_float(value)
            for key, value in _mapping(diagnostics_payload["retrieval_stage_ms"]).items()
        },
        generator_invoked=_required_bool(diagnostics_payload["generator_invoked"]),
        citations_normalized=_required_bool(diagnostics_payload["citations_normalized"]),
    )
    return ReasoningResponse(
        schema_version=_required_int(payload["schema_version"]),
        outcome=cast(ReasoningOutcome, payload["outcome"]),
        answer=_optional_str(payload.get("answer")),
        clarification_request=_optional_str(payload.get("clarification_request")),
        trusted_evidence=bundle,
        inference_proposals=proposals,
        reasoning_trace=_optional_trace_from_dict(payload.get("reasoning_trace")),
        contradictions=contradictions,
        unsupported_gaps=tuple(
            _gap_from_dict(_mapping(item)) for item in _sequence(payload["unsupported_gaps"])
        ),
        citations=citations,
        calibration_id=_optional_str(payload.get("calibration_id")),
        calibration_status=str(payload["calibration_status"]),
        tenant_id=_optional_str(payload.get("tenant_id")),
        generation_id=_optional_str(payload.get("generation_id")),
        pipeline_fingerprint=_optional_str(payload.get("pipeline_fingerprint")),
        corpus_fingerprint=_optional_str(payload.get("corpus_fingerprint")),
        query_set_digest=_optional_str(payload.get("query_set_digest")),
        trust_state=trust_state,
        refusal_reason=_optional_str(payload.get("refusal_reason")),
        diagnostics=diagnostics,
    )


def _validate_retrieval_binding(request: ReasoningRequest, retrieval: TrustedResult) -> None:
    checks = (
        ("tenant_id", request.tenant_id, retrieval.tenant_id),
        ("generation_id", request.generation.generation_id, retrieval.generation_id),
        (
            "pipeline_fingerprint",
            request.generation.pipeline_fingerprint,
            retrieval.pipeline_fingerprint,
        ),
        ("corpus_fingerprint", request.generation.corpus_fingerprint, retrieval.corpus_fingerprint),
    )
    for name, expected, actual in checks:
        if expected is not None and actual != expected:
            raise ReasoningValidationError(f"retrieval {name} does not match request {name}")
    if request.policy.require_certified_evidence and retrieval.trust_state != "trusted":
        return
    trusted = [hit for hit in retrieval.hits if is_trusted(hit)]
    if not retrieval.abstained and not trusted:
        raise ReasoningValidationError("non-abstained retrieval contains no trusted evidence")


def _validate_graph_binding(
    request: ReasoningRequest, retrieval: TrustedResult, graph: ReasoningGraphProjection
) -> None:
    checks = (
        ("tenant_id", request.tenant_id, graph.tenant_id),
        ("generation_id", request.generation.generation_id or retrieval.generation_id, graph.generation_id),
        (
            "pipeline_fingerprint",
            request.generation.pipeline_fingerprint or retrieval.pipeline_fingerprint,
            graph.pipeline_fingerprint,
        ),
        (
            "corpus_fingerprint",
            request.generation.corpus_fingerprint or retrieval.corpus_fingerprint,
            graph.corpus_fingerprint,
        ),
    )
    for name, expected, actual in checks:
        if expected is not None and actual != expected:
            raise ReasoningValidationError(f"reasoning graph {name} does not match request")
    _validate_graph_members(graph)


def _proposal_tuple(
    request: ReasoningRequest,
    graph: ReasoningGraphProjection,
    retrieval: TrustedResult,
) -> tuple[InferenceProposal, ...]:
    provider = request.providers.proposal_provider
    if provider is None:
        return ()
    raw = provider(request, graph, retrieval)
    if isinstance(raw, ProposalProtocolReport):
        return raw.proposals
    return tuple(raw)


def _validate_proposals(
    graph: ReasoningGraphProjection, proposals: Sequence[InferenceProposal]
) -> None:
    evidence_ids = {node.id for node in graph.nodes}
    for proposal in proposals:
        if proposal.generation_id != graph.generation_id:
            raise ReasoningValidationError("proposal generation_id does not match graph")
        missing = sorted(set(proposal.source_evidence_ids) - evidence_ids)
        if missing:
            raise ReasoningValidationError(
                "proposal cites evidence outside the reasoning graph: " + ", ".join(missing)
            )


def _validate_graph_members(graph: ReasoningGraphProjection) -> None:
    for node in graph.nodes:
        _check_member_identity("node", node.id, node.tenant_id, node.generation_id, graph)
    for edge in (*graph.authored_edges, *graph.inferred_candidate_edges):
        _check_member_identity("edge", edge.id, edge.tenant_id, edge.generation_id, graph)
    for diagnostic in graph.diagnostics:
        _check_member_identity(
            "diagnostic",
            diagnostic.id,
            diagnostic.tenant_id,
            diagnostic.generation_id,
            graph,
        )


def _check_member_identity(
    kind: str,
    member_id: str,
    tenant_id: str,
    generation_id: str,
    graph: ReasoningGraphProjection,
) -> None:
    if tenant_id != graph.tenant_id:
        raise ReasoningValidationError(f"{kind} {member_id} tenant_id does not match graph")
    if generation_id != graph.generation_id:
        raise ReasoningValidationError(f"{kind} {member_id} generation_id does not match graph")


def _response(
    *,
    request: ReasoningRequest,
    retrieval: TrustedResult | None,
    bundle: EvidenceBundle,
    outcome: ReasoningOutcome,
    answer: str | None,
    proposals: Sequence[InferenceProposal],
    plan: ReasoningPlan | None,
    started: float,
    clarification_request: str | None = None,
    citations: Sequence[str] = (),
    refusal_reason: str | None = None,
    generator_invoked: bool,
    citations_normalized: bool,
) -> ReasoningResponse:
    cited = _citations(bundle, citations)
    contradictions = tuple(
        Contradiction(
            proposal_id=proposal.id,
            subject_id=proposal.subject_id,
            object_id=proposal.object_id,
            evidence_ids=proposal.source_evidence_ids,
            explanation=proposal.explanation,
        )
        for proposal in proposals
        if proposal.proposed_relation == "contradicts"
    )
    return ReasoningResponse(
        schema_version=REASONING_API_VERSION,
        outcome=outcome,
        answer=answer,
        clarification_request=clarification_request,
        trusted_evidence=bundle,
        inference_proposals=tuple(proposals),
        reasoning_trace=plan.trace if plan is not None else None,
        contradictions=contradictions,
        unsupported_gaps=plan.trace.unresolved_gaps if plan is not None else (),
        citations=cited,
        calibration_id=retrieval.calibration_id if retrieval is not None else None,
        calibration_status=retrieval.calibration_status if retrieval is not None else "missing",
        tenant_id=retrieval.tenant_id if retrieval is not None else request.tenant_id,
        generation_id=retrieval.generation_id if retrieval is not None else request.generation_id,
        pipeline_fingerprint=retrieval.pipeline_fingerprint if retrieval is not None else None,
        corpus_fingerprint=retrieval.corpus_fingerprint if retrieval is not None else None,
        query_set_digest=retrieval.query_set_digest if retrieval is not None else None,
        trust_state=bundle.trust_state,
        refusal_reason=refusal_reason,
        diagnostics=ReasoningDiagnostics(
            latency_ms=max(0, int((datetime.now().timestamp() - started) * 1000)),
            budget=request.budget,
            budget_used=plan.budget_used if plan is not None else None,
            retrieval_stage_ms=bundle_stage_ms(retrieval),
            generator_invoked=generator_invoked,
            citations_normalized=citations_normalized,
        ),
    )


def bundle_stage_ms(retrieval: TrustedResult | None) -> Mapping[str, float]:
    if retrieval is None:
        return {}
    return retrieval.diagnostics.stage_ms


def _citations(bundle: EvidenceBundle, citation_ids: Sequence[str]) -> tuple[Citation, ...]:
    items = {item.chunk_id: item for item in bundle.items}
    citations: list[Citation] = []
    for chunk_id in citation_ids:
        item = items.get(chunk_id)
        if item is None:
            raise ReasoningValidationError(f"citation {chunk_id} is not trusted retrieved evidence")
        citations.append(Citation(chunk_id=item.chunk_id, source=item.source, ordinal=item.ordinal))
    return tuple(citations)


def _empty_bundle(request: ReasoningRequest) -> EvidenceBundle:
    generation = request.generation
    return EvidenceBundle(
        query=request.query,
        decision="abstain",
        reason_code="needs_clarification",
        calibrated=False,
        stale=False,
        embedding_profile="unknown",
        retrieval_profile="unknown",
        index_generation=generation.generation_id or "unknown",
        items=(),
    )


def _to_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_value(item) for item in value]
    return value


def _evidence_bundle_from_dict(payload: Mapping[str, object]) -> EvidenceBundle:
    from recall.evidence import EvidenceItem

    if "trust_state" not in payload:
        raise EvidenceValidationError("trusted_evidence trust_state is required")
    trust_state = _trust_state(payload["trust_state"])
    items = tuple(
        EvidenceItem(
            chunk_id=str(item["chunk_id"]),
            text=str(item["text"]),
            source=str(item["source"]),
            ordinal=_optional_int(item.get("ordinal")),
            indexed_at=_optional_datetime(item.get("indexed_at")),
            valid_from=_optional_datetime(item.get("valid_from")),
            valid_until=_optional_datetime(item.get("valid_until")),
            cosine=_required_float(item["cosine"]),
            confidence=_required_float(item["confidence"]),
        )
        for item in (_mapping(value) for value in _sequence(payload["items"]))
    )
    return EvidenceBundle(
        query=str(payload["query"]),
        decision=cast(Literal["answer", "abstain"], payload["decision"]),
        reason_code=_optional_str(payload.get("reason_code")),
        calibrated=_required_bool(payload["calibrated"]),
        stale=_required_bool(payload["stale"]),
        embedding_profile=str(payload["embedding_profile"]),
        retrieval_profile=str(payload["retrieval_profile"]),
        index_generation=str(payload["index_generation"]),
        items=items,
        trust_state=trust_state,
        failure_code=_optional_str(payload.get("failure_code")),
    )


def _proposal_from_dict(payload: Mapping[str, object]) -> InferenceProposal:
    return InferenceProposal(
        id=str(payload["id"]),
        source_evidence_ids=tuple(str(item) for item in _sequence(payload["source_evidence_ids"])),
        proposed_relation=cast(Any, payload["proposed_relation"]),
        subject_id=str(payload["subject_id"]),
        object_id=str(payload["object_id"]),
        explanation=str(payload["explanation"]),
        model_id=str(payload["model_id"]),
        pipeline_id=str(payload["pipeline_id"]),
        provider_id=str(payload["provider_id"]),
        provider_revision=str(payload["provider_revision"]),
        confidence=_optional_float(payload.get("confidence")),
        uncertainty=tuple(str(item) for item in _sequence(payload["uncertainty"])),
        generation_id=str(payload["generation_id"]),
        status=cast(Any, payload.get("status", "candidate")),
        rule_id=_optional_str(payload.get("rule_id")),
        metadata=_mapping(payload.get("metadata", {})),
    )


def _budget_from_dict(payload: Mapping[str, object]) -> ReasoningBudget:
    return ReasoningBudget(
        max_steps=_required_int(payload["max_steps"]),
        max_graph_nodes=_required_int(payload["max_graph_nodes"]),
        max_model_calls=_required_int(payload["max_model_calls"]),
        max_evidence_tokens=_required_int(payload["max_evidence_tokens"]),
        max_wall_time_ms=_required_int(payload["max_wall_time_ms"]),
    )


def _optional_budget_usage_from_dict(value: object) -> ReasoningBudgetUsage | None:
    if value is None:
        return None
    payload = _mapping(value)
    return ReasoningBudgetUsage(
        steps=_required_int(payload["steps"]),
        graph_nodes=_required_int(payload["graph_nodes"]),
        model_calls=_required_int(payload["model_calls"]),
        evidence_tokens=_required_int(payload["evidence_tokens"]),
        wall_time_ms=_required_int(payload["wall_time_ms"]),
    )


def _gap_from_dict(payload: Mapping[str, object]) -> UnresolvedGap:
    return UnresolvedGap(
        id=str(payload["id"]),
        kind=str(payload["kind"]),
        node_ids=tuple(str(item) for item in _sequence(payload.get("node_ids", ()))),
        proposal_ids=tuple(str(item) for item in _sequence(payload.get("proposal_ids", ()))),
        reason=str(payload.get("reason", "")),
    )


def _optional_trace_from_dict(value: object) -> ReasoningTrace | None:
    if value is None:
        return None
    payload = _mapping(value)
    return ReasoningTrace(
        initial_retrieval=_initial_retrieval_from_dict(
            _mapping(payload["initial_retrieval"])
        ),
        expansion_steps=tuple(
            _expansion_step_from_dict(_mapping(item))
            for item in _sequence(payload["expansion_steps"])
        ),
        evidence_accepted=tuple(
            _evidence_decision_from_dict(_mapping(item))
            for item in _sequence(payload["evidence_accepted"])
        ),
        evidence_rejected=tuple(
            _evidence_decision_from_dict(_mapping(item))
            for item in _sequence(payload["evidence_rejected"])
        ),
        unresolved_gaps=tuple(
            _gap_from_dict(_mapping(item)) for item in _sequence(payload["unresolved_gaps"])
        ),
        inference_proposals_used_for_exploration=tuple(
            _proposal_trace_from_dict(_mapping(item))
            for item in _sequence(payload["inference_proposals_used_for_exploration"])
        ),
    )


def _initial_retrieval_from_dict(payload: Mapping[str, object]) -> PlannerInitialRetrieval:
    return PlannerInitialRetrieval(
        query=str(payload["query"]),
        generation_id=_optional_str(payload.get("generation_id")),
        trusted_hit_ids=tuple(str(item) for item in _sequence(payload["trusted_hit_ids"])),
        rejected_hit_ids=tuple(str(item) for item in _sequence(payload["rejected_hit_ids"])),
        abstained=_required_bool(payload["abstained"]),
        reason=str(payload["reason"]),
    )


def _expansion_step_from_dict(payload: Mapping[str, object]) -> ExpansionStep:
    return ExpansionStep(
        step_index=_required_int(payload["step_index"]),
        operation=cast(Any, payload["operation"]),
        input_node_ids=tuple(str(item) for item in _sequence(payload["input_node_ids"])),
        output_node_ids=tuple(str(item) for item in _sequence(payload["output_node_ids"])),
        accepted_node_ids=tuple(str(item) for item in _sequence(payload["accepted_node_ids"])),
        rejected_node_ids=tuple(str(item) for item in _sequence(payload["rejected_node_ids"])),
        inference_proposal_ids=tuple(
            str(item) for item in _sequence(payload.get("inference_proposal_ids", ()))
        ),
        gap_ids=tuple(str(item) for item in _sequence(payload.get("gap_ids", ()))),
        notes=tuple(str(item) for item in _sequence(payload.get("notes", ()))),
    )


def _evidence_decision_from_dict(payload: Mapping[str, object]) -> EvidenceDecision:
    return EvidenceDecision(
        kind=cast(Any, payload["kind"]),
        node_id=_optional_str(payload.get("node_id")),
        chunk_id=_optional_str(payload.get("chunk_id")),
        source=_optional_str(payload.get("source")),
        file=_optional_str(payload.get("file")),
        reason=str(payload["reason"]),
    )


def _proposal_trace_from_dict(payload: Mapping[str, object]) -> InferenceProposalTrace:
    return InferenceProposalTrace(
        proposal_id=str(payload["proposal_id"]),
        relation=cast(Any, payload["relation"]),
        subject_id=str(payload["subject_id"]),
        object_id=str(payload["object_id"]),
        source_evidence_ids=tuple(
            str(item) for item in _sequence(payload["source_evidence_ids"])
        ),
        used_for_exploration=_required_bool(payload["used_for_exploration"]),
        trusted_evidence=_required_bool(payload["trusted_evidence"]),
        reason=str(payload["reason"]),
    )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceValidationError("expected object")
    return value


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EvidenceValidationError("expected array")
    return value


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidenceValidationError("expected ISO datetime string")
    return datetime.fromisoformat(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _required_int(value)


def _required_int(value: object) -> int:
    if isinstance(value, bool):
        raise EvidenceValidationError("expected integer")
    if not isinstance(value, (str, bytes, bytearray, int)):
        raise EvidenceValidationError("expected integer")
    return int(value)


def _required_float(value: object) -> float:
    if isinstance(value, bool):
        raise EvidenceValidationError("expected number")
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        raise EvidenceValidationError("expected number")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvidenceValidationError("expected finite number") from exc
    if not math.isfinite(parsed):
        raise EvidenceValidationError("expected finite number")
    return parsed


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _required_float(value)


def _required_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise EvidenceValidationError("expected boolean")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidenceValidationError("expected string or null")
    return value


def _trust_state(value: object) -> str:
    if value not in {"trusted", "degraded"}:
        raise EvidenceValidationError("trust_state must be trusted or degraded")
    return value


__all__ = [
    "Citation",
    "Contradiction",
    "GenerationSelection",
    "REASONING_API_VERSION",
    "ReasoningAnswerProvider",
    "ReasoningDiagnostics",
    "ReasoningGraphProvider",
    "ReasoningOutcome",
    "ReasoningPolicy",
    "ReasoningPolicyName",
    "ReasoningProviderPorts",
    "ReasoningRequest",
    "ReasoningResponse",
    "ReasoningRetriever",
    "ReasoningValidationError",
    "reason",
    "reasoning_response_from_dict",
]
