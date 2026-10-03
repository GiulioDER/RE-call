"""The reasoning response and its wire format.

``ReasoningResponse`` and the records it carries, with the JSON encoder (``to_dict``) and the
strict decoder (``reasoning_response_from_dict``) that reads one back. Kept apart from
``recall.reasoning``, which runs a reasoning request, because the format is used by every reader of
a stored or transmitted response and the decoder alone is about four hundred lines. Every name here
is re-exported from ``recall.reasoning``, its public home.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, cast, get_args

from recall.evidence import (
    ANSWER_PROFILES,
    EvidenceBundle,
    EvidenceValidationError,
)
from recall.provider_metadata import ProviderMetadata
from recall.reasoning_expansion import (
    ExpansionMode,
    ExpansionProposal,
    RetrievalExpansionTrace,
)
from recall.reasoning_planner import (
    EvidenceDecision,
    ExpansionStep,
    InferenceProposalTrace,
    PlannerInitialRetrieval,
    ReasoningBudget,
    ReasoningBudgetUsage,
    ReasoningTrace,
    UnresolvedGap,
)
from recall.reasoning_proposals import (
    InferenceProposal,
    ProposalStatus,
    ProposedRelation,
    ProviderFailure,
)
from recall.types import AtomicFact, DecisionState, EvidenceCard

ReasoningOutcome = Literal["answered", "abstained", "needs_clarification", "needs_review"]
GraphExpansionMode = Literal["off", "one_hop"]


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
    retrieval_expansion: RetrievalExpansionTrace | None = None
    provider_failures: tuple[ProviderFailure, ...] = ()
    provider_metadata: tuple[ProviderMetadata, ...] = ()
    graph_expansion_mode: GraphExpansionMode = "off"
    graph_readiness: str = "not_requested"
    graph_entities_inspected: int = 0
    graph_relations_inspected: int = 0
    graph_candidates_discovered: int = 0
    graph_candidates_rejected: int = 0
    graph_relation_seed_activations: Mapping[str, int] = dataclass_field(default_factory=dict)
    graph_relation_candidates_accepted: Mapping[str, int] = dataclass_field(default_factory=dict)
    graph_relation_new_trusted_evidence: Mapping[str, int] = dataclass_field(default_factory=dict)
    graph_diagnostics_encountered: int = 0
    graph_expansion_latency_ms: float = 0.0
    graph_admission_rejections: Mapping[str, int] = dataclass_field(default_factory=dict)
    graph_expansion_refusals: Mapping[str, int] = dataclass_field(default_factory=dict)
    graph_gate_reason: str | None = None
    graph_policy_fingerprint: str | None = None
    performance: Mapping[str, object] = dataclass_field(default_factory=dict)
    #: Which answer prompt produced the answer, so an audit of a response can tell the profiles apart.
    #: ``plain`` when unrecorded: every response serialized before this field existed was plain.
    answer_profile: str = "plain"


@dataclass(frozen=True)
class ReasoningResponse:
    """Typed public response for one reasoning run."""

    schema_version: int
    outcome: ReasoningOutcome
    answer: str | None
    clarification_request: str | None
    trusted_evidence: EvidenceBundle
    inference_proposals: tuple[InferenceProposal, ...]
    provider_failures: tuple[ProviderFailure, ...]
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


def reasoning_response_from_dict(payload: Mapping[str, object]) -> ReasoningResponse:
    """Deserialize the JSON form emitted by :meth:`ReasoningResponse.to_dict`."""
    bundle = _evidence_bundle_from_dict(_mapping(payload["trusted_evidence"]))
    trust_state = _trust_state(payload["trust_state"])
    if bundle.trust_state != trust_state:
        raise EvidenceValidationError("trust_state mismatch between response and trusted_evidence")
    refusal_reason = _optional_str(payload.get("refusal_reason"))
    if trust_state == "refused" and (bundle.items or not (refusal_reason or bundle.failure_code)):
        raise EvidenceValidationError("refused responses must carry no evidence and a refusal code")
    proposals = tuple(
        _proposal_from_dict(_mapping(item)) for item in _sequence(payload["inference_proposals"])
    )
    provider_failures = tuple(
        _provider_failure_from_dict(_mapping(item))
        for item in _sequence(payload.get("provider_failures", ()))
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
        provider_failures=tuple(
            _provider_failure_from_dict(_mapping(item))
            for item in _sequence(diagnostics_payload.get("provider_failures", ()))
        ),
        provider_metadata=tuple(
            _provider_metadata_from_dict(_mapping(item))
            for item in _sequence(diagnostics_payload.get("provider_metadata", ()))
        ),
        retrieval_expansion=_optional_expansion_trace(
            diagnostics_payload.get("retrieval_expansion")
        ),
        graph_expansion_mode=_checked_literal(
            diagnostics_payload.get("graph_expansion_mode", "off"),
            get_args(GraphExpansionMode),
            "graph_expansion_mode",
        ),
        graph_readiness=str(diagnostics_payload.get("graph_readiness", "not_requested")),
        graph_entities_inspected=_required_int(
            diagnostics_payload.get("graph_entities_inspected", 0)
        ),
        graph_relations_inspected=_required_int(
            diagnostics_payload.get("graph_relations_inspected", 0)
        ),
        graph_candidates_discovered=_required_int(
            diagnostics_payload.get("graph_candidates_discovered", 0)
        ),
        graph_candidates_rejected=_required_int(
            diagnostics_payload.get("graph_candidates_rejected", 0)
        ),
        graph_relation_seed_activations={
            str(key): _required_int(value)
            for key, value in _mapping(
                diagnostics_payload.get("graph_relation_seed_activations", {})
            ).items()
        },
        graph_relation_candidates_accepted={
            str(key): _required_int(value)
            for key, value in _mapping(
                diagnostics_payload.get("graph_relation_candidates_accepted", {})
            ).items()
        },
        graph_relation_new_trusted_evidence={
            str(key): _required_int(value)
            for key, value in _mapping(
                diagnostics_payload.get("graph_relation_new_trusted_evidence", {})
            ).items()
        },
        graph_diagnostics_encountered=_required_int(
            diagnostics_payload.get("graph_diagnostics_encountered", 0)
        ),
        graph_expansion_latency_ms=_required_float(
            diagnostics_payload.get("graph_expansion_latency_ms", 0.0)
        ),
        graph_admission_rejections={
            str(key): _required_int(value)
            for key, value in _mapping(
                diagnostics_payload.get("graph_admission_rejections", {})
            ).items()
        },
        graph_expansion_refusals={
            str(key): _required_int(value)
            for key, value in _mapping(
                diagnostics_payload.get("graph_expansion_refusals", {})
            ).items()
        },
        graph_gate_reason=_optional_str(diagnostics_payload.get("graph_gate_reason")),
        graph_policy_fingerprint=_optional_str(
            diagnostics_payload.get("graph_policy_fingerprint")
        ),
        performance=dict(_mapping(diagnostics_payload.get("performance", {}))),
        answer_profile=_checked_literal(
            diagnostics_payload.get("answer_profile", "plain"), ANSWER_PROFILES, "answer_profile"
        ),
    )
    return ReasoningResponse(
        schema_version=_required_int(payload["schema_version"]),
        outcome=_checked_literal(payload["outcome"], get_args(ReasoningOutcome), "outcome"),
        answer=_optional_str(payload.get("answer")),
        clarification_request=_optional_str(payload.get("clarification_request")),
        trusted_evidence=bundle,
        inference_proposals=proposals,
        provider_failures=provider_failures,
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
        refusal_reason=refusal_reason,
        diagnostics=diagnostics,
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
    decision = _checked_literal(payload["decision"], ("answer", "abstain"), "decision")
    reason_code = _optional_str(payload.get("reason_code"))
    raw_decision_state = payload.get("decision_state")
    decision_state: DecisionState = cast(
        DecisionState,
        (
            _checked_literal(
                raw_decision_state,
                ("supported", "corpus_gap", "no_supporting_evidence"),
                "decision_state",
            )
            if raw_decision_state is not None
            else "supported"
            if decision == "answer"
            else "corpus_gap"
            if reason_code == "corpus_gap"
            else "no_supporting_evidence"
        ),
    )
    return EvidenceBundle(
        query=str(payload["query"]),
        decision=decision,
        reason_code=reason_code,
        decision_state=decision_state,
        calibrated=_required_bool(payload["calibrated"]),
        stale=_required_bool(payload["stale"]),
        embedding_profile=str(payload["embedding_profile"]),
        retrieval_profile=str(payload["retrieval_profile"]),
        index_generation=str(payload["index_generation"]),
        items=items,
        trust_state=trust_state,
        failure_code=_optional_str(payload.get("failure_code")),
        cards=tuple(
            EvidenceCard(
                card_id=str(card["card_id"]),
                chunk_id=str(card["chunk_id"]),
                source=str(card["source"]),
                source_digest=str(card["source_digest"]),
                valid_from=_optional_datetime(card.get("valid_from")),
                valid_until=_optional_datetime(card.get("valid_until")),
                first_indexed_at=_optional_datetime(card.get("first_indexed_at")),
                indexed_at=_optional_datetime(card.get("indexed_at")),
                tenant_id=str(card["tenant_id"]),
                generation_id=str(card["generation_id"]),
                pipeline_fingerprint=_optional_str(card.get("pipeline_fingerprint")),
                corpus_fingerprint=_optional_str(card.get("corpus_fingerprint")),
                calibration_id=_optional_str(card.get("calibration_id")),
                calibration_status=str(card["calibration_status"]),
                trust_state=str(card["trust_state"]),
                verdict=cast(Any, card["verdict"]),
                confidence=_required_float(card["confidence"]),
                rank=_required_int(card["rank"]),
                supersession_links=tuple(
                    str(value) for value in _sequence(card.get("supersession_links", ()))
                ),
                contradiction_links=tuple(
                    str(value) for value in _sequence(card.get("contradiction_links", ()))
                ),
                support_refs=tuple(
                    str(value) for value in _sequence(card.get("support_refs", ()))
                ),
                structured_facts=tuple(
                    AtomicFact.from_payload(_mapping(value))
                    for value in _sequence(card.get("structured_facts", ()))
                ),
                schema_version=_required_int(card.get("schema_version", 1)),
            )
            for card in (_mapping(value) for value in _sequence(payload.get("cards", ())))
        ),
    )


def _proposal_from_dict(payload: Mapping[str, object]) -> InferenceProposal:
    return InferenceProposal(
        id=str(payload["id"]),
        source_evidence_ids=tuple(str(item) for item in _sequence(payload["source_evidence_ids"])),
        proposed_relation=_checked_literal(
            payload["proposed_relation"], get_args(ProposedRelation), "proposed_relation"
        ),
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
        status=_checked_literal(
            payload.get("status", "candidate"), get_args(ProposalStatus), "status"
        ),
        rule_id=_optional_str(payload.get("rule_id")),
        metadata=_mapping(payload.get("metadata", {})),
    )


def _provider_failure_from_dict(payload: Mapping[str, object]) -> ProviderFailure:
    return ProviderFailure(
        kind=cast(Any, payload["kind"]),
        provider_id=str(payload["provider_id"]),
        model_id=str(payload["model_id"]),
        provider_revision=str(payload["provider_revision"]),
        message=str(payload["message"]),
    )


def _provider_metadata_from_dict(payload: Mapping[str, object]) -> ProviderMetadata:
    try:
        return ProviderMetadata.from_mapping(payload)
    except ValueError as exc:
        raise EvidenceValidationError(str(exc)) from exc


def _optional_expansion_trace(value: object) -> RetrievalExpansionTrace | None:
    if value is None:
        return None
    payload = _mapping(value)
    proposals = tuple(
        ExpansionProposal(
            id=str(item["id"]),
            mode=_checked_literal(item["mode"], get_args(ExpansionMode), "mode"),
            query=str(item["query"]),
            rationale=str(item.get("rationale", "")),
            parent_chunk_ids=tuple(
                str(chunk_id) for chunk_id in _sequence(item.get("parent_chunk_ids", ()))
            ),
        )
        for item in (_mapping(raw) for raw in _sequence(payload.get("proposals", ())))
    )
    return RetrievalExpansionTrace(
        attempted=_required_bool(payload["attempted"]),
        rounds=_required_int(payload["rounds"]),
        proposals=proposals,
        executed_queries=tuple(
            str(query) for query in _sequence(payload.get("executed_queries", ()))
        ),
        accepted_chunk_ids=tuple(
            str(chunk_id) for chunk_id in _sequence(payload.get("accepted_chunk_ids", ()))
        ),
        fallback_reason=_optional_str(payload.get("fallback_reason")),
        provider_skipped_reason=_optional_str(payload.get("provider_skipped_reason")),
    )


def _budget_from_dict(payload: Mapping[str, object]) -> ReasoningBudget:
    return ReasoningBudget(
        max_steps=_required_int(payload["max_steps"]),
        max_graph_nodes=_required_int(payload["max_graph_nodes"]),
        max_model_calls=_required_int(payload["max_model_calls"]),
        max_evidence_tokens=_required_int(payload["max_evidence_tokens"]),
        max_wall_time_ms=_required_int(payload["max_wall_time_ms"]),
        max_graph_hops=_required_int(payload.get("max_graph_hops", 0)),
        max_graph_entities=_required_int(payload.get("max_graph_entities", 8)),
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
        initial_retrieval=_initial_retrieval_from_dict(_mapping(payload["initial_retrieval"])),
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
        source_evidence_ids=tuple(str(item) for item in _sequence(payload["source_evidence_ids"])),
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
    try:
        return int(value)
    except ValueError as exc:
        raise EvidenceValidationError("expected integer") from exc


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


def _checked_literal(value: object, allowed: tuple[str, ...], field_name: str) -> Any:
    """Reject a serialized value that is outside its Literal vocabulary.

    A bare `cast` would let an unknown value flow into a typed field silently, so every
    deserialized enum-like field goes through here and fails loudly instead.
    """
    if value not in allowed:
        raise EvidenceValidationError(f"{field_name} must be one of: {', '.join(allowed)}")
    return value


def _trust_state(value: object) -> str:
    if value not in {"trusted", "degraded", "refused"}:
        raise EvidenceValidationError("trust_state must be trusted, degraded, or refused")
    return value
