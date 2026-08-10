"""Optional provider execution and validation for inference proposals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from recall.reasoning_graph import ReasoningGraphProjection
from recall.reasoning_proposals._deterministic import (
    _context,
    _make_proposal,
    _proposal_id,
    _resolve_pipeline_id,
    deterministic_inference_proposals,
)
from recall.reasoning_proposals.types import (
    PROPOSAL_SCHEMA_VERSION,
    InferenceProposal,
    ModelBackedProposalProvider,
    ProposalContext,
    ProposalProtocolReport,
    ProposalStatus,
    ProposedRelation,
    ProviderFailure,
    ProviderFailureKind,
)

def _coerce_provider_proposal(
    raw: InferenceProposal | Mapping[str, Any],
    *,
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    known_evidence_ids: set[str],
) -> InferenceProposal:
    if isinstance(raw, InferenceProposal):
        proposal = raw
        if proposal.provider_id != context.provider_id:
            raise ValueError("provider proposal provider_id does not match request context")
        if proposal.model_id != context.model_id:
            raise ValueError("provider proposal model_id does not match request context")
        if proposal.provider_revision != context.provider_revision:
            raise ValueError(
                "provider proposal provider_revision does not match request context"
            )
        _checked_relation(proposal.proposed_relation)
        _checked_status(proposal.status)
        _checked_confidence(proposal.confidence)
    else:
        required = {
            "source_evidence_ids",
            "proposed_relation",
            "subject_id",
            "object_id",
            "explanation",
            "confidence",
            "uncertainty",
        }
        missing = sorted(key for key in required if key not in raw)
        if missing:
            raise ValueError(f"provider proposal missing fields: {', '.join(missing)}")
        proposal = _make_proposal(
            graph=graph,
            context=context,
            source_evidence_ids=tuple(str(item) for item in raw["source_evidence_ids"]),
            proposed_relation=_checked_relation(raw["proposed_relation"]),
            subject_id=str(raw["subject_id"]),
            object_id=str(raw["object_id"]),
            explanation=str(raw["explanation"]),
            confidence=_checked_confidence(raw["confidence"]),
            uncertainty=tuple(str(item) for item in raw["uncertainty"]),
            status=_checked_status(raw.get("status", "candidate")),
            rule_id=str(raw["rule_id"]) if raw.get("rule_id") is not None else None,
            metadata=raw["metadata"] if isinstance(raw.get("metadata"), Mapping) else {},
        )
    if proposal.generation_id != graph.generation_id:
        raise ValueError("provider proposal generation_id does not match graph generation")
    if proposal.pipeline_id != context.pipeline_id:
        raise ValueError("provider proposal pipeline_id does not match request context")
    if not proposal.source_evidence_ids:
        raise ValueError("provider proposal has no source evidence")
    unknown = sorted(set(proposal.source_evidence_ids) - known_evidence_ids)
    if unknown:
        raise ValueError(f"provider proposal cites unknown evidence ids: {', '.join(unknown)}")
    expected_id = _proposal_id(
        {
            "tenant_id": graph.tenant_id,
            "generation_id": graph.generation_id,
            "pipeline_id": context.pipeline_id,
            "provider_id": context.provider_id,
            "model_id": context.model_id,
            "provider_revision": context.provider_revision,
            "evidence": tuple(sorted(proposal.source_evidence_ids)),
            "relation": proposal.proposed_relation,
            "subject_id": proposal.subject_id,
            "object_id": proposal.object_id,
            "rule_id": proposal.rule_id,
            "status": proposal.status,
        }
    )
    if proposal.id != expected_id:
        raise ValueError("provider proposal id does not match canonical identity")
    return proposal


def _checked_relation(value: Any) -> ProposedRelation:
    if value not in {"supersedes", "contradicts", "same_entity", "references"}:
        raise ValueError(f"unknown proposed_relation: {value!r}")
    return cast(ProposedRelation, value)


def _checked_status(value: Any) -> ProposalStatus:
    if value not in {"candidate", "rejected", "requires_review"}:
        raise ValueError(f"unknown status: {value!r}")
    return cast(ProposalStatus, value)


def _checked_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("confidence must be a number or null")
    score = float(value)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return score


def _reject_duplicate_ids(
    existing: set[str],
    proposals: Sequence[InferenceProposal],
) -> None:
    seen: set[str] = set()
    for proposal in proposals:
        if proposal.id in existing or proposal.id in seen:
            raise ValueError(f"duplicate provider proposal id: {proposal.id}")
        seen.add(proposal.id)


def _safe_failure_message(kind: ProviderFailureKind, exc: BaseException) -> str:
    if kind == "wrong_cardinality":
        return "provider returned too many proposals"
    if kind == "timeout":
        return "provider timed out"
    if kind == "malformed_output":
        return f"provider returned malformed output: {exc.__class__.__name__}"
    return f"provider error: {exc.__class__.__name__}"


def proposal_report(
    graph: ReasoningGraphProjection,
    *,
    pipeline_id: str | None = None,
    model_provider: ModelBackedProposalProvider | None = None,
) -> ProposalProtocolReport:
    """Run deterministic and optional model backed proposal generation.

    Provider failures are returned in band. A malformed provider cannot silently add or drop
    proposals, and deterministic proposals still survive provider failure.
    """
    pipeline = _resolve_pipeline_id(graph, pipeline_id)
    deterministic = deterministic_inference_proposals(graph, pipeline_id=pipeline)
    accepted: list[InferenceProposal] = list(deterministic)
    rejected: list[InferenceProposal] = []
    failures: list[ProviderFailure] = []
    if model_provider is not None:
        provider_context = _context(
            graph,
            pipeline_id=pipeline,
            provider_id=model_provider.provider_id,
            model_id=model_provider.model_id,
            provider_revision=model_provider.provider_revision,
        )
        try:
            raw_proposals = tuple(model_provider.propose(graph, provider_context))
            if len(raw_proposals) > model_provider.max_proposals:
                raise ValueError(
                    f"provider returned {len(raw_proposals)} proposals, "
                    f"maximum is {model_provider.max_proposals}"
                )
            known_evidence_ids = {node.id for node in graph.nodes}
            provider_accepted: list[InferenceProposal] = []
            provider_rejected: list[InferenceProposal] = []
            for raw in raw_proposals:
                proposal = _coerce_provider_proposal(
                    raw,
                    graph=graph,
                    context=provider_context,
                    known_evidence_ids=known_evidence_ids,
                )
                if proposal.status == "rejected":
                    provider_rejected.append(proposal)
                else:
                    provider_accepted.append(proposal)
            _reject_duplicate_ids(
                {proposal.id for proposal in (*accepted, *rejected)},
                [*provider_accepted, *provider_rejected],
            )
            accepted.extend(provider_accepted)
            rejected.extend(provider_rejected)
        except TimeoutError as exc:
            failures.append(
                _provider_failure(
                    model_provider, "timeout", _safe_failure_message("timeout", exc)
                )
            )
        except ValueError as exc:
            message = str(exc)
            kind: ProviderFailureKind = (
                "wrong_cardinality" if "maximum is" in message else "malformed_output"
            )
            failures.append(
                _provider_failure(model_provider, kind, _safe_failure_message(kind, exc))
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                _provider_failure(
                    model_provider,
                    "provider_error",
                    _safe_failure_message("provider_error", exc),
                )
            )
    return ProposalProtocolReport(
        schema_version=PROPOSAL_SCHEMA_VERSION,
        generation_id=graph.generation_id,
        pipeline_id=pipeline,
        proposals=tuple(sorted(accepted, key=lambda proposal: proposal.id)),
        rejected_proposals=tuple(sorted(rejected, key=lambda proposal: proposal.id)),
        provider_failures=tuple(failures),
    )


def _provider_failure(
    provider: ModelBackedProposalProvider, kind: ProviderFailureKind, message: str
) -> ProviderFailure:
    return ProviderFailure(
        kind=kind,
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        provider_revision=provider.provider_revision,
        message=message,
    )
