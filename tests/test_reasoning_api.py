from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from recall.evidence import EvidenceValidationError
from recall.reasoning import (
    GenerationSelection,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    ReasoningValidationError,
    reason,
    reasoning_response_from_dict,
)
from recall.reasoning_graph import ReasoningGraphNode, build_reasoning_graph
from recall.reasoning_planner import ReasoningBudgetUsage, ReasoningTrace
from recall.reasoning_proposals import InferenceProposal
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)

NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _chunk(
    cid: str,
    file: str,
    text: str,
    *,
    supersedes: str | None = None,
) -> Chunk:
    metadata: dict[str, Any] = {"file": file, "ord": 0, "valid_from": "2026-01-01"}
    if supersedes is not None:
        metadata["supersedes"] = supersedes
    return Chunk(cid, f"/corpus/{file}", text, metadata)


def _hit(chunk: Chunk, *, verdict: str = "ok") -> TrustedHit:
    return TrustedHit(
        chunk=chunk,
        cosine=0.91,
        confidence=0.97,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(chunk.source, chunk.metadata["file"], 0, NOW),
        validity=Validity(NOW, None, "new.md" if verdict == "superseded" else None),
    )


def _result(*hits: TrustedHit, tenant: str = "acme", generation: str = "gen_1") -> TrustedResult:
    return TrustedResult(
        query="who owns rollout?",
        hits=list(hits),
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(
            embedding_profile="embed-a",
            retrieval_profile="quality",
            index_generation=generation,
            stage_ms={"dense_retrieval": 1.0},
        ),
        calibration_id="cal-1",
        calibration_status="certified",
        tenant_id=tenant,
        generation_id=generation,
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        query_set_digest="query-a",
    )


def _request(
    retrieval: TrustedResult,
    *,
    policy: ReasoningPolicy | None = None,
    graph=None,
    proposals=(),
    answer=None,
) -> ReasoningRequest:
    def retrieve(_request: ReasoningRequest) -> TrustedResult:
        return retrieval

    def graph_provider(_request: ReasoningRequest, _retrieval: TrustedResult):
        assert graph is not None
        return graph

    def proposal_provider(_request: ReasoningRequest, _graph, _retrieval: TrustedResult):
        return proposals

    return ReasoningRequest(
        query=retrieval.query,
        tenant_id="acme",
        generation=GenerationSelection(
            generation_id="gen_1",
            pipeline_fingerprint="pipe-a",
            corpus_fingerprint="corpus-a",
        ),
        providers=ReasoningProviderPorts(
            retriever=retrieve,
            graph_provider=graph_provider if graph is not None else None,
            proposal_provider=proposal_provider if proposals else None,
            answer_provider=answer,
        ),
        policy=policy or ReasoningPolicy(),
    )


def test_reasoning_answers_with_trusted_citation() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    request = _request(
        _result(_hit(chunk)),
        answer=lambda _system, _user: {
            "answer": "Ada owns rollout.",
            "citations": ["c1"],
            "insufficient_evidence": False,
        },
    )

    response = reason(request)

    assert response.outcome == "answered"
    assert response.answer == "Ada owns rollout."
    assert response.citations[0].chunk_id == "c1"
    assert response.trusted_evidence.items[0].chunk_id == "c1"
    assert response.calibration_status == "certified"
    assert response.generation_id == "gen_1"


def test_reasoning_rejects_demoted_and_unretrieved_citations() -> None:
    trusted = _chunk("trusted", "current.md", "Ada owns rollout.")
    demoted = _chunk("old", "old.md", "Bea owns rollout.")
    retrieval = _result(_hit(trusted), _hit(demoted, verdict="superseded"))
    request = _request(
        retrieval,
        answer=lambda _system, _user: {
            "answer": "Bea owns rollout.",
            "citations": ["old"],
            "insufficient_evidence": False,
        },
    )

    with pytest.raises(EvidenceValidationError, match="unknown citation ids"):
        reason(request)

    request = _request(
        retrieval,
        answer=lambda _system, _user: {
            "answer": "No retrieved citation.",
            "citations": ["missing"],
            "insufficient_evidence": False,
        },
    )

    with pytest.raises(EvidenceValidationError, match="unknown citation ids"):
        reason(request)


def test_reasoning_rejects_cross_tenant_or_generation_retrieval() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")

    with pytest.raises(ReasoningValidationError, match="tenant_id"):
        reason(_request(_result(_hit(chunk), tenant="other")))

    with pytest.raises(ReasoningValidationError, match="generation_id"):
        reason(_request(_result(_hit(chunk), generation="gen_2")))


def test_reasoning_rejects_missing_requested_retrieval_boundaries() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    retrieval = _result(_hit(chunk))
    missing_tenant = TrustedResult(
        query=retrieval.query,
        hits=retrieval.hits,
        abstained=retrieval.abstained,
        reason=retrieval.reason,
        gap_warning=retrieval.gap_warning,
        staleness=retrieval.staleness,
        diagnostics=retrieval.diagnostics,
        calibration_id=retrieval.calibration_id,
        calibration_status=retrieval.calibration_status,
        tenant_id=None,
        generation_id=retrieval.generation_id,
        pipeline_fingerprint=retrieval.pipeline_fingerprint,
        corpus_fingerprint=retrieval.corpus_fingerprint,
    )
    missing_pipeline = TrustedResult(
        query=retrieval.query,
        hits=retrieval.hits,
        abstained=retrieval.abstained,
        reason=retrieval.reason,
        gap_warning=retrieval.gap_warning,
        staleness=retrieval.staleness,
        diagnostics=retrieval.diagnostics,
        calibration_id=retrieval.calibration_id,
        calibration_status=retrieval.calibration_status,
        tenant_id=retrieval.tenant_id,
        generation_id=retrieval.generation_id,
        pipeline_fingerprint=None,
        corpus_fingerprint=retrieval.corpus_fingerprint,
    )

    with pytest.raises(ReasoningValidationError, match="tenant_id"):
        reason(_request(missing_tenant))
    with pytest.raises(ReasoningValidationError, match="pipeline_fingerprint"):
        reason(_request(missing_pipeline))


def test_reasoning_rejects_foreign_identity_inside_graph() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    graph = build_reasoning_graph(
        [chunk],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        include_text=True,
    )
    foreign = ReasoningGraphNode(
        id="foreign-node",
        kind="chunk",
        tenant_id="other",
        generation_id="gen_1",
        source="/corpus/foreign.md",
        chunk_id="foreign",
        file="foreign.md",
    )
    forged = type(graph)(
        schema_version=graph.schema_version,
        graph_id=graph.graph_id,
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        pipeline_fingerprint=graph.pipeline_fingerprint,
        corpus_fingerprint=graph.corpus_fingerprint,
        nodes=(*graph.nodes, foreign),
        authored_edges=graph.authored_edges,
        inferred_candidate_edges=graph.inferred_candidate_edges,
        diagnostics=graph.diagnostics,
    )

    with pytest.raises(ReasoningValidationError, match="node foreign-node tenant_id"):
        reason(
            _request(
                _result(_hit(chunk)),
                policy=ReasoningPolicy(name="proposal_assisted"),
                graph=forged,
            )
        )


def test_retrieval_only_policy_does_not_invoke_answer_provider() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    called = False

    def answer(_system: str, _user: str):
        nonlocal called
        called = True
        return {}

    response = reason(
        _request(
            _result(_hit(chunk)),
            policy=ReasoningPolicy(name="retrieval_only"),
            answer=answer,
        )
    )

    assert response.outcome == "abstained"
    assert response.refusal_reason == "retrieval_only_policy"
    assert called is False
    assert response.trusted_evidence.items[0].chunk_id == "c1"


def test_strict_policy_abstains_on_degraded_result() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    retrieval = _result(_hit(chunk))
    degraded = TrustedResult(
        query=retrieval.query,
        hits=retrieval.hits,
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=retrieval.staleness,
        diagnostics=retrieval.diagnostics,
        tenant_id=retrieval.tenant_id,
        generation_id=retrieval.generation_id,
        pipeline_fingerprint=retrieval.pipeline_fingerprint,
        corpus_fingerprint=retrieval.corpus_fingerprint,
        trust_state="degraded",
        failure_code="CALIBRATION_UNCERTIFIED",
    )

    response = reason(_request(degraded, answer=lambda _s, _u: {}))

    assert response.outcome == "abstained"
    assert response.refusal_reason == "uncertified_evidence"
    assert response.diagnostics.generator_invoked is False


def test_proposal_assisted_policy_records_proposals_but_never_cites_them() -> None:
    old = _chunk("old", "rollout_v1.md", "decision: rollout owner. Ada owns it.")
    new = _chunk("new", "rollout_v2.md", "decision: rollout owner. Bea owns it.")
    graph = build_reasoning_graph(
        [old, new],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        include_text=True,
    )
    evidence_ids = tuple(node.id for node in graph.nodes if node.kind == "source")[:2]
    proposal = InferenceProposal(
        id="proposal1",
        source_evidence_ids=evidence_ids,
        proposed_relation="supersedes",
        subject_id="rollout_v1.md",
        object_id="rollout_v2.md",
        explanation="candidate only",
        model_id="rules",
        pipeline_id="pipe-a",
        provider_id="test",
        provider_revision="rev",
        confidence=0.8,
        uncertainty=(),
        generation_id="gen_1",
    )
    request = _request(
        _result(_hit(old)),
        policy=ReasoningPolicy(name="proposal_assisted"),
        graph=graph,
        proposals=(proposal,),
        answer=lambda _system, _user: {
            "answer": "The proposal says Bea owns it.",
            "citations": ["proposal1"],
            "insufficient_evidence": False,
        },
    )

    with pytest.raises(EvidenceValidationError, match="unknown citation ids"):
        reason(request)


def test_review_required_policy_distinguishes_review_from_abstain() -> None:
    old = _chunk("old", "rollout_v1.md", "decision: rollout owner. Ada owns it.")
    new = _chunk("new", "rollout_v2.md", "decision: rollout owner. Bea owns it.")
    graph = build_reasoning_graph(
        [old, new],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        include_text=True,
    )
    evidence_ids = tuple(node.id for node in graph.nodes if node.kind == "source")[:2]
    proposal = InferenceProposal(
        id="proposal_review",
        source_evidence_ids=evidence_ids,
        proposed_relation="supersedes",
        subject_id="rollout_v1.md",
        object_id="rollout_v2.md",
        explanation="candidate only",
        model_id="rules",
        pipeline_id="pipe-a",
        provider_id="test",
        provider_revision="rev",
        confidence=0.8,
        uncertainty=(),
        generation_id="gen_1",
    )

    response = reason(
        _request(
            _result(_hit(old)),
            policy=ReasoningPolicy(name="review_required"),
            graph=graph,
            proposals=(proposal,),
            answer=lambda _s, _u: {},
        )
    )

    assert response.outcome == "needs_review"
    assert response.refusal_reason == "review_required_policy"
    assert response.diagnostics.generator_invoked is False


def test_reasoning_rejects_malformed_provider_response() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")

    with pytest.raises(EvidenceValidationError, match="answer envelope"):
        reason(_request(_result(_hit(chunk)), answer=lambda _s, _u: {"answer": "x"}))


def test_empty_query_requests_clarification_without_retrieval() -> None:
    def retrieve(_request: ReasoningRequest) -> TrustedResult:
        raise AssertionError("retriever should not be called")

    response = reason(
        ReasoningRequest(
            query=" ",
            tenant_id="acme",
            generation=GenerationSelection(generation_id="gen_1"),
            providers=ReasoningProviderPorts(retriever=retrieve),
        )
    )

    assert response.outcome == "needs_clarification"
    assert response.refusal_reason == "empty_query"


def test_reasoning_response_serializes_to_strict_json_and_round_trips() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    response = reason(
        _request(
            _result(_hit(chunk)),
            answer=lambda _system, _user: {
                "answer": "Ada owns rollout.",
                "citations": ["c1"],
                "insufficient_evidence": False,
            },
        )
    )

    payload = response.to_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    decoded = reasoning_response_from_dict(json.loads(encoded))

    assert decoded.outcome == response.outcome
    assert decoded.citations == response.citations
    assert decoded.trusted_evidence.items == response.trusted_evidence.items


def test_planner_trace_and_budget_usage_round_trip_as_typed_objects() -> None:
    old = _chunk("old", "rollout_v1.md", "decision: rollout owner. Ada owns it.")
    new = _chunk("new", "rollout_v2.md", "decision: rollout owner. Bea owns it.")
    graph = build_reasoning_graph(
        [old, new],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        include_text=True,
    )
    response = reason(
        _request(
            _result(_hit(old)),
            policy=ReasoningPolicy(name="proposal_assisted"),
            graph=graph,
            answer=lambda _system, _user: {
                "answer": "Ada owns rollout.",
                "citations": ["old"],
                "insufficient_evidence": False,
            },
        )
    )

    decoded = reasoning_response_from_dict(json.loads(json.dumps(response.to_dict())))

    assert isinstance(decoded.reasoning_trace, ReasoningTrace)
    assert isinstance(decoded.diagnostics.budget_used, ReasoningBudgetUsage)
    assert decoded.reasoning_trace.initial_retrieval.trusted_hit_ids == ("old",)
    assert decoded.diagnostics.budget_used.graph_nodes >= 1


def test_deserialization_requires_nested_trust_state_to_match_top_level() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    response = reason(
        _request(
            _result(_hit(chunk)),
            answer=lambda _system, _user: {
                "answer": "Ada owns rollout.",
                "citations": ["c1"],
                "insufficient_evidence": False,
            },
        )
    )
    payload = response.to_dict()
    trusted_evidence = payload["trusted_evidence"]
    assert isinstance(trusted_evidence, dict)
    trusted_evidence.pop("trust_state")

    with pytest.raises(EvidenceValidationError, match="trust_state"):
        reasoning_response_from_dict(payload)

    trusted_evidence["trust_state"] = "degraded"
    payload["trust_state"] = "trusted"
    with pytest.raises(EvidenceValidationError, match="trust_state mismatch"):
        reasoning_response_from_dict(payload)


def test_deserialization_rejects_nonfinite_float_strings() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    response = reason(
        _request(
            _result(_hit(chunk)),
            answer=lambda _system, _user: {
                "answer": "Ada owns rollout.",
                "citations": ["c1"],
                "insufficient_evidence": False,
            },
        )
    )
    payload = response.to_dict()
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    stage_ms = diagnostics["retrieval_stage_ms"]
    assert isinstance(stage_ms, dict)
    stage_ms["dense_retrieval"] = "NaN"

    with pytest.raises(EvidenceValidationError, match="finite"):
        reasoning_response_from_dict(payload)
