from __future__ import annotations

import json
from dataclasses import replace
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
    SemanticGraphExpansionResult,
    reason,
    reasoning_response_from_dict,
)
from recall.provider_metadata import ProviderMetadata
from recall.reasoning_expansion import ExpansionProposal, ExpansionReport
from recall.reasoning_graph import ReasoningGraphNode, build_reasoning_graph
from recall.reasoning_planner import ReasoningBudget, ReasoningBudgetUsage, ReasoningTrace
from recall.reasoning_proposals import InferenceProposal, ProposalProtocolReport
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


def _result(
    *hits: TrustedHit,
    tenant: str = "acme",
    generation: str = "gen_1",
    gap_warning: bool = False,
    abstained: bool = False,
) -> TrustedResult:
    return TrustedResult(
        query="who owns rollout?",
        hits=list(hits),
        abstained=abstained,
        reason="",
        gap_warning=gap_warning,
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
    budget: ReasoningBudget | None = None,
    graph=None,
    proposals=(),
    answer=None,
    expansion_provider=None,
    expansion_retriever=None,
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
            expansion_provider=expansion_provider,
            expansion_retriever=expansion_retriever,
            answer_provider=answer,
        ),
        policy=policy or ReasoningPolicy(),
        budget=budget or ReasoningBudget(),
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


def test_retrieval_expansion_rebuilds_evidence_before_answering() -> None:
    first = _chunk("first", "first.md", "The first document identifies the project.")
    second = _chunk("second", "second.md", "The second document contains the owner: Ada.")
    initial = _result(_hit(first))
    expanded = _result(_hit(second))
    calls: list[str] = []

    def expansion_provider(_request):
        return ExpansionReport(
            proposals=(
                ExpansionProposal(
                    id="rewrite_1",
                    mode="rewrite",
                    query="who owns the project",
                    rationale="the initial evidence does not contain an owner",
                    parent_chunk_ids=("first",),
                ),
            )
        )

    def expansion_retriever(_request, proposal, _initial):
        calls.append(proposal.query)
        return expanded

    def answer(_system: str, user: str):
        assert "second" in user
        return {
            "answer": "Ada owns the project.",
            "citations": ["second"],
            "insufficient_evidence": False,
        }

    response = reason(
        _request(
            initial,
            policy=ReasoningPolicy(allow_retrieval_expansion=True),
            budget=ReasoningBudget(max_model_calls=1),
            expansion_provider=expansion_provider,
            expansion_retriever=expansion_retriever,
            answer=answer,
        )
    )

    assert response.outcome == "answered"
    assert response.answer == "Ada owns the project."
    assert calls == ["who owns the project"]
    assert response.diagnostics.retrieval_expansion is not None
    assert response.diagnostics.retrieval_expansion.accepted_chunk_ids == ("second",)
    assert [item.chunk_id for item in response.trusted_evidence.items] == ["first", "second"]


def test_retrieval_expansion_provider_failure_preserves_baseline() -> None:
    chunk = _chunk("first", "first.md", "The first document contains the answer.")

    def expansion_provider(_request):
        raise TimeoutError("cheap model unavailable")

    response = reason(
        _request(
            _result(_hit(chunk)),
            policy=ReasoningPolicy(allow_retrieval_expansion=True),
            budget=ReasoningBudget(max_model_calls=1),
            expansion_provider=expansion_provider,
            expansion_retriever=lambda *_args: _result(_hit(chunk)),
            answer=lambda _system, _user: {
                "answer": "The first document contains the answer.",
                "citations": ["first"],
                "insufficient_evidence": False,
            },
        )
    )

    assert response.outcome == "answered"
    assert response.citations[0].chunk_id == "first"
    assert response.refusal_reason is None
    assert response.provider_failures[0].message == "TimeoutError"
    assert response.diagnostics.retrieval_expansion is not None
    assert response.diagnostics.retrieval_expansion.fallback_reason == "provider_failure"


def test_retrieval_expansion_tries_depth_before_the_cheap_provider() -> None:
    first = _chunk("first", "first.md", "The first document is incomplete.")
    second = _chunk("second", "second.md", "The second document contains the owner: Ada.")
    initial = _result(_hit(first), gap_warning=True)
    depth = _result(_hit(second))
    calls: list[str] = []
    provider_evidence: list[str] = []

    def expansion_provider(request):
        provider_evidence.extend(item["chunk_id"] for item in request.evidence)
        return ExpansionReport(
            proposals=(ExpansionProposal("rewrite", "rewrite", "owner of the project"),)
        )

    def expansion_retriever(_request, proposal, _initial):
        calls.append(proposal.mode)
        return depth

    response = reason(
        _request(
            initial,
            policy=ReasoningPolicy(allow_retrieval_expansion=True),
            budget=ReasoningBudget(max_model_calls=1),
            expansion_provider=expansion_provider,
            expansion_retriever=expansion_retriever,
            answer=lambda _system, _user: {
                "answer": "Ada owns the project.",
                "citations": ["second"],
                "insufficient_evidence": False,
            },
        )
    )

    assert response.outcome == "answered"
    assert calls == ["depth"]
    assert provider_evidence == []
    assert response.diagnostics.retrieval_expansion is not None
    assert response.diagnostics.retrieval_expansion.provider_skipped_reason == "depth_resolved"


def test_depth_expansion_can_run_without_a_cheap_provider() -> None:
    first = _chunk("first", "first.md", "The first document is incomplete.")
    second = _chunk("second", "second.md", "The second document contains the owner: Ada.")

    response = reason(
        _request(
            _result(_hit(first), gap_warning=True),
            policy=ReasoningPolicy(allow_retrieval_expansion=True),
            expansion_retriever=lambda *_args: _result(_hit(second)),
            answer=lambda _system, _user: {
                "answer": "Ada owns the project.",
                "citations": ["second"],
                "insufficient_evidence": False,
            },
        )
    )

    assert response.outcome == "answered"
    assert response.provider_failures == ()
    assert response.diagnostics.retrieval_expansion is not None
    assert response.diagnostics.retrieval_expansion.accepted_chunk_ids == ("second",)


def test_retrieval_expansion_rejects_mismatched_generation() -> None:
    first = _chunk("first", "first.md", "The first document contains the answer.")
    second = _chunk("second", "second.md", "Foreign generation evidence.")
    foreign = _result(_hit(second), generation="gen_2")

    response = reason(
        _request(
            _result(_hit(first)),
            policy=ReasoningPolicy(allow_retrieval_expansion=True),
            budget=ReasoningBudget(max_model_calls=1),
            expansion_provider=lambda _request: ExpansionReport(
                proposals=(ExpansionProposal("depth_1", "depth", "who owns the project"),)
            ),
            expansion_retriever=lambda *_args: foreign,
            answer=lambda _system, _user: {
                "answer": "The first document contains the answer.",
                "citations": ["first"],
                "insufficient_evidence": False,
            },
        )
    )

    assert response.outcome == "answered"
    assert [item.chunk_id for item in response.trusted_evidence.items] == ["first"]
    assert response.diagnostics.retrieval_expansion is not None
    assert response.diagnostics.retrieval_expansion.fallback_reason == "expanded_retrieval_failure"


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


def test_reasoning_rejects_foreign_proposal_report_identity_before_answering() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    graph = build_reasoning_graph(
        [chunk],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        include_text=True,
    )
    called = False

    def answer(_system: str, _user: str) -> dict[str, object]:
        nonlocal called
        called = True
        return {"answer": "Ada owns rollout.", "citations": ["c1"], "insufficient_evidence": False}

    response = reason(
        _request(
            _result(_hit(chunk)),
            policy=ReasoningPolicy(name="proposal_assisted"),
            graph=graph,
            proposals=ProposalProtocolReport(
                schema_version=1,
                generation_id="gen_1",
                pipeline_id="other-pipeline",
                proposals=(),
                rejected_proposals=(),
                provider_failures=(),
            ),
            answer=answer,
        )
    )

    assert response.outcome == "needs_review"
    assert response.refusal_reason == "provider_failure"
    assert response.provider_failures[0].message == "report_pipeline_mismatch"
    assert called is False


def test_reasoning_rejects_proposal_pipeline_mismatch() -> None:
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
        id="foreign_pipeline",
        source_evidence_ids=evidence_ids,
        proposed_relation="supersedes",
        subject_id="rollout_v1.md",
        object_id="rollout_v2.md",
        explanation="candidate only",
        model_id="rules",
        pipeline_id="other-pipeline",
        provider_id="test",
        provider_revision="rev",
        confidence=0.8,
        uncertainty=(),
        generation_id="gen_1",
    )

    with pytest.raises(ReasoningValidationError, match="proposal pipeline_id"):
        reason(
            _request(
                _result(_hit(old)),
                policy=ReasoningPolicy(name="proposal_assisted"),
                graph=graph,
                proposals=(proposal,),
            )
        )


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
    # No retrieval ran and no trust evaluation happened, so the bundle must not carry the
    # dataclass default of "trusted": the gate refused this request before evaluating it.
    assert response.trust_state == "refused"
    assert response.trusted_evidence.trust_state == "refused"


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


def test_reasoning_diagnostics_round_trip_provider_metadata() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")

    class _Answer:
        def __call__(self, _system: str, _user: str) -> dict[str, object]:
            return {
                "answer": "Ada owns rollout.",
                "citations": ["c1"],
                "insufficient_evidence": False,
            }

        def provider_metadata(self) -> ProviderMetadata:
            return ProviderMetadata(
                provider_id="fixture",
                model_id="fixture-model",
                model_revision="rev-1",
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                latency_ms=5,
                monetary_cost_usd=0.001,
            )

    response = reason(_request(_result(_hit(chunk)), answer=_Answer()))
    decoded = reasoning_response_from_dict(json.loads(json.dumps(response.to_dict())))

    assert decoded.diagnostics.provider_metadata[0].provider_id == "fixture"
    assert decoded.diagnostics.provider_metadata[0].model_revision == "rev-1"
    assert decoded.diagnostics.provider_metadata[0].monetary_cost_usd == 0.001


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


def test_graph_precision_diagnostics_round_trip_as_additive_fields() -> None:
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    response = reason(_request(_result(_hit(chunk))))
    response = replace(
        response,
        diagnostics=replace(
            response.diagnostics,
            graph_admission_rejections={"hub_entity": 2, "cosine_admission": 1},
            graph_gate_reason="graph_gate_not_met",
            graph_policy_fingerprint="f" * 64,
        ),
    )
    decoded = reasoning_response_from_dict(json.loads(json.dumps(response.to_dict())))
    assert decoded.diagnostics.graph_admission_rejections == {
        "hub_entity": 2,
        "cosine_admission": 1,
    }
    assert decoded.diagnostics.graph_gate_reason == "graph_gate_not_met"
    assert decoded.diagnostics.graph_policy_fingerprint == "f" * 64


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


def _graph_expansion_request(
    seed: TrustedResult,
    provider,
    *,
    answer=None,
) -> ReasoningRequest:
    return ReasoningRequest(
        query=seed.query,
        tenant_id="acme",
        generation=GenerationSelection(
            generation_id="gen_1",
            pipeline_fingerprint="pipe-a",
            corpus_fingerprint="corpus-a",
        ),
        providers=ReasoningProviderPorts(
            retriever=lambda _request: seed,
            graph_expansion_provider=provider,
            answer_provider=answer,
        ),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_hops=1),
    )


def test_graph_expansion_with_foreign_binding_fails_closed() -> None:
    seed = _result(_hit(_chunk("c1", "rollout.md", "Ada owns rollout.")))
    foreign = replace(
        _result(
            _hit(_chunk("f1", "foreign.md", "Evidence from another tenant.")),
            tenant="OTHER-TENANT",
            generation="gen_other",
        ),
        pipeline_fingerprint="pipe-other",
        corpus_fingerprint="corpus-other",
    )

    def provider(_request: ReasoningRequest, _retrieval: TrustedResult):
        return SemanticGraphExpansionResult(retrieval=foreign, readiness="ready")

    response = reason(_graph_expansion_request(seed, provider))

    assert response.outcome == "abstained"
    assert response.refusal_reason == "GRAPH_PROVIDER_ERROR"
    assert response.tenant_id == "acme"
    assert response.generation_id == "gen_1"
    assert all(item.chunk_id != "f1" for item in response.trusted_evidence.items)
    assert all(citation.chunk_id != "f1" for citation in response.citations)
    assert response.provider_failures
    assert response.provider_failures[0].message == "ReasoningValidationError"


def test_graph_expansion_with_correct_binding_is_adopted() -> None:
    seed_hit = _hit(_chunk("c1", "rollout.md", "Ada owns rollout."))
    neighbour_hit = _hit(_chunk("c2", "graph.md", "The rollout supersedes the pilot."))
    seed = _result(seed_hit)
    expanded = _result(seed_hit, neighbour_hit)

    def provider(_request: ReasoningRequest, retrieval: TrustedResult):
        assert retrieval is seed
        return SemanticGraphExpansionResult(retrieval=expanded, readiness="ready")

    response = reason(
        _graph_expansion_request(
            seed,
            provider,
            answer=lambda _system, _user: {
                "answer": "Ada owns rollout.",
                "citations": ["c1"],
                "insufficient_evidence": False,
            },
        )
    )

    assert response.outcome == "answered"
    assert not response.provider_failures
    assert {item.chunk_id for item in response.trusted_evidence.items} == {"c1", "c2"}


def _answered_payload() -> dict:
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
    return response.to_dict()


def test_deserialization_rejects_unknown_outcome() -> None:
    payload = _answered_payload()
    payload["outcome"] = "answred"

    with pytest.raises(EvidenceValidationError, match="outcome"):
        reasoning_response_from_dict(payload)


def test_deserialization_rejects_unknown_bundle_decision() -> None:
    payload = _answered_payload()
    trusted_evidence = payload["trusted_evidence"]
    assert isinstance(trusted_evidence, dict)
    trusted_evidence["decision"] = "maybe"

    with pytest.raises(EvidenceValidationError, match="decision"):
        reasoning_response_from_dict(payload)


def test_deserialization_rejects_unknown_graph_expansion_mode() -> None:
    payload = _answered_payload()
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["graph_expansion_mode"] = "two_hop"

    with pytest.raises(EvidenceValidationError, match="graph_expansion_mode"):
        reasoning_response_from_dict(payload)


def test_deserialization_rejects_unknown_proposal_relation_and_status() -> None:
    proposal_payload = {
        "id": "p1",
        "source_evidence_ids": ["c1"],
        "proposed_relation": "supersedes",
        "subject_id": "s",
        "object_id": "o",
        "explanation": "x",
        "model_id": "m",
        "pipeline_id": "pipe",
        "provider_id": "prov",
        "provider_revision": "rev",
        "confidence": 0.5,
        "uncertainty": [],
        "generation_id": "gen_1",
        "status": "candidate",
    }

    payload = _answered_payload()
    payload["inference_proposals"] = [dict(proposal_payload, proposed_relation="friend_of")]
    with pytest.raises(EvidenceValidationError, match="proposed_relation"):
        reasoning_response_from_dict(payload)

    payload["inference_proposals"] = [dict(proposal_payload, status="approved")]
    with pytest.raises(EvidenceValidationError, match="status"):
        reasoning_response_from_dict(payload)


def test_deserialization_rejects_unknown_expansion_proposal_mode() -> None:
    payload = _answered_payload()
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["retrieval_expansion"] = {
        "attempted": True,
        "rounds": 1,
        "proposals": [{"id": "p1", "mode": "sideways", "query": "who owns rollout?"}],
    }

    with pytest.raises(EvidenceValidationError, match="mode"):
        reasoning_response_from_dict(payload)


def test_deserialization_rejects_non_numeric_int_string() -> None:
    payload = _answered_payload()
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["latency_ms"] = "abc"

    with pytest.raises(EvidenceValidationError, match="integer"):
        reasoning_response_from_dict(payload)


def test_deserialization_round_trip_still_works_after_enum_checks() -> None:
    payload = _answered_payload()

    round_tripped = reasoning_response_from_dict(payload)

    assert round_tripped.to_dict() == payload
