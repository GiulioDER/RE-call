from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from recall.reasoning_graph import build_reasoning_graph
from recall.reasoning_planner import ReasoningBudget, plan_multi_hop_evidence
import pytest

from recall.reasoning_proposals import InferenceProposal, deterministic_inference_proposals
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


def _trusted_hit(chunk: Chunk, *, verdict: str = "ok") -> TrustedHit:
    return TrustedHit(
        chunk=chunk,
        cosine=0.91,
        confidence=0.98,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(
            source=chunk.source,
            file=chunk.metadata["file"],
            ord=0,
            indexed_at=NOW,
        ),
        validity=Validity(valid_from=NOW, valid_until=None, superseded_by=None),
    )


def _result(*hits: TrustedHit, abstained: bool = False, reason: str = "") -> TrustedResult:
    return TrustedResult(
        query="what is the current rollout policy?",
        hits=list(hits),
        abstained=abstained,
        reason=reason,
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="gen_1"),
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
    )


def _proposal(graph, subject: str, obj: str, relation: str = "supersedes") -> InferenceProposal:
    evidence = tuple(node.id for node in graph.nodes if node.kind == "chunk")[:2]
    return InferenceProposal(
        id=f"proposal_{subject}_{relation}_{obj}",
        source_evidence_ids=evidence,
        proposed_relation=relation,  # type: ignore[arg-type]
        subject_id=subject,
        object_id=obj,
        explanation="fixture proposal",
        model_id="rules",
        pipeline_id="pipe-a",
        provider_id="test",
        provider_revision="rev",
        confidence=0.5,
        uncertainty=(),
        generation_id=graph.generation_id,
    )


def test_planner_expands_trusted_hit_through_authored_graph_edge() -> None:
    old = _chunk("old", "rollout_v1.md", "rollout is off")
    new = _chunk("new", "rollout_v2.md", "rollout is on", supersedes="rollout_v1.md")
    graph = build_reasoning_graph([old, new], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(_result(_trusted_hit(old)), graph)

    accepted_files = {decision.file for decision in plan.trace.evidence_accepted}
    assert plan.outcome == "completed"
    assert {"rollout_v1.md", "rollout_v2.md"} <= accepted_files
    assert [step.operation for step in plan.trace.expansion_steps] == [
        "retrieve_related_claims",
        "follow_authored_relationships",
        "compare_candidate_memories",
        "search_missing_intermediate_evidence",
        "check_temporal_consistency",
        "check_contradiction",
    ]
    assert plan.trace.initial_retrieval.trusted_hit_ids == ("old",)


def test_planner_uses_inference_proposals_for_exploration_not_trust() -> None:
    old = _chunk("old", "cache_v1.md", "cache ttl is 30")
    new = _chunk("new", "cache_v2.md", "cache ttl is 60")
    graph = build_reasoning_graph([old, new], tenant_id="acme", generation_id="gen_1")
    proposal = _proposal(graph, "cache_v1.md", "cache_v2.md")

    plan = plan_multi_hop_evidence(_result(_trusted_hit(old)), graph, proposals=(proposal,))

    proposal_trace = plan.trace.inference_proposals_used_for_exploration
    assert plan.outcome == "completed"
    assert {decision.file for decision in plan.trace.evidence_accepted} == {"cache_v1.md"}
    assert proposal_trace[0].proposal_id == proposal.id
    assert proposal_trace[0].used_for_exploration is True
    assert proposal_trace[0].trusted_evidence is False
    assert any(gap.kind == "missing_authored_edge" for gap in plan.trace.unresolved_gaps)


def test_planner_fails_closed_when_retrieval_abstained() -> None:
    graph = build_reasoning_graph([], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(
        _result(abstained=True, reason="no trusted evidence"),
        graph,
    )

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "retrieval_abstained"


def test_planner_fails_closed_when_retrieval_and_graph_generation_do_not_match() -> None:
    chunk = _chunk("old", "rollout_v1.md", "rollout")
    graph = build_reasoning_graph([chunk], tenant_id="acme", generation_id="gen_2")

    plan = plan_multi_hop_evidence(_result(_trusted_hit(chunk)), graph)

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "unsupported_evidence"
    assert plan.trace.unresolved_gaps[0].kind == "retrieval_graph_binding_mismatch"


def test_planner_fails_closed_when_initial_evidence_has_blocking_graph_diagnostic() -> None:
    first = _chunk("first", "shared.md", "first copy")
    second = Chunk(
        "second",
        "/other/shared.md",
        "second copy",
        {"file": "shared.md", "ord": 0, "valid_from": "2026-01-01"},
    )
    graph = build_reasoning_graph([first, second], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(_result(_trusted_hit(first)), graph)

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "ambiguous_evidence"
    assert plan.trace.expansion_steps == ()
    assert plan.trace.evidence_rejected[0].chunk_id == "first"


def test_planner_fails_closed_for_non_canonical_cycle_member() -> None:
    first = _chunk("first", "cycle_a.md", "cycle a", supersedes="cycle_b.md")
    second = _chunk("second", "cycle_b.md", "cycle b", supersedes="cycle_a.md")
    graph = build_reasoning_graph([first, second], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(_result(_trusted_hit(second)), graph)

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "ambiguous_evidence"


def test_planner_enforces_node_budget_explicitly() -> None:
    old = _chunk("old", "rollout_v1.md", "rollout is off")
    new = _chunk("new", "rollout_v2.md", "rollout is on", supersedes="rollout_v1.md")
    graph = build_reasoning_graph([old, new], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(
        _result(_trusted_hit(old)),
        graph,
        budget=ReasoningBudget(max_graph_nodes=1),
    )

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "budget_exhausted"
    assert plan.budget_used.graph_nodes == 1


def test_planner_fails_closed_when_reachable_candidates_overflow_node_budget() -> None:
    old = _chunk("old", "rollout_v1.md", "rollout is off")
    extra = Chunk(
        "extra",
        old.source,
        "more rollout evidence",
        {"file": "rollout_v1.md", "ord": 1, "valid_from": "2026-01-01"},
    )
    graph = build_reasoning_graph([old, extra], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(
        _result(_trusted_hit(old)),
        graph,
        budget=ReasoningBudget(max_graph_nodes=1),
    )

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "budget_exhausted"


def test_planner_enforces_step_budget_explicitly() -> None:
    old = _chunk("old", "rollout_v1.md", "rollout is off")
    new = _chunk("new", "rollout_v2.md", "rollout is on", supersedes="rollout_v1.md")
    graph = build_reasoning_graph([old, new], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(
        _result(_trusted_hit(old)),
        graph,
        budget=ReasoningBudget(max_steps=1),
    )

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "budget_exhausted"
    assert [step.operation for step in plan.trace.expansion_steps] == ["retrieve_related_claims"]


def test_planner_enforces_token_budget_on_initial_evidence() -> None:
    chunk = _chunk("old", "rollout_v1.md", "one two three four five")
    graph = build_reasoning_graph(
        [chunk],
        tenant_id="acme",
        generation_id="gen_1",
        include_text=True,
    )

    plan = plan_multi_hop_evidence(
        _result(_trusted_hit(chunk)),
        graph,
        budget=ReasoningBudget(max_evidence_tokens=2),
    )

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "budget_exhausted"
    assert plan.trace.evidence_rejected[0].chunk_id == "old"


def test_planner_enforces_model_call_budget_before_expansion() -> None:
    chunk = _chunk("old", "rollout_v1.md", "rollout")
    graph = build_reasoning_graph([chunk], tenant_id="acme", generation_id="gen_1")

    plan = plan_multi_hop_evidence(
        _result(_trusted_hit(chunk)),
        graph,
        budget=ReasoningBudget(max_model_calls=0),
        model_calls_used=1,
    )

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "budget_exhausted"
    assert plan.trace.expansion_steps == ()


def test_planner_rejects_negative_model_call_usage() -> None:
    chunk = _chunk("old", "rollout_v1.md", "rollout")
    graph = build_reasoning_graph([chunk], tenant_id="acme", generation_id="gen_1")

    with pytest.raises(ValueError, match="model_calls_used"):
        plan_multi_hop_evidence(
            _result(_trusted_hit(chunk)),
            graph,
            model_calls_used=-1,
        )


def test_planner_enforces_wall_time_budget() -> None:
    chunk = _chunk("old", "rollout_v1.md", "rollout")
    graph = build_reasoning_graph([chunk], tenant_id="acme", generation_id="gen_1")
    ticks = iter((0.0, 0.2, 0.2))

    plan = plan_multi_hop_evidence(
        _result(_trusted_hit(chunk)),
        graph,
        budget=ReasoningBudget(max_wall_time_ms=100),
        clock=lambda: next(ticks),
    )

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "budget_exhausted"


def test_planner_fails_closed_on_unsupported_proposal_citation() -> None:
    old = _chunk("old", "cache_v1.md", "cache ttl is 30")
    new = _chunk("new", "cache_v2.md", "cache ttl is 60")
    graph = build_reasoning_graph([old, new], tenant_id="acme", generation_id="gen_1")
    proposal = InferenceProposal(
        id="bad_proposal",
        source_evidence_ids=("missing-node",),
        proposed_relation="supersedes",
        subject_id="cache_v1.md",
        object_id="cache_v2.md",
        explanation="bad citation",
        model_id="rules",
        pipeline_id="pipe-a",
        provider_id="test",
        provider_revision="rev",
        confidence=0.5,
        uncertainty=(),
        generation_id=graph.generation_id,
    )

    plan = plan_multi_hop_evidence(_result(_trusted_hit(old)), graph, proposals=(proposal,))

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "unsupported_evidence"


def test_planner_detects_deterministic_contradictions_that_cite_source_nodes() -> None:
    left = _chunk(
        "left",
        "feature_left.md",
        "decision: feature. Status: enabled.",
    )
    right = _chunk(
        "right",
        "feature_right.md",
        "decision: feature. Status: disabled.",
    )
    graph = build_reasoning_graph(
        [left, right],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        include_text=True,
    )
    proposals = deterministic_inference_proposals(graph)

    plan = plan_multi_hop_evidence(_result(_trusted_hit(left)), graph, proposals=proposals)

    assert plan.outcome == "failed_closed"
    assert plan.stop_reason == "ambiguous_evidence"
    assert any(gap.kind == "contradiction" for gap in plan.trace.unresolved_gaps)
