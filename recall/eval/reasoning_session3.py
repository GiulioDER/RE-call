"""Session 3 proposal protocol control artifact."""

from __future__ import annotations

import json
from typing import Any

from recall.eval.metrics import nan_to_null
from recall.reasoning_graph import ReasoningGraphProjection, build_reasoning_graph
from recall.reasoning_proposals import (
    ProposalContext,
    proposal_precision_recall,
    proposal_report,
)
from recall.types import Chunk


class _RejectedProvider:
    provider_id = "offline-model-control"
    model_id = "offline-proposal-model"
    provider_revision = "rev-rejected-example"
    max_proposals = 3

    def propose(
        self, graph: ReasoningGraphProjection, context: ProposalContext
    ) -> list[dict[str, Any]]:
        return [
            {
                "source_evidence_ids": (graph.nodes[0].id,),
                "proposed_relation": "references",
                "subject_id": "search_policy_v1.md",
                "object_id": "retry_policy.md",
                "explanation": "Rejected example: lexical overlap is not a relation.",
                "confidence": 0.12,
                "uncertainty": ["no second supporting evidence item"],
                "status": "rejected",
                "rule_id": "offline.rejected_lexical_overlap",
            }
        ]


class _FailureProvider:
    provider_id = "offline-model-control"
    model_id = "offline-proposal-model"
    provider_revision = "rev-failure-example"
    max_proposals = 1

    def __init__(self, output: Any) -> None:
        self.output = output

    def propose(self, graph: ReasoningGraphProjection, context: ProposalContext) -> Any:
        if isinstance(self.output, BaseException):
            raise self.output
        return self.output


def _chunk(
    cid: str,
    file: str,
    text: str,
    *,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> Chunk:
    metadata: dict[str, Any] = {"file": file, "ord": 0}
    if valid_from is not None:
        metadata["valid_from"] = valid_from
    if valid_until is not None:
        metadata["valid_until"] = valid_until
    return Chunk(cid, f"/session3/{file}", text, metadata)


def _fixture_graph() -> ReasoningGraphProjection:
    return build_reasoning_graph(
        [
            _chunk(
                "search-v1",
                "search_policy_v1.md",
                "decision: search policy. Status: active.",
                valid_from="2026-01-01",
            ),
            _chunk(
                "search-v2",
                "search_policy_v2.md",
                "decision: search policy. Status: active. This updates search_policy_v1.md.",
                valid_from="2026-02-01",
            ),
            _chunk(
                "cache-v1",
                "cache_policy_v1.md",
                "decision: cache policy. Status: enabled.",
                valid_from="2026-01-01",
                valid_until="2026-03-31",
            ),
            _chunk(
                "cache-v2",
                "cache_policy_v2.md",
                "decision: cache policy. Status: disabled.",
                valid_from="2026-02-01",
                valid_until="2026-04-30",
            ),
            _chunk(
                "retry",
                "retry_policy.md",
                "decision: retry policy. Status: active.",
                valid_from="2026-03-01",
            ),
        ],
        tenant_id="session3",
        generation_id="reasoning-session3-fixture-gen1",
        pipeline_fingerprint="reasoning-session3-pipeline-v1",
        include_text=True,
    )


def _proposal_json(proposal: Any) -> dict[str, Any]:
    return {
        "id": proposal.id,
        "source_evidence_ids": list(proposal.source_evidence_ids),
        "proposed_relation": proposal.proposed_relation,
        "subject_id": proposal.subject_id,
        "object_id": proposal.object_id,
        "explanation": proposal.explanation,
        "provider_id": proposal.provider_id,
        "model_id": proposal.model_id,
        "provider_revision": proposal.provider_revision,
        "pipeline_id": proposal.pipeline_id,
        "generation_id": proposal.generation_id,
        "confidence": proposal.confidence,
        "uncertainty": list(proposal.uncertainty),
        "status": proposal.status,
        "rule_id": proposal.rule_id,
        "metadata": dict(proposal.metadata),
    }


def reasoning_session3_artifact() -> dict[str, Any]:
    graph = _fixture_graph()
    deterministic_report = proposal_report(graph)
    rejected_report = proposal_report(graph, model_provider=_RejectedProvider())
    expected_pairs = {
        ("search_policy_v1.md", "search_policy_v2.md"),
        ("cache_policy_v1.md", "cache_policy_v2.md"),
    }
    failures = [
        proposal_report(graph, model_provider=_FailureProvider(TimeoutError("timeout"))),
        proposal_report(graph, model_provider=_FailureProvider([{"source_evidence_ids": []}])),
        proposal_report(
            graph,
            model_provider=_FailureProvider(
                [
                    {
                        "source_evidence_ids": [],
                        "proposed_relation": "references",
                        "subject_id": "a",
                        "object_id": "b",
                        "explanation": "too many",
                        "confidence": None,
                        "uncertainty": [],
                    },
                    {
                        "source_evidence_ids": [],
                        "proposed_relation": "references",
                        "subject_id": "c",
                        "object_id": "d",
                        "explanation": "too many",
                        "confidence": None,
                        "uncertainty": [],
                    },
                ]
            ),
        ),
        proposal_report(graph, model_provider=_FailureProvider(RuntimeError("boom"))),
    ]
    failure_matrix = dict(deterministic_report.failure_matrix)
    for report in failures:
        for key, value in report.failure_matrix.items():
            failure_matrix[key] += value
    return {
        "version": "reasoning-session3-v0.1.0",
        "protocol_spec": "docs/INFERENCE_PROPOSALS.md",
        "generation_id": graph.generation_id,
        "pipeline_id": deterministic_report.pipeline_id,
        "deterministic_rules": sorted(
            {proposal.rule_id for proposal in deterministic_report.proposals if proposal.rule_id}
        ),
        "precision_recall": dict(
            proposal_precision_recall(deterministic_report.proposals, expected_pairs)
        ),
        "proposal_count": len(deterministic_report.proposals),
        "proposals": [_proposal_json(proposal) for proposal in deterministic_report.proposals],
        "rejected_proposal_examples": [
            _proposal_json(proposal) for proposal in rejected_report.rejected_proposals
        ],
        "provider_failure_matrix": failure_matrix,
        "audit": {
            "side_effect_free": True,
            "authored_edges_unchanged": graph.authored_edges == (),
            "proposal_output_promotes_trust": False,
            "all_proposals_trace_to_evidence": all(
                proposal.source_evidence_ids for proposal in deterministic_report.proposals
            ),
        },
    }


def _artifact_json() -> str:
    """Serialise strictly. `proposal_precision_recall` reports NaN on no data, and `NaN` is not
    valid JSON, so sanitise to null first and let `allow_nan=False` catch anything missed."""

    return json.dumps(
        nan_to_null(reasoning_session3_artifact()), indent=2, sort_keys=True, allow_nan=False
    )


def main() -> None:
    print(_artifact_json())


if __name__ == "__main__":
    main()
