from pathlib import Path

from recall.semantic_graph import build_semantic_graph
from recall.types import Chunk
from benchmarks.evidence_graph_eval import (
    EvidenceGraphEvaluationArtifact,
    GraphEvaluationObservation,
    relation_control,
)
from scripts.summarize_graph_precision_eval import _non_increase_guardrail, _relation_totals


def _graph():
    return build_semantic_graph(
        (
            Chunk(
                "c1",
                "memo.md",
                "",
                {
                    "project": "A",
                    "service": "B",
                    "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
                },
            ),
        ),
        tenant_id="tenant-a",
        generation_id="generation-a",
    )


def test_relation_controls_are_deterministic_and_detached():
    graph = _graph()
    shuffled = relation_control(graph, "shuffled_relation_control", seed=7)
    removed = relation_control(graph, "removed_relation_control", seed=7)
    assert shuffled is not graph
    assert shuffled.graph_id != graph.graph_id
    assert len(shuffled.relations) == len(graph.relations)
    assert removed.relations == ()
    assert graph.relations


def test_precision_summary_reports_typed_activation_and_quality_guardrails():
    rows = [
        {
            "graph_relation_seed_activations": {"supports": 2},
            "graph_relation_candidates_accepted": {"supports": 1, "caused": 1},
            "graph_relation_new_trusted_evidence": {"supports": 1},
        }
    ]
    assert _relation_totals(rows, "graph_relation_seed_activations")["supports"] == 2
    assert _relation_totals(rows, "graph_relation_candidates_accepted")["caused"] == 1
    assert _relation_totals(rows, "graph_relation_new_trusted_evidence")["supports"] == 1
    baseline = {"q1": {"false_abstention": False, "unsupported_claim_count": 0}}
    candidate = {"q1": {"false_abstention": False, "unsupported_claim_count": 0}}
    assert _non_increase_guardrail(baseline, candidate, "false_abstention")["passes"] is True
    candidate["q1"]["false_abstention"] = True
    assert _non_increase_guardrail(baseline, candidate, "false_abstention")["passes"] is False


def test_evaluation_artifact_retains_sanitized_per_query_observations(tmp_path: Path):
    artifact = EvidenceGraphEvaluationArtifact()
    artifact.add(
        GraphEvaluationObservation(
            query_id="q1",
            arm="deterministic_graph",
            tenant_id="tenant-a",
            generation_id="generation-a",
            pipeline_fingerprint=None,
            corpus_fingerprint=None,
            calibration_id=None,
            graph_expansion_mode="one_hop",
            graph_readiness="ready",
            graph_fingerprint="g1",
            relation_control_seed=None,
            initial_trusted_chunk_ids=("c1",),
            appended_trusted_chunk_ids=("c2",),
        )
    )
    output = tmp_path / "evaluation.json"
    artifact.write(output)
    text = output.read_text(encoding="utf-8")
    assert '"query_id": "q1"' in text
    assert "corpus text" not in text
