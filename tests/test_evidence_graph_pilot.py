from collections import Counter

from benchmarks.evidence_graph_eval import relation_control
from benchmarks.evidence_graph_pilot import _graph, run_deepseek_judge, run_pilot


def _observation(artifact, query_id: str, arm: str):
    return next(item for item in artifact.observations if item.query_id == query_id and item.arm == arm)


def test_shuffled_control_rewires_edges_without_changing_endpoint_degrees():
    graph = _graph()
    shuffled = relation_control(graph, "shuffled_relation_control", seed=1)
    original_pairs = {(relation.subject_id, relation.object_id) for relation in graph.relations}
    shuffled_pairs = {(relation.subject_id, relation.object_id) for relation in shuffled.relations}
    assert original_pairs != shuffled_pairs
    assert Counter(relation.subject_id for relation in graph.relations) == Counter(
        relation.subject_id for relation in shuffled.relations
    )
    assert Counter(relation.object_id for relation in graph.relations) == Counter(
        relation.object_id for relation in shuffled.relations
    )


def test_pilot_is_deterministic_except_for_timing():
    first = run_pilot().to_dict()
    second = run_pilot().to_dict()
    for artifact in (first, second):
        for observation in artifact["observations"]:
            observation.pop("latency_ms", None)
    assert first == second


def test_pilot_shows_graph_gain_and_trust_refusal():
    artifact = run_pilot()
    baseline = _observation(artifact, "supports_relation", "hybrid_retrieval")
    graph = _observation(artifact, "supports_relation", "deterministic_graph")
    assert baseline.appended_trusted_chunk_ids == ()
    assert graph.appended_trusted_chunk_ids == ("c2", "c8")
    assert set(graph.gold_evidence_chunk_ids).issubset(graph.citation_chunk_ids)

    guarded = _observation(artifact, "untrusted_neighbor", "deterministic_graph")
    assert guarded.appended_trusted_chunk_ids == ()
    assert guarded.rejected_candidate_count == 1


def test_deepseek_judge_compares_the_same_fixture_without_network_calls():
    calls = []

    def fake_completion(prompt: str, model: str) -> str:
        calls.append((prompt, model))
        return (
            '{"baseline_answer":"baseline", "graph_answer":"graph", '
            '"added_evidence_value":"useful", "winner":"graph", '
            '"rationale":"The added chunk answers the dependency question."}'
        )

    comparisons = run_deepseek_judge(run_pilot(), model="deepseek/test", completion=fake_completion)

    assert len(comparisons) == 6
    assert len(calls) == 6
    assert all(model == "deepseek/test" for _, model in calls)
    supports = next(item for item in comparisons if item.query_id == "supports_relation")
    assert supports.baseline_chunk_ids == ("c1",)
    assert supports.graph_chunk_ids == ("c1", "c2", "c8")
    assert "What evidence supports" in calls[1][0]
