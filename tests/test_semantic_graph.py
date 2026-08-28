from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from recall.semantic_graph import (
    SemanticRelation,
    build_semantic_graph,
    normalize_entity_name,
)
from recall.types import Chunk, Provenance, StalenessReport, TrustedHit, TrustedResult, Validity


def _graph(*chunks: Chunk):
    return build_semantic_graph(
        chunks,
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )


def _run_precision_graph(
    chunks: list[Chunk],
    seed_ids: tuple[str, ...] = ("seed",),
    *,
    projection=None,
    query_scores=None,
    query: str = "q",
    gap_warning: bool = False,
    max_graph_nodes: int = 32,
):
    from recall_mcp.service import _expand_semantic_graph
    from recall.reasoning import (
        GenerationSelection,
        ReasoningPolicy,
        ReasoningProviderPorts,
        ReasoningRequest,
    )
    from recall.reasoning_planner import ReasoningBudget

    graph = projection or _graph(*chunks)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    scores = query_scores or {}

    class Store:
        tenant = "tenant-a"
        generation_id = "generation-a"

        def iter_chunks(self):
            return iter(chunks)

        def load_semantic_graph(self, generation_id=None):
            del generation_id
            return graph

        def graph_readiness(self):
            return graph.readiness()

        def supersession_all(self):
            return {}, frozenset(), {}

        def supersession(self):
            return {}, frozenset()

        def cosines_for(self, ids, vec):
            del vec
            return {chunk_id: scores.get(chunk_id, 0.95) for chunk_id in ids}

    hits = [
        TrustedHit(
            chunk_by_id[chunk_id],
            1.0 - index * 0.05,
            1.0,
            "ok",
            Provenance(chunk_by_id[chunk_id].source, chunk_by_id[chunk_id].source, 0, None),
            Validity(None, None, None),
        )
        for index, chunk_id in enumerate(seed_ids)
    ]
    retrieval = TrustedResult(
        query=query,
        hits=hits,
        abstained=False,
        reason="",
        gap_warning=gap_warning,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="legacy_unbound",
    )
    request = ReasoningRequest(
        query=query,
        tenant_id="tenant-a",
        generation=GenerationSelection("generation-a", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_hops=1, max_graph_nodes=max_graph_nodes),
    )

    class Embedder:
        def embed_query(self, text):
            del text
            return [1.0]

    return _expand_semantic_graph(Store(), request, retrieval, None, Embedder())


def test_entity_normalization_is_unicode_and_whitespace_stable():
    assert normalize_entity_name("  RE-call\u00a0API  ") == "re call api"
    assert normalize_entity_name("RE-call API") == "re call api"


def test_graph_ids_are_stable_and_bound_to_tenant_and_generation():
    chunks = (Chunk("c1", "memo.md", "# Rollout", {"project": "RE-call"}),)
    first = _graph(*chunks)
    second = _graph(*chunks)
    other_tenant = build_semantic_graph(
        chunks,
        tenant_id="tenant-b",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )
    assert first.graph_id == second.graph_id
    assert first.fingerprint == second.fingerprint
    assert first.graph_id != other_tenant.graph_id


def test_ambiguous_exact_entity_kinds_do_not_create_implicit_merge():
    graph = _graph(
        Chunk("c1", "one.md", "", {"person": "Alex", "relations": []}),
        Chunk("c2", "two.md", "", {"project": "Alex", "relations": []}),
    )
    assert len([entity for entity in graph.entities if entity.normalized_name == "alex"]) == 2
    assert any(diagnostic.kind == "ambiguous_entity" for diagnostic in graph.diagnostics)


def test_relations_are_deduplicated_and_require_supporting_mentions():
    graph = _graph(
        Chunk(
            "c1",
            "memo.md",
            "",
            {
                "project": "RE-call",
                "service": "API",
                "relations": [
                    {"relation": "supports", "subject": "RE-call", "object": "API"},
                    {"relation": "supports", "subject": "RE-call", "object": "API"},
                    {"relation": "unknown", "subject": "RE-call", "object": "API"},
                ],
            },
        ),
        Chunk(
            "c2",
            "other.md",
            "",
            {"relations": [{"relation": "supports", "subject": "Missing", "object": "API"}]},
        ),
    )
    assert len(graph.relations) == 1
    assert sum(diagnostic.kind == "invalid_relation" for diagnostic in graph.diagnostics) == 1
    assert any(diagnostic.kind == "missing_evidence" for diagnostic in graph.diagnostics)


def test_projection_is_immutable():
    graph = _graph(Chunk("c1", "memo.md", "", {"project": "RE-call"}))
    with pytest.raises(FrozenInstanceError):
        graph.entities[0].canonical_name = "changed"
    with pytest.raises(TypeError):
        graph.entities[0].metadata["new"] = "value"


def test_graph_fingerprint_changes_when_source_changes():
    first = _graph(Chunk("c1", "memo.md", "", {"project": "RE-call"}))
    second = _graph(Chunk("c1", "memo.md", "", {"project": "Different"}))
    assert first.fingerprint != second.fingerprint


def test_explicit_aliases_resolve_to_one_canonical_entity():
    graph = _graph(
        Chunk(
            "c1",
            "project.md",
            "",
            {"project": "RE-call", "entity_aliases": {"RE-call": ["Recall", "re call"]}},
        ),
        Chunk("c2", "note.md", "", {"entities": ["Recall"]}),
    )
    entities = [entity for entity in graph.entities if entity.normalized_name == "re call"]
    assert len(entities) == 1
    assert {mention.entity_id for mention in graph.mentions if mention.mention_text == "Recall"} == {
        entities[0].id
    }
    assert not any(diagnostic.kind == "ambiguous_entity" for diagnostic in graph.diagnostics)


def test_frontmatter_graph_annotations_create_authored_relations():
    graph = _graph(
        Chunk(
            "c1",
            "decision.md",
            "",
            {
                "file": "decision.md",
                "recall_graph": {
                    "entities": [
                        {"name": "Rate Limits", "kind": "decision"},
                        {"name": "Gateway", "kind": "service"},
                    ],
                    "relations": [
                        {
                            "relation": "supports",
                            "subject": "Rate Limits",
                            "object": "Gateway",
                        }
                    ],
                },
            },
        )
    )
    assert len(graph.relations) == 1
    assert graph.relations[0].relation == "supports"
    assert graph.relations[0].evidence_chunk_ids == ("c1",)


def test_explicit_markdown_references_create_deterministic_reference_edges():
    graph = _graph(
        Chunk("c1", "decision.md", "See [the policy](policy.md).", {"file": "decision.md"}),
        Chunk("c2", "policy.md", "The policy.", {"file": "policy.md"}),
    )
    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert relation.relation == "references"
    assert relation.extraction_method == "explicit_reference"
    assert relation.evidence_chunk_ids == ("c1",)
    assert any(
        mention.chunk_id == "c1"
        and mention.mention_text == "policy.md"
        and mention.extraction_method == "explicit_reference"
        for mention in graph.mentions
    )


def test_ambiguous_file_reference_does_not_create_a_reference_edge():
    graph = _graph(
        Chunk("c1", "notes/decision.md", "See [policy](policy.md).", {"file": "decision.md"}),
        Chunk("c2", "one/policy.md", "one", {"file": "policy.md"}),
        Chunk("c3", "two/policy.md", "two", {"file": "policy.md"}),
    )
    assert not graph.relations
    assert any(
        diagnostic.kind == "ambiguous_entity" and diagnostic.reference == "policy.md"
        for diagnostic in graph.diagnostics
    )


def test_malformed_frontmatter_graph_annotation_is_diagnostic():
    graph = _graph(
        Chunk(
            "c1",
            "memo.md",
            "",
            {"file": "memo.md", "recall_graph": {"__parse_error__": "bad"}},
        )
    )
    assert any(diagnostic.kind == "invalid_relation" for diagnostic in graph.diagnostics)


def test_one_hop_expansion_appends_only_candidates_that_pass_trust():
    from recall_mcp.service import _expand_semantic_graph
    from recall.reasoning import (
        GenerationSelection,
        ReasoningPolicy,
        ReasoningProviderPorts,
        ReasoningRequest,
    )
    from recall.reasoning_planner import ReasoningBudget

    chunks = [
        Chunk(
            "c1",
            "seed.md",
            "seed",
            {
                "file": "seed.md",
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("c2", "neighbor.md", "neighbor", {"file": "neighbor.md", "project": "B"}),
        Chunk(
            "c3",
            "invalid.md",
            "invalid",
            {"file": "invalid.md", "project": "B", "valid_from": "not-a-date"},
        ),
    ]
    projection = _graph(*chunks)

    class Store:
        tenant = "tenant-a"
        generation_id = "generation-a"

        def iter_chunks(self):
            return iter(chunks)

        def load_semantic_graph(self, generation_id=None):
            return projection

        def graph_readiness(self):
            return projection.readiness()

        def supersession_all(self):
            return {}, frozenset(), {}

        def supersession(self):
            return {}, frozenset()

        def cosines_for(self, ids, vec):
            del vec
            return {chunk_id: 0.9 for chunk_id in ids}

    seed = TrustedHit(
        chunks[0],
        1.0,
        1.0,
        "ok",
        Provenance("seed.md", "seed.md", 0, None),
        Validity(None, None, None),
    )
    retrieval = TrustedResult(
        query="q",
        hits=[seed],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="legacy_unbound",
    )
    request = ReasoningRequest(
        query="q",
        tenant_id="tenant-a",
        generation=GenerationSelection("generation-a", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_hops=1),
    )
    class Embedder:
        def embed_query(self, text):
            del text
            return [1.0]

    result = _expand_semantic_graph(Store(), request, retrieval, None, Embedder())
    assert result.readiness == "ready"
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["c1", "c2"]
    assert result.candidates_discovered == 2
    assert result.candidates_rejected == 1


def test_directional_traversal_refuses_reverse_relation():
    chunks = [
        Chunk("seed", "seed.md", "seed", {"project": "A"}),
        Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
    ]
    graph = _graph(*chunks)
    entities = {entity.normalized_name: entity.id for entity in graph.entities}
    reverse = SemanticRelation(
        id="reverse-relation",
        tenant_id="tenant-a",
        generation_id="generation-a",
        subject_id=entities["b"],
        object_id=entities["a"],
        relation="supports",
        evidence_chunk_ids=("seed",),
        extraction_method="explicit_relation",
        confidence=1.0,
    )
    graph = replace(
        graph,
        mentions=tuple(mention for mention in graph.mentions if mention.entity_id == entities["a"]),
        relations=(reverse,),
    )
    result = _run_precision_graph(chunks, projection=graph)
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed"]
    assert dict(result.admission_rejections)["relation_direction"] == 1


@pytest.mark.parametrize("relation_kind", ("supports", "references", "depends_on", "caused"))
def test_directional_traversal_allows_supported_outgoing_relations(relation_kind):
    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "project": ["A", "B"],
                "relations": [{"relation": relation_kind, "subject": "A", "object": "B"}],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
    ]
    result = _run_precision_graph(chunks)
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed", "neighbor"]
    assert result.relations_inspected == 1


def test_contradiction_and_identity_relations_do_not_expand():
    for relation_kind in ("contradicts", "same_entity"):
        chunks = [
            Chunk(
                "seed",
                "seed.md",
                "seed",
                {
                    "project": ["A", "B"],
                    "relations": [
                        {"relation": relation_kind, "subject": "A", "object": "B"}
                    ],
                },
            ),
            Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
        ]
        result = _run_precision_graph(chunks)
        assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed"]
        assert dict(result.admission_rejections)["relation_type"] == 1


def test_relative_cosine_gate_rejects_distant_candidates():
    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
    ]
    result = _run_precision_graph(chunks, query_scores={"neighbor": 0.7})
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed"]
    assert dict(result.admission_rejections)["cosine_admission"] == 1


def test_hub_entity_is_refused_without_exact_query_alias(monkeypatch):
    monkeypatch.setenv("RECALL_GRAPH_PRECISION_VARIANT", "hub")
    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
    ] + [Chunk(f"hub-{index}", f"hub-{index}.md", "hub", {"project": "A"}) for index in range(32)]
    result = _run_precision_graph(chunks)
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed"]
    assert dict(result.admission_rejections)["hub_entity"] == 1


def test_hub_entity_exact_query_alias_can_expand(monkeypatch):
    monkeypatch.setenv("RECALL_GRAPH_PRECISION_VARIANT", "hub")
    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
    ] + [Chunk(f"hub-{index}", f"hub-{index}.md", "hub", {"project": "A"}) for index in range(32)]
    result = _run_precision_graph(chunks, query="about A")
    assert result.candidates_discovered >= 1
    assert "hub_entity" not in dict(result.admission_rejections)


def test_selective_gate_refuses_when_initial_retrieval_is_sufficient():
    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("seed-2", "seed-2.md", "seed two", {"project": "A"}),
        Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
    ]
    result = _run_precision_graph(chunks, seed_ids=("seed", "seed-2"))
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed", "seed-2"]
    assert result.gate_reason == "graph_gate_not_met"
    assert dict(result.admission_rejections)["selective_gate"] == 1


def test_selective_gate_allows_expansion_for_single_seed_with_gap():
    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"project": "B"}),
    ]
    result = _run_precision_graph(chunks, gap_warning=True)
    assert result.gate_reason is None
    assert result.candidates_discovered == 1


@pytest.mark.parametrize(
    "bad_confidence",
    [None, "high", 1.5, -0.5, float("nan"), True],
    ids=["none", "string", "above-range", "below-range", "nan", "bool"],
)
def test_invalid_authored_relation_confidence_is_a_diagnostic_not_an_error(bad_confidence):
    graph = _graph(
        Chunk(
            "c1",
            "memo.md",
            "",
            {
                "project": "RE-call",
                "service": "API",
                "relations": [
                    {
                        "relation": "supports",
                        "subject": "RE-call",
                        "object": "API",
                        "confidence": bad_confidence,
                    }
                ],
            },
        )
    )
    assert not graph.relations
    assert any(
        diagnostic.kind == "invalid_relation" and diagnostic.reference == "c1"
        for diagnostic in graph.diagnostics
    )


def test_valid_authored_relation_confidence_is_preserved():
    graph = _graph(
        Chunk(
            "c1",
            "memo.md",
            "",
            {
                "project": "RE-call",
                "service": "API",
                "relations": [
                    {
                        "relation": "supports",
                        "subject": "RE-call",
                        "object": "API",
                        "confidence": 0.25,
                    }
                ],
            },
        )
    )
    assert len(graph.relations) == 1
    assert graph.relations[0].confidence == 0.25
    assert not any(diagnostic.kind == "invalid_relation" for diagnostic in graph.diagnostics)


def test_load_semantic_graph_runs_all_reads_inside_one_transaction():
    from recall.semantic_graph import load_semantic_graph

    class Result:
        def fetchone(self):
            return None

        def fetchall(self):
            return []

    class Transaction:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            self._conn.in_transaction = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self._conn.in_transaction = False
            return False

    class Connection:
        def __init__(self):
            self.in_transaction = False
            self.reads_in_transaction = []

        def transaction(self):
            return Transaction(self)

        def execute(self, sql, params=None):
            del sql, params
            self.reads_in_transaction.append(self.in_transaction)
            return Result()

    conn = Connection()
    assert load_semantic_graph(conn, "tenant-a", "generation-a") is None
    assert len(conn.reads_in_transaction) == 4
    assert all(conn.reads_in_transaction), "every read must run inside one transaction"
    assert conn.in_transaction is False
