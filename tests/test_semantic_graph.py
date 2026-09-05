from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from recall.semantic_graph import build_semantic_graph, normalize_entity_name
from recall.types import Chunk, Provenance, StalenessReport, TrustedHit, TrustedResult, Validity


def _graph(*chunks: Chunk):
    return build_semantic_graph(
        chunks,
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )


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

    class Embedder:
        def embed_query(self, text):
            assert text == "q"
            return [1.0]

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
    result = _expand_semantic_graph(Store(), request, retrieval, None, Embedder())
    assert result.readiness == "ready"
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["c1", "c2"]
    assert result.candidates_discovered == 2
    assert result.candidates_rejected == 1


def test_graph_relation_must_be_evidenced_by_a_trusted_seed_chunk():
    from recall_mcp.service import _expand_semantic_graph
    from recall.reasoning import (
        GenerationSelection,
        ReasoningPolicy,
        ReasoningProviderPorts,
        ReasoningRequest,
    )
    from recall.reasoning_planner import ReasoningBudget

    chunks = [
        Chunk("seed", "seed.md", "seed", {"file": "seed.md", "project": "A"}),
        Chunk(
            "relation",
            "relation.md",
            "relation",
            {
                "file": "relation.md",
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"file": "neighbor.md", "project": "B"}),
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

    request = ReasoningRequest(
        query="q",
        tenant_id="tenant-a",
        generation=GenerationSelection("generation-a", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: None),  # type: ignore[arg-type]
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_hops=1),
    )
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

    class Embedder:
        def embed_query(self, text):
            assert text == "q"
            return [1.0]

    result = _expand_semantic_graph(Store(), request, retrieval, None, Embedder())
    assert result.readiness == "ready"
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed"]
    assert result.candidates_discovered == 0


def test_graph_candidate_uses_query_cosine_not_relation_confidence():
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
            "seed",
            "seed.md",
            "seed",
            {
                "file": "seed.md",
                "project": ["A", "B"],
                "relations": [
                    {
                        "relation": "supports",
                        "subject": "A",
                        "object": "B",
                        "confidence": 1.0,
                    }
                ],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"file": "neighbor.md", "project": "B"}),
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
            return {chunk_id: 0.1 for chunk_id in ids}

    request = ReasoningRequest(
        query="q",
        tenant_id="tenant-a",
        generation=GenerationSelection("generation-a", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: None),  # type: ignore[arg-type]
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_hops=1),
    )
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

    class Embedder:
        def embed_query(self, text):
            assert text == "q"
            return [1.0]

    result = _expand_semantic_graph(Store(), request, retrieval, None, Embedder())
    assert result.readiness == "ready"
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed"]
    assert result.candidates_rejected == 1
