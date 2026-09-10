import hashlib
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from recall.semantic_graph import build_semantic_graph, normalize_entity_name, relation_coverage
from recall.reasoning import (
    GenerationSelection,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
)
from recall.reasoning_graph import build_reasoning_graph
from recall.reasoning_planner import ReasoningBudget
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.trust_policy import TrustPolicy
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


def test_typed_authored_relation_coverage_reports_before_and_after_ingestion():
    """All five typed authored relations ingest through exact unique file endpoints.

    Invariant: coverage is zero filled before ingestion, then contains one authored row for
    each of supports, contradicts, depends_on, caused, and same_entity after ingestion. Red
    proof for node ``recall/semantic_graph.py::build_semantic_graph``: temporarily removing
    the ``file_entity_by_name`` fallback made this test fail at
    ``assert after["supports"]["authored"] == 1`` with a missing_evidence diagnostic. The
    fallback was restored before the green run. The frozen evaluation reused by this behavior
    is ``benchmarks/PREREGISTRATION-evidence-graph-precision-tuning-v1.md``.
    """
    targets = (
        Chunk("c2", "support.md", "support", {"file": "support.md"}),
        Chunk("c3", "contradiction.md", "contradiction", {"file": "contradiction.md"}),
        Chunk("c4", "dependency.md", "dependency", {"file": "dependency.md"}),
        Chunk("c5", "cause.md", "cause", {"file": "cause.md"}),
        Chunk("c6", "entity.md", "entity", {"file": "entity.md"}),
    )
    source = Chunk("c1", "decision.md", "decision", {"file": "decision.md"})
    before = relation_coverage(_graph(source, *targets))
    assert all(before[relation]["authored"] == 0 for relation in before)

    typed = replace(
        source,
        metadata={
            "file": "decision.md",
            "recall_graph": {
                "relations": [
                    {"relation": "supports", "subject": "decision.md", "object": "support.md"},
                    {
                        "relation": "contradicts",
                        "subject": "decision.md",
                        "object": "contradiction.md",
                    },
                    {
                        "relation": "depends_on",
                        "subject": "decision.md",
                        "object": "dependency.md",
                    },
                    {"relation": "caused", "subject": "decision.md", "object": "cause.md"},
                    {
                        "relation": "same_entity",
                        "subject": "decision.md",
                        "object": "entity.md",
                    },
                ]
            },
        },
    )
    after = relation_coverage(_graph(typed, *targets))
    assert {
        relation: after[relation]["authored"]
        for relation in ("supports", "contradicts", "depends_on", "caused", "same_entity")
    } == {relation: 1 for relation in ("supports", "contradicts", "depends_on", "caused", "same_entity")}
    assert all(after[relation]["candidate"] == 0 for relation in after)


def test_typed_relation_file_endpoint_ambiguity_fails_closed():
    graph = _graph(
        Chunk(
            "c1",
            "decision.md",
            "decision",
            {
                "file": "decision.md",
                "recall_graph": {
                    "relations": [
                        {"relation": "supports", "subject": "decision.md", "object": "policy.md"}
                    ]
                },
            },
        ),
        Chunk("c2", "one/policy.md", "one", {"file": "policy.md"}),
        Chunk("c3", "two/policy.md", "two", {"file": "policy.md"}),
    )
    assert not graph.relations
    assert any(diagnostic.kind == "missing_evidence" for diagnostic in graph.diagnostics)


def test_candidate_semantic_relations_are_not_traversed_or_trusted():
    """Candidate semantic edges remain exploratory and cannot add trusted evidence."""
    from recall_mcp.service import _expand_semantic_graph

    chunks = (
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "file": "seed.md",
                "recall_graph": {
                    "relations": [
                        {"relation": "supports", "subject": "seed.md", "object": "neighbor.md"}
                    ]
                },
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"file": "neighbor.md"}),
    )
    authored = _graph(*chunks)
    candidate = replace(authored.relations[0], status="candidate")
    projection = replace(authored, relations=(candidate,))
    assert relation_coverage(projection)["supports"] == {"authored": 0, "candidate": 1}

    class Store:
        tenant = "tenant-a"
        generation_id = "generation-a"

        def load_semantic_graph(self, generation_id=None):
            del generation_id
            return projection

        def graph_readiness(self):
            return projection.readiness()

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
    result = _expand_semantic_graph(
        Store(), request, retrieval, None, type("Embedder", (), {"embed_query": lambda self, _: [1.0]})()
    )
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed"]
    assert result.relation_seed_activations["supports"] == 0
    assert result.relation_new_trusted_evidence["supports"] == 0


def test_one_hop_expansion_applies_source_authorization_before_admission(monkeypatch):
    from recall_mcp import service

    chunks = (
        Chunk(
            "seed",
            "public/seed.md",
            "seed",
            {
                "file": "seed.md",
                "project": ["Seed", "Secret"],
                "relations": [{"relation": "supports", "subject": "Seed", "object": "Secret"}],
            },
        ),
        Chunk("secret", "secret/secret.md", "secret text", {"file": "secret.md", "project": "Secret"}),
    )
    semantic = _graph(*chunks)
    policy = SourceSecurityPolicy(
        (
            SourceRule("public", classification="public"),
            SourceRule("secret", classification="confidential", principals=frozenset({"allowed"})),
        )
    )
    context = AccessContext("denied", "tenant-a", clearance="confidential")

    class Store:
        tenant = "tenant-a"
        generation_id = "generation-a"

        def graph_readiness(self):
            return semantic.readiness()

        def load_semantic_graph(self, generation_id=None):
            assert generation_id == self.generation_id
            return semantic

        def iter_chunks(self):
            return iter(chunks)

        def cosines_for(self, ids, vec):
            del vec
            return {chunk_id: 0.95 for chunk_id in ids}

        def supersession(self):
            return {}, frozenset()

        def supersession_all(self):
            return {}, frozenset(), {}

    seed = TrustedHit(
        chunks[0],
        1.0,
        1.0,
        "ok",
        Provenance("public/seed.md", "seed.md", 0, None),
        Validity(None, None, None),
    )
    retrieval = TrustedResult(
        query="Seed",
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

    def fake_retrieve(*_args, **_kwargs):
        return type("Retrieved", (), {"result": retrieval})()

    monkeypatch.setattr(service, "_retrieve_trusted", fake_retrieve)
    response = service.reasoning_query(
        Store(),
        type("Embedder", (), {"embed_query": lambda self, _: [1.0]})(),
        "Seed",
        mode="evidence_assembly",
        graph_expansion="one_hop",
        policy=TrustPolicy.development(),
        security_policy=policy,
        access_context=context,
    )

    assert [item.chunk_id for item in response.trusted_evidence.items] == ["seed"]


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
    assert result.relation_seed_activations["supports"] == 1
    assert result.relation_candidates_accepted["supports"] == 2
    assert result.relation_new_trusted_evidence["supports"] == 1


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


def test_active_one_hop_serving_path_exposes_documented_policy_fingerprint(monkeypatch):
    """Parity guard for the serving provider described in docs/REASONING_GRAPH.md.

    Red proof recorded before the implementation for node
    ``tests/test_semantic_graph.py::test_active_one_hop_serving_path_exposes_documented_policy_fingerprint``:
    baseline ``HEAD=823ebb880f06f52a4a1ae336e3e3ebf213507892`` returned
    ``policy_fingerprint=None`` because ``_expand_semantic_graph`` projected the store directly
    and never populated the policy result field. The mutation restores the documented admission
    policy before projection. The production symbol under test is
    ``recall_mcp.service._expand_semantic_graph`` at line 2455, called by the
    ``reasoning_query`` graph expansion provider. The baseline failure was
    ``assert None == bd95d38fc1603f7f591096172bfd69a576ae99340afd1bda0d593a6b852e1854``.
    """
    from recall_mcp import service

    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "file": "seed.md",
                "project": ["A", "B"],
                "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"file": "neighbor.md", "project": "B"}),
    ]
    semantic = _graph(*chunks)
    projected = build_reasoning_graph(
        chunks,
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        include_text=True,
        semantic_graph=semantic,
    )

    class Store:
        tenant = "tenant-a"
        generation_id = "generation-a"

        def graph_readiness(self):
            return semantic.readiness()

        def cosines_for(self, ids, vec):
            del vec
            return {chunk_id: 0.95 for chunk_id in ids}

        def supersession(self):
            return {}, frozenset()

    monkeypatch.setattr(service, "project_store_graph", lambda *_args, **_kwargs: projected)
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

    result = service._expand_semantic_graph(
        Store(),
        request,
        retrieval,
        None,
        type("Embedder", (), {"embed_query": lambda self, _: [1.0]})(),
    )

    documented_policy = (
        "semantic_graph_precision_v1|combined|none|20260825|32|0.10|"
        "caused,depends_on,references,supports|contradicts,same_entity"
    )
    expected = hashlib.sha256(documented_policy.encode("utf-8")).hexdigest()
    assert result.policy_fingerprint == expected

    class Retrieved:
        result = retrieval

    monkeypatch.setattr(service, "_retrieve_trusted", lambda *_args, **_kwargs: Retrieved())
    response = service.reasoning_query(
        Store(),
        type("Embedder", (), {"embed_query": lambda self, _: [1.0]})(),
        "q",
        mode="evidence_assembly",
        graph_expansion="one_hop",
        policy=TrustPolicy.development(),
    )
    assert response.diagnostics.graph_policy_fingerprint == expected


def test_combined_no_selective_isolates_the_admission_gate(monkeypatch):
    """The diagnostic variant keeps every combined filter except selective admission.

    Red proof for node ``tests/test_semantic_graph.py::test_combined_no_selective_isolates_the_admission_gate``:
    before the variant was added, ``recall_mcp.service._graph_precision_settings`` normalized
    ``combined_no_selective`` to ``combined``. The failure reason was that the diagnostic could
    not distinguish selective admission from the other combined filters.
    """
    from recall_mcp import service

    monkeypatch.setenv("RECALL_GRAPH_PRECISION_VARIANT", "combined_no_selective")
    variant, relation_control, seed, hub_threshold, cosine_margin = (
        service._graph_precision_settings()
    )

    assert variant == "combined_no_selective"
    assert (relation_control, seed, hub_threshold, cosine_margin) == (
        "none",
        20260825,
        32,
        0.10,
    )
    assert service._graph_precision_feature_flags(variant) == (True, True, True, True, False)
    assert service._graph_precision_feature_flags("combined") == (True, True, True, True, True)

    monkeypatch.setenv("RECALL_GRAPH_COSINE_MARGIN", "0.20")
    assert service._graph_precision_settings()[-1] == 0.20
