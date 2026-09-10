import hashlib
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from recall.calibration import Calibration
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


def test_supersession_relation_carries_effective_date_and_validity_window():
    graph = _graph(
        Chunk("old", "old.md", "old", {"file": "old.md", "project": "A"}),
        Chunk(
            "new",
            "new.md",
            "new",
            {
                "file": "new.md",
                "project": "A",
                "supersedes": "old.md",
                "valid_from": "2026-01-01",
                "valid_until": "2026-12-31",
            },
        ),
    )
    relation = next(item for item in graph.relations if item.relation == "supersedes")
    assert relation.effective_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert relation.valid_from == datetime(2026, 1, 1, tzinfo=UTC)
    assert relation.valid_until == datetime(2026, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
    assert relation.metadata["edge_kind"] == "supersession"


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
    assert result.candidates_discovered == 1
    assert result.candidates_rejected == 1
    assert dict(result.admission_rejections)["invalid_temporal_metadata"] == 1
    assert result.relation_seed_activations["supports"] == 1
    assert result.relation_candidates_accepted["supports"] == 1
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


def test_graph_expansion_enforces_the_category_entity_budget():
    """A breadth budget must cap distinct neighboring entities, not only chunk count.

    Invariant: one graph pass with ``max_graph_entities=1`` admits candidates from at most one
    neighboring entity and records the overflow. The failure mode is a node only budget that lets
    one high degree entity consume the whole list recall expansion. The baseline mutation is to
    remove the ``neighboring_entities`` check in ``recall_mcp.service._expand_semantic_graph``;
    this test then returns both neighbors instead of one.
    """
    from recall_mcp.service import _expand_semantic_graph

    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "file": "seed.md",
                "project": ["A", "B", "C"],
                "relations": [
                    {"relation": "supports", "subject": "A", "object": "B"},
                    {"relation": "supports", "subject": "A", "object": "C"},
                ],
            },
        ),
        Chunk("neighbor-b", "b.md", "b", {"file": "b.md", "project": "B"}),
        Chunk("neighbor-c", "c.md", "c", {"file": "c.md", "project": "C"}),
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
        query="list every project",
        hits=[seed],
        abstained=False,
        reason="",
        gap_warning=True,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="legacy_unbound",
    )
    request = ReasoningRequest(
        query="list every project",
        tenant_id="tenant-a",
        generation=GenerationSelection("generation-a", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=10, max_graph_entities=1, max_graph_hops=1),
    )

    result = _expand_semantic_graph(
        Store(), request, retrieval, None, type("Embedder", (), {"embed_query": lambda self, _: [1.0]})()
    )

    assert len([hit for hit in result.retrieval.hits if hit.chunk.id != "seed"]) == 1
    assert dict(result.admission_rejections)["entity_budget"] >= 1


def test_temporal_and_supersession_neighbors_are_filtered_before_budget():
    """Stale graph neighbors cannot consume the bounded candidate or ranking path.

    Invariant: with one candidate slot, an expired neighbor followed by a current neighbor leaves
    the current neighbor admitted and sends only that neighbor to cosine scoring. The failure mode
    is the pre fix loop, which charged the first neighbor before trust evaluation and therefore
    starved the current neighbor. The required red proof is a deliberate mutation that moves the
    temporal filter below the candidate budget admission in ``recall_mcp.service._expand_semantic_graph``;
    that mutation makes the intended live neighbor assertion fail. This test also checks the
    supersession filter at the same pre ranking boundary.
    """
    from recall_mcp.service import _expand_semantic_graph

    as_of = datetime(2026, 6, 1, tzinfo=UTC)
    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "file": "seed.md",
                "project": ["A", "Expired", "Superseded", "Live"],
                "relations": [
                    {"relation": "supports", "subject": "A", "object": "Expired"},
                    {"relation": "supports", "subject": "A", "object": "Superseded"},
                    {"relation": "supports", "subject": "A", "object": "Live"},
                ],
            },
        ),
        Chunk(
            "expired",
            "expired.md",
            "expired",
            {"file": "expired.md", "project": "Expired", "valid_until": "2025-12-31"},
        ),
        Chunk(
            "superseded",
            "superseded.md",
            "superseded",
            {"file": "superseded.md", "project": "Superseded"},
        ),
        Chunk(
            "live",
            "live.md",
            "live",
            {"file": "live.md", "project": "Live", "valid_from": "2026-01-01"},
        ),
    ]
    projection = _graph(*chunks)
    scored_ids: list[str] = []

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
            return {"superseded": "replacement"}, frozenset(), {}

        def cosines_for(self, ids, vec):
            del vec
            scored_ids.extend(ids)
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
        query="which project is live",
        hits=[seed],
        abstained=False,
        reason="",
        gap_warning=True,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="legacy_unbound",
    )
    request = ReasoningRequest(
        query="which project is live",
        tenant_id="tenant-a",
        generation=GenerationSelection("generation-a", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=2, max_graph_hops=1),
        as_of=as_of,
    )

    result = _expand_semantic_graph(
        Store(), request, retrieval, None, type("Embedder", (), {"embed_query": lambda self, _: [1.0]})()
    )

    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed", "live"]
    assert scored_ids == ["live"]
    assert dict(result.admission_rejections)["temporal_expired"] >= 1
    assert dict(result.admission_rejections)["superseded"] >= 1
    assert dict(result.admission_rejections).get("budget", 0) == 0


def test_query_entity_resolution_activates_alias_and_date_seed_without_llm():
    """A query can activate a graph endpoint that the trusted seed does not mention verbatim.

    Invariant: a unique file entity resolved from an alias and an equivalent date spelling joins
    the seed entity set, so its authored outgoing relation is traversed. The failure mode is the
    pre fix ``seed_entities`` set, which contains only mentions on the trusted seed chunk and
    reports ``relation_not_seeded``. The required red proof is this exact test against the
    pre change consumer boundary, not collection failure for a new helper; the production symbol
    under test is ``recall_mcp.service._expand_semantic_graph``.
    """
    from recall_mcp.service import _expand_semantic_graph

    chunks = [
        Chunk(
            "seed",
            "seed.md",
            "seed",
            {
                "file": "seed.md",
                "recall_graph": {
                    "relations": [
                        {
                            "relation": "supports",
                            "subject": "target.md",
                            "object": "neighbor.md",
                        }
                    ]
                },
            },
        ),
        Chunk(
            "target",
            "target.md",
            "target",
            {
                "file": "target.md",
                "entity_aliases": {"target.md": ["Target Run", "2026-08-25"]},
            },
        ),
        Chunk("neighbor", "neighbor.md", "neighbor", {"file": "neighbor.md"}),
    ]
    projection = _graph(*chunks)

    class Store:
        tenant = "tenant-a"
        generation_id = "generation-a"

        def iter_chunks(self):
            return iter(chunks)

        def load_semantic_graph(self, generation_id=None):
            assert generation_id == self.generation_id
            return projection

        def graph_readiness(self):
            return projection.readiness()

        def supersession_all(self):
            return {}, frozenset(), {}

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
        query="what changed for the target run on August 25, 2026",
        hits=[seed],
        abstained=False,
        reason="",
        gap_warning=True,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="legacy_unbound",
    )
    request = ReasoningRequest(
        query=retrieval.query,
        tenant_id="tenant-a",
        generation=GenerationSelection("generation-a", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=2, max_graph_hops=1),
    )

    result = _expand_semantic_graph(
        Store(),
        request,
        retrieval,
        None,
        type("Embedder", (), {"embed_query": lambda self, _: [1.0]})(),
    )

    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed", "neighbor"]
    assert result.relation_seed_activations["supports"] == 1


def test_graph_candidate_uses_calibrated_rerank_without_cosine_admission():
    """A low cosine is reranked and then judged by trust instead of hard rejected.

    Invariant: graph admission must not apply ``best_seed_cosine - margin``. The regression is a
    candidate with cosine ``0.10`` and relation confidence ``1.0`` that clears an explicit
    calibration threshold but is far below the old seed margin. A mutation that restores the old
    ``cosine_admission`` branch must fail the ``neighbor`` evidence assertion. The production
    symbol under test is ``recall_mcp.service._expand_semantic_graph``.
    """
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

    result = _expand_semantic_graph(
        Store(),
        request,
        retrieval,
        Calibration("test", threshold=0.0, scale=0.1),
        Embedder(),
    )
    assert result.readiness == "ready"
    assert [hit.chunk.id for hit in result.retrieval.hits] == ["seed", "neighbor"]
    assert result.retrieval.hits[1].cosine == 0.1
    assert result.candidates_rejected == 0


def test_graph_rerank_combines_structural_features_and_preserves_direct_retrieval():
    """Graph reranking orders graph candidates without reordering direct retrieval.

    The production symbol is ``recall_mcp.service._merge_graph_hits``. Its baseline behavior at
    ``HEAD=8203c0c8`` kept only two direct anchors, so the graph candidate appeared before
    ``original-3``. The expected order below is the red proof for that mutation's failure mode;
    the graph fill policy restores every direct hit before considering graph evidence.
    """
    from recall_mcp import service

    calibration = Calibration("test", threshold=0.65, scale=0.1)
    structurally_supported = service._GraphCandidate(
        trusted_seed_chunk_ids={"seed-a", "seed-b"},
        relation_ids={"relation-a", "relation-b"},
        best_confidence=1.0,
        path_length=1,
    )
    weakly_supported = service._GraphCandidate(
        best_confidence=0.1,
        path_length=2,
    )

    supported_score = service._graph_candidate_rerank_score(
        structurally_supported, 0.68, calibration
    )
    weak_score = service._graph_candidate_rerank_score(weakly_supported, 0.85, calibration)
    assert supported_score > weak_score

    def hit(chunk_id: str, cosine: float) -> TrustedHit:
        return TrustedHit(
            Chunk(chunk_id, f"{chunk_id}.md", chunk_id),
            cosine,
            calibration.confidence(cosine),
            "ok",
            Provenance(f"{chunk_id}.md", f"{chunk_id}.md", 0, None),
            Validity(None, None, None),
        )

    baseline = TrustedResult(
        query="q",
        hits=[hit("original-1", 0.99), hit("original-2", 0.90), hit("original-3", 0.55)],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="certified",
    )
    graph_hit = hit("graph", 0.68)
    merged = service._merge_graph_hits(
        baseline,
        [graph_hit],
        {"graph": supported_score},
        calibration,
    )
    assert [item.chunk.id for item in merged] == [
        "original-1",
        "original-2",
        "original-3",
        "graph",
    ]


def test_graph_fill_policy_preserves_direct_hits_before_filling_remaining_slots():
    """Graph evidence fills unused evidence capacity without displacing direct evidence.

    Invariant: with four trusted direct hits and a five item evidence capacity, all four direct
    hits remain in retrieval order and only one graph hit is admitted. The failure mode is the
    previous anchor reranker, which let graph candidates outrank the weaker direct hits and could
    return more than the evidence capacity. Red proof was run for node
    ``tests/test_semantic_graph.py::test_graph_fill_policy_preserves_direct_hits_before_filling_remaining_slots``
    against baseline ``HEAD=8203c0c8`` before the production change. It failed at the expected
    list assertion because the old ``recall_mcp.service._merge_graph_hits`` returned six items
    with ``graph-1`` ahead of ``direct-3``.
    """
    from recall_mcp import service

    def hit(chunk_id: str, cosine: float) -> TrustedHit:
        return TrustedHit(
            Chunk(chunk_id, f"{chunk_id}.md", chunk_id),
            cosine,
            cosine,
            "ok",
            Provenance(f"{chunk_id}.md", f"{chunk_id}.md", 0, None),
            Validity(None, None, None),
        )

    direct = [hit(f"direct-{index}", 0.90 - index * 0.05) for index in range(1, 5)]
    baseline = TrustedResult(
        query="q",
        hits=direct,
        abstained=False,
        reason="",
        gap_warning=True,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="certified",
    )
    graph_hits = [hit("graph-1", 0.99), hit("graph-2", 0.98)]

    merged = service._merge_graph_hits(
        baseline,
        graph_hits,
        {"graph-1": 1.0, "graph-2": 0.99},
        None,
    )

    assert [item.chunk.id for item in merged] == [
        "direct-1",
        "direct-2",
        "direct-3",
        "direct-4",
        "graph-1",
    ]


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
        "semantic_graph_precision_v2|combined|none|20260825|32|"
        "caused,depends_on,references,supersedes,supports|contradicts,same_entity|"
        "rerank=0.60,0.20,0.10,0.10|corroboration_cap=2|"
        "fill_policy=direct_first_fill_missing|fill_slots=5"
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
