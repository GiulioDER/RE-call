"""Regression coverage for query scoped semantic graph expansion.

The invariant is that graph metadata may scale with a generation, but query work and text
payload must scale only with the admitted graph budget. The pre-fix production line under test
was ``recall_mcp.service._expand_semantic_graph`` calling ``_store_graph(..., include_text=True)``:
that streamed every chunk before candidate admission. Red proof for
``test_graph_serving_is_lazy_and_budgeted`` used a deliberate mutation restoring that behavior
from baseline commit ``01be3f9d`` and the 1,000 chunk parameterization. It failed in the intended
assertion because the baseline projected all chunks and the fake store's batch boundary was never
used.
The production symbols under test are ``_expand_semantic_graph`` and
``GenerationStore.chunks_by_ids``.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest

from recall.semantic_graph import (
    SemanticEntity,
    SemanticGraphProjection,
    SemanticMention,
    SemanticRelation,
)
from recall.reasoning import GenerationSelection, ReasoningPolicy, ReasoningProviderPorts, ReasoningRequest
from recall.reasoning_planner import ReasoningBudget
from recall.types import Chunk, Provenance, StalenessReport, TrustedHit, TrustedResult, Validity
from recall_mcp import service


TEXT = "candidate text " * 20
GRAPH_BUDGET = 7
GENERATION = "generation-a"
PIPELINE = "p" * 64
CORPUS = "c" * 64


def _semantic_graph(candidate_count: int) -> SemanticGraphProjection:
    seed_entity = SemanticEntity(
        id="entity-seed",
        tenant_id="tenant-a",
        generation_id=GENERATION,
        canonical_name="Seed",
        normalized_name="seed",
        kind="project",
    )
    neighbor_entity = SemanticEntity(
        id="entity-neighbor",
        tenant_id="tenant-a",
        generation_id=GENERATION,
        canonical_name="Neighbor",
        normalized_name="neighbor",
        kind="service",
    )
    mentions = [
        SemanticMention(
            id="mention-seed",
            tenant_id="tenant-a",
            generation_id=GENERATION,
            entity_id=seed_entity.id,
            chunk_id="seed",
            mention_text="Seed",
            extraction_method="metadata",
        )
    ]
    mentions.extend(
        SemanticMention(
            id=f"mention-candidate-{index}",
            tenant_id="tenant-a",
            generation_id=GENERATION,
            entity_id=neighbor_entity.id,
            chunk_id=f"candidate-{index:06d}",
            mention_text="Neighbor",
            extraction_method="metadata",
        )
        for index in range(candidate_count)
    )
    return SemanticGraphProjection(
        schema_version=1,
        graph_id=f"semantic-graph-{candidate_count}",
        tenant_id="tenant-a",
        generation_id=GENERATION,
        pipeline_fingerprint=PIPELINE,
        corpus_fingerprint=CORPUS,
        entities=(seed_entity, neighbor_entity),
        mentions=tuple(mentions),
        relations=(
            SemanticRelation(
                id="relation-seed-neighbor",
                tenant_id="tenant-a",
                generation_id=GENERATION,
                subject_id=seed_entity.id,
                object_id=neighbor_entity.id,
                relation="supports",
                evidence_chunk_ids=("seed",),
                extraction_method="explicit_relation",
                confidence=1.0,
            ),
        ),
        diagnostics=(),
    )


def _retrieval() -> tuple[TrustedResult, Chunk]:
    seed = Chunk("seed", "seed.md", "seed", {"file": "seed.md"})
    hit = TrustedHit(
        chunk=seed,
        cosine=1.0,
        confidence=1.0,
        verdict="ok",
        provenance=Provenance("seed.md", "seed.md", 0, datetime.now(UTC)),
        validity=Validity(None, None, None),
    )
    return (
        TrustedResult(
            query="q",
            hits=[hit],
            abstained=False,
            reason="",
            gap_warning=True,
            staleness=StalenessReport(False, None, timedelta(days=1), timedelta(days=2)),
            tenant_id="tenant-a",
            generation_id=GENERATION,
            pipeline_fingerprint=PIPELINE,
            corpus_fingerprint=CORPUS,
            calibration_status="certified",
        ),
        seed,
    )


class _Store:
    tenant = "tenant-a"
    generation_id = GENERATION

    def __init__(self, semantic: SemanticGraphProjection) -> None:
        self.semantic = semantic
        self.operations: list[str] = []
        self.batch_ids: tuple[str, ...] = ()
        self.text_bytes = 0
        self.pinned_generations: list[str] = []

    @contextmanager
    def pin_generation(self, generation_id):  # type: ignore[no-untyped-def]
        self.pinned_generations.append(generation_id)
        yield generation_id

    def graph_readiness(self):  # type: ignore[no-untyped-def]
        self.operations.append("graph_readiness")
        return self.semantic.readiness()

    def load_semantic_graph(self, generation_id=None):  # type: ignore[no-untyped-def]
        assert generation_id == GENERATION
        self.operations.append("load_semantic_graph")
        return self.semantic

    def generation_binding(self):  # type: ignore[no-untyped-def]
        return {
            "pipeline_fingerprint": PIPELINE,
            "corpus_fingerprint": CORPUS,
        }

    def iter_chunks(self):  # type: ignore[no-untyped-def]
        self.operations.append("iter_chunks")
        yield Chunk("seed", "seed.md", "seed", {"file": "seed.md"})
        for mention in self.semantic.mentions:
            if mention.chunk_id != "seed":
                yield Chunk(
                    mention.chunk_id,
                    f"{mention.chunk_id}.md",
                    TEXT,
                    {"file": f"{mention.chunk_id}.md"},
                )

    def supersession_all(self):  # type: ignore[no-untyped-def]
        self.operations.append("supersession")
        return {}, frozenset(), {}

    def cosines_for(self, ids, vec):  # type: ignore[no-untyped-def]
        del vec
        self.operations.append("cosines_for")
        assert len(ids) <= GRAPH_BUDGET - 1
        return {chunk_id: 0.9 for chunk_id in ids}

    def chunks_by_ids(self, ids):  # type: ignore[no-untyped-def]
        self.operations.append("chunks_by_ids")
        self.batch_ids = tuple(ids)
        assert len(self.batch_ids) <= GRAPH_BUDGET - 1
        self.text_bytes += len(self.batch_ids) * len(TEXT.encode("utf-8"))
        return {
            chunk_id: Chunk(chunk_id, f"{chunk_id}.md", TEXT, {"file": f"{chunk_id}.md"})
            for chunk_id in self.batch_ids
        }

    def supersession(self):
        self.operations.append("supersession")
        return {}, frozenset()


@pytest.mark.parametrize("corpus_size", [1_000, 10_000, 100_000])
def test_graph_serving_is_lazy_and_budgeted(corpus_size: int, monkeypatch) -> None:
    service._reset_graph_projection_cache()
    semantic = _semantic_graph(corpus_size)
    store = _Store(semantic)
    retrieval, _seed = _retrieval()
    request = ReasoningRequest(
        query="q",
        tenant_id="tenant-a",
        generation=GenerationSelection(GENERATION, PIPELINE, CORPUS),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=GRAPH_BUDGET, max_graph_hops=1),
    )
    trust_ids: list[str] = []
    original_evaluate = service.evaluate

    def traced_evaluate(*args, **kwargs):  # type: ignore[no-untyped-def]
        evaluated = original_evaluate(*args, **kwargs)
        trust_ids.extend(hit.chunk.id for hit in evaluated.hits)
        return evaluated

    monkeypatch.setattr(service, "evaluate", traced_evaluate)
    result = service._expand_semantic_graph(
        store,
        request,
        retrieval,
        None,
        type("Embedder", (), {"embed_query": lambda self, _: [1.0]})(),
    )

    candidate_ids = tuple(hit.chunk.id for hit in result.retrieval.hits if hit.chunk.id != "seed")
    assert result.candidates_discovered <= GRAPH_BUDGET - 1
    assert len(candidate_ids) <= GRAPH_BUDGET - 1
    assert store.batch_ids == candidate_ids
    assert trust_ids == list(candidate_ids)
    assert store.text_bytes == len(candidate_ids) * len(TEXT.encode("utf-8"))
    assert "iter_chunks" not in store.operations
    assert store.operations == [
        "graph_readiness",
        "load_semantic_graph",
        "supersession",
        "chunks_by_ids",
        "cosines_for",
    ]
    assert store.pinned_generations == [GENERATION] * 3


def test_lazy_semantic_graph_is_reused_for_one_generation() -> None:
    service._reset_graph_projection_cache()
    semantic = _semantic_graph(10)
    store = _Store(semantic)
    retrieval, _seed = _retrieval()
    request = ReasoningRequest(
        query="q",
        tenant_id="tenant-a",
        generation=GenerationSelection(GENERATION, PIPELINE, CORPUS),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=GRAPH_BUDGET, max_graph_hops=1),
    )
    embedder = type("Embedder", (), {"embed_query": lambda self, _: [1.0]})()

    service._expand_semantic_graph(store, request, retrieval, None, embedder)
    service._expand_semantic_graph(store, request, retrieval, None, embedder)

    assert store.operations.count("load_semantic_graph") == 1


def test_graph_adjacency_indexes_are_reused_for_one_generation() -> None:
    service._reset_graph_projection_cache()
    semantic = _semantic_graph(10)
    first = service._semantic_graph_indexes(semantic)
    second = service._semantic_graph_indexes(semantic)
    assert first is second
