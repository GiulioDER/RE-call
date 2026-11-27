"""The reasoning query's graph expansion follows LLM mentions only when the tenant is active.

Invariant: an authored relation Seed to Neighbor, and no authored chunk mentioning Neighbor. With
the tenant's compiled graph ``active`` and an artifact that gives ``llm-chunk`` an LLM mention of
Neighbor, expansion reaches ``llm-chunk`` and labels its hit ``graph_link: llm_compiled``; with
``off`` or ``shadow`` it reaches nothing. The authored-only relation rule is unchanged.

Red proof, recorded 2026-11-26: with the ``admit_llm_mentions`` call removed from
``recall_mcp.graph_expansion.expand_semantic_graph``, ``test_active_follows_the_llm_mention`` fails
on the candidate ids; with the ``label_hits`` call removed from its ``finish``, it fails on the
label.

Red proof, recorded 2026-11-27 on the remote test host (CCA run 20261127T110213Z): with the retrieval hit ids no
longer subtracted from ``llm_labels`` in ``expand_semantic_graph``,
``test_a_hit_retrieval_returned_is_never_labelled`` fails on the label of the returned
``llm-chunk`` hit, and passes once restored.

LLM mentions only widen the chunks a neighbor reaches (owner's choice, 2026-11-27). Red proof,
recorded 2026-11-27 on the remote test host by mutating ``recall_mcp/graph_expansion.py`` and
restoring it; each failed on its own assertion and passed once restored:

- ``test_an_llm_mention_on_the_seed_chunk_starts_no_relation``: seeds read from the merged
  ``indexes.mentions_by_chunk`` fails on ``target-chunk`` being reached.
- ``test_llm_mentions_do_not_make_a_seed_a_hub``: the hub count read from the merged
  ``chunks_by_entity`` fails on ``llm-chunk`` no longer being reached.
- ``test_the_budget_is_spent_on_authored_candidates_first``: the candidates sorted by id alone
  fails on the authored set ``off`` returns not being a subset of what ``active`` returns.
- ``test_a_candidate_an_authored_mention_reaches_is_not_labelled``: ``llm_only`` taken as every
  chunk with an LLM relation fails on the label of ``candidate-000000``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from recall.compiled_graph.artifact import ArtifactRecord, CompiledGraphArtifact, write_artifact
from recall.compiled_graph.mentions import LABEL_KEY, LABEL_VALUE
from recall.compiled_graph.service import serving_lineage, tenant_folder
from recall.reasoning import GenerationSelection, ReasoningPolicy, ReasoningProviderPorts, ReasoningRequest
from recall.reasoning_planner import ReasoningBudget
from recall_mcp import service
from tests.test_graph_lazy_loading import CORPUS, GENERATION, PIPELINE, _retrieval, _semantic_graph, _Store


class _BoundStore(_Store):
    def generation_binding(self):  # type: ignore[no-untyped-def]  # mirrors the store double
        return {"pipeline_fingerprint": PIPELINE, "corpus_fingerprint": CORPUS, "generation_id": GENERATION}


class _Embedder:
    name = "fake-1"
    dim = 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


def _env(root: Path, mode: str) -> dict[str, str]:
    return {"RECALL_COMPILED_GRAPH_MODE": mode, "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": str(root),
            "RECALL_COMPILED_GRAPH_MODEL": "openai/gpt-6-luna-pro"}


def _expand(store: _BoundStore, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], retrieval=None):  # type: ignore[no-untyped-def]  # returns the expansion result
    monkeypatch.setattr("recall_mcp.settings.runtime_environment", lambda: env)
    service._reset_graph_projection_cache()
    if retrieval is None:
        retrieval, _ = _retrieval()
    request = ReasoningRequest(
        query="q",
        tenant_id="tenant-a",
        generation=GenerationSelection(GENERATION, PIPELINE, CORPUS),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=7, max_graph_hops=1),
    )
    return service._expand_semantic_graph(cast(Any, store), request, retrieval, None, cast(Any, _Embedder()))


def _store_with(root: Path, semantic, *mentions: tuple[str, str]) -> _BoundStore:  # type: ignore[no-untyped-def]  # semantic is a SemanticGraphProjection
    store = _BoundStore(semantic)
    lineage = serving_lineage("tenant-a", store.generation_binding(), _Embedder(), _env(root, "active"))
    assert lineage is not None
    record = ArtifactRecord("n.md#0", "n.md", "fact", ("x",), tuple({name for _, name in mentions}),
                            tuple(sorted({chunk for chunk, _ in mentions})), mentions=mentions)
    write_artifact(tenant_folder(root, "tenant-a"),
                   CompiledGraphArtifact(lineage, (record,), np.ones((1, 1), dtype=np.float32)))
    return store


@pytest.fixture
def store_with_artifact(tmp_path: Path) -> _BoundStore:
    return _store_with(tmp_path, _semantic_graph(0), ("llm-chunk", "Neighbor"))


def test_active_follows_the_llm_mention(store_with_artifact: _BoundStore, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    result = _expand(store_with_artifact, monkeypatch, _env(tmp_path, "active"))
    reached = {hit.chunk.id: hit for hit in result.retrieval.hits if hit.chunk.id != "seed"}
    assert set(reached) == {"llm-chunk"}
    assert reached["llm-chunk"].chunk.metadata.get(LABEL_KEY) == LABEL_VALUE


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_off_and_shadow_reach_nothing(store_with_artifact: _BoundStore, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    result = _expand(store_with_artifact, monkeypatch, _env(tmp_path, mode))
    assert [hit.chunk.id for hit in result.retrieval.hits if hit.chunk.id != "seed"] == []


def test_a_hit_retrieval_returned_is_never_labelled(store_with_artifact: _BoundStore, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from recall.types import Chunk

    retrieval, _ = _retrieval()
    (seed,) = retrieval.hits
    # Retrieval itself returned llm-chunk, untrusted, so it is a graph candidate as well.
    untrusted = replace(seed, chunk=Chunk("llm-chunk", "n.md", "n", {"file": "n.md"}), verdict="low_confidence")
    result = _expand(store_with_artifact, monkeypatch, _env(tmp_path, "active"),
                     replace(retrieval, hits=[seed, untrusted]))
    returned = [hit for hit in result.retrieval.hits if hit.chunk.id == "llm-chunk"]
    assert returned
    assert all(hit.chunk.metadata.get(LABEL_KEY) is None for hit in returned)


def _entity(entity_id: str, name: str):  # type: ignore[no-untyped-def]  # returns a SemanticEntity
    from recall.semantic_graph import SemanticEntity

    return SemanticEntity(id=entity_id, tenant_id="tenant-a", generation_id=GENERATION,
                          canonical_name=name, normalized_name=name.casefold(), kind="service")


def _relation(relation_id: str, subject: str, obj: str):  # type: ignore[no-untyped-def]  # returns a SemanticRelation
    from recall.semantic_graph import SemanticRelation

    return SemanticRelation(id=relation_id, tenant_id="tenant-a", generation_id=GENERATION,
                            subject_id=subject, object_id=obj, relation="supports",
                            evidence_chunk_ids=("seed",), extraction_method="explicit_relation",
                            confidence=1.0)


def _reached(result) -> set[str]:  # type: ignore[no-untyped-def]  # result is the expansion result
    return {hit.chunk.id for hit in result.retrieval.hits if hit.chunk.id != "seed"}


def test_an_llm_mention_on_the_seed_chunk_starts_no_relation(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from recall.semantic_graph import SemanticMention

    base = _semantic_graph(0)
    # Other -> Target is authored, with evidence in the seed chunk; only an LLM mention says the
    # seed chunk names Other, so Other may widen nothing and must not become a seed.
    semantic = replace(
        base,
        graph_id="semantic-graph-seed-side",
        entities=(*base.entities, _entity("entity-other", "Other"), _entity("entity-target", "Target")),
        mentions=(*base.mentions, SemanticMention(
            id="mention-target", tenant_id="tenant-a", generation_id=GENERATION, entity_id="entity-target",
            chunk_id="target-chunk", mention_text="Target", extraction_method="metadata")),
        relations=(*base.relations, _relation("relation-other-target", "entity-other", "entity-target")),
    )
    store = _store_with(tmp_path, semantic, ("seed", "Other"))
    assert "target-chunk" not in _reached(_expand(store, monkeypatch, _env(tmp_path, "active")))


def test_llm_mentions_do_not_make_a_seed_a_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_copies = tuple((f"seed-copy-{index:03d}", "Seed") for index in range(40))
    store = _store_with(tmp_path, _semantic_graph(0), ("llm-chunk", "Neighbor"), *seed_copies)
    assert "llm-chunk" in _reached(_expand(store, monkeypatch, _env(tmp_path, "active")))


def test_the_budget_is_spent_on_authored_candidates_first(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    # Six authored candidates fill the budget; two LLM-only ones sort before them by chunk id.
    store = _store_with(tmp_path, _semantic_graph(6), ("a-llm-0", "Neighbor"), ("a-llm-1", "Neighbor"))
    authored_off = _reached(_expand(store, monkeypatch, _env(tmp_path, "off")))
    assert authored_off
    assert authored_off <= _reached(_expand(store, monkeypatch, _env(tmp_path, "active")))


def test_a_candidate_an_authored_mention_reaches_is_not_labelled(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    base = _semantic_graph(1)
    # candidate-000000 holds Neighbor through an authored mention, and Extra only through an LLM one.
    semantic = replace(
        base,
        graph_id="semantic-graph-mixed",
        entities=(*base.entities, _entity("entity-extra", "Extra")),
        relations=(*base.relations, _relation("relation-seed-extra", "entity-seed", "entity-extra")),
    )
    store = _store_with(tmp_path, semantic, ("candidate-000000", "Extra"))
    result = _expand(store, monkeypatch, _env(tmp_path, "active"))
    hits = {hit.chunk.id: hit for hit in result.retrieval.hits}
    assert "candidate-000000" in hits
    assert hits["candidate-000000"].chunk.metadata.get(LABEL_KEY) is None
