"""The reasoning query's graph expansion follows LLM mentions only when the tenant is active.

Invariant: an authored relation Seed to Neighbor, and no authored chunk mentioning Neighbor. With
the tenant's compiled graph ``active`` and an artifact that gives ``llm-chunk`` an LLM mention of
Neighbor, expansion reaches ``llm-chunk`` and labels its hit ``graph_link: llm_compiled``; with
``off`` or ``shadow`` it reaches nothing. The authored-only relation rule is unchanged.

Red proof, recorded 2026-11-26: with the ``admit_llm_mentions`` call removed from
``recall_mcp.graph_expansion.expand_semantic_graph``, ``test_active_follows_the_llm_mention`` fails
on the candidate ids; with the ``label_hits`` call removed from its ``finish``, it fails on the
label.
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


def _expand(store: _BoundStore, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]):  # type: ignore[no-untyped-def]  # returns the expansion result
    monkeypatch.setattr("recall_mcp.settings.runtime_environment", lambda: env)
    service._reset_graph_projection_cache()
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


@pytest.fixture
def store_with_artifact(tmp_path: Path) -> _BoundStore:
    store = _BoundStore(_semantic_graph(0))
    lineage = serving_lineage("tenant-a", store.generation_binding(), _Embedder(), _env(tmp_path, "active"))
    assert lineage is not None
    record = ArtifactRecord("n.md#0", "n.md", "fact", ("x",), ("Neighbor",), ("llm-chunk",),
                            mentions=(("llm-chunk", "Neighbor"),))
    write_artifact(tenant_folder(tmp_path, "tenant-a"),
                   CompiledGraphArtifact(lineage, (record,), np.ones((1, 1), dtype=np.float32)))
    return store


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
