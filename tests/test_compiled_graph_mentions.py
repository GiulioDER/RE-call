"""LLM mentions in graph expansion: admitted only when active, only onto existing entities, labelled.

Red proof, recorded 2026-11-26 (production symbols in ``recall.compiled_graph.mentions`` and
``recall.compiled_graph.build``):

- ``test_mentions_attach_only_to_existing_unambiguous_entities``: accepting an ambiguous name
  (``if not matches`` for ``len(matches) != 1``) fails on the pair set.
- ``test_existing_authored_mentions_are_not_duplicated``: the existing-pair check removed fails
  on the mention count and ``llm_pairs``.
- ``test_admission_requires_active``: admitting in every mode but ``off`` fails on the shadow case.
- ``test_candidate_reached_through_an_llm_mention_is_labelled``: the neighbor-side check removed
  fails on the label set; the seed-side check removed fails on the seed case.
- ``test_label_marks_hits``: ``label_hits`` returning hits unchanged fails on the metadata.
- Each of the two halves of the labelling condition removed in turn fails
  ``test_candidate_reached_through_an_llm_mention_is_labelled``.
- ``test_build_records_mentions_only_where_the_chunk_holds_the_entity``: every edge chunk paired
  with every entity fails on the mention pairs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recall.compiled_graph.artifact import ArtifactRecord, CompiledGraphArtifact, write_artifact
from recall.compiled_graph.mentions import (
    LABEL_KEY,
    LABEL_VALUE,
    admit_llm_mentions,
    label_hits,
    llm_dependent_chunks,
    merge_llm_mentions,
)
from recall.compiled_graph.service import serving_lineage, tenant_folder
from recall.semantic_graph import SemanticEntity, SemanticGraphProjection, SemanticMention
from recall.types import Chunk, ScoredChunk


def _entity(entity_id: str, name: str) -> SemanticEntity:
    return SemanticEntity(id=entity_id, tenant_id="t1", generation_id="g1", canonical_name=name,
                          normalized_name=name.casefold(), kind="person")


def _mention(entity_id: str, chunk_id: str) -> SemanticMention:
    return SemanticMention(id=f"m-{entity_id}-{chunk_id}", tenant_id="t1", generation_id="g1",
                           entity_id=entity_id, chunk_id=chunk_id, mention_text="x",
                           extraction_method="metadata")


def _graph() -> SemanticGraphProjection:
    return SemanticGraphProjection(
        schema_version=3, graph_id="gid", tenant_id="t1", generation_id="g1",
        pipeline_fingerprint=None, corpus_fingerprint=None,
        entities=(_entity("e-alice", "alice"), _entity("e-bob1", "bob"), _entity("e-bob2", "bob")),
        mentions=(_mention("e-alice", "c1"),), relations=(), diagnostics=(),
    )


def _artifact(*mentions: tuple[str, str]) -> CompiledGraphArtifact:
    record = ArtifactRecord("a.md#0", "a.md", "fact", ("x",), ("Alice", "Bob", "Carol"), ("c1", "c2"),
                            mentions=mentions)
    return CompiledGraphArtifact({"tenant": "t1"}, (record,), np.ones((1, 4), dtype=np.float32))


def test_mentions_attach_only_to_existing_unambiguous_entities() -> None:
    merged, pairs = merge_llm_mentions(_graph(), _artifact(("c2", "Alice"), ("c2", "Bob"), ("c2", "Carol")))
    assert len(merged.entities) == 3
    assert pairs == frozenset({("c2", "e-alice")})
    added = [m for m in merged.mentions if m.extraction_method == "llm_compiled"]
    assert [(m.chunk_id, m.entity_id) for m in added] == [("c2", "e-alice")]


def test_existing_authored_mentions_are_not_duplicated() -> None:
    merged, pairs = merge_llm_mentions(_graph(), _artifact(("c1", "Alice")))
    assert len(merged.mentions) == 1
    assert pairs == frozenset()


class _Embedder:
    dim = 4
    name = "fake-4"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _store() -> SimpleNamespace:
    return SimpleNamespace(tenant="t1", generation_binding=lambda: {"generation_id": "g1", "corpus_fingerprint": "cf"})


def test_admission_requires_active(tmp_path: Path) -> None:
    env = {"RECALL_COMPILED_GRAPH_MODE": "active", "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": str(tmp_path),
           "RECALL_COMPILED_GRAPH_MODEL": "m"}
    lineage = serving_lineage("t1", _store().generation_binding(), _Embedder(), env)
    assert lineage is not None
    artifact = _artifact(("c2", "Alice"))
    write_artifact(tenant_folder(tmp_path, "t1"), CompiledGraphArtifact(lineage, artifact.records, artifact.vectors))
    merged, pairs = admit_llm_mentions(_store(), _graph(), _Embedder(), env)
    assert pairs == frozenset({("c2", "e-alice")}) and len(merged.mentions) == 2
    for mode in ("shadow", "off"):
        graph = _graph()
        same, none = admit_llm_mentions(_store(), graph, _Embedder(), {**env, "RECALL_COMPILED_GRAPH_MODE": mode})
        assert same is graph and none == frozenset()
    missing, none = admit_llm_mentions(_store(), _graph(), _Embedder(), {**env, "RECALL_COMPILED_GRAPH_MODEL": "other"})
    assert none == frozenset()


def test_candidate_reached_through_an_llm_mention_is_labelled() -> None:
    relation = SimpleNamespace(subject_id="e-alice", object_id="e-bob1")
    candidates = {"c9": (relation,), "c8": (relation,)}
    mentions_by_chunk = {"c1": frozenset({"e-alice"})}
    # Neighbor side: c9 holds the neighbor e-bob1 only through an LLM mention.
    labels = llm_dependent_chunks(candidates, {"e-alice"}, {"c1"}, set(), mentions_by_chunk,
                                  frozenset({("c9", "e-bob1")}))
    assert labels == frozenset({"c9"})
    # Seed side: the seed chunk c1 holds e-alice only through an LLM mention, so both are labelled.
    labels = llm_dependent_chunks(candidates, {"e-alice"}, {"c1"}, set(), mentions_by_chunk,
                                  frozenset({("c1", "e-alice")}))
    assert labels == frozenset({"c8", "c9"})
    # A query that names the entity makes the seed side authored again.
    labels = llm_dependent_chunks(candidates, {"e-alice"}, {"c1"}, {"e-alice"}, mentions_by_chunk,
                                  frozenset({("c1", "e-alice")}))
    assert labels == frozenset()


def test_label_marks_hits() -> None:
    hits = [ScoredChunk(chunk=Chunk(id=f"c{i}", source="a.md", text="t", metadata={"file": "a.md"}), score=0.5)
            for i in range(3)]
    labelled = label_hits(hits, frozenset({"c1"}))
    assert [hit.chunk.metadata.get(LABEL_KEY) for hit in labelled] == [None, LABEL_VALUE, None]
    assert labelled[1].score == 0.5 and hits[1].chunk.metadata.get(LABEL_KEY) is None


def test_build_records_mentions_only_where_the_chunk_holds_the_entity(tmp_path: Path) -> None:
    import json

    from recall.compiled_graph.anchors import build_source_anchors
    from recall.compiled_graph.budget import Budget
    from recall.compiled_graph.build import SourceInput, build_compiled_graph
    from recall.compiled_graph.cache import CompileCache
    from recall.compiled_graph.provider import ProviderConfig

    body = "Alice owns the release checklist. " + "x" * 1_000 + " Bob reviews it."
    chunks = (Chunk(id="a:0", source="a.md", text=body[:600], metadata={"text_start": 0, "text_end": 600}),
              Chunk(id="a:1", source="a.md", text=body[600:], metadata={"text_start": 600, "text_end": len(body)}))
    anchor = build_source_anchors("a.md", body)[0].id
    record = {"kind": "fact", "summary": "Alice owns the release checklist.", "detail": "",
              "entities": ["Alice", "Bob"], "evidence_anchor_ids": [anchor]}
    reply = json.dumps({"choices": [{"message": {"content": json.dumps({"records": [record]})}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10}}).encode()

    def transport(url, body_bytes, headers, timeout):  # type: ignore[no-untyped-def]  # matches provider.Transport
        return 200, {}, reply

    artifact, _ = build_compiled_graph(
        [SourceInput("a.md", body, chunks)], _Embedder(), ProviderConfig(api_key="k", model="m"), transport,
        CompileCache(tmp_path / "c.sqlite"), Budget(max_usd=1.0, cost_in_per_1k=0.0, cost_out_per_1k=0.0),
        lineage={"tenant": "t1"}, sleep=lambda _: None,
    )
    assert artifact is not None
    (built,) = artifact.records
    assert built.edges == ("a:0", "a:1")
    assert set(built.mentions) == {("a:0", "Alice"), ("a:1", "Bob")}
