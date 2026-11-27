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

Red proof, recorded 2026-11-27 on the remote test host by mutating the fixed line and restoring it (CCA run
20261127T110213Z); each failed on its own assertion and passed once restored:

- ``test_an_older_schema_directory_is_replaced_not_kept``: ``artifact_dir`` digesting the lineage
  without the schema fails on ``written != old`` (the schema 1 directory is kept).
- ``test_a_chunk_also_reached_through_an_authored_relation_is_not_labelled``: ``all`` replaced by
  ``any`` in ``llm_dependent_chunks`` fails on the label set (``c9`` labelled).
- ``test_a_scoped_request_admits_mentions_only_on_chunks_its_graph_holds``: the
  ``visible_chunks`` check removed from ``merge_llm_mentions`` fails on the scoped pair set.
- ``test_a_cache_hit_does_not_load_the_artifact_again``: the cache hit branch of
  ``admit_llm_mentions`` disabled fails on the returned projection identity.
- ``test_the_cache_tells_a_relation_control_copy_from_its_graph``: the ``_MERGED`` key back on
  ``(graph_id, fingerprint, lineage)`` without the identity check fails on the relations.
- ``test_a_cache_hit_sees_an_artifact_removed_since``: the manifest check dropped from the cache
  hit branch of ``admit_llm_mentions`` fails on the pair set (the removed artifact's mentions
  are still admitted).
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
    merged, pairs = admit_llm_mentions(_store(), _graph(), _Embedder(), env, scoped=False)
    assert pairs == frozenset({("c2", "e-alice")}) and len(merged.mentions) == 2
    for mode in ("shadow", "off"):
        graph = _graph()
        same, none = admit_llm_mentions(_store(), graph, _Embedder(), {**env, "RECALL_COMPILED_GRAPH_MODE": mode}, scoped=False)
        assert same is graph and none == frozenset()
    missing, none = admit_llm_mentions(_store(), _graph(), _Embedder(), {**env, "RECALL_COMPILED_GRAPH_MODEL": "other"},
                                       scoped=False)
    assert none == frozenset()


def test_candidate_reached_through_an_llm_mention_is_labelled() -> None:
    relation = SimpleNamespace(subject_id="e-alice", object_id="e-bob1")
    candidates = {"c9": (relation,), "c8": (relation,)}
    mentions_by_chunk = {"c1": frozenset({"e-alice"})}
    # Neighbor side: c9 holds the neighbor e-bob1 only through an LLM mention.
    labels = llm_dependent_chunks(candidates, seed_entities={"e-alice"}, trusted_seed_ids={"c1"},
                                  resolved_query_entities=set(), mentions_by_chunk=mentions_by_chunk,
                                  llm_pairs=frozenset({("c9", "e-bob1")}))
    assert labels == frozenset({"c9"})
    # Seed side: the seed chunk c1 holds e-alice only through an LLM mention, so both are labelled.
    labels = llm_dependent_chunks(candidates, seed_entities={"e-alice"}, trusted_seed_ids={"c1"},
                                  resolved_query_entities=set(), mentions_by_chunk=mentions_by_chunk,
                                  llm_pairs=frozenset({("c1", "e-alice")}))
    assert labels == frozenset({"c8", "c9"})
    # A query that names the entity makes the seed side authored again.
    labels = llm_dependent_chunks(candidates, seed_entities={"e-alice"}, trusted_seed_ids={"c1"},
                                  resolved_query_entities={"e-alice"}, mentions_by_chunk=mentions_by_chunk,
                                  llm_pairs=frozenset({("c1", "e-alice")}))
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


def _active_env(root: Path) -> dict[str, str]:
    return {"RECALL_COMPILED_GRAPH_MODE": "active", "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": str(root),
            "RECALL_COMPILED_GRAPH_MODEL": "m"}


def _write_served(root: Path, *mentions: tuple[str, str]) -> None:
    lineage = serving_lineage("t1", _store().generation_binding(), _Embedder(), _active_env(root))
    assert lineage is not None
    artifact = _artifact(*mentions)
    write_artifact(tenant_folder(root, "t1"), CompiledGraphArtifact(lineage, artifact.records, artifact.vectors))


def test_an_older_schema_directory_is_replaced_not_kept(tmp_path: Path) -> None:
    import hashlib
    import json

    from recall.compiled_graph.artifact import load_artifact

    lineage = {"tenant": "t1", "generation": "g1"}
    folder = tmp_path / "f"
    # The directory a schema 1 build wrote: named from the lineage alone, with a schema 1 manifest.
    old = folder / f"cg-{hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()[:24]}"
    old.mkdir(parents=True)
    (old / "manifest.json").write_text(json.dumps({"schema": 1, "lineage": lineage}), encoding="utf-8")
    artifact = _artifact(("c2", "Alice"))
    written = write_artifact(folder, CompiledGraphArtifact(lineage, artifact.records, artifact.vectors))
    assert written != old
    assert load_artifact(folder, lineage).records[0].mentions == (("c2", "Alice"),)


def test_a_chunk_also_reached_through_an_authored_relation_is_not_labelled() -> None:
    through_llm = SimpleNamespace(subject_id="e-alice", object_id="e-bob1")
    authored = SimpleNamespace(subject_id="e-alice", object_id="e-carol")
    labels = llm_dependent_chunks(
        {"c9": (through_llm, authored), "c8": (through_llm,)},
        seed_entities={"e-alice"}, trusted_seed_ids={"c1"}, resolved_query_entities=set(),
        mentions_by_chunk={"c1": frozenset({"e-alice"})}, llm_pairs=frozenset({("c9", "e-bob1"), ("c8", "e-bob1")}),
    )
    assert labels == frozenset({"c8"})


def test_a_scoped_request_admits_mentions_only_on_chunks_its_graph_holds(tmp_path: Path) -> None:
    _write_served(tmp_path, ("c2", "Alice"))
    graph = _graph()  # authored mention on c1 only; c2 is a chunk the policy hid from this graph
    _, scoped = admit_llm_mentions(_store(), graph, _Embedder(), _active_env(tmp_path), scoped=True)
    _, unscoped = admit_llm_mentions(_store(), graph, _Embedder(), _active_env(tmp_path), scoped=False)
    assert scoped == frozenset()
    assert unscoped == frozenset({("c2", "e-alice")})


def test_the_cache_tells_a_relation_control_copy_from_its_graph(tmp_path: Path) -> None:
    from dataclasses import replace

    from recall.semantic_graph import SemanticRelation

    _write_served(tmp_path, ("c2", "Alice"))
    relation = SemanticRelation(id="r1", tenant_id="t1", generation_id="g1", subject_id="e-alice",
                                object_id="e-bob1", relation="references", evidence_chunk_ids=("c1",),
                                extraction_method="explicit_relation", confidence=1.0)
    graph = replace(_graph(), relations=(relation,))
    shuffled = replace(graph, relations=(replace(relation, subject_id="e-bob1", object_id="e-alice"),))
    assert shuffled.fingerprint == graph.fingerprint and shuffled.graph_id == graph.graph_id
    admit_llm_mentions(_store(), graph, _Embedder(), _active_env(tmp_path), scoped=False)
    merged, _ = admit_llm_mentions(_store(), shuffled, _Embedder(), _active_env(tmp_path), scoped=False)
    assert merged.relations == shuffled.relations


def test_a_cache_hit_does_not_load_the_artifact_again(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # pytest fixture
    from recall.compiled_graph import mentions

    _write_served(tmp_path, ("c2", "Alice"))
    loads: list[str] = []
    real = mentions.load_serving_artifact

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]  # wraps the real loader
        loads.append("load")
        return real(*args, **kwargs)

    monkeypatch.setattr(mentions, "load_serving_artifact", counting)
    graph = _graph()
    first = admit_llm_mentions(_store(), graph, _Embedder(), _active_env(tmp_path), scoped=False)
    second = admit_llm_mentions(_store(), graph, _Embedder(), _active_env(tmp_path), scoped=False)
    assert second[0] is first[0] and second[1] == frozenset({("c2", "e-alice")})
    assert loads == ["load"]


def test_a_cache_hit_sees_an_artifact_removed_since(tmp_path: Path) -> None:
    """A forget erases every artifact holding the source, in any tenant, with the lineage unchanged."""
    import shutil

    _write_served(tmp_path, ("c2", "Alice"))
    graph = _graph()
    _, before = admit_llm_mentions(_store(), graph, _Embedder(), _active_env(tmp_path), scoped=False)
    assert before == frozenset({("c2", "e-alice")})
    shutil.rmtree(tenant_folder(tmp_path, "t1"))
    after_graph, after = admit_llm_mentions(_store(), graph, _Embedder(), _active_env(tmp_path), scoped=False)
    assert after == frozenset()
    assert after_graph is graph
