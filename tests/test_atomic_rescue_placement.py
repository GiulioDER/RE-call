"""Where an atomic rescue lands: dense rank six before fusion, or final rank six after it.

Red proofs (2026-09-23, each mutation applied and reverted):

* ``test_fused_insert_keeps_the_final_top_five_and_places_the_winner_sixth`` failed when
  ``insert_atomic_rescue_fused`` inserted at index 4 instead of 5.
* ``test_fused_insert_leaves_a_winner_already_in_the_top_five_alone`` failed when the
  top-five membership check was removed (the winner was duplicated at rank six).
* ``test_fused_placement_reaches_trust_after_fusion`` failed when ``recall.trust`` ignored
  ``RECALL_ATOMIC_RESCUE_PLACEMENT`` and always built the dense transform.
* ``test_an_unknown_placement_is_refused`` failed when the placement check in ``recall.trust``
  was removed (the search ran with the dense transform).
* ``test_c8_fused_placement_keeps_the_fused_top_five`` failed when the C8 retriever applied the
  dense transform for ``placement="fused"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    clear_atomic_rescue_artifact_cache,
    insert_atomic_rescue_fused,
    load_atomic_rescue_artifact,
    write_atomic_rescue_artifact,
)
from recall.calibration import Calibration
from recall.retriever import HybridRetriever
from recall.trust import TrustPolicy, trusted_search
from recall.types import Chunk, ScoredChunk
from recall_aml.retrieval import AtomicRescueBinding, HostedRetriever
from tests.test_aml_atomic_rescue import _Embedder as _C8Embedder
from tests.test_aml_atomic_rescue import _Reranker
from tests.test_atomic_rescue_production_shadow import _ActiveStore, _Embedder


def _hit(name: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(Chunk(name, f"{name}.md", name, {}), score)


def _artifact(tmp_path, parent: str, *, dim: int = 2):
    clear_atomic_rescue_artifact_cache()
    vector = np.zeros(dim, dtype=np.float32)
    vector[0] = 1.0
    manifest = write_atomic_rescue_artifact(
        tmp_path / f"artifact-{parent}",
        matrix=np.stack([vector]),
        views=[{"chunk_id": parent, "source": f"{parent}.md", "parent_ordinal": 0, "view_ordinal": 0}],
        generation_id="g",
        calibration_id="c",
        pipeline_fingerprint="p",
        corpus_fingerprint="0" * 64,
        embedding_profile="test",
        embedding_fingerprint="f",
        ordinary_chunk_count=10,
        source_commit="test",
    )
    return load_atomic_rescue_artifact(manifest)


def test_fused_insert_keeps_the_final_top_five_and_places_the_winner_sixth(tmp_path) -> None:
    artifact = _artifact(tmp_path, "winner")
    dense = [_hit(f"d{index}") for index in range(1, 8)]
    ranked = [_hit(name) for name in ("f1", "f2", "f3", "f4", "f5", "f6", "winner", "f7")]
    placed = insert_atomic_rescue_fused(artifact, [1.0, 0.0], dense, ranked, lambda c, s: _hit(c, s))
    assert [hit.chunk.id for hit in placed] == ["f1", "f2", "f3", "f4", "f5", "winner", "f6", "f7"]


def test_fused_insert_leaves_a_winner_already_in_the_top_five_alone(tmp_path) -> None:
    artifact = _artifact(tmp_path, "winner")
    dense = [_hit(f"d{index}") for index in range(1, 8)]
    ranked = [_hit(name) for name in ("f1", "winner", "f3", "f4", "f5", "f6")]
    placed = insert_atomic_rescue_fused(artifact, [1.0, 0.0], dense, ranked, lambda c, s: _hit(c, s))
    assert [hit.chunk.id for hit in placed] == [hit.chunk.id for hit in ranked]


def _trusted(tmp_path, monkeypatch, placement: str) -> tuple[list[str], dict[str, list[object]]]:
    from recall import trust

    class Artifact:
        def assert_lineage(self, **kwargs):
            del kwargs

    calls: dict[str, list[object]] = {"dense": [], "fused": []}

    def dense_insert(artifact, query_vector, dense, loader):
        calls["dense"].append([hit.chunk.id for hit in dense])
        return [*dense[:5], loader("rescued", 0.75), *dense[5:]]

    def fused_insert(artifact, query_vector, dense, ranked, loader):
        calls["fused"].append([hit.chunk.id for hit in ranked])
        return [*ranked[:5], loader("rescued", 0.75), *ranked[5:]]

    monkeypatch.setattr(trust, "load_atomic_rescue_artifact", lambda path: Artifact())
    monkeypatch.setattr(trust, "insert_atomic_rescue_dense", dense_insert)
    monkeypatch.setattr(trust, "insert_atomic_rescue_fused", fused_insert)
    result = trusted_search(
        _ActiveStore(),
        _Embedder(),
        "query",
        k=6,
        candidate_k=7,
        calibration=Calibration(embedder="test-profile", threshold=0.1, scale=0.1),
        policy=TrustPolicy.development(),
        env={
            "RECALL_ATOMIC_RESCUE_MODE": "active",
            "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
            "RECALL_ATOMIC_RESCUE_PLACEMENT": placement,
        },
    )
    return [hit.chunk.id for hit in result.hits], calls


def test_fused_placement_reaches_trust_after_fusion(tmp_path, monkeypatch) -> None:
    hits, calls = _trusted(tmp_path, monkeypatch, "fused")
    assert calls["dense"] == []
    assert calls["fused"] == [[f"dense-{index}" for index in range(1, 8)]]
    assert hits == [f"dense-{index}" for index in range(1, 6)] + ["rescued"]


def test_dense_placement_stays_the_default_path(tmp_path, monkeypatch) -> None:
    _, calls = _trusted(tmp_path, monkeypatch, "dense")
    assert calls["fused"] == []
    assert len(calls["dense"]) == 1


def test_an_unknown_placement_is_refused(tmp_path, monkeypatch) -> None:
    with pytest.raises(AtomicRescueArtifactError, match="PLACEMENT"):
        _trusted(tmp_path, monkeypatch, "sideways")


def test_settings_refuse_an_unknown_placement() -> None:
    from recall_mcp.settings import Settings

    with pytest.raises(ValueError, match="RECALL_ATOMIC_RESCUE_PLACEMENT"):
        Settings.from_env({"RECALL_ATOMIC_RESCUE_PLACEMENT": "sideways"})


def test_a_retriever_refuses_both_placements_at_once() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        HybridRetriever(
            _ActiveStore(),
            _Embedder(),
            dense_transform=lambda vector, dense: dense,
            post_fusion_transform=lambda vector, dense, ranked: ranked,
        )


class _BoostedStore:
    """Dense ranks parent-0..6; the lexical leg ranks parent-6 first, so it is near the top."""

    def __init__(self) -> None:
        self.chunks = {
            f"parent-{i}": Chunk(f"parent-{i}", "corpus", f"parent {i}", {"segment": 0})
            for i in range(7)
        }

    def query_dense(self, _vector, k):
        return [ScoredChunk(self.chunks[f"parent-{i}"], 1.0 - i / 100) for i in range(min(k, 7))]

    def query_sparse(self, _query, k, vec=None):
        order = [6, 0, 1, 2, 3, 4, 5]
        return [ScoredChunk(self.chunks[f"parent-{i}"], 0.5) for i in order[:k]]

    def iter_chunks(self):
        yield from self.chunks.values()

    def explicit_superseded_chunk_ids(self):
        return frozenset()

    def chunks_by_ids(self, ids):
        return {i: self.chunks[i] for i in ids if i in self.chunks}


def _c8_top(tmp_path, placement: str | None) -> list[str]:
    clear_atomic_rescue_artifact_cache()
    generation, scope, corpus = "gen", "scope", "c" * 64
    root = tmp_path / "c8"
    if not (root / scope / generation / corpus / "manifest.json").exists():
        write_atomic_rescue_artifact(
            root / scope / generation / corpus,
            matrix=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            views=[{"chunk_id": "parent-6", "source": "corpus", "parent_ordinal": 6, "view_ordinal": 0}],
            generation_id=generation,
            calibration_id="cal",
            pipeline_fingerprint="pipe",
            corpus_fingerprint=corpus,
            embedding_profile="aml-test-code4",
            embedding_fingerprint=_C8Embedder.profile.fingerprint(),
            ordinary_chunk_count=7,
            source_commit="test",
        )
    binding = None
    if placement is not None:
        binding = AtomicRescueBinding(
            mode="active",
            artifact_root=str(root),
            scope_id=scope,
            generation_id=generation,
            calibration_id="cal",
            pipeline_fingerprint="pipe",
            corpus_fingerprint=corpus,
            placement=placement,
        )
    run = HostedRetriever(_C8Embedder(), _Reranker()).search(
        _BoostedStore(), "query", [], rerank=False, atomic_rescue=binding
    )
    assert placement is None or run.atomic_rescue_active is True
    return [hit.chunk.id for hit in run.hits[:5]]


def test_c8_fused_placement_keeps_the_fused_top_five(tmp_path) -> None:
    off = _c8_top(tmp_path, None)
    dense = _c8_top(tmp_path, "dense")
    fused = _c8_top(tmp_path, "fused")
    # The dense placement gives parent-6 a dense-rank-six vote and reorders the fused top five;
    # the fused placement cannot.
    assert dense != off
    assert fused == off
