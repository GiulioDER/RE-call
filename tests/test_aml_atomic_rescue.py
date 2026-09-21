"""Hosted C8 atomic rescue integration proofs."""

from __future__ import annotations

import numpy as np

from recall.atomic_rescue import write_atomic_rescue_artifact
from recall.types import Chunk, ScoredChunk
from recall_aml.retrieval import AtomicRescueBinding, HostedRetriever
from recall_aml.service import HostedService
from recall_aml.variants import variant


class _Embedder:
    dim = 3
    name = "aml-test-code4"

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _Reranker:
    def rerank(self, _query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return hits


class _CandidateStore:
    """Seven independent parents, so the artifact has a genuine outside-top-five candidate."""

    def __init__(self) -> None:
        self.chunks = {
            f"parent-{index}": Chunk(
                id=f"parent-{index}",
                source="candidate-corpus",
                text=f"atomic candidate parent {index}",
                metadata={"ordinal": index, "segment": 0},
            )
            for index in range(7)
        }
        self.loaded: list[str] = []

    def query_dense(self, _vector: list[float], k: int) -> list[ScoredChunk]:
        return [
            ScoredChunk(self.chunks[f"parent-{index}"], 1.0 - index / 100)
            for index in range(min(k, 7))
        ]

    def query_sparse(self, _query: str, k: int, vec=None) -> list[ScoredChunk]:
        return self.query_dense([1.0, 0.0, 0.0], k)

    def iter_chunks(self):
        yield from self.chunks.values()

    def explicit_superseded_chunk_ids(self) -> frozenset[str]:
        return frozenset()

    def chunks_by_ids(self, ids: list[str]) -> dict[str, Chunk]:
        self.loaded.extend(ids)
        return {chunk_id: self.chunks[chunk_id] for chunk_id in ids if chunk_id in self.chunks}


def test_active_c8_atomic_rescue_loads_a_real_outside_top_five_candidate(tmp_path) -> None:
    """A valid C8 artifact rescues parent six, not merely a configuration-only no-op.

    Red proof: removing the sixth-rank atomic insertion leaves ``store.loaded`` empty and
    fails the assertion that the candidate parent is materialized from this seven-parent corpus.
    """
    generation = "aml-c8-candidate-generation"
    artifact_root = tmp_path / "artifacts"
    write_atomic_rescue_artifact(
        artifact_root / generation,
        matrix=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        views=[
            {
                "chunk_id": "parent-6",
                "source": "candidate-corpus",
                "parent_ordinal": 6,
                "view_ordinal": 0,
            }
        ],
        generation_id=generation,
        calibration_id="aml-c8-calibration-v1",
        pipeline_fingerprint="aml-c8-pipeline-v1",
        corpus_fingerprint="c" * 64,
        embedding_profile="aml-test-code4",
        ordinary_chunk_count=7,
        source_commit="test",
    )
    store = _CandidateStore()
    run = HostedRetriever(_Embedder(), _Reranker()).search(
        store,
        "find the atomic candidate",
        [],
        rerank=False,
        atomic_rescue=AtomicRescueBinding(
            mode="active",
            artifact_root=str(artifact_root),
            generation_id=generation,
            calibration_id="aml-c8-calibration-v1",
            pipeline_fingerprint="aml-c8-pipeline-v1",
            corpus_fingerprint="c" * 64,
        ),
    )

    assert store.loaded == ["parent-6"]
    assert run.atomic_rescue_attempted is True
    assert run.atomic_rescue_active is True
    assert run.atomic_rescue_candidate_available is True
    assert run.atomic_rescue_fallback is False


def test_c8_alone_exposes_the_generation_bound_atomic_adapter(monkeypatch) -> None:
    """C8, rather than C7, turns the hosted artifact adapter on.

    Red proof: changing C8's ``atomic_rescue`` flag to false returns ``None`` and fails the
    binding assertion, showing that the test observes the service-to-retriever wiring.
    """
    c8 = variant("C8_routed_specialists_grounded_graph")
    service = HostedService(
        object(),
        object(),
        HostedRetriever(_Embedder(), _Reranker()),
        behavior=c8,
        multimodal_embedder=object(),
        specialist_retrievers={
            c8.context_embedding_profile: HostedRetriever(_Embedder(), _Reranker())
        },
    )
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "active")
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", "/atomic-artifacts")
    monkeypatch.setenv("RECALL_AML_ATOMIC_RESCUE_CALIBRATION_ID", "calibration")
    monkeypatch.setenv("RECALL_AML_ATOMIC_RESCUE_PIPELINE_FINGERPRINT", "pipeline")

    binding = service._atomic_rescue_binding(
        {"generation_id": "c8-generation", "corpus_sha256": "f" * 64}
    )

    assert binding == AtomicRescueBinding(
        mode="active",
        artifact_root="/atomic-artifacts",
        generation_id="c8-generation",
        calibration_id="calibration",
        pipeline_fingerprint="pipeline",
        corpus_fingerprint="f" * 64,
    )
    assert variant("C7_routed_specialists").atomic_rescue is False


def test_invalid_active_artifact_falls_back_without_changing_hosted_candidates(tmp_path) -> None:
    """A lineage mismatch disables the optional rescue instead of serving altered results."""
    store = _CandidateStore()
    run = HostedRetriever(_Embedder(), _Reranker()).search(
        store,
        "ordinary retrieval survives bad artifact lineage",
        [],
        rerank=False,
        atomic_rescue=AtomicRescueBinding(
            mode="active",
            artifact_root=str(tmp_path),
            generation_id="missing-generation",
            calibration_id="calibration",
            pipeline_fingerprint="pipeline",
            corpus_fingerprint="f" * 64,
        ),
    )

    assert [hit.chunk.id for hit in run.hits[:7]] == [
        f"parent-{index}" for index in range(7)
    ]
    assert run.atomic_rescue_attempted is True
    assert run.atomic_rescue_active is False
    assert run.atomic_rescue_fallback is True
    assert run.atomic_rescue_candidate_available is False
    assert store.loaded == []
