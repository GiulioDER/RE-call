"""Hosted C8 atomic rescue integration proofs."""

from __future__ import annotations

import numpy as np

from recall.atomic_rescue import write_atomic_rescue_artifact
from recall.embeddings import EmbeddingProfile
from recall.types import Chunk, ScoredChunk
from recall_aml.retrieval import AtomicRescueBinding, HostedRetriever
from recall_aml.service import HostedService
from recall_aml.variants import variant


class _Embedder:
    dim = 3
    name = "aml-test-code4"
    profile = EmbeddingProfile(
        profile_id="aml-test-code4",
        model_name="test-model",
        artifact_digest="test-digest",
        dimension=3,
        query_mode="embed",
        passage_mode="embed",
    )

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
    """A matching scoped C8 artifact rescues parent six, not merely a configuration-only no-op.

    Red proof receipt ``aml-c8-scoped-atomic-01`` targets the ``scope_id`` argument passed to
    ``resolve_atomic_rescue_manifest``. Replacing it with ``None`` leaves the manifest unreadable,
    sets ``atomic_rescue_fallback``, and fails the active assertion below.
    """
    generation = "aml-c8-candidate-generation"
    scope = "aml_scope_candidate"
    artifact_root = tmp_path / "artifacts"
    write_atomic_rescue_artifact(
        artifact_root / scope / generation / ("c" * 64),
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
        embedding_fingerprint=_Embedder.profile.fingerprint(),
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
            scope_id=scope,
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
        {"generation_id": "c8-generation", "corpus_sha256": "f" * 64},
        "aml_scope_one",
    )

    assert binding == AtomicRescueBinding(
        mode="active",
        artifact_root="/atomic-artifacts",
        scope_id="aml_scope_one",
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
            scope_id="aml_scope_missing",
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


def test_scoped_artifacts_with_one_generation_do_not_collide_in_one_process(tmp_path) -> None:
    """C8 resolves distinct corpus fingerprints inside one opaque scope.

    Red proof receipt ``aml-c8-scoped-atomic-02`` targets fingerprint aware artifact resolution.
    Replacing ``binding.corpus_fingerprint`` with ``"a" * 64`` makes the second binding load the
    first artifact and fail the two active assertions below.
    """
    generation = "shared-generation"
    root = tmp_path / "artifacts"
    bindings: list[AtomicRescueBinding] = []
    for scope, corpus in (("aml_scope_one", "a" * 64), ("aml_scope_one", "b" * 64)):
        write_atomic_rescue_artifact(
            root / scope / generation / corpus,
            matrix=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            views=[
                {
                    "chunk_id": "parent-6",
                    "source": f"candidate-{scope}",
                    "parent_ordinal": 6,
                    "view_ordinal": 0,
                }
            ],
            generation_id=generation,
            calibration_id="calibration",
            pipeline_fingerprint="pipeline",
            corpus_fingerprint=corpus,
            embedding_profile="aml-test-code4",
            embedding_fingerprint=_Embedder.profile.fingerprint(),
            ordinary_chunk_count=7,
            source_commit="test",
        )
        bindings.append(
            AtomicRescueBinding(
                mode="active",
                artifact_root=str(root),
                scope_id=scope,
                generation_id=generation,
                calibration_id="calibration",
                pipeline_fingerprint="pipeline",
                corpus_fingerprint=corpus,
            )
        )

    results = [
        HostedRetriever(_Embedder(), _Reranker()).search(
            _CandidateStore(), "find the atomic candidate", [], rerank=False, atomic_rescue=binding
        )
        for binding in bindings
    ]

    assert [run.atomic_rescue_active for run in results] == [True, True]
    assert [run.atomic_rescue_fallback for run in results] == [False, False]


def test_matching_manifest_in_another_scope_fails_closed(tmp_path) -> None:
    """An artifact is not reusable across opaque scopes even with matching lineage.

    Red proof receipt ``aml-c8-scoped-atomic-03`` targets the scope passed to artifact resolution.
    Replacing it with the stored scope activates the artifact and fails the fallback assertion.
    """
    generation = "shared-generation"
    corpus = "c" * 64
    root = tmp_path / "artifacts"
    write_atomic_rescue_artifact(
        root / "aml_scope_stored" / generation / corpus,
        matrix=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        views=[
            {
                "chunk_id": "parent-6",
                "source": "candidate-stored",
                "parent_ordinal": 6,
                "view_ordinal": 0,
            }
        ],
        generation_id=generation,
        calibration_id="calibration",
        pipeline_fingerprint="pipeline",
        corpus_fingerprint=corpus,
        embedding_profile="aml-test-code4",
        embedding_fingerprint=_Embedder.profile.fingerprint(),
        ordinary_chunk_count=7,
        source_commit="test",
    )

    run = HostedRetriever(_Embedder(), _Reranker()).search(
        _CandidateStore(),
        "find the atomic candidate",
        [],
        rerank=False,
        atomic_rescue=AtomicRescueBinding(
            mode="active",
            artifact_root=str(root),
            scope_id="aml_scope_requested",
            generation_id=generation,
            calibration_id="calibration",
            pipeline_fingerprint="pipeline",
            corpus_fingerprint=corpus,
        ),
    )

    assert run.atomic_rescue_active is False
    assert run.atomic_rescue_fallback is True
