"""Certified inventories of at most five chunks use ordinary trusted retrieval.

NUM-003 red proof runs the two small-inventory cases against the pre-fix trusted_search.
The remaining cases preserve existing refusals and pass on that baseline by design.
"""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    AtomicRescueSelectionError,
    clear_atomic_rescue_artifact_cache,
    write_atomic_rescue_artifact,
)
from recall.calibration_v2 import canonical_sha256
from recall.embeddings import embedding_profile, embedding_profile_id
from recall.lite import LiteStore
from recall.lite.calibration import pipeline_fingerprint
from recall.trust import trusted_search
from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal
from recall.types import Chunk
from tests.test_calibration_v2_hunt_fixes import _artifact


class _Embedder:
    name = "synthetic-small-corpus"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _store(tmp_path: Path, count: int) -> LiteStore:
    store = LiteStore(tmp_path / "memory.db", dim=2)
    store.upsert(
        [Chunk(str(i), f"memo-{i}.md", "alpha bravo charlie delta", {"file": f"memo-{i}.md", "ord": 0})
         for i in range(count)],
        [[1.0, 0.0]] * count,
    )
    fit = _artifact(
        tenant_id=store.tenant, generation_id=store.lite_generation_id(),
        embedder_identity={"model": _Embedder.name, "dimension": _Embedder.dim},
        pipeline_fingerprint=pipeline_fingerprint(_Embedder.name, _Embedder.dim),
        corpus_fingerprint=store.corpus_fingerprint(), lifecycle_state="published",
    )
    fit = replace(fit, checksum=canonical_sha256(fit.immutable_payload()))
    store.save_calibration(
        json.dumps(fit.to_dict()), created_at=fit.created_at, model=_Embedder.name, dimension=_Embedder.dim,
    )
    return store


@pytest.mark.parametrize("count", [1, 5])
def test_a_certified_small_inventory_serves_without_a_rescue_artifact(tmp_path: Path, count: int) -> None:
    with _store(tmp_path, count) as store:
        ordinary = trusted_search(store, _Embedder(), "alpha", k=6, env={"RECALL_ATOMIC_RESCUE_MODE": "off"})
        assert ordinary.calibrated and ordinary.trust_state == "trusted" and len(ordinary.hits) == count
        try:
            active = trusted_search(store, _Embedder(), "alpha", k=6, env={"RECALL_ATOMIC_RESCUE_MODE": "active"})
        except AtomicRescueArtifactError as exc:
            actual = type(exc).__name__
        else:
            actual = ([hit.chunk.id for hit in active.hits], active.calibration_id, active.trust_state)
        expected = ([hit.chunk.id for hit in ordinary.hits], ordinary.calibration_id, ordinary.trust_state)
        assert actual == expected, "a certified small inventory required an impossible sixth-parent rescue"


def test_six_actual_chunks_still_require_an_artifact(tmp_path: Path) -> None:
    """This supporting artifact refusal control passes before the repair."""
    with _store(tmp_path, 6) as store:
        with pytest.raises(AtomicRescueArtifactError, match="artifact root and generation identity"):
            trusted_search(store, _Embedder(), "alpha", env={"RECALL_ATOMIC_RESCUE_MODE": "active"})


def test_a_truncated_dense_result_does_not_make_a_six_chunk_inventory_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a valid artifact, the existing selector refusal remains a supporting control."""
    with _store(tmp_path, 6) as store:
        root = tmp_path / "atomic"
        binding = store.generation_binding()
        fit = store.resolve_calibration().artifact
        assert fit is not None
        write_atomic_rescue_artifact(
            store.atomic_rescue_manifest(root, binding["generation_id"], fit.calibration_id).parent,
            matrix=[[1.0, 0.0]] * 6,
            views=[{"chunk_id": str(i), "source": f"memo-{i}.md", "parent_ordinal": 0, "view_ordinal": 0}
                   for i in range(6)],
            generation_id=binding["generation_id"], calibration_id=fit.calibration_id,
            pipeline_fingerprint=binding["pipeline_fingerprint"], corpus_fingerprint=binding["corpus_fingerprint"],
            embedding_profile=embedding_profile_id(_Embedder()),
            embedding_fingerprint=embedding_profile(_Embedder()).fingerprint(),
            ordinary_chunk_count=6, source_commit="synthetic",
        )
        query = store.query_dense
        monkeypatch.setattr(store, "query_dense", lambda *args, **kwargs: query(*args, **kwargs)[:1])
        try:
            with pytest.raises(AtomicRescueSelectionError, match="five dense candidates"):
                trusted_search(store, _Embedder(), "alpha", env={
                    "RECALL_ATOMIC_RESCUE_MODE": "active", "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(root),
                })
        finally:
            clear_atomic_rescue_artifact_cache()


def test_a_small_inventory_still_requires_certified_trust(tmp_path: Path) -> None:
    """The existing strict calibration refusal is a supporting control."""
    with LiteStore(tmp_path / "memory.db", dim=2) as store:
        store.upsert([Chunk("one", "memo.md", "alpha bravo charlie delta", {})], [[1.0, 0.0]])
        with pytest.raises(TrustRefusal) as refused:
            trusted_search(store, _Embedder(), "alpha", env={"RECALL_ATOMIC_RESCUE_MODE": "active"})
        assert refused.value.code is TrustFailureCode.CALIBRATION_MISSING
        with pytest.raises(AtomicRescueArtifactError, match="artifact root and generation identity"):
            trusted_search(
                store, _Embedder(), "alpha", policy=TrustPolicy.development(),
                env={"RECALL_ATOMIC_RESCUE_MODE": "active"},
            )


def test_an_adapter_without_a_count_keeps_the_artifact_requirement(tmp_path: Path) -> None:
    """Missing inventory support preserves the existing artifact refusal."""
    with _store(tmp_path, 1) as store:
        class Adapter:
            count = None

            def __getattr__(self, name: str):  # noqa: ANN204
                return getattr(store, name)

        with pytest.raises(AtomicRescueArtifactError, match="artifact root and generation identity"):
            trusted_search(Adapter(), _Embedder(), "alpha", env={"RECALL_ATOMIC_RESCUE_MODE": "active"})
