from __future__ import annotations

import pytest

from recall.embedding_registry import registered_profile
from recall.embeddings import EmbeddingProfile, resolve_embedder, resolve_registered_embedder
from recall.generation_build import BuildRequest, pipeline_identity
from recall.lineage import PipelineIdentity


def test_profile_selection_is_shared_by_the_generic_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    sentinel = object()

    def fake(profile_id: str, env, *, shadow: bool = False):
        seen.update(profile_id=profile_id, env=env, shadow=shadow)
        return sentinel

    monkeypatch.setattr("recall.embeddings.resolve_registered_embedder", fake)

    result = resolve_embedder(
        "fastembed",
        {"RECALL_EMBED_PROFILE": "bge-small-context-section-v1"},
    )

    assert result is sentinel
    assert seen["profile_id"] == "bge-small-context-section-v1"
    assert seen["shadow"] is False


def test_profile_selection_rejects_non_fastembed_backends() -> None:
    with pytest.raises(ValueError, match="RECALL_EMBEDDER=fastembed"):
        resolve_embedder(
            "hashing",
            {"RECALL_EMBED_PROFILE": "bge-small-context-section-v1"},
        )


def test_registered_profile_resolver_uses_the_profile_artifact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    entry = registered_profile("bge-small-context-section-v1")

    def fake_build(self, *, artifact_path, artifact_digest):
        seen.update(profile_id=self.profile_id, artifact_path=artifact_path, digest=artifact_digest)
        return "built"

    monkeypatch.setattr(type(entry), "build", fake_build)

    result = resolve_registered_embedder(
        entry.profile_id,
        {
            "RECALL_MODEL_CACHE": "C:/models/bge",
            "RECALL_MODEL_SHA256": "a" * 64,
        },
    )

    assert result == "built"
    assert seen == {
        "profile_id": entry.profile_id,
        "artifact_path": "C:/models/bge",
        "digest": "a" * 64,
    }


class _ProfiledEmbedder:
    name = "BAAI/bge-small-en-v1.5"
    dim = 384

    def __init__(self) -> None:
        self.profile: EmbeddingProfile = registered_profile(
            "bge-small-context-section-v1"
        ).identity(artifact_digest="a" * 64)


def test_generation_pipeline_records_context_identity() -> None:
    pipeline = pipeline_identity(
        _ProfiledEmbedder(),
        BuildRequest(artifact_digest="a" * 64),
    )

    assert pipeline.embedder.profile_id == "bge-small-context-section-v1"
    assert pipeline.embedder.context_mode == "section"
    assert pipeline.embedder.context_version == "context-section-v1"

    restored = PipelineIdentity.from_dict(pipeline.to_dict())
    assert restored.fingerprint == pipeline.fingerprint
    assert restored.embedder.context_mode == "section"


def test_raw_generation_identity_keeps_legacy_serialization_shape() -> None:
    raw = PipelineIdentity.from_dict(
        {
            "embedder": {
                "provider": "fixture",
                "model": "model-a",
                "dimension": 64,
                "revision": "commit-a",
                "artifact_digest": None,
                "verified": True,
                "unverified_reason": None,
            },
            "chunker": {
                "algorithm": "paragraph-pack",
                "schema_version": 1,
                "configuration": {},
            },
        }
    )

    assert "profile_id" not in raw.to_dict()["embedder"]
    assert "context_mode" not in raw.to_dict()["embedder"]

