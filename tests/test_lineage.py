from __future__ import annotations

import json

import pytest

from recall.lineage import (
    ChunkerIdentity,
    EmbedderIdentity,
    IndexManifestV1,
    LineageError,
    ManifestObjectV1,
    PipelineIdentity,
    UnverifiedPipelineError,
)


def _embedder(model: str, revision: str = "commit-a") -> EmbedderIdentity:
    return EmbedderIdentity(
        provider="test-provider",
        model=model,
        revision=revision,
        dimension=384,
    )


def _chunker(*, overlap: int = 80) -> ChunkerIdentity:
    return ChunkerIdentity(
        algorithm="paragraph-pack",
        schema_version=1,
        configuration={"max_chars": 800, "overlap": overlap},
    )


def test_same_dimension_different_models_and_revisions_never_share_a_pipeline() -> None:
    first = PipelineIdentity(_embedder("model-a"), _chunker())
    other_model = PipelineIdentity(_embedder("model-b"), _chunker())
    other_revision = PipelineIdentity(_embedder("model-a", "commit-b"), _chunker())

    assert len({first.fingerprint, other_model.fingerprint, other_revision.fingerprint}) == 3


def test_every_chunker_and_fts_input_changes_the_pipeline_fingerprint() -> None:
    base = PipelineIdentity(_embedder("model-a"), _chunker(overlap=80))
    overlap = PipelineIdentity(_embedder("model-a"), _chunker(overlap=81))
    fts = PipelineIdentity(
        _embedder("model-a"),
        _chunker(overlap=80),
        fts_configuration={"language": "simple", "schema_version": 1},
    )

    assert len({base.fingerprint, overlap.fingerprint, fts.fingerprint}) == 3


def test_identity_configuration_is_deeply_immutable() -> None:
    config = {"max_chars": 800, "nested": {"separators": ["\n\n", "\n"]}}
    chunker = ChunkerIdentity("paragraph-pack", 1, config)
    config["max_chars"] = 1
    config["nested"]["separators"].append(" ")

    assert chunker.to_dict()["configuration"] == {
        "max_chars": 800,
        "nested": {"separators": ["\n\n", "\n"]},
    }
    with pytest.raises(TypeError):
        chunker.configuration["max_chars"] = 1  # type: ignore[index]


def test_unverified_embedder_is_explicit_and_rejected_for_production() -> None:
    identity = EmbedderIdentity(
        provider="custom",
        model="mutable-alias",
        dimension=384,
        unverified_reason="development fixture",
    )
    pipeline = PipelineIdentity(identity, _chunker())

    assert not pipeline.verified
    with pytest.raises(UnverifiedPipelineError):
        pipeline.require_production_identity()
    with pytest.raises(LineageError, match="revision or artifact"):
        EmbedderIdentity(provider="custom", model="alias", dimension=384)


def test_manifest_is_canonical_regardless_of_input_object_order() -> None:
    a = ManifestObjectV1(
        uri="s3://corpus/memos/a.md",
        version_id="v1",
        media_type="text/markdown",
        size=4,
        sha256="a" * 64,
    )
    b = ManifestObjectV1(
        uri="s3://corpus/memos/b.md",
        version_id="v9",
        media_type="text/markdown",
        size=8,
        sha256="b" * 64,
    )
    first = IndexManifestV1("tenant-a", "2026-08-03", (b, a))
    second = IndexManifestV1("tenant-a", "2026-08-03", (a, b))

    assert first.digest == second.digest
    assert first.to_json() == second.to_json()
    assert IndexManifestV1.from_json(first.to_json()) == first
    assert json.loads(first.to_json())["objects"][0]["uri"].endswith("a.md")


def test_manifest_rejects_mutable_or_ambiguous_object_references() -> None:
    with pytest.raises(LineageError, match="version_id"):
        ManifestObjectV1("s3://corpus/a.md", "", "text/markdown", 1, "a" * 64)
    with pytest.raises(LineageError, match="immutable object version"):
        ManifestObjectV1("s3://corpus/a.md", "null", "text/markdown", 1, "a" * 64)
    with pytest.raises(LineageError, match="duplicate"):
        item = ManifestObjectV1(
            "s3://corpus/a.md", "v1", "text/markdown", 1, "a" * 64
        )
        IndexManifestV1("tenant", "v1", (item, item))
    with pytest.raises(LineageError, match="duplicate object URI"):
        first = ManifestObjectV1(
            "s3://corpus/a.md", "v1", "text/markdown", 1, "a" * 64
        )
        second = ManifestObjectV1(
            "s3://corpus/a.md", "v2", "text/markdown", 2, "b" * 64
        )
        IndexManifestV1("tenant", "v1", (first, second))
