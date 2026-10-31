"""Hosted provenance must be honest and survive generation storage.

Red proof on pre-fix lineage.py/generations.py: STAKES-001 accepts the
invented revision and fails the refusal assertion. DAT-001 raises LineageError
during build reload, which is caught and reported by the intended assertion.
"""
from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest

from recall.embeddings import EmbeddingProfile, HOSTED_UNVERIFIED_DIGEST
from recall.generation_build import BuildRequest, embedder_identity
from recall.generations import GenerationManager
from recall.index import chunk_text
from recall.lineage import (
    ChunkerIdentity, EmbedderIdentity, IndexManifestV1, LineageError,
    ManifestObjectV1, PipelineIdentity,
)
from recall.manifest import LocalObjectReader
from recall_mcp.factories import make_embedder
from tests.conftest import TEST_DSN, requires_db


@pytest.mark.parametrize("through_assembly", [False, True])
def test_hosted_revision_cannot_claim_pinned_bytes(through_assembly) -> None:
    class Hosted:
        name = "voyage:voyage-4"
        dim = 64
        profile = EmbeddingProfile(
            profile_id="test-hosted", model_name="voyage-4",
            artifact_digest=HOSTED_UNVERIFIED_DIGEST, dimension=64,
            query_mode="query", passage_mode="document",
        )

    refusal = None
    try:
        if through_assembly:
            embedder_identity(Hosted(), BuildRequest(revision="provider-release"))
        else:
            EmbedderIdentity(
                provider="voyage", model=Hosted.name, dimension=64,
                hosted=True, revision="provider-release",
            )
    except LineageError as exc:
        refusal = str(exc)
    assert refusal is not None, "a hosted revision falsely claimed pinned bytes"
    assert "revision" in refusal
    control = embedder_identity(Hosted(), BuildRequest())
    assert control.production_admissible and not control.verified


@requires_db
def test_profileless_hosted_generation_builds_after_creation(tmp_path) -> None:
    tenant = "hosted-reload-" + uuid.uuid4().hex[:10]
    embedder = make_embedder("hashing")
    source = tmp_path / "memo.md"
    data = b"a hosted identity must survive generation persistence"
    source.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    manifest = IndexManifestV1(tenant, "v1", (
        ManifestObjectV1(source.as_uri(), digest, "text/markdown", len(data), digest),
    ))
    pipeline = PipelineIdentity(
        EmbedderIdentity(provider="test-hosted", model=embedder.name, dimension=64, hosted=True),
        ChunkerIdentity("recall.chunk_text", 1),
    )
    fingerprint = pipeline.fingerprint
    wire = pipeline.to_dict()
    manager = GenerationManager(TEST_DSN, tenant, environment="production")
    try:
        record = manager.create(manifest, pipeline, allow_unverified=True)
        failure = None
        try:
            stats = manager.build(record.generation_id, LocalObjectReader([tmp_path]), embedder, chunk_text)
        except LineageError as exc:
            failure = str(exc)
        assert failure is None, f"accepted hosted identity could not be reloaded: {failure}"
        assert stats.chunks > 0
        assert manager.get(record.generation_id).pipeline_fingerprint == fingerprint
        with manager._connect() as conn:
            _, restored = manager._load_identity(conn, record.generation_id)
        assert restored.embedder.hosted and not restored.verified
        assert restored.fingerprint == fingerprint
        assert restored.to_dict() == wire
        assert "hosted" not in wire["embedder"]
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            conn.execute("DELETE FROM recall_chunks_v1 WHERE tenant_id = %s", (tenant,))
            conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))
