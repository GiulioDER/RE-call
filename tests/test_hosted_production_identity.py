"""A hosted embedder may back a production generation without pinning an artifact it cannot have.

⛔ **What these tests are worth is decided by the mutation run, not by their passing.** Every one
of them was watched go red with its fix removed; the commands are in the module docstring of
`tests/test_hosted_production_identity_mutations.txt`. The failure this file exists to prevent is
not "hosted is refused" but the subtler one recorded in `recall/generations.py`: a fix that admits
hosted at `require_production_identity` and then refuses it one line lower at `allow_unverified`,
leaving behaviour unchanged while every unit test of the gate turns green.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.embeddings import (
    HOSTED_UNVERIFIED_DIGEST,
    LEGACY_UNVERIFIED_DIGEST,
    EmbeddingProfile,
    embedder_is_hosted,
)
from recall.generations import GenerationManager
from recall.lineage import (
    ChunkerIdentity,
    EmbedderIdentity,
    IndexManifestV1,
    LineageError,
    ManifestObjectV1,
    PipelineIdentity,
    UnverifiedPipelineError,
)
from tests.conftest import TEST_DSN, requires_db


def _hosted() -> EmbedderIdentity:
    return EmbedderIdentity(provider="voyage", model="voyage:voyage-4", dimension=1024, hosted=True)


def _pipeline(embedder: EmbedderIdentity) -> PipelineIdentity:
    return PipelineIdentity(
        embedder=embedder, chunker=ChunkerIdentity(algorithm="recall.chunk_text", schema_version=1)
    )


# --------------------------------------------------------------------------- the identity itself


def test_a_hosted_identity_needs_no_digest_revision_or_unverified_reason() -> None:
    """The construction that raised before this work, and the whole point of the change.

    A hosted provider has no artifact to hash and no revision to pin, so demanding one of the three
    forced every hosted caller to declare itself a development build in order to exist at all.
    """
    identity = _hosted()
    assert identity.artifact_digest is None
    assert identity.revision is None
    assert identity.unverified_reason is None


def test_a_non_hosted_identity_still_needs_one_of_them() -> None:
    """The exemption must not leak to local artifacts, which CAN be pinned and therefore must be."""
    with pytest.raises(LineageError, match="immutable revision or artifact digest"):
        EmbedderIdentity(provider="fastembed", model="bge-small", dimension=384)


def test_hosted_is_admissible_but_is_never_verified() -> None:
    """🔑 The distinction the whole change rests on: admissible and pinned are different claims.

    If `verified` were flipped instead, every caller asking "are these bytes pinned?" would start
    getting a false yes, which is the false-immutability trap `RegisteredProfile` already refuses.
    """
    identity = _hosted()
    assert identity.production_admissible is True
    assert identity.verified is False
    assert identity.to_dict()["verified"] is False


def test_a_hosted_identity_cannot_also_pin_an_artifact_digest() -> None:
    with pytest.raises(LineageError, match="cannot pin an artifact digest"):
        EmbedderIdentity(
            provider="voyage",
            model="voyage:voyage-4",
            dimension=1024,
            hosted=True,
            artifact_digest="a" * 64,
        )


def test_hosted_is_absent_from_the_serialized_shape_so_fingerprints_do_not_move() -> None:
    """⛔ The guard on the live corpus: shipping this must not re-key existing pipelines.

    A new serialized field would give every generation in existence a new `pipeline_fingerprint`
    and strand it from its own calibration history. The value is a build-time question, so it is
    kept out of `to_dict` on purpose and this pins that decision.
    """
    assert "hosted" not in _hosted().to_dict()

    # ⚠️ Toggle ONLY the field under test. The first version of this compared a hosted identity
    # carrying no `unverified_reason` against a local one carrying "explicit development build",
    # and failed — correctly, because `unverified_reason` IS serialized. That failure was the test
    # measuring the wrong pair, not the code moving a fingerprint, and it is recorded here because
    # a green version of that comparison would have proved nothing about `hosted` at all.
    fields = dict(
        provider="voyage",
        model="voyage:voyage-4",
        dimension=1024,
        unverified_reason="explicit development build",
    )
    before = EmbedderIdentity(**fields)
    after = EmbedderIdentity(**fields, hosted=True)
    assert _pipeline(before).fingerprint == _pipeline(after).fingerprint

    # And the identity the INGEST path actually builds is the `after` shape: it passes
    # `unverified=not embedder_digest`, which is True for a hosted embedder, so the reason is still
    # stamped. That is what keeps a live corpus on its existing pipeline lineage across this change.
    assert after.production_admissible is True
    assert after.to_dict() == before.to_dict()


# ------------------------------------------------------------------------------------- the gate


def test_require_production_identity_admits_hosted() -> None:
    _pipeline(_hosted()).require_production_identity()  # must not raise


def test_require_production_identity_still_refuses_an_unpinned_local_artifact() -> None:
    local = EmbedderIdentity(
        provider="fastembed",
        model="bge-small",
        dimension=384,
        unverified_reason="explicit development build",
    )
    with pytest.raises(UnverifiedPipelineError):
        _pipeline(local).require_production_identity()


# ------------------------------------------------------------- the runtime predicate that feeds it


def test_a_registered_hosted_profile_is_recognised() -> None:
    class _WithProfile:
        profile = EmbeddingProfile(
            profile_id="voyage-4-v1",
            model_name="voyage-4",
            artifact_digest=HOSTED_UNVERIFIED_DIGEST,
            dimension=1024,
            query_mode="query",
            passage_mode="document",
        )

    assert embedder_is_hosted(_WithProfile()) is True


def test_a_legacy_local_profile_is_not_hosted() -> None:
    class _Local:
        profile = EmbeddingProfile(
            profile_id="bge-small",
            model_name="bge-small",
            artifact_digest=LEGACY_UNVERIFIED_DIGEST,
            dimension=384,
            query_mode="query",
            passage_mode="document",
        )

    assert embedder_is_hosted(_Local()) is False


def test_an_object_that_merely_looks_hosted_is_not_hosted() -> None:
    """No duck-typing. An arbitrary object growing a `profile` attribute must not pass the gate."""

    class _Impostor:
        profile = "voyage"
        name = "voyage:voyage-4"

    assert embedder_is_hosted(_Impostor()) is False


# ------------------------------------------------------------------ the gate as callers reach it
#
# ⛔ **Everything above this line can pass while the product stays broken**, and that is not a
# hypothetical: `GenerationManager.create` refuses in TWO places under production, and both callers
# pass `allow_unverified=not pipeline.verified`, which is permanently True for a hosted endpoint.
# Admitting hosted at `require_production_identity` alone moves the refusal one line lower and
# changes nothing a user can observe. These tests call `create` for that reason.

def _one_object_manifest(tenant: str) -> IndexManifestV1:
    return IndexManifestV1(
        tenant,
        "corpus-v1",
        (ManifestObjectV1("s3://approved/corpora/x/memo.md", "v1", "text/markdown", 5, "a" * 64),),
    )


def _production(tenant: str) -> GenerationManager:
    return GenerationManager(TEST_DSN, tenant, actor="pytest", environment="production")


@requires_db
def test_a_hosted_pipeline_can_create_a_generation_in_production() -> None:
    """The behaviour the user asked for, asserted where it is actually decided.

    Before this change every hosted corpus had to run under `RECALL_ENV=development` to get past
    this call, which also redirects reads to the legacy `chunks` table — so the workaround for the
    gate silently split indexing and serving across two tables.
    """
    tenant = "hosted-prod-" + uuid.uuid4().hex[:10]
    manager = _production(tenant)
    pipeline = _pipeline(
        EmbedderIdentity(
            provider="voyage",
            model="voyage:voyage-4",
            # ⚠️ 64 because this checkout's container is `vector(64)`. The first version passed 1024
            # and died on the dimension check, which sits BELOW both refusals — so it proved the
            # gate was passed while reading as a failure, and its sibling below would have passed
            # on that same unrelated error and proved nothing at all.
            dimension=64,
            hosted=True,
            # What the ingest path stamps today. Kept so this test exercises the real shape.
            unverified_reason="explicit development build",
        )
    )
    try:
        record = manager.create(
            _one_object_manifest(tenant), pipeline, allow_unverified=not pipeline.verified
        )
        assert record.generation_id.startswith("gen_")
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


@requires_db
def test_an_unpinned_LOCAL_pipeline_is_still_refused_in_production() -> None:
    """The exemption must be exactly one provider-shaped hole, not a hole in the gate."""
    tenant = "local-prod-" + uuid.uuid4().hex[:10]
    pipeline = _pipeline(
        EmbedderIdentity(
            provider="fastembed",
            model="bge-small",
            dimension=64,
            unverified_reason="explicit development build",
        )
    )
    # The SPECIFIC refusal, not "something raised". `create` can also raise GenerationError for a
    # dimension mismatch, and accepting that would let this test pass with the gate wide open.
    with pytest.raises(UnverifiedPipelineError, match="production generation builds require"):
        _production(tenant).create(
            _one_object_manifest(tenant), pipeline, allow_unverified=True
        )


# --------------------------------------------------------- the wiring, which nothing above covers
#
# ⛔ **Added because a mutation SURVIVED.** Replacing `hosted=embedder_is_hosted(embedder)` with
# `hosted=False` in `recall/generation_build.py` left all twelve tests above green, because every
# one of them builds an `EmbedderIdentity` by hand and none goes through `embedder_identity()`.
# That is the whole fix shipping inert: the flag exists, the gate honours it, and nothing ever sets
# it — which is precisely the defect class this repository keeps rediscovering.


def test_a_live_hosted_embedder_produces_an_admissible_identity() -> None:
    """From the EMBEDDER to the gate's answer, which is the only path a real caller takes."""
    from recall.generation_build import BuildRequest, embedder_identity

    class _HostedEmbedder:
        name = "voyage:voyage-4"
        dim = 1024
        profile = EmbeddingProfile(
            profile_id="voyage-4-v1",
            model_name="voyage-4",
            artifact_digest=HOSTED_UNVERIFIED_DIGEST,
            dimension=1024,
            query_mode="query",
            passage_mode="document",
        )

    # `unverified=True` is what the ingest path passes for an embedder with no digest.
    identity = embedder_identity(_HostedEmbedder(), BuildRequest(unverified=True))
    assert identity.hosted is True
    assert identity.production_admissible is True
    assert identity.verified is False


def test_a_live_LOCAL_embedder_stays_inadmissible() -> None:
    """The other half. Without this, `hosted=True` hard-coded would also pass the test above."""
    from recall.generation_build import BuildRequest, embedder_identity

    class _LocalEmbedder:
        name = "bge-small"
        dim = 384

    identity = embedder_identity(_LocalEmbedder(), BuildRequest(unverified=True))
    assert identity.hosted is False
    assert identity.production_admissible is False


# -------------------------------------------------------------- end to end, through the real path
#
# 🔑 **The claim this file is really making**, and the only one that matters to a user: a hosted
# corpus can accept an upload under `RECALL_ENV=production`. Everything above tests a predicate or
# a gate in isolation; this drives `generation_ingest`, which is what `recall_ingest` calls, against
# a real database with production actually set.
#
# The sibling file `test_mcp_ingest_routing.py` monkeypatches `generation_ingest` and inspects its
# source, so it can prove routing and cannot prove this.


@requires_db
def test_a_hosted_upload_is_accepted_under_production(tmp_path, monkeypatch) -> None:
    """Before this change this raised `UnverifiedPipelineError` and no generation was created."""
    from recall.store import PgVectorStore
    from recall_mcp.service import generation_ingest

    monkeypatch.setenv("RECALL_ENV", "production")
    tenant = "hosted-e2e-" + uuid.uuid4().hex[:10]

    staged = tmp_path / tenant / "job1"
    staged.mkdir(parents=True)
    (staged / "memo.md").write_text(
        "The kingfisher protocol reconciles ledger drift at dawn.", encoding="utf-8"
    )

    class _HostedEmbedder:
        """Stands in for VoyageEmbedder: hosted identity, deterministic vectors, no network.

        ⚠️ A stub is used for the VECTORS, never for the gate: the gate under test runs for real,
        against a real `GenerationManager` with `RECALL_ENV=production` genuinely set. Faking the
        thing under test is what makes a test agree with itself.
        """

        name = "voyage:voyage-4"
        dim = 64
        profile = EmbeddingProfile(
            profile_id="voyage-4-v1",
            model_name="voyage-4",
            artifact_digest=HOSTED_UNVERIFIED_DIGEST,
            dimension=64,
            query_mode="query",
            passage_mode="document",
        )

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(t) % 7) for _ in range(64)] for t in texts]

    store = PgVectorStore(TEST_DSN, dim=64, tenant=tenant)
    try:
        result = generation_ingest(store, _HostedEmbedder(), str(staged), "text")
    finally:
        store.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            for table in ("recall_chunks_v1", "recall_generations"):
                conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))

    # The upload was BUILT. Whether it also went live depends on `_certify_upload` finding enough
    # content to certify, which one memo will not; that is a separate, documented outcome and is
    # reported rather than raised. What must be true is that production no longer REFUSES it.
    assert result.chunks >= 1
