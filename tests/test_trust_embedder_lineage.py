"""A query embedded by one model must never be scored against another model's vectors.

⛔ The case these tests exist for, measured on VPS2 2026-08-20 before the check existed: a server
running `voyage:voyage-4` against a `bge-large` generation returned `trust_state: trusted`,
`failure_code: null`, and bound a certified calibration. Both models emit 1024 dimensions, so the
only comparison that ran was the dimension one, and it passed.

Nothing on the stdio path compared the MODEL. `check_enterprise_readiness` does, but only
`if control_plane is not None`, and a stdio server has none; its sibling check on the calibration's
identity is documented in `recall_mcp/server.py` as unreachable from startup.

`test_same_dimension_different_model_is_refused` is the one that matters: an equal-dimension
mismatch is the only kind that ever reached production, because an unequal one already failed.

The PROFILE tests below close the same hole one level down. Two profiles of one model share
`name` and `dim`: `voyage-4-v1` and `voyage-4-retrieval-v1` are both `voyage:voyage-4` at 1024 and
differ only in whether Voyage is sent an `input_type`. Comparing name and dimension let either
serve the other's generation, every query vector from one encoder scored against passages from
the other. Red proof for each is in its docstring; "pre-fix" means origin/master 63bbe161, where
`generation_binding` carried no profile and the gate compared `embedder_model` and
`embedder_dimension` only.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.calibration_v2 import (
    CalibrationBindingError,
    CalibrationRepository,
    CalibrationStatus,
)
from recall.embedding_core import EmbeddingProfile
from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
from recall.lineage import ChunkerIdentity, EmbedderIdentity, PipelineIdentity
from recall.trust import trusted_search
from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal
from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import (
    _CarryEmbedder,
    _bodies,
    _labels,
    _manifest,
    _reader,
)
from tests.test_generations import _pipeline


@pytest.fixture
def lineage_tenant():
    tenant = "lineage-test-" + uuid.uuid4().hex[:10]
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield tenant, manager
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_calibrations WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_calibration_query_sets WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_source_tombstones WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_ingest_jobs WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


def _serving_generation(manager, embedder):
    """A promoted generation with a published, certified calibration: the healthy starting state."""
    bodies = _bodies()
    manifest = _manifest(manager.tenant_id, bodies, "v1")
    generation = manager.create(manifest, _pipeline(embedder.name))
    manager.build(generation.generation_id, _reader(manifest, bodies), embedder, lambda t: [t])
    manager.validate(generation.generation_id)
    manager.promote(generation.generation_id, unsafe_development=True)
    repo = CalibrationRepository(TEST_DSN, manager.tenant_id, actor="pytest")
    artifact = repo.calibrate(generation.generation_id, _labels(), embedder)
    repo.publish(artifact.calibration_id)
    return generation.generation_id


@requires_db
def test_the_binding_reports_which_model_wrote_the_vectors(lineage_tenant) -> None:
    tenant, manager = lineage_tenant
    embedder = _CarryEmbedder()
    generation_id = _serving_generation(manager, embedder)

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        binding = store.generation_binding()

    assert binding["generation_id"] == generation_id
    assert binding["embedder_model"] == embedder.name
    assert binding["embedder_dimension"] == str(embedder.dim)


@requires_db
def test_the_matching_embedder_is_still_trusted(lineage_tenant) -> None:
    """The check must not refuse the CORRECT embedder.

    Worth its own test because the obvious implementation compares `embedding_profile_id(embedder)`
    against a `provider:model` string, and that spelling never matches, so a mismatch check written
    that way refuses everything while looking right.
    """
    tenant, manager = lineage_tenant
    embedder = _CarryEmbedder()
    _serving_generation(manager, embedder)

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        result = trusted_search(
            store, embedder, "answer-0", k=3, policy=TrustPolicy.strict_policy()
        )

    assert result.failure_code is None
    assert result.trust_state == "trusted"
    assert result.calibration_status == CalibrationStatus.CERTIFIED.value
    assert result.calibrated is True


@requires_db
def test_same_dimension_different_model_is_refused(lineage_tenant) -> None:
    """THE VPS2 CASE. Same dimension, different model, strict policy: refuse before retrieval."""
    tenant, manager = lineage_tenant
    built_with = _CarryEmbedder()
    _serving_generation(manager, built_with)

    impostor = _CarryEmbedder(model="a-different-model")
    assert impostor.dim == built_with.dim, "the whole point is that the dimensions agree"

    with GenerationStore(TEST_DSN, built_with.dim, tenant=tenant) as store:
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, impostor, "answer-0", k=3, policy=TrustPolicy.strict_policy())

    assert excinfo.value.code is TrustFailureCode.LINEAGE_MISMATCH


@requires_db
def test_development_mode_degrades_and_drops_the_calibration(lineage_tenant) -> None:
    """Degraded mode may still answer, but must not apply or name a threshold fitted elsewhere.

    Before the check, this exact call reported `trust_state: trusted`, `failure_code: null` and a
    bound `calibration_id`, which is the shape that makes a wrong answer indistinguishable from a
    right one.
    """
    tenant, manager = lineage_tenant
    built_with = _CarryEmbedder()
    _serving_generation(manager, built_with)

    impostor = _CarryEmbedder(model="a-different-model")
    with GenerationStore(TEST_DSN, built_with.dim, tenant=tenant) as store:
        result = trusted_search(
            store, impostor, "answer-0", k=3, policy=TrustPolicy.development()
        )

    assert result.failure_code == TrustFailureCode.LINEAGE_MISMATCH.value
    assert result.trust_state == "degraded"
    assert result.calibration_id is None, "must not name an artifact it did not apply"
    assert result.calibrated is False
    # The status stays as the repository reported it: a certified artifact really does exist for
    # this generation. The mismatch is with the RUNTIME embedder, and the failure code says so.
    assert result.calibration_status == CalibrationStatus.CERTIFIED.value


@requires_db
def test_a_legacy_store_without_a_binding_is_unaffected(lineage_tenant) -> None:
    """No generation, no recorded model, so the check has nothing to compare and must not fire.

    Guards against the mismatch branch turning every uncalibrated legacy search into
    LINEAGE_MISMATCH, which would be a far louder regression than the bug it fixes.
    """
    from recall.store import PgVectorStore

    tenant, _manager = lineage_tenant
    embedder = _CarryEmbedder()
    with PgVectorStore(TEST_DSN, dim=embedder.dim, tenant=tenant) as store:
        store.ensure_schema()
        result = trusted_search(
            store, embedder, "answer-0", k=3, policy=TrustPolicy.development()
        )

    assert result.failure_code != TrustFailureCode.LINEAGE_MISMATCH.value


# --------------------------------------------------------------------------------------------
# Two profiles of ONE model: same `name`, same `dim`, different encoder.
# --------------------------------------------------------------------------------------------


class _ProfiledEmbedder(_CarryEmbedder):
    """`_CarryEmbedder` carrying a profile, the way `RegisteredProfile.build` hands one out.

    The vectors ignore the profile on purpose. The defect is that two profiles of one model agree
    on `name` and `dim`, so the identity is the only thing that can tell them apart; a fixture
    whose numbers differed would let a check on the numbers pass for a check on the identity.
    """

    def __init__(
        self,
        profile_id: str = "fixture-untyped-v1",
        *,
        query_mode: str = "embed",
        passage_mode: str = "embed",
        client_version: str = "1.0.0",
    ) -> None:
        super().__init__()
        self._profile = EmbeddingProfile(
            profile_id=profile_id,
            model_name=self.model,
            artifact_digest="fixture-unverified",
            dimension=self.dim,
            query_mode=query_mode,
            passage_mode=passage_mode,
            dependencies=(("fixture-client", client_version),),
        )

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile


def _typed() -> _ProfiledEmbedder:
    """The retrieval spelling of the same model: `voyage-4-retrieval-v1` against `voyage-4-v1`."""
    return _ProfiledEmbedder("fixture-typed-v1", query_mode="query", passage_mode="document")


def _profiled_pipeline(embedder: _ProfiledEmbedder) -> PipelineIdentity:
    """`_pipeline`, plus the profile fields `generation_build.embedder_identity` records."""
    profile = embedder.profile
    return PipelineIdentity(
        EmbedderIdentity(
            "fixture",
            embedder.name,
            embedder.dim,
            revision=f"{embedder.name}-commit",
            profile_id=profile.profile_id,
            profile_fingerprint=profile.fingerprint(),
            context_version=profile.context_version,
        ),
        ChunkerIdentity("paragraph-pack", 1, {"max_chars": 800, "overlap": 80}),
        fts_configuration={"language": "english", "schema_version": 1},
    )


def _built(manager, embedder, bodies: dict[str, bytes], version: str, *, profiled: bool) -> str:
    manifest = _manifest(manager.tenant_id, bodies, version)
    pipeline = _profiled_pipeline(embedder) if profiled else _pipeline(embedder.name)
    generation = manager.create(manifest, pipeline)
    manager.build(generation.generation_id, _reader(manifest, bodies), embedder, lambda t: [t])
    manager.validate(generation.generation_id)
    return generation.generation_id


def _serving(manager, embedder, *, profiled: bool = True) -> str:
    """`_serving_generation`, built and calibrated under `embedder`'s recorded profile."""
    generation_id = _built(manager, embedder, _bodies(), "v1", profiled=profiled)
    manager.promote(generation_id, unsafe_development=True)
    repo = CalibrationRepository(TEST_DSN, manager.tenant_id, actor="pytest")
    repo.publish(repo.calibrate(generation_id, _labels(), embedder).calibration_id)
    return generation_id


@requires_db
def test_the_binding_reports_which_profile_wrote_the_vectors(lineage_tenant) -> None:
    """Red proof: pre-fix, `binding.get("embedder_profile_id")` was None, failing the first assert."""
    tenant, manager = lineage_tenant
    embedder = _ProfiledEmbedder()
    _serving(manager, embedder)

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        binding = store.generation_binding()

    assert binding.get("embedder_profile_id") == embedder.profile.profile_id
    assert binding.get("embedder_profile_fingerprint") == embedder.profile.fingerprint()


@requires_db
@pytest.mark.parametrize(
    ("built_with", "served_by"),
    [(_ProfiledEmbedder, _typed), (_typed, _ProfiledEmbedder)],
    ids=["untyped-generation-typed-runtime", "typed-generation-untyped-runtime"],
)
def test_same_model_different_profile_is_refused(lineage_tenant, built_with, served_by) -> None:
    """THE PROFILE CASE. Same model name, same dimension, different encoder: refuse.

    Red proof, both ids: pre-fix, `trusted_search` returned a trusted result and the test failed
    with `Failed: DID NOT RAISE TrustRefusal`.
    """
    tenant, manager = lineage_tenant
    original = built_with()
    _serving(manager, original)
    runtime = served_by()
    assert (runtime.name, runtime.dim) == (original.name, original.dim)
    assert runtime.profile.profile_id != original.profile.profile_id

    with GenerationStore(TEST_DSN, original.dim, tenant=tenant) as store:
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, runtime, "answer-0", k=3, policy=TrustPolicy.strict_policy())

    assert excinfo.value.code is TrustFailureCode.LINEAGE_MISMATCH


@requires_db
def test_a_legacy_runtime_cannot_serve_a_generation_built_under_a_profile(lineage_tenant) -> None:
    """`voyage:<model>` without RECALL_EMBED_PROFILE builds a profile-less, untyped embedder.

    Against a generation built under a typed profile it is the same wrong pairing as above, and
    `GenerationManager._check_build_identity` already refuses it at build time.

    Red proof: pre-fix, `Failed: DID NOT RAISE TrustRefusal`.
    """
    tenant, manager = lineage_tenant
    original = _typed()
    _serving(manager, original)
    legacy = _CarryEmbedder()
    assert not hasattr(legacy, "profile")

    with GenerationStore(TEST_DSN, original.dim, tenant=tenant) as store:
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, legacy, "answer-0", k=3, policy=TrustPolicy.strict_policy())

    assert excinfo.value.code is TrustFailureCode.LINEAGE_MISMATCH


@requires_db
def test_same_profile_with_a_different_fingerprint_is_refused(lineage_tenant) -> None:
    """The recorded fingerprint binds too, as it already does at build time.

    Red proof: pre-fix, `Failed: DID NOT RAISE TrustRefusal`. Mutation proof: deleting the
    fingerprint branch of `recall.embedding_core.profile_lineage_mismatch` failed it the same way.
    """
    tenant, manager = lineage_tenant
    original = _ProfiledEmbedder(client_version="1.0.0")
    _serving(manager, original)
    upgraded = _ProfiledEmbedder(client_version="2.0.0")
    assert upgraded.profile.profile_id == original.profile.profile_id

    with GenerationStore(TEST_DSN, original.dim, tenant=tenant) as store:
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, upgraded, "answer-0", k=3, policy=TrustPolicy.strict_policy())

    assert excinfo.value.code is TrustFailureCode.LINEAGE_MISMATCH


@requires_db
def test_the_matching_profile_is_still_trusted(lineage_tenant) -> None:
    """The check must not refuse the CORRECT embedder, which is the live memory tenant's case.

    Passes pre-fix by construction. Mutation proof: replacing the gate's `profile_lineage_mismatch`
    call in `recall.trust.trusted_search` with the obvious spelling, recorded profile against
    `embedder.name`, raised `TrustRefusal: LINEAGE_MISMATCH` from `trusted_search` here.
    """
    tenant, manager = lineage_tenant
    _serving(manager, _ProfiledEmbedder())

    with GenerationStore(TEST_DSN, _ProfiledEmbedder.dim, tenant=tenant) as store:
        result = trusted_search(
            store, _ProfiledEmbedder(), "answer-0", k=3, policy=TrustPolicy.strict_policy()
        )

    assert result.failure_code is None
    assert result.trust_state == "trusted"
    assert result.calibration_status == CalibrationStatus.CERTIFIED.value
    assert result.calibrated is True


@requires_db
def test_a_generation_that_recorded_no_profile_serves_as_before(lineage_tenant) -> None:
    """Legacy generations (the live code tenant's case) are compared on name and dimension only.

    Passes pre-fix by construction. Mutation proof: replacing the gate's `profile_lineage_mismatch`
    call in `recall.trust.trusted_search` with a comparison that lacks the "nothing recorded"
    guard (`binding.get("embedder_profile_id") != embedding_profile_id(embedder)`) raised
    `TrustRefusal: LINEAGE_MISMATCH` from `trusted_search` here.
    """
    tenant, manager = lineage_tenant
    _serving(manager, _CarryEmbedder(), profiled=False)

    with GenerationStore(TEST_DSN, _CarryEmbedder.dim, tenant=tenant) as store:
        assert "embedder_profile_id" not in store.generation_binding()
        result = trusted_search(
            store, _ProfiledEmbedder(), "answer-0", k=3, policy=TrustPolicy.strict_policy()
        )

    assert result.failure_code is None
    assert result.trust_state == "trusted"
    assert result.calibrated is True


@requires_db
def test_calibration_refuses_a_runtime_on_another_profile(lineage_tenant) -> None:
    """A threshold fitted on cosines across two encoders would certify numbers that mean nothing.

    Red proof: pre-fix, `calibrate` fitted and returned an artifact, and the test failed with
    `Failed: DID NOT RAISE CalibrationBindingError`.
    """
    _tenant, manager = lineage_tenant
    generation_id = _built(manager, _ProfiledEmbedder(), _bodies(), "v1", profiled=True)
    repo = CalibrationRepository(TEST_DSN, manager.tenant_id, actor="pytest")

    with pytest.raises(CalibrationBindingError, match="profile"):
        repo.calibrate(generation_id, _labels(), _typed())


@requires_db
def test_carry_forward_refuses_a_runtime_on_another_profile(lineage_tenant) -> None:
    """Carry-forward re-scores the parent's labels with the runtime embedder, so the same rule.

    Red proof: pre-fix, the threshold was carried forward and certified, and the test failed with
    `Failed: DID NOT RAISE CalibrationBindingError`.
    """
    _tenant, manager = lineage_tenant
    original = _ProfiledEmbedder()
    _serving(manager, original)
    child = _built(manager, original, _bodies(added=2), "v2", profiled=True)
    repo = CalibrationRepository(TEST_DSN, manager.tenant_id, actor="pytest")

    with pytest.raises(CalibrationBindingError, match="profile"):
        repo.carry_forward(child, _typed())
