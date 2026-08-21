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
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.calibration_v2 import CalibrationRepository, CalibrationStatus
from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
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
