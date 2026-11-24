"""A generation's sparse leg must not be served by a server that stems words differently.

PostgreSQL 19 changed eight Snowball stemmers. Measured 2026-11-24 with `to_tsvector('english', w)`
on `pgvector/pgvector:pg18` and `:pg19` over the 9,553 distinct words of this repository's
documentation: 13 stem differently ("organization" is `organ` on 18 and `organiz` on 19). A
database moved to 19 with `pg_upgrade` keeps the `tsv` its generations stored under 18, while
every query is parsed by 19, so the keyword leg silently misses those chunks and the calibrated
threshold was fitted on the other ranking. Each generation now records a hash of
`FTS_STEMMER_PROBE`'s lexemes, and the trust gate treats a different hash on the serving server as
LINEAGE_MISMATCH, the code it already uses for a different embedder.

One server cannot stem two ways, so drift is simulated by rewriting the recorded hash.

Red proof, each against a deliberate mutation of the production line named, run serially on
pgvector/pgvector:pg18 at the branch's base `9d965896a` plus this change:

- `test_a_generation_records_this_servers_stemmer_outside_its_fingerprint`: mutation deletes
  `stored_identity["fts_stemmer_probe"] = ...` in `GenerationManager.create`; fails on the
  `recorded == expected` assertion (recorded is None).
- `test_a_generation_stemmed_by_another_server_is_refused`: mutation sets `stemmer_drift = False`
  in `recall.trust`; fails with `Failed: DID NOT RAISE TrustRefusal`.
- `test_a_drifted_generation_degrades_without_its_calibration`: same mutation; fails on the
  `failure_code` assertion (None).
- `test_a_generation_built_before_the_probe_is_unaffected`: mutation replaces the
  `isinstance(recorded_probe, str)` gate in `GenerationStore.generation_binding` with `True` and
  records `str(recorded_probe)`; fails on the `failure_code is None` assertion (LINEAGE_MISMATCH).
- `test_doctor_names_the_drift_and_its_repair`: mutation makes `_stemmer_check` compare
  `row[0] != row[0]`; fails on the `status == "fail"` assertion.
"""

from __future__ import annotations

import psycopg
import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.doctor import _stemmer_check
from recall.generation_store import FTS_STEMMER_PROBE, GenerationStore
from recall.lineage import PipelineIdentity
from recall.trust import trusted_search
from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal
from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import _CarryEmbedder
from tests.test_trust_embedder_lineage import _serving_generation, lineage_tenant  # noqa: F401


def _rewrite_probe(tenant: str, generation_id: str, expression: str) -> None:
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            f"UPDATE recall_generations SET pipeline_identity = {expression} "
            "WHERE tenant_id = %s AND generation_id = %s",
            (tenant, generation_id),
        )


def _stemmed_elsewhere(tenant: str, generation_id: str) -> None:
    _rewrite_probe(
        tenant,
        generation_id,
        "jsonb_set(pipeline_identity, '{fts_stemmer_probe}', '\"stemmed-by-another-server\"')",
    )


@requires_db
def test_a_generation_records_this_servers_stemmer_outside_its_fingerprint(lineage_tenant) -> None:  # noqa: F811
    tenant, manager = lineage_tenant
    embedder = _CarryEmbedder()
    generation_id = _serving_generation(manager, embedder)

    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        stored, fingerprint = conn.execute(
            "SELECT pipeline_identity, pipeline_fingerprint FROM recall_generations "
            "WHERE tenant_id = %s AND generation_id = %s",
            (tenant, generation_id),
        ).fetchone()
        (expected,) = conn.execute(
            "SELECT md5(to_tsvector(%s::regconfig, %s)::text)",
            (stored["fts_configuration"]["language"], FTS_STEMMER_PROBE),
        ).fetchone()

    recorded = stored.get("fts_stemmer_probe")
    assert recorded == expected, "the generation must record how THIS server stems the probe"
    assert PipelineIdentity.from_dict(stored).fingerprint == fingerprint, (
        "the probe describes the server, not the configured pipeline, so it must stay outside the "
        "fingerprint that calibrations and build reuse bind to"
    )
    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        result = trusted_search(store, embedder, "answer-0", k=3, policy=TrustPolicy.strict_policy())
    assert result.failure_code is None, "the server that built the generation must still serve it"


@requires_db
def test_a_generation_stemmed_by_another_server_is_refused(lineage_tenant) -> None:  # noqa: F811
    tenant, manager = lineage_tenant
    embedder = _CarryEmbedder()
    generation_id = _serving_generation(manager, embedder)
    _stemmed_elsewhere(tenant, generation_id)

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, embedder, "answer-0", k=3, policy=TrustPolicy.strict_policy())

    assert excinfo.value.code is TrustFailureCode.LINEAGE_MISMATCH


@requires_db
def test_a_drifted_generation_degrades_without_its_calibration(lineage_tenant) -> None:  # noqa: F811
    tenant, manager = lineage_tenant
    embedder = _CarryEmbedder()
    generation_id = _serving_generation(manager, embedder)
    _stemmed_elsewhere(tenant, generation_id)

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        result = trusted_search(store, embedder, "answer-0", k=3, policy=TrustPolicy.development())

    assert result.failure_code == TrustFailureCode.LINEAGE_MISMATCH.value
    assert result.trust_state == "degraded"
    assert result.calibration_id is None, "a threshold fitted on the other stemmer must not apply"
    assert result.calibration_status == CalibrationStatus.CERTIFIED.value


@requires_db
def test_a_generation_built_before_the_probe_is_unaffected(lineage_tenant) -> None:  # noqa: F811
    """Every generation built before this change has no probe, and must serve exactly as before."""
    tenant, manager = lineage_tenant
    embedder = _CarryEmbedder()
    generation_id = _serving_generation(manager, embedder)
    _rewrite_probe(tenant, generation_id, "pipeline_identity - 'fts_stemmer_probe'")

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        result = trusted_search(store, embedder, "answer-0", k=3, policy=TrustPolicy.development())

    assert result.failure_code is None
    assert result.trust_state == "trusted"


@requires_db
def test_doctor_names_the_drift_and_its_repair(lineage_tenant) -> None:  # noqa: F811
    tenant, manager = lineage_tenant
    generation_id = _serving_generation(manager, _CarryEmbedder())

    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        healthy = _stemmer_check(conn, tenant, generation_id)
    _stemmed_elsewhere(tenant, generation_id)
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        drifted = _stemmer_check(conn, tenant, generation_id)

    assert healthy.status == "ok"
    assert drifted.status == "fail"
    assert drifted.fix is not None and "recall generation build" in drifted.fix
