"""PostgreSQL does what lite does: the unknown-term check and near-miss coverage at each fit.

Invariants and the failure each one catches:
- P1 `GenerationStore.unknown_terms` reads the pinned generation's own vocabulary with its own
  text configuration: a term present only in another generation is unknown, and a token that
  configuration drops entirely (a capitalised stop word) is never reported unknown.
- P2 strict trusted search on a generation store refuses a question naming an absent code by name
  (`unknown_term`, abstained, the term in the reason), and answers one naming a present code.
- P3 a scoped request reports the check unavailable and is not refused, so a whole-store
  vocabulary never discloses what lies outside the scope.
- P4 every fit records a lite-shaped coverage report bound to its calibration id, with the
  unknown-entity probes rejected by the gate; status reports it under `near_miss_coverage` and in
  its message; the fit's certification and lifecycle are what they were before the measurement.
- P5 a measurement that fails never withdraws the fit, and status then says coverage is
  unavailable.
- P6 every search on the served calibration carries that report as `calibration_coverage`.

Node IDs, all in `tests/test_postgres_near_miss_parity.py`:
P1 `test_p1_unknown_terms_reads_the_pinned_generation_vocabulary`,
P2 `test_p2_strict_search_refuses_an_unknown_term_by_name`,
P3 `test_p3_a_scoped_request_reports_the_check_unavailable`,
P4 `test_p4_each_fit_records_lite_shaped_coverage`,
P5 `test_p5_a_failing_measurement_never_withdraws_the_fit`,
P6 `test_p6_search_carries_the_fits_coverage`.

Red proof, 2026-10-21, on a Linux host (not a workstation) against a throwaway pg18 container, each
mutation alone, the named test failing in the named assertion (exit 1, "1 failed", no error), then
restored byte for byte and the whole file green (6 passed):
- PU1 (P1) `GenerationStore.unknown_terms` without the generation filter: "a term from another
  generation was known".
- PU1b (P1) the lexemes taken with `simple` instead of the generation's configuration: "a stop word
  the configuration drops was reported unknown".
- PU2 (P2) `GenerationStore.unknown_terms` renamed away: "the generation store was not checked".
- PU3 (P3) the scoped-request rule removed from `check_unknown_terms`: "a scoped request used the
  vocabulary".
- PU4 (P4) `calibrate` no longer recording coverage: "the fit recorded no near-miss coverage".
- PU4b (P4) status without the coverage keys: "status does not report the fit's coverage".
- PU4c (P4) the measurement superseding the fit it measured: "the measurement changed the fit". Its
  first form (marking the fit rejected) made the artifact inconsistent, so `get` raised before the
  assertion; the consistent form is the one recorded.
- PU5 (P5) the fail-open catch narrowed to ImportError: "a failing near-miss measurement withdrew
  the fit".
- PU6 (P6) `GenerationStore.near_miss_coverage` renamed away: "a search did not carry the fit's
  coverage".
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.calibration_v2 import CalibrationRepository
from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
from recall.near_miss import UNAVAILABLE_MESSAGE
from recall.support import validated_coverage_report
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy
from recall_mcp.status import calibration_status
from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import _CarryEmbedder, _labels, _ready

#: Twenty sources, each naming its own code, so a code is present and an invented one absent.
BODIES = {f"memo-{i}.md": f"document body {i} mentions PGV{i:03d} settings".encode() for i in range(20)}
STRICT = TrustPolicy.strict_policy()


@pytest.fixture
def parity_tenant():
    tenant = "parity-test-" + uuid.uuid4().hex[:10]
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield tenant, manager
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for table in ("recall_calibrations", "recall_calibration_query_sets",
                      "recall_source_tombstones", "recall_audit_events", "recall_ingest_jobs",
                      "recall_tenant_state", "recall_generations"):
            conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))


def _served(parity_tenant):
    tenant, manager = parity_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    generation_id = _ready(manager, embedder, BODIES, "v1")
    fitted = repository.calibrate(generation_id, _labels(), embedder)
    repository.publish(fitted.calibration_id)
    manager.promote(generation_id, unsafe_development=True)
    return tenant, manager, generation_id, repository, embedder, fitted.calibration_id


@requires_db
def test_p1_unknown_terms_reads_the_pinned_generation_vocabulary(parity_tenant) -> None:
    tenant, manager = parity_tenant
    embedder = _CarryEmbedder()
    first = _ready(manager, embedder, BODIES, "v1")
    second = _ready(manager, embedder, {**BODIES, "extra.md": b"only here ZXQ999 lives"}, "v2")
    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        with store.pin_generation(first):
            in_first = store.unknown_terms(["PGV003", "QX7777", "THE", "ZXQ999"])
        with store.pin_generation(second):
            in_second = store.unknown_terms(["PGV003", "QX7777", "THE", "ZXQ999"])
    finally:
        store.close()
    assert "ZXQ999" in in_first, "a term from another generation was known"
    assert "THE" not in in_first, "a stop word the configuration drops was reported unknown"
    assert in_first == ["QX7777", "ZXQ999"]
    assert in_second == ["QX7777"]


@requires_db
def test_p2_strict_search_refuses_an_unknown_term_by_name(parity_tenant) -> None:
    tenant, _manager, _generation_id, _repository, embedder, _cal = _served(parity_tenant)
    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        absent = trusted_search(store, embedder, "what does QX7777 configure", k=3, policy=STRICT)
        present = trusted_search(store, embedder, "what does PGV003 configure", k=3, policy=STRICT)
    finally:
        store.close()
    assert absent.diagnostics.unknown_term_check == "checked", "the generation store was not checked"
    assert absent.abstained is True, "a question naming an absent code was answered"
    assert absent.diagnostics.unknown_terms == ("QX7777",)
    assert "'QX7777'" in absent.reason
    assert {h.verdict for h in absent.hits} == {"unknown_term"}
    assert present.abstained is False and present.diagnostics.unknown_terms == ()


@requires_db
def test_p3_a_scoped_request_reports_the_check_unavailable(parity_tenant) -> None:
    tenant, _manager, _generation_id, _repository, embedder, _cal = _served(parity_tenant)
    source = f"s3://approved/corpora/{tenant}/memo-3.md"
    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        scoped = trusted_search(
            store, embedder, "what does QX7777 configure", k=3, source=source, policy=STRICT
        )
    finally:
        store.close()
    assert scoped.diagnostics.unknown_term_check == "unavailable", "a scoped request used the vocabulary"
    assert scoped.diagnostics.unknown_terms == ()


@requires_db
def test_p4_each_fit_records_lite_shaped_coverage(parity_tenant) -> None:
    tenant, manager = parity_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    generation_id = _ready(manager, embedder, BODIES, "v1")
    cal = repository.calibrate(generation_id, _labels(), embedder).calibration_id
    stored = repository.get(cal)
    assert stored.certified is True and stored.lifecycle_state == "draft", (
        "the measurement changed the fit"
    )
    report = repository.near_miss_coverage(cal)
    assert report is not None, "the fit recorded no near-miss coverage"
    assert validated_coverage_report(report, cal) == report
    unknown = report["unknown_entity"]
    assert unknown["n"] == 22 and unknown["rejected"] == 22, "invented codes were not rejected"
    assert report["unknown_term_gate"] == "checked"
    repository.publish(cal)
    manager.promote(generation_id, unsafe_development=True)
    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        status = calibration_status(store)
    finally:
        store.close()
    assert status.get("near_miss_coverage") == report, "status does not report the fit's coverage"
    assert "near-miss coverage: unknown-entity 22/22 rejected" in str(status.get("message"))


@requires_db
def test_p5_a_failing_measurement_never_withdraws_the_fit(parity_tenant, monkeypatch) -> None:
    tenant, manager = parity_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    generation_id = _ready(manager, embedder, BODIES, "v1")

    def broken(*_args, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr("recall.near_miss.near_miss_coverage", broken)
    try:
        fitted = repository.calibrate(generation_id, _labels(), embedder)
    except RuntimeError:
        pytest.fail("a failing near-miss measurement withdrew the fit")
    assert fitted.certified is True and fitted.lifecycle_state == "draft"
    assert repository.near_miss_coverage(fitted.calibration_id) is None
    repository.publish(fitted.calibration_id)
    manager.promote(generation_id, unsafe_development=True)
    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        status = calibration_status(store)
    finally:
        store.close()
    assert UNAVAILABLE_MESSAGE in str(status.get("message"))


@requires_db
def test_p6_search_carries_the_fits_coverage(parity_tenant) -> None:
    tenant, _manager, _generation_id, repository, embedder, cal = _served(parity_tenant)
    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        result = trusted_search(store, embedder, "what does PGV003 configure", k=3, policy=STRICT)
    finally:
        store.close()
    coverage = result.calibration_coverage
    assert coverage.calibration_id == cal
    assert coverage.method == "synthetic_corpus_probes", "a search did not carry the fit's coverage"
    assert coverage.report is not None and coverage.report["unknown_entity"]["rejected"] == 22
