"""The PostgreSQL near-miss certificate: measured, reported beside the calibration, never a gate.

Invariants and the failure each one catches:
- M1 a near-miss question scoring exactly the threshold counts as confirmed, because serving marks
  a hit `low_confidence` only strictly below it: an off-by-one here understates false confirms.
- M2 the report keeps the lite shape and its rules: it passes the lite validator, and fewer than
  `MIN_PROBES` questions is unmeasured and provisional however many were rejected.
- M3 a stored report whose counts disagree with its own scores is refused, not printed.
- M4 status reports the certificate of the calibration actually serving, and only that one.
- M5 a failing certificate changes no serving decision: calibration resolution, strict trusted
  search, the MCP search service, status and carry-forward come out exactly as before it was
  recorded.
- M6 the audit row holds no question text: near-miss sets over a private corpus carry its content.
- M7 a set holding an answerable entry is refused, not scored as a near-miss.
- M8 the CLI measures and prints the certificate and the population the calibration's own
  certificate covers; on a lite store the subcommand is refused by name with the reason.

Node IDs, all in `tests/test_near_miss_certificate.py`:
M1 `test_m1_a_near_miss_at_the_threshold_is_confirmed`,
M2 `test_m2_the_report_keeps_the_lite_shape_and_rules`,
M3 `test_m3_a_report_that_disagrees_with_its_scores_is_refused`,
M4 `test_m4_status_reports_the_serving_calibrations_certificate_only`,
M5 `test_m5_a_failing_certificate_changes_no_serving_decision`,
M6 `test_m6_the_audit_row_holds_no_question_text`,
M7 `test_m7_an_answerable_entry_is_refused`,
M8 `test_m8_cli_prints_the_certificate_and_its_population`,
`test_m8_lite_refuses_near_miss_by_name`.

Red proof, 2026-10-21, on a Linux host (not a workstation) against a throwaway pg18 container, each
mutation alone, the named test failing in the named assertion (exit 1, "1 failed", no error), then
restored byte for byte and the whole file green (9 passed):
- MU1 (M1) `certificate_report` counting `score <= threshold` as rejected: "a score equal to the
  threshold was rejected".
- MU2 (M2) `provisional_in_vocabulary` without the `n < MIN_PROBES` term: "an unmeasured class
  read as covered".
- MU2b (M2) the per-class sample minimum removed from `certified`: "five near-miss questions
  certified a threshold".
- MU3 (M3) `validated_certificate` without the counts-against-scores check: DID NOT RAISE.
- MU4 (M4) `CalibrationRepository.near_miss_report` not filtering on the calibration id: the
  served calibration's summary was not "unmeasured" before its own measurement.
- MU4b (M4) `calibration_status` without `_near_miss_status`: the same assertion.
- MU5 (M5) `resolve_within` returning UNCERTIFIED when a failing certificate exists: "recording a
  failing near-miss certificate changed a serving decision".
- MU5b (M5) `carry_forward` raising when the parent's certificate fails: "a failing near-miss
  certificate blocked carry-forward".
- MU6 (M6) the question texts added to the recorded report: "a near-miss question's text reached
  the audit ledger".
- MU7 (M7) the answerable-entry refusal removed: DID NOT RAISE.
- MU8 (M8) the CLI's population line removed: "the CLI does not say what the certificate covers".
- MU8b (M8) the lite refusal's near-miss branch removed: the generic refusal lacks the reason.
The runner applies each mutation as one exact string replacement and is kept with the
maintainer's research records.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import psycopg
import pytest

from recall.calibration_v2 import (
    CalibrationBindingError,
    CalibrationError,
    CalibrationRepository,
    CalibrationStatus,
)
from recall.cli import main as cli_main
from recall.generation_store import GenerationStore
from recall.near_miss_certificate import (
    CERTIFICATE_POPULATION,
    EVENT_TYPE,
    certificate_report,
    validated_certificate,
)
from recall.support import validated_coverage_report
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall_mcp.service import search_memory
from recall_mcp.status import calibration_status
from tests.conftest import TEST_DSN, requires_db
from recall.generations import GenerationManager
from tests.test_calibration_carry_forward import _bodies, _CarryEmbedder, _labels, _ready
from tests.test_lite_cli import _run

#: The carry embedder puts every text that is neither `answer-` nor `missing-` on the documents'
#: axis, so these score cosine 1.0, exactly as high as an answerable question: a failing certificate.
NEAR_MISS = [{"query": f"near-secret-{index}", "answerable": False} for index in range(20)]
QUERIES = ("answer-1", "missing-1", "near-secret-1")


# --------------------------------------------------------------------------------------------
# The report, from scores alone.
# --------------------------------------------------------------------------------------------


def test_m1_a_near_miss_at_the_threshold_is_confirmed() -> None:
    report = certificate_report([0.7] * 20, [0.41, 0.41, 0.4099], 0.41)
    assert report["in_vocabulary"]["rejected"] == 1, "a score equal to the threshold was rejected"
    assert report["false_confirm_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_m2_the_report_keeps_the_lite_shape_and_rules() -> None:
    # Overlapping classes: AUC (25 * 15 + 25 * 45) / (50 * 45) = 0.667.
    failing = certificate_report([0.40] * 25 + [0.70] * 25, [0.45] * 30 + [0.30] * 15, 0.41)
    clean = validated_coverage_report({"calibration_id": "cal_x", **failing}, "cal_x")
    assert clean["in_vocabulary"] == {"n": 45, "rejected": 15, "rate": 0.333, "state": "measured"}
    assert clean["unknown_entity"]["state"] == "unmeasured"
    assert failing["provisional_in_vocabulary"] is True
    assert failing["certified"] is False
    assert failing["separability_ci"][0] < 0.90
    few = certificate_report([0.7] * 50, [0.1] * 5, 0.41)
    assert few["in_vocabulary"]["state"] == "unmeasured"
    assert few["provisional_in_vocabulary"] is True, "an unmeasured class read as covered"
    assert few["certified"] is False, "five near-miss questions certified a threshold"


def test_m3_a_report_that_disagrees_with_its_scores_is_refused() -> None:
    report = {"calibration_id": "cal_x", **certificate_report([0.7] * 20, [0.45] * 20, 0.41)}
    assert validated_certificate(report, "cal_x")["in_vocabulary"]["rejected"] == 0
    report["near_miss_scores"] = [0.30] * 20
    with pytest.raises(ValueError, match="disagree"):
        validated_certificate(report, "cal_x")


# --------------------------------------------------------------------------------------------
# Against a real tenant.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def near_miss_tenant():
    tenant = "near-miss-test-" + uuid.uuid4().hex[:10]
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield tenant, manager
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for table in ("recall_calibrations", "recall_calibration_query_sets",
                      "recall_source_tombstones", "recall_audit_events", "recall_ingest_jobs",
                      "recall_tenant_state", "recall_generations"):
            conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))


def _served(near_miss_tenant) -> tuple[str, str, CalibrationRepository, _CarryEmbedder, str]:
    tenant, manager = near_miss_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    generation_id = _ready(manager, embedder, _bodies(), "v1")
    served = repository.publish(repository.calibrate(generation_id, _labels(), embedder).calibration_id)
    manager.promote(generation_id, unsafe_development=True)
    return tenant, generation_id, repository, embedder, served.calibration_id


def _outcome(search):
    """A search's decision, or the strict refusal that replaced it: both are serving decisions."""
    try:
        return search()
    except TrustRefusal as refusal:
        return ("refused", refusal.code.value)


def _decisions(tenant: str, generation_id: str, repository, embedder) -> dict[str, object]:
    """Everything that decides what a caller is served, for a fixed set of queries."""
    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    strict_policy = TrustPolicy.strict_policy()
    try:
        strict = [
            _outcome(lambda q=q: _strict_view(
                trusted_search(store, embedder, q, k=3, policy=strict_policy)))
            for q in QUERIES
        ]
        service = [
            _outcome(lambda q=q: _service_view(
                search_memory(store, embedder, q, k=3, policy=strict_policy)))
            for q in QUERIES
        ]
        status = calibration_status(store)
    finally:
        store.close()
    return {
        "resolution": repository.resolve(generation_id).status,
        "strict": strict,
        "service": service,
        "status": (status["status"], status["calibration_id"], status["threshold"], status["certified"]),
    }


def _strict_view(r) -> tuple[object, ...]:
    return (r.abstained, r.trust_state, r.calibration_status, r.calibration_id,
            [(h.chunk.id, h.verdict, round(h.cosine, 6)) for h in r.hits])


def _service_view(r) -> tuple[object, ...]:
    return (r.abstained, r.trust_state, r.calibration_id, [(h.chunk_id, h.verdict) for h in r.hits])


@requires_db
def test_m4_status_reports_the_serving_calibrations_certificate_only(near_miss_tenant) -> None:
    tenant, generation_id, repository, embedder, served_id = _served(near_miss_tenant)
    draft = repository.calibrate(generation_id, _labels(), embedder)
    repository.measure_near_miss(draft.calibration_id, NEAR_MISS, embedder)

    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        before = calibration_status(store)
        repository.measure_near_miss(served_id, NEAR_MISS, embedder)
        after = calibration_status(store)
    finally:
        store.close()

    assert before.get("calibration_id") == served_id
    assert before.get("near_miss_certificate") is None, "another calibration's certificate was reported"
    assert str(before.get("near_miss_certificate_summary", "")).startswith("near-miss certificate: unmeasured")
    report = after.get("near_miss_certificate")
    assert isinstance(report, dict), "status does not carry the serving calibration's certificate"
    assert report["calibration_id"] == served_id
    assert report["in_vocabulary"]["n"] == len(NEAR_MISS)
    assert report["false_confirm_rate"] == 1.0
    assert after.get("certificate_population") == CERTIFICATE_POPULATION


@requires_db
def test_m5_a_failing_certificate_changes_no_serving_decision(near_miss_tenant) -> None:
    tenant, generation_id, repository, embedder, served_id = _served(near_miss_tenant)
    before = _decisions(tenant, generation_id, repository, embedder)
    assert before["resolution"] == CalibrationStatus.CERTIFIED

    report = repository.measure_near_miss(served_id, NEAR_MISS, embedder)
    # The certificate really fails, so a gate reading it would have something to act on.
    assert report["certified"] is False
    assert report["provisional_in_vocabulary"] is True
    assert report["false_confirm_rate"] == 1.0

    after = _decisions(tenant, generation_id, repository, embedder)
    assert after == before, "recording a failing near-miss certificate changed a serving decision"

    _tenant, manager = near_miss_tenant
    child = _ready(manager, embedder, _bodies(added=2), "v2")
    try:
        carried = repository.carry_forward(child, embedder)
    except CalibrationError as exc:
        pytest.fail(f"a failing near-miss certificate blocked carry-forward: {exc}")
    assert carried.certified is True, "a failing near-miss certificate uncertified carry-forward"
    assert carried.threshold == repository.get(served_id).threshold


@requires_db
def test_m6_the_audit_row_holds_no_question_text(near_miss_tenant) -> None:
    tenant, _generation_id, repository, embedder, served_id = _served(near_miss_tenant)
    repository.measure_near_miss(served_id, NEAR_MISS, embedder)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute(
            "SELECT payload::text FROM recall_audit_events WHERE tenant_id = %s AND event_type = %s",
            (tenant, EVENT_TYPE),
        ).fetchall()
    assert len(rows) == 1
    assert "secret" not in rows[0][0], "a near-miss question's text reached the audit ledger"


@requires_db
def test_m7_an_answerable_entry_is_refused(near_miss_tenant) -> None:
    _tenant, _generation_id, repository, embedder, served_id = _served(near_miss_tenant)
    mixed = [*NEAR_MISS, {"query": "answer-99", "answerable": True}]
    with pytest.raises(CalibrationBindingError, match="unanswerable questions only"):
        repository.measure_near_miss(served_id, mixed, embedder, record=False)


@requires_db
def test_m8_cli_prints_the_certificate_and_its_population(
    near_miss_tenant, tmp_path: Path, monkeypatch, capsys
) -> None:
    tenant, _generation_id, _repository, embedder, served_id = _served(near_miss_tenant)
    monkeypatch.setattr("recall.cli_commands.calibration_cmd._make_embedder", lambda _name: embedder)
    query_path = tmp_path / "near_miss.json"
    query_path.write_text(json.dumps(NEAR_MISS), encoding="utf-8")
    base = ["--serving-dsn", TEST_DSN, "--tenant", tenant, "calibration", "near-miss"]

    cli_main([*base, "--queries", str(query_path)])
    measured = capsys.readouterr().out
    cli_main(base)
    stored = capsys.readouterr().out

    for out in (measured, stored):
        assert f"calibration: {served_id}" in out
        assert "20 of 20 held-out near-miss question(s) clear threshold" in out, out
        assert CERTIFICATE_POPULATION in out, "the CLI does not say what the certificate covers"
    assert "recorded in the audit ledger" in measured


def test_m8_lite_refuses_near_miss_by_name(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    out = _run(["--dsn", "sqlite:///" + path.as_posix(), "calibration", "near-miss"])
    assert "EXIT" in out and "calibration near-miss" in out, out
    assert "measures its own near-miss coverage" in out, out
    assert not path.exists(), "a refused subcommand created the lite file"
