"""Served against exact best cosine: compared per stored labelled query, reported, never a gate.

Invariants and the failure each one catches:
- G1 a served score equal to the threshold clears it, as in `recall.trust_gate`; a flip is counted
  only when the exact and served scores fall on different sides.
- G2 agreement is a gap no larger than the tolerance, and a query served no dense hit disagrees;
  counts are kept per labelled class.
- G3 an abstain disagreement is an exact score that clears the threshold beside a served abstention,
  or the reverse; an abstention below the threshold is agreement.
- G4 `CalibrationRepository.served_gap` pairs every stored query with its OWN exact score, although
  the stored set is sorted with every unanswerable query first.
- G5 it reads only: no audit event, no calibration row, no change to resolution.
- G6 the CLI runs the MCP service search over the stored set and prints the report, and a lite
  store refuses the subcommand by name without creating a file.

Node IDs, all in `tests/test_served_gap.py`:
G1 `test_g1_a_served_score_at_the_threshold_clears_it`,
G2 `test_g2_agreement_is_within_tolerance_and_nothing_served_disagrees`,
G3 `test_g3_abstain_disagreement_has_the_right_polarity`,
G4 `test_g4_each_query_is_paired_with_its_own_exact_score`,
G5 `test_g5_served_gap_reads_only`,
G6 `test_g6_cli_reports_over_the_mcp_service_path`, `test_g6_lite_refuses_served_gap_by_name`.

Red proof, 2026-10-21, on a Linux host (not a workstation) against a throwaway pg18 container, each
mutation alone, the named test failing in the named assertion (exit 1, "1 failed", no error), then
restored byte for byte and the whole file green (7 passed):
- GU1 (G1) `served_gap_report` counting a served score as clearing only strictly above the
  threshold: "a served score equal to the threshold was counted below it".
- GU2 (G2) a query served nothing counted as agreeing: "a query served nothing agreed". Its first
  run failed on the class total instead; the test was reordered so the named assertion comes first.
- GU3 (G3) abstain disagreement with its polarity inverted: `report["abstain_disagreements"] == 2`.
- GU4 (G4) `CalibrationRepository.served_gap` pairing answerable queries with the unanswerable
  scores: "a query was paired with another query's exact score".
- GU5 (G5) `served_gap` appending an audit event: "served_gap wrote to the database".
- GU6 (G6) the CLI's search returning the abstention inverted: the output reports abstain
  disagreements.
- GU6b (G6) `served-gap` added to the lite-served subcommands: the lite path answers "calibration
  None not found" instead of refusing the subcommand by name.
Analysis checked against a known answer: the rows of the 2026-10-21 live measurement reproduce
72 of 78 and 47 of 48 agreeing, 0 flips, gap median 0.0321 and largest 0.0726.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
import pytest

from recall.calibration_v2 import CalibrationRepository, CalibrationStatus
from recall.cli import main as cli_main
from recall.generations import GenerationManager
from recall.served_gap import ServedGapRow, describe, served_gap_report
from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import _bodies, _CarryEmbedder, _labels, _ready
from tests.test_lite_cli import _run

# --------------------------------------------------------------------------------------------
# The report, from scores alone.
# --------------------------------------------------------------------------------------------


def test_g1_a_served_score_at_the_threshold_clears_it() -> None:
    report = served_gap_report(
        [ServedGapRow(True, 0.60, 0.41, False), ServedGapRow(True, 0.60, 0.4099, True)], 0.41
    )
    assert report["score_flips"] == 1, "a served score equal to the threshold was counted below it"
    assert report["flips"] == [{"answerable": True, "exact": 0.6, "served": 0.4099}]


def test_g2_agreement_is_within_tolerance_and_nothing_served_disagrees() -> None:
    rows = [
        ServedGapRow(True, 0.700, 0.6995, False),   # within 0.001: agrees
        ServedGapRow(True, 0.700, 0.6500, False),   # fused out: gap 0.05
        ServedGapRow(False, 0.300, None, True),     # nothing served: disagrees
        ServedGapRow(False, 0.300, 0.3000, True),
    ]
    report = served_gap_report(rows, 0.41)
    assert report["by_class"]["unanswerable"]["agree"] == 1, "a query served nothing agreed"
    assert report["by_class"]["answerable"]["agree"] == 1, "a gap beyond the tolerance agreed"
    assert report["agree"] == 2
    assert report["nothing_served"] == 1
    assert report["gap_max"] == 0.05
    assert report["score_flips"] == 0


def test_g3_abstain_disagreement_has_the_right_polarity() -> None:
    rows = [
        ServedGapRow(True, 0.70, 0.70, True),    # clears, yet abstained: disagreement
        ServedGapRow(False, 0.20, 0.20, True),   # below, abstained: agreement
        ServedGapRow(False, 0.20, 0.20, False),  # below, answered (a rule promoted it): disagreement
    ]
    report = served_gap_report(rows, 0.41)
    assert report["abstain_disagreements"] == 2
    assert report["by_class"]["unanswerable"]["abstain_disagreements"] == 1
    assert "2 abstain disagreement(s)" in describe(report)


# --------------------------------------------------------------------------------------------
# Against a real tenant.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def gap_tenant():
    tenant = "served-gap-test-" + uuid.uuid4().hex[:10]
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield tenant, manager
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for table in ("recall_calibrations", "recall_calibration_query_sets",
                      "recall_source_tombstones", "recall_audit_events", "recall_ingest_jobs",
                      "recall_tenant_state", "recall_generations"):
            conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))


def _served(gap_tenant) -> tuple[str, str, CalibrationRepository, _CarryEmbedder, str]:
    tenant, manager = gap_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    generation_id = _ready(manager, embedder, _bodies(), "v1")
    served = repository.publish(repository.calibrate(generation_id, _labels(), embedder).calibration_id)
    manager.promote(generation_id, unsafe_development=True)
    return tenant, generation_id, repository, embedder, served.calibration_id


def _search(query: str) -> tuple[float | None, bool]:
    """Answerable queries score 1.0 exactly and unanswerable 0.0; `answer-3` is served 0.2."""
    if query == "answer-3":
        return 0.2, True
    return (1.0, False) if query.startswith("answer-") else (0.0, True)


def _counts(tenant: str) -> tuple[int, int]:
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        events = conn.execute(
            "SELECT count(*) FROM recall_audit_events WHERE tenant_id = %s", (tenant,)
        ).fetchone()
        calibrations = conn.execute(
            "SELECT count(*) FROM recall_calibrations WHERE tenant_id = %s", (tenant,)
        ).fetchone()
    assert events is not None and calibrations is not None
    return int(events[0]), int(calibrations[0])


@requires_db
def test_g4_each_query_is_paired_with_its_own_exact_score(gap_tenant) -> None:
    _tenant, _generation_id, repository, embedder, served_id = _served(gap_tenant)
    report = repository.served_gap(served_id, embedder, _search)
    assert report["by_class"]["answerable"] == {
        "n": 22, "agree": 21, "score_flips": 1, "abstain_disagreements": 1
    }, "a query was paired with another query's exact score"
    assert report["by_class"]["unanswerable"] == {
        "n": 22, "agree": 22, "score_flips": 0, "abstain_disagreements": 0
    }
    assert report["flips"] == [{"answerable": True, "exact": 1.0, "served": 0.2}]
    assert report["calibration_id"] == served_id


@requires_db
def test_g5_served_gap_reads_only(gap_tenant) -> None:
    tenant, generation_id, repository, embedder, served_id = _served(gap_tenant)
    before = (_counts(tenant), repository.resolve(generation_id).status)
    repository.served_gap(served_id, embedder, _search)
    after = (_counts(tenant), repository.resolve(generation_id).status)
    assert after == before, "served_gap wrote to the database"
    assert after[1] == CalibrationStatus.CERTIFIED


@requires_db
def test_g6_cli_reports_over_the_mcp_service_path(gap_tenant, monkeypatch, capsys) -> None:
    tenant, _generation_id, _repository, embedder, served_id = _served(gap_tenant)
    monkeypatch.setattr("recall.cli_commands.calibration_cmd._make_embedder", lambda _name: embedder)
    cli_main(["--serving-dsn", TEST_DSN, "--tenant", tenant, "calibration", "served-gap"])
    out = capsys.readouterr().out
    assert f"calibration: {served_id}" in out
    assert "44 of 44 agree within 0.001; 0 verdict flip(s)" in out, out
    assert "abstain disagreement" not in out, out


def test_g6_lite_refuses_served_gap_by_name(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    out = _run(["--dsn", "sqlite:///" + path.as_posix(), "calibration", "served-gap"])
    assert "EXIT" in out and "calibration served-gap" in out and "lite store" in out, out
    assert not path.exists(), "a refused subcommand created the lite file"
