"""Deferred CCA fixes for PR 945 (query support and calibration coverage), no database.

Invariants and the failure each one catches:
- Q1 a measured `CalibrationCoverage` hashes: its report is compared but not hashed (CCA CODE-003).
- Q2 decoding a reasoning response whose coverage names another calibration than the response is
  refused (CCA SEC-001).
- Q3 the CLI prints the query-support status the result carries, not a fixed literal (CCA ENV-001).

Node IDs, all in `tests/test_query_support_deferred.py`:
Q1 `test_q1_a_measured_coverage_hashes`,
Q2 `test_q2_coverage_naming_another_fit_is_refused`,
Q3 `test_q3_the_cli_prints_the_carried_support_status`.

Red proof, 2026-10-28, on this workstation, CCA deferred run 20261028T185351Z, by
`cca_checks tautology verify` against the pre-fix files; see that run's L5.6 record.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from recall.evidence import EvidenceValidationError
from recall.reasoning_response import reasoning_response_from_dict
from recall.support import CalibrationCoverage, QuerySupport
from tests.test_query_support_reasoning import _response


def test_q1_a_measured_coverage_hashes() -> None:
    coverage = CalibrationCoverage(calibration_id="c", status="unavailable", report={"calibration_id": "c"})
    hashed = True
    try:
        hash(coverage)
    except TypeError:
        hashed = False
    assert hashed, "a coverage carrying a report could not be hashed"


def test_q2_coverage_naming_another_fit_is_refused() -> None:
    payload = _response().to_dict()
    payload["trusted_evidence"]["calibration_coverage"] = {"calibration_id": "another-fit", "status": "unmeasured"}
    with pytest.raises(EvidenceValidationError, match="another calibration"):
        reasoning_response_from_dict(payload)


def test_q3_the_cli_prints_the_carried_support_status(capsys) -> None:
    from dataclasses import replace

    from recall.cli_commands._shared import _print_result
    from tests.test_cli_terminal_injection import _result_with

    result = replace(_result_with("notes.md"), query_support=QuerySupport(status=cast(Any, "checked")))
    _print_result(result)
    out = capsys.readouterr().out
    assert "query support: checked" in out, "the CLI printed a fixed support status"
