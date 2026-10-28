"""The calibration CLI's optional reports never fail a command, and served-gap stays read only.

Invariants and the failure each one catches:
- C1 `recall calibration calibrate` prints the fit and then the coverage line; a coverage reader
  that raises prints the unavailable line instead, and the command still succeeds, because the fit
  is already saved.
- C2 `recall calibration near-miss` without a question file prints a stored report that cannot be
  read as unavailable, naming only the error's type, instead of failing; with `--json` it prints
  that as JSON, never as text.
- C3 `recall calibration served-gap` refuses by name when `RECALL_SOURCE_POLICY_FILE` is set, before
  any search runs: a policy-filtered search is not the population calibration sampled.
- C4 served-gap drives the MCP service search with the decision ledger off even when the
  environment turns it on, so a diagnostic never writes a served-search decision row.
- C5 served-gap compares the served side's unrounded cosine, the value the trust gate compares
  with the threshold, never the response's 4-place score, which near the threshold reports a
  verdict flip serving never made (CCA NUM-001, PR 969).
- C6 a stored coverage row that is damaged (a class missing its `rejected` count) reads as
  unavailable in the status line instead of raising: Lite's reader returns the row unvalidated
  (CCA DAT-001, PR 969).
- C7 a coverage report measured with the unknown-term gate `checked` says so on the status line
  when this process would not run that gate (`RECALL_UNKNOWN_TERM_GATE` off, or a source policy
  scoping every request), and says nothing when it would (CCA STAKES-001, PR 969, owner yes
  2026-10-28).

None needs a database: the repository, store and search are doubles at the CLI's own boundary.

Node IDs, all in `tests/test_calibration_cli_report_only.py`:
C1 `test_c1_calibrate_survives_an_unreadable_coverage_report`,
C2 `test_c2_near_miss_reports_an_unreadable_certificate_as_unavailable`,
C3 `test_c3_served_gap_refuses_under_a_source_policy`,
C4 `test_c4_served_gap_never_turns_the_decision_ledger_on`,
C5 `test_c5_served_gap_compares_the_unrounded_served_cosine`,
C6 `test_c6_a_damaged_stored_coverage_row_reads_as_unavailable`,
C7 `test_c7_status_says_when_this_process_does_not_run_the_measured_gate`,
C8 `test_c8_served_gap_refuses_a_report_when_a_promotion_lands_mid_run` (CCA DAT-002, PR 969: the
  served side searches the active generation per query, the exact side is pinned, and the report
  was printed under the calibration's generation after a promotion).

Red proof, 2026-10-28, on this workstation (no database needed), by `cca_checks tautology verify`
against the pre-fix files of CCA deferred run 20261028T130504Z, then green on the fixed tree:
- C8 against `_cmd_served_gap` without the post-run re-read: RED, 1 failed (DID NOT RAISE
  SystemExit).

Red proof, 2026-10-28, on this workstation (no database needed): C5 and C6 run by
`cca_checks tautology verify` against the pre-fix files (both RED, 1 failed each); C7 written first
and run against `recall.near_miss.coverage_summary` before `_serving_gate_note` existed, failing in
its named assertion for both parameters ("status showed the fit's checked gate as if this process
ran it"), then green with the fix.

Red proof for the CCA audit fixes, 2026-10-21, on a Linux host against a throwaway pg18 container,
each mutation alone, the named test failing in the named assertion (exit 1, "1 failed", no
error), then restored byte for byte and the four touched test files green (30 passed):
- FU3 (C1) `coverage_summary`'s fail-open catch narrowed to ImportError: "an unreadable coverage
  report failed calibrate".
- FU4 (C2) the near-miss stored-read catch narrowed to ImportError: "an unreadable stored
  certificate failed the near-miss command".
- FU4b (C2) the `--json` branch of that fallback removed: "--json printed a damaged report as text".
- FU5 (C3) the `RECALL_SOURCE_POLICY_FILE` refusal removed: DID NOT RAISE.
- FU6 (C4) the environment passed whole (`dict(os.environ)`): "served-gap searched with the
  decision ledger on".
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

import recall.calibration_v2 as calibration_v2
import recall.cli_commands.calibration_cmd as calibration_cmd
from recall.decision_ledger import LEDGER_ENV, DecisionLedger
from recall.near_miss import UNAVAILABLE_MESSAGE
from recall.served_gap import served_gap_report

CAL = "cal_cli_double"
SERVED_COSINE = 0.50946
DSN = "postgresql://cli-double@127.0.0.1:1/none"


class _Embedder:
    name = "double"
    dim = 4


class _Repository:
    """The CalibrationRepository surface these commands touch, and nothing else."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.envs: list[dict[str, str]] = []

    def calibrate(self, generation_id, _labels, _embedder):
        return SimpleNamespace(
            calibration_id=CAL, status=SimpleNamespace(value="certified"),
            generation_id=generation_id, pipeline_fingerprint="p", corpus_fingerprint="c",
            query_set_digest="q",
        )

    def near_miss_coverage(self, _calibration_id):
        raise RuntimeError("private provider text that must not be printed")

    def near_miss_report(self, _calibration_id):
        raise ValueError("private stored text that must not be printed")

    def get(self, _calibration_id):
        return SimpleNamespace(generation_id="gen_double")

    def served_gap(self, _calibration_id, _embedder, search, *, tolerance=None):
        self.served = search("answer-1")
        return served_gap_report([], 0.41)


class _Store:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def _args(command: str, **extra) -> argparse.Namespace:
    base = {"dsn": DSN, "tenant": "t", "calibration_cmd": command, "embedder": "double",
            "json": False}
    return argparse.Namespace(**{**base, **extra})


@pytest.fixture
def doubles(monkeypatch):
    import recall.generation_store as generation_store
    import recall_mcp.service as service

    repository = _Repository()

    # One served hit whose cosine sits within rounding of a threshold: the response model rounds
    # it to 4 places, the trusted retrieval underneath keeps it whole.
    def search_memory(_store, _embedder, _query, *, k, policy, env):
        repository.envs.append(dict(env))
        return SimpleNamespace(hits=[SimpleNamespace(score=round(SERVED_COSINE, 4))], abstained=True)

    def retrieve_trusted(_store, _embedder, _query, _source, _k, _calibration, _policy, *, env):
        repository.envs.append(dict(env))
        hits = [SimpleNamespace(cosine=SERVED_COSINE)]
        return SimpleNamespace(result=SimpleNamespace(hits=hits, abstained=True))

    monkeypatch.setattr(generation_store, "GenerationStore", _Store)
    monkeypatch.setattr(service, "search_memory", search_memory)
    monkeypatch.setattr(service, "_retrieve_trusted", retrieve_trusted)
    monkeypatch.setattr(calibration_v2, "CalibrationRepository", lambda *a, **k: repository)
    monkeypatch.setattr(calibration_v2, "load_query_set", lambda _path: ([], "digest"))
    monkeypatch.setattr(calibration_cmd, "_make_embedder", lambda _name: _Embedder())
    monkeypatch.setattr(calibration_cmd, "_serving_calibration_id", lambda _a, _r, _active=None: CAL)
    monkeypatch.setattr(calibration_cmd, "_active_generation", lambda _a: "gen_double")
    return repository


def test_c1_calibrate_survives_an_unreadable_coverage_report(doubles, capsys) -> None:
    args = _args("calibrate", generation="gen_double", query_file="q.json", publish=False)
    try:
        calibration_cmd._cmd_calibration(args)
    except (RuntimeError, SystemExit):
        pytest.fail("an unreadable coverage report failed calibrate after the fit was saved")
    out = capsys.readouterr().out
    assert f"calibration: {CAL}" in out
    assert UNAVAILABLE_MESSAGE in out
    assert "private" not in out


def test_c2_near_miss_reports_an_unreadable_certificate_as_unavailable(doubles, capsys) -> None:
    args = _args("near-miss", query_file=None, record=False)
    try:
        calibration_cmd._cmd_calibration(args)
    except (ValueError, SystemExit):
        pytest.fail("an unreadable stored certificate failed the near-miss command")
    out = capsys.readouterr().out
    assert "near-miss certificate unavailable" in out and "(ValueError)" in out
    assert "private" not in out
    calibration_cmd._cmd_calibration(_args("near-miss", query_file=None, record=False, json=True))
    raw = capsys.readouterr().out
    assert raw.lstrip().startswith("{"), "--json printed a damaged report as text"
    parsed = json.loads(raw)
    assert parsed["near_miss_certificate"] is None
    assert parsed["error"] == "ValueError"


def test_c3_served_gap_refuses_under_a_source_policy(doubles, monkeypatch) -> None:
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", "/policies/sources.toml")
    with pytest.raises(SystemExit, match="RECALL_SOURCE_POLICY_FILE"):
        calibration_cmd._cmd_calibration(_args("served-gap", k=5, tolerance=None))
    assert doubles.envs == [], "served-gap searched under a source policy"


def test_c4_served_gap_never_turns_the_decision_ledger_on(doubles, monkeypatch) -> None:
    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    monkeypatch.setenv(LEDGER_ENV, "1")
    calibration_cmd._cmd_calibration(_args("served-gap", k=5, tolerance=None))
    assert len(doubles.envs) == 1
    assert DecisionLedger.from_env(_Store(), env=doubles.envs[0]) is None, (
        "served-gap searched with the decision ledger on"
    )


def test_c5_served_gap_compares_the_unrounded_served_cosine(doubles, monkeypatch) -> None:
    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    calibration_cmd._cmd_calibration(_args("served-gap", k=5, tolerance=None))
    assert doubles.served == (SERVED_COSINE, True), (
        "served-gap compared the 4-place response score, not the cosine the gate compares"
    )


def test_c6_a_damaged_stored_coverage_row_reads_as_unavailable() -> None:
    from recall.near_miss import coverage_summary

    stored = {"unknown_entity": {"n": 3}, "in_vocabulary": {"rejected": 1, "n": 3}}
    try:
        report, line = coverage_summary(lambda _calibration_id: stored, CAL)
    except KeyError:
        pytest.fail("a damaged stored coverage row failed the status line that prints it")
    assert (report, line) == (None, UNAVAILABLE_MESSAGE)


_CHECKED_REPORT = {
    "unknown_entity": {"rejected": 3, "n": 3},
    "in_vocabulary": {"rejected": 1, "n": 3},
    "unknown_term_gate": "checked",
}


@pytest.mark.parametrize(
    ("variable", "value", "state"),
    [("RECALL_UNKNOWN_TERM_GATE", "0", "disabled"),
     ("RECALL_SOURCE_POLICY_FILE", "/policies/sources.toml", "unavailable")],
)
def test_c7_status_says_when_this_process_does_not_run_the_measured_gate(
    monkeypatch, variable, value, state
) -> None:
    from recall.near_miss import coverage_summary

    monkeypatch.delenv("RECALL_UNKNOWN_TERM_GATE", raising=False)
    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    _report, measured_here = coverage_summary(lambda _calibration_id: _CHECKED_REPORT, CAL)
    assert "in this process" not in measured_here, "a matching process was told it differs"
    monkeypatch.setenv(variable, value)
    _report, line = coverage_summary(lambda _calibration_id: _CHECKED_REPORT, CAL)
    assert f"unknown-term gate {state} in this process" in line, (
        "status showed the fit's checked gate as if this process ran it"
    )


def test_c8_served_gap_refuses_a_report_when_a_promotion_lands_mid_run(doubles, monkeypatch) -> None:
    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    reads = iter(["gen_double", "gen_promoted", "gen_promoted"])
    monkeypatch.setattr(calibration_cmd, "_active_generation", lambda _a: next(reads))
    with pytest.raises(SystemExit, match="moved off gen_double"):
        calibration_cmd._cmd_calibration(_args("served-gap", k=5, tolerance=None))
