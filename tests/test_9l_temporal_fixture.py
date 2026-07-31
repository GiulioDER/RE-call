"""The committed §9l fixture must stay true to the runs it was derived from.

`results/beam_9l_temporal.json` is what any future temporal selector will be scored against
(issue #167 step 1), and `results/FINDINGS.md` §9l cites it through `benchmarks/claim_gate.py`
markers. Both uses assume the file is internally coherent, so that assumption is checked here
rather than trusted.

The two BEAM runs behind it live under `benchmarks/results/`, which is gitignored, so CI cannot
re-derive the fixture. That is exactly why `validate()` exists separately from `build()`: every
invariant that does not need the runs is checked on the committed file, on every run of the
suite. The one test that DOES need the runs skips when they are absent, and says so.

`test_validate_rejects_*` are the negative controls. Without them this module would assert that a
correct file passes, which a validator that returns unconditionally also does — and a validator
that cannot reject is not a guard. Each control corrupts one field of an in-memory copy and
requires a rejection; all of them were confirmed to fail against a `validate()` that had not yet
learned the corresponding rule.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.build_9l_temporal_fixture import (
    DATE_SELECTION_MECHANISMS,
    DEFAULT_RUNS,
    FIXTURE_PATH,
    MEM0_RUN_NAME,
    MEM0_RUN_SHA256,
    RECALL_RUN_NAME,
    RECALL_RUN_SHA256,
    BuildError,
    build,
    load_rows,
    main,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_committed_fixture_is_internally_coherent(fixture: dict) -> None:
    validate(fixture)


def test_it_carries_the_seven_cases_9l_describes(fixture: dict) -> None:
    assert len(fixture["cases"]) == 7
    assert fixture["summary"]["badly_lost"] == 7


def test_the_published_headline_numbers_round_as_9l_prints_them(fixture: dict) -> None:
    """§9l prints `0.408 vs Mem0's 0.567`. The claim gate rounds to published precision, so a
    fixture holding a mean that no longer rounds that way would break the document silently."""
    summary = fixture["summary"]
    assert f"{summary['recall_mean']:.3f}" == "0.408"
    assert f"{summary['mem0_mean']:.3f}" == "0.567"
    assert summary["paired_n"] == 30


def test_supersession_reaches_exactly_one_of_the_five(fixture: dict) -> None:
    """The load-bearing count for scoping. `check_p1_supersession_density.py` gates a supersession
    selector, and this says how many of §9l's own failures such a selector could touch even if it
    worked perfectly. If a reclassification ever moves it, that should be a visible change."""
    summary = fixture["summary"]
    assert summary["supersession_reachable"] == 1
    assert summary["confident_wrong"] == 5


def test_every_confident_wrong_case_did_correct_arithmetic(fixture: dict) -> None:
    """The finding that makes a retrieval-side fix the right shape at all: the day counts we
    published were right for the dates we chose, so the defect is date selection, not reasoning."""
    confident = [c for c in fixture["cases"] if c["our_interval"] is not None]
    assert len(confident) == 5
    for case in confident:
        assert str(case["our_interval"]["days"]) in case["our_answer"]
        assert case["our_interval"]["days"] != case["gold_interval"]["days"]


def test_mechanisms_use_the_declared_vocabulary(fixture: dict) -> None:
    allowed = set(DATE_SELECTION_MECHANISMS) | {"abstained_with_retrieval", "retrieval_empty"}
    assert {c["mechanism"] for c in fixture["cases"]} <= allowed


def test_provenance_names_the_runs_by_hash(fixture: dict) -> None:
    provenance = fixture["provenance"]
    assert provenance["recall_run_sha256"] == RECALL_RUN_SHA256
    assert provenance["mem0_run_sha256"] == MEM0_RUN_SHA256
    assert len(provenance["recall_run_sha256"]) == 64


# --------------------------------------------------------------------------------------------
# Negative controls. Each must be REJECTED.
# --------------------------------------------------------------------------------------------


def test_validate_rejects_invented_operands_on_an_abstention(fixture: dict) -> None:
    """The corruption that a mutation pass caught this module missing: giving a question we
    abstained on a pair of dates makes the fixture report a date-selection failure that never
    happened."""
    corrupt = copy.deepcopy(fixture)
    case = next(c for c in corrupt["cases"] if c["abstained"] and not c["retrieval_empty"])
    case["our_interval"] = {"start": "2024-02-04", "end": "2024-04-09", "days": 65}
    with pytest.raises(BuildError, match="S6"):
        validate(corrupt)


def test_validate_rejects_operands_that_do_not_match_our_answer(fixture: dict) -> None:
    corrupt = copy.deepcopy(fixture)
    case = next(c for c in corrupt["cases"] if c["our_interval"] is not None)
    case["our_interval"]["days"] += 1
    with pytest.raises(BuildError, match="S5"):
        validate(corrupt)


def test_validate_rejects_a_gold_interval_the_rubric_does_not_state(fixture: dict) -> None:
    corrupt = copy.deepcopy(fixture)
    corrupt["cases"][0]["gold_interval"] = {"start": "2024-03-25", "end": "2024-04-02", "days": 8}
    with pytest.raises(BuildError, match="S7"):
        validate(corrupt)


def test_validate_rejects_a_summary_that_disagrees_with_its_cases(fixture: dict) -> None:
    corrupt = copy.deepcopy(fixture)
    corrupt["summary"]["supersession_reachable"] = 3
    with pytest.raises(BuildError, match="supersession_reachable"):
        validate(corrupt)


def test_validate_rejects_an_unknown_mechanism_label(fixture: dict) -> None:
    corrupt = copy.deepcopy(fixture)
    corrupt["cases"][0]["mechanism"] = "recency"
    with pytest.raises(BuildError):
        validate(corrupt)


def test_validate_rejects_a_duplicated_case(fixture: dict) -> None:
    corrupt = copy.deepcopy(fixture)
    corrupt["cases"].append(copy.deepcopy(corrupt["cases"][0]))
    with pytest.raises(BuildError):
        validate(corrupt)


# --------------------------------------------------------------------------------------------
# Re-derivation. Needs the gitignored runs; skips loudly when they are absent.
# --------------------------------------------------------------------------------------------


def _runs_present() -> bool:
    return (DEFAULT_RUNS / RECALL_RUN_NAME).is_file() and (DEFAULT_RUNS / MEM0_RUN_NAME).is_file()


@pytest.mark.skipif(
    not _runs_present(),
    reason=f"BEAM runs absent from {DEFAULT_RUNS} (gitignored); fixture checked structurally only",
)
def test_fixture_matches_a_fresh_rebuild_from_the_pinned_runs() -> None:
    assert main(["--check"]) == 0


@pytest.mark.skipif(not _runs_present(), reason="BEAM runs absent (gitignored)")
def test_a_run_whose_hash_moved_is_refused() -> None:
    """A path says where a file was; a hash says which file it was. Rebuilding from a different
    run would silently re-target the cases §9l was drawn from."""
    with pytest.raises(BuildError, match="sha256"):
        load_rows(DEFAULT_RUNS / RECALL_RUN_NAME, "0" * 64, "recall")


@pytest.mark.skipif(not _runs_present(), reason="BEAM runs absent (gitignored)")
def test_rebuild_reproduces_the_published_section() -> None:
    rebuilt = build(
        load_rows(DEFAULT_RUNS / RECALL_RUN_NAME, RECALL_RUN_SHA256, "recall"),
        load_rows(DEFAULT_RUNS / MEM0_RUN_NAME, MEM0_RUN_SHA256, "mem0"),
    )
    committed = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert rebuilt["cases"] == committed["cases"]
    assert rebuilt["summary"] == committed["summary"]
