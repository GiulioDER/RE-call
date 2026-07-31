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
that cannot reject is not a guard.

Every control asserts `match=` against a string unique to the rule it names. That is not
decoration: a bug audit found three controls here passing because an EARLIER rule fired, so they
were green while the rule in their own name was never executed. `match=` is what makes "this
control tests that rule" a checked claim rather than a naming convention.

Verified by rule ablation, not by assertion: every `raise` in `validate()` and `_require_shape`
was replaced by `pass` in turn and the module re-run, expecting a red test. **33 guards ablated,
27 pinned.** The enumeration below is the measured result, and it is written out because an
inventory that quietly omits a check is how a stale label starts — the first version of this
docstring named three exceptions when there were eleven.

The six that no test can turn red, each because a different rule reaches the corruption first:

  1. the generic `malformed fixture:` wrapper — every concrete malformation now has a specific
     `_require_shape` message, so the wrapper only catches what nothing else names
  2. S3, the empty-retrieval count
  3. S4, the confident-wrong count — 2 and 3 both need the flags changed, which trips S6, S6b or
     S8 before a count can drift
  4. S7's gold `days`-vs-its-own-endpoints pre-check — the rubric branch fires first
  5. S5's operand `days`-vs-its-own-endpoints pre-check — the `_stated_days` branch fires first
  6. S5's `matches gold` check — reaching it needs a two-field corruption

These are defence in depth, not guards this module proves can fire. That distinction is the point
of running the ablation at all.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.build_9l_temporal_fixture import (
    CLASSIFICATION,
    DATE_SELECTION_MECHANISMS,
    DEFAULT_RUNS,
    FIXTURE_PATH,
    MEM0_RUN_NAME,
    MEM0_RUN_SHA256,
    RECALL_RUN_NAME,
    RECALL_RUN_SHA256,
    BuildError,
    _stated_days,
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
    published were right for the dates we chose, so the defect is date selection, not reasoning.

    Uses `_stated_days` rather than a substring test. `str(days) in our_answer` was unfalsifiable
    for `1M_1_q18`, whose count is 0 and whose answer contains "2024" — the assertion held no
    matter what the transcription said.
    """
    confident = [c for c in fixture["cases"] if c["our_interval"] is not None]
    assert len(confident) == 5
    for case in confident:
        assert _stated_days(case["our_answer"]) == case["our_interval"]["days"]
        assert case["our_interval"]["days"] != case["gold_interval"]["days"]


def test_mechanisms_use_the_declared_vocabulary(fixture: dict) -> None:
    allowed = set(DATE_SELECTION_MECHANISMS) | {"abstained_with_retrieval", "retrieval_empty"}
    assert {c["mechanism"] for c in fixture["cases"]} <= allowed


def test_every_case_is_actually_a_full_point_loss(fixture: dict) -> None:
    """The property that defines this set. It was checked only in `build` until a bug audit
    demonstrated a fixture claiming RE-call WON all seven, accepted with the suite green."""
    gate = fixture["provenance"]["badly_lost_delta"]
    for case in fixture["cases"]:
        assert case["recall_score"] - case["mem0_score"] <= gate


def test_provenance_names_the_runs_by_hash(fixture: dict) -> None:
    provenance = fixture["provenance"]
    assert provenance["recall_run_sha256"] == RECALL_RUN_SHA256
    assert provenance["mem0_run_sha256"] == MEM0_RUN_SHA256
    assert len(provenance["recall_run_sha256"]) == 64


# --------------------------------------------------------------------------------------------
# Negative controls. Each must be REJECTED.
# --------------------------------------------------------------------------------------------


def _corrupt(fixture: dict, question_id_prefix: str) -> tuple[dict, dict]:
    """A deep copy plus the case named by id.

    Addressing by id rather than by `cases[0]`: index 0 binds to whatever sorts first, so a
    reclassification would silently retarget the control to a different case with nothing failing.

    The uniqueness assertion matters for the same reason: `cases` is sorted, so a prefix that
    matches two ids picks the first one — reintroducing exactly the bug this helper replaced.
    `1M_1` matches `1M_12_q18` before `1M_1_q18`, and `1M_13` matches q18 before q19.
    """
    corrupt = copy.deepcopy(fixture)
    hits = [c for c in corrupt["cases"] if c["question_id"].startswith(question_id_prefix)]
    assert len(hits) == 1, f"{question_id_prefix!r} matches {[h['question_id'] for h in hits]}"
    return corrupt, hits[0]


def test_validate_rejects_a_case_that_was_not_badly_lost(fixture: dict) -> None:
    """A fixture asserting RE-call won the questions it publishes as losses. Accepted before S2b,
    with all CI-visible tests green, because the criterion lived only in `build`."""
    corrupt, case = _corrupt(fixture, "1M_8")
    case["recall_score"], case["mem0_score"] = 1.0, 0.0
    with pytest.raises(BuildError, match="S2b"):
        validate(corrupt)


def test_validate_rejects_a_mechanism_that_contradicts_the_flags(fixture: dict) -> None:
    """Relabelling the empty-retrieval case as a date-selection failure moves a published count.
    `mechanism_tally` is recomputed from the labels, so it can never disagree with them — which
    is why the label has to be checked against the flags instead."""
    corrupt, case = _corrupt(fixture, "1M_14")
    case["mechanism"] = "instance_disambiguation"
    corrupt["summary"]["mechanism_tally"] = {"instance_disambiguation": 3, "supersession": 1,
                                             "event_vs_utterance_time": 1,
                                             "abstained_with_retrieval": 1,
                                             "role_value_vs_assertion_time": 1}
    corrupt["summary"]["date_selection_failures"] = 6
    with pytest.raises(BuildError, match="S6b"):
        validate(corrupt)


def test_validate_rejects_invented_operands_on_an_abstention(fixture: dict) -> None:
    """The corruption that a mutation pass caught this module missing: giving a question we
    abstained on a pair of dates makes the fixture report a date-selection failure that never
    happened."""
    corrupt, case = _corrupt(fixture, "1M_13_q19")
    case["our_interval"] = {"start": "2024-02-04", "end": "2024-04-09", "days": 65}
    with pytest.raises(BuildError, match="S6:"):
        validate(corrupt)


def test_validate_rejects_operands_that_do_not_match_our_answer(fixture: dict) -> None:
    """Self-consistent operands whose span disagrees with the count the answer leads with. Bumping
    `days` alone tested the endpoint-consistency branch instead, leaving `_stated_days` — the whole
    point of S5 — unpinned."""
    corrupt, case = _corrupt(fixture, "1M_0")
    case["our_interval"] = {"start": "2024-04-01", "end": "2024-04-16", "days": 15}
    with pytest.raises(BuildError, match="answered 14 days but"):
        validate(corrupt)


def test_validate_rejects_operands_absent_from_our_own_answer(fixture: dict) -> None:
    """A delta-preserving shift: the span is still 14 days, so every length check passes, but the
    dates appear nowhere in the answer they were supposedly read from."""
    corrupt, case = _corrupt(fixture, "1M_0")
    case["our_interval"] = {"start": "2020-01-01", "end": "2020-01-15", "days": 14}
    with pytest.raises(BuildError, match="not named anywhere in our own answer"):
        validate(corrupt)


def test_validate_rejects_an_answer_with_no_leading_day_count(fixture: dict) -> None:
    """`_stated_days` returning None is what makes S5 unable to tie operands to the answer."""
    corrupt, case = _corrupt(fixture, "1M_0")
    case["our_answer"] = "It was about two weeks, from April 1, 2024 to April 15, 2024."
    with pytest.raises(BuildError, match="does not lead with a day count"):
        validate(corrupt)


def test_validate_rejects_a_gold_interval_the_rubric_does_not_state(fixture: dict) -> None:
    corrupt, case = _corrupt(fixture, "1M_0")
    case["gold_interval"] = {"start": "2024-03-25", "end": "2024-04-02", "days": 8}
    with pytest.raises(BuildError, match="does not state that as its answer"):
        validate(corrupt)


def test_validate_rejects_a_gold_span_that_collides_with_a_date_in_its_rubric(fixture: dict) -> None:
    """The vacuity the old bare `\\b{days}\\b` allowed. `1M_12`'s rubric reads "from March 25, 2024
    till April 10, 2024", so a wrong 10-day gold matched the "10" of April 10 and shipped."""
    corrupt, case = _corrupt(fixture, "1M_12")
    case["gold_interval"] = {"start": "2024-03-31", "end": "2024-04-10", "days": 10}
    with pytest.raises(BuildError, match="does not state that as its answer"):
        validate(corrupt)


def test_validate_rejects_gold_endpoints_the_rubric_contradicts(fixture: dict) -> None:
    """Both ends shifted, length preserved: only comparing the endpoints catches it."""
    corrupt, case = _corrupt(fixture, "1M_0")
    case["gold_interval"] = {"start": "2020-01-01", "end": "2020-01-08", "days": 7}
    with pytest.raises(BuildError, match="but the rubric says"):
        validate(corrupt)


def test_validate_rejects_a_whole_year_shift_of_our_operands(fixture: dict) -> None:
    """The hole the prose checks cannot see, and the reason S8 exists.

    `_date_is_mentioned`'s year group is optional and trailing, so "April 1, 2020" satisfies a
    pattern written for "April 1, 2024". The span is still 14 days and the months and days still
    appear in the answer, so every length and prose check passes — this shipped green until the
    endpoints were pinned to `CLASSIFICATION` in code.
    """
    corrupt, case = _corrupt(fixture, "1M_0")
    case["our_interval"] = {"start": "2020-04-01", "end": "2020-04-15", "days": 14}
    with pytest.raises(BuildError, match="S8"):
        validate(corrupt)


def test_validate_rejects_a_year_shift_of_gold_whose_rubric_omits_the_year(fixture: dict) -> None:
    """`1M_14_q18`'s rubric says "from March 15 till April 6" and names no year, so the rubric
    check cannot constrain one. Two of the seven are like this."""
    corrupt, case = _corrupt(fixture, "1M_14")
    case["gold_interval"] = {"start": "2020-03-15", "end": "2020-04-06", "days": 22}
    with pytest.raises(BuildError, match="S8"):
        validate(corrupt)


def test_validate_rejects_a_fixture_that_moved_its_own_gate(fixture: dict) -> None:
    """S2b read its threshold from the file it was checking. At 0.5 a TIE counts as a loss."""
    corrupt = copy.deepcopy(fixture)
    corrupt["provenance"]["badly_lost_delta"] = 0.5
    with pytest.raises(BuildError, match="does not get to move its own gate"):
        validate(corrupt)


def test_validate_rejects_a_summary_that_disagrees_with_its_cases(fixture: dict) -> None:
    corrupt = copy.deepcopy(fixture)
    corrupt["summary"]["supersession_reachable"] = 3
    with pytest.raises(BuildError, match="supersession_reachable"):
        validate(corrupt)


def test_validate_rejects_an_unknown_mechanism_label(fixture: dict) -> None:
    corrupt, case = _corrupt(fixture, "1M_0")
    case["mechanism"] = "recency"
    with pytest.raises(BuildError, match="S6b"):
        validate(corrupt)


def test_validate_rejects_an_id_outside_the_classification_table(fixture: dict) -> None:
    corrupt, case = _corrupt(fixture, "1M_8")
    case["question_id"] = "1M_99_q42_temporal_reasoning"
    with pytest.raises(BuildError, match="classification table"):
        validate(corrupt)


def test_validate_rejects_a_duplicated_case(fixture: dict) -> None:
    """The summary is adjusted so only the duplicate-id rule can be what fires — otherwise this
    passed on S4's confident-wrong count instead."""
    corrupt, case = _corrupt(fixture, "1M_0")
    corrupt["cases"].append(copy.deepcopy(case))
    corrupt["summary"]["badly_lost"] = 8
    corrupt["summary"]["confident_wrong"] = 6
    with pytest.raises(BuildError, match="duplicate question_id"):
        validate(corrupt)


def test_validate_rejects_a_relabelled_mechanism(fixture: dict) -> None:
    """`mechanism` is published through a claim-gate marker in §9l's table, so an unpinned
    relabel moves a figure in a released document. S6b only partitions empty / abstained /
    any-date-selection, which leaves the four date-selection labels interchangeable."""
    corrupt, case = _corrupt(fixture, "1M_8")
    case["mechanism"] = "event_vs_utterance_time"
    corrupt["summary"]["mechanism_tally"] = {"instance_disambiguation": 1, "supersession": 1,
                                             "event_vs_utterance_time": 2,
                                             "abstained_with_retrieval": 1,
                                             "retrieval_empty": 1,
                                             "role_value_vs_assertion_time": 1}
    with pytest.raises(BuildError, match=r"S8: .*\.mechanism does not match"):
        validate(corrupt)


def test_validate_rejects_rewritten_reasoning(fixture: dict) -> None:
    """The reasoning is the evidence a reader checks the label against."""
    corrupt, case = _corrupt(fixture, "1M_0")
    case["reasoning"] = "Because it seemed likely."
    with pytest.raises(BuildError, match=r"S8: .*\.reasoning does not match"):
        validate(corrupt)


def test_validate_rejects_a_rubric_with_no_span(fixture: dict) -> None:
    """The day count still reads "7 days", so S7's first branch passes and the span branch is
    the one under test."""
    corrupt, case = _corrupt(fixture, "1M_0")
    case["gold_rubric"] = ["LLM response should state: 7 days"]
    with pytest.raises(BuildError, match="states no 'from X till Y' span"):
        validate(corrupt)


def test_validate_rejects_a_classification_entry_with_an_impossible_date(fixture, monkeypatch) -> None:
    """Corrupts the TABLE, not the artifact, and the message must say so — a typo in this
    module's own data reported as a FAIL of the committed JSON blames the file that is correct."""
    broken = copy.deepcopy(CLASSIFICATION)
    broken["1M_0_q18_temporal_reasoning"]["gold"] = ("2024-03-25", "2024-04-31")
    monkeypatch.setattr(
        "benchmarks.build_9l_temporal_fixture.CLASSIFICATION", broken, raising=True
    )
    with pytest.raises(BuildError, match="CLASSIFICATION.*is not a valid date pair"):
        validate(copy.deepcopy(fixture))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda f, c: f["cases"].append("not an object"), "case is str"),
        (lambda f, c: c.__setitem__("gold_interval", {"start": "2024-03-25"}),
         "must hold start, end and days"),
        (lambda f, c: c["gold_interval"].__setitem__("days", 7.0), "days is not an int"),
        (lambda f, c: c.__setitem__("our_answer", None), "our_answer is NoneType"),
        (lambda f, c: c.__setitem__("gold_rubric", "should state: 7 days"), "gold_rubric is str"),
        (lambda f, c: c.__setitem__("gold_rubric", [1, 2]), "non-string line"),
        (lambda f, c: c.__setitem__("gold_interval", None), "gold_interval is null"),
        (lambda f, c: c["gold_interval"].__setitem__("start", "March 25, 2024"), "not an ISO date"),
        (lambda f, c: c.pop("our_interval"), "missing 'our_interval'"),
        (lambda f, c: c.pop("conversation_idx"), "missing 'conversation_idx'"),
        (lambda f, c: f.__setitem__("summary", []), "summary is list"),
        (lambda f, c: f["provenance"].__setitem__("badly_lost_delta", "-1.0"),
         "badly_lost_delta is str"),
    ],
    ids=["case-not-an-object", "interval-missing-a-key", "days-not-an-int", "null-answer",
         "rubric-as-string", "rubric-non-string-line", "null-gold-interval", "prose-date",
         "missing-interval", "missing-conv-idx", "summary-not-object", "gate-not-a-number"],
)
def test_validate_reports_malformed_content_as_BuildError(fixture, mutate, expected) -> None:
    """These raised TypeError / ValueError / KeyError, which `main` does not catch, so a slip in
    the hand-typed table produced a traceback instead of the FAIL line. The rubric-as-a-string
    case was worse than a crash: `" ".join` spaced out every character and the old regex still
    matched, so it was accepted."""
    corrupt, case = _corrupt(fixture, "1M_0")
    mutate(corrupt, case)
    with pytest.raises(BuildError, match=expected):
        validate(corrupt)


# --------------------------------------------------------------------------------------------
# Re-derivation. Needs the gitignored runs; skips loudly when they are absent.
# --------------------------------------------------------------------------------------------


def test_validate_only_mode_needs_no_runs_and_passes() -> None:
    """The CLI path CI can actually reach. `--check` cannot: it re-derives from the gitignored
    runs and exits 2 when they are absent, which is every CI run and every fresh clone. The
    docstring advertised `--check` as the no-runs mode; it never was."""
    assert main(["--validate-only"]) == 0


def test_validate_only_reports_failure_rather_than_passing_quietly(tmp_path) -> None:
    """A guard whose failure path is untested is a guard with an untested failure path."""
    broken = tmp_path / "broken.json"
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["summary"]["supersession_reachable"] = 99
    broken.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--validate-only", "--out", str(broken)]) == 1


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
