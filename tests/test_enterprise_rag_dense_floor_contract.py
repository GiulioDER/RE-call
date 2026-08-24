"""The dense-floor artifact's summary must be exactly what its own rows produce.

This suite exists because two successive versions of that artifact failed properties nobody was
checking.

First: every `median` in the hand-written `dense_floor_summary` was `statistics.median_high` rather
than the median, biased upward by up to 0.045, while the commit message quoted the TRUE median for
one category, a figure that appeared nowhere in the file. The disagreement inverted a published
ranking. Nothing caught it, because a summary beside a body is two statements of the same fact and
no test compared them.

Second: the module that fixed it described itself as a write-site guard in five places while every
caller lived in this file. A validator nothing calls is a comment. `test_the_runner_writes_through
_the_validator` is the one that would have caught that, and it is the reason it exists.

One test per rejection path, plus one proving the validator does not reject a correct artifact,
because a validator that refuses everything passes every rejection test it has.
"""
from __future__ import annotations

import ast
import json
import statistics
from pathlib import Path

import pytest

from benchmarks.enterprise_rag_contract import (
    require_provenance,
    SCORE_FIELD,
    POPULATION,
    summarize,
    validate_dense_floor_artifact,
    write_dense_floor_artifact,
)

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "results" / "enterprise_rag" / "dense_floor_strat100.retrieval.json"
#: A committed judge summary carries the benchmark's own per-type counts, so the population weights
#: are pinned against a file in the tree rather than against a constant restating itself.
JUDGE_SUMMARY = (
    ROOT
    / "benchmarks"
    / "artifacts"
    / "enterprise_rag"
    / "re_call_voyage_splade_gpt4o.no_correction.summary.json"
)


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_committed_artifact_passes_its_own_contract() -> None:
    """The non-over-rejection case. Without it, `raise ValueError` unconditionally would pass."""
    validate_dense_floor_artifact(_payload())


def test_the_runner_writes_through_the_validator() -> None:
    """The wiring, asserted against the runner's source rather than against a docstring.

    The previous version of the contract module claimed a write-site guard in five places while
    `benchmarks/enterprise_rag.py` still called `write_text` directly, so a drifted summary would
    have been written happily and only a commit-time test stood between it and the tree. Parsed,
    not grepped, so a mention inside a comment or a string cannot satisfy it.
    """
    tree = ast.parse((ROOT / "benchmarks" / "enterprise_rag.py").read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "write_dense_floor_artifact" in called, (
        "the calibration write must go through the validating writer, not write_text"
    )
    assert "summarize" in called, "the runner must derive dense_floor_summary, not leave it to hand"


def test_the_committed_medians_are_the_median_not_the_median_high() -> None:
    """The specific defect this file was written for, pinned by name.

    The inequality half is guarded: median and median_high round to the same 4 decimals whenever a
    category's 5th and 6th order statistics fall within 1e-4, and asserting inequality there would
    go red on a CORRECT artifact. Today's smallest gap is 0.0208, 200x the tolerance, but a re-run
    is not obliged to keep that.
    """
    payload = _payload()
    by_type: dict[str, list[float]] = {}
    for row in payload["rows"]:
        # The field the summary derives from. Reading `best_dense_score` here compared the
        # summary's medians against a DIFFERENT column and went red on a correct artifact.
        by_type.setdefault(row["question_type"], []).append(row[SCORE_FIELD])

    distinguishing = 0
    for cat, vals in by_type.items():
        got = payload["dense_floor_summary"]["by_question_type"][cat]["median"]
        assert got == pytest.approx(statistics.median(vals), abs=5e-5), cat
        if abs(statistics.median(vals) - statistics.median_high(vals)) > 1e-4:
            distinguishing += 1
            assert got != pytest.approx(statistics.median_high(vals), abs=5e-5), (
                f"{cat}: the summary is carrying median_high, the defect this test exists for"
            )
    assert distinguishing >= 5, (
        "too few categories where median and median_high are distinguishable at this tolerance; "
        "this test would be vacuous"
    )


def test_a_wrong_median_is_refused() -> None:
    payload = _payload()
    payload["dense_floor_summary"]["by_question_type"]["basic"]["median"] = 0.9999
    with pytest.raises(ValueError, match=r"by_question_type\[basic\]\.median"):
        validate_dense_floor_artifact(payload)


def test_a_wrong_count_is_refused() -> None:
    payload = _payload()
    payload["dense_floor_summary"]["sample_below_0_50"] = 1
    with pytest.raises(ValueError, match="sample_below_0_50"):
        validate_dense_floor_artifact(payload)


def test_a_wrong_population_weighting_is_refused() -> None:
    payload = _payload()
    payload["dense_floor_summary"]["population_weighted_estimate_below_0_50"] = 1.0
    with pytest.raises(ValueError, match="population_weighted_estimate_below_0_50"):
        validate_dense_floor_artifact(payload)


def test_a_missing_summary_is_refused() -> None:
    payload = _payload()
    del payload["dense_floor_summary"]
    with pytest.raises(ValueError, match="dense_floor_summary"):
        validate_dense_floor_artifact(payload)


def test_empty_rows_are_refused() -> None:
    payload = _payload()
    payload["rows"] = []
    with pytest.raises(ValueError, match="non-empty rows"):
        validate_dense_floor_artifact(payload)


def test_a_category_missing_from_by_question_type_is_refused() -> None:
    payload = _payload()
    del payload["dense_floor_summary"]["by_question_type"]["basic"]
    with pytest.raises(ValueError, match="by_question_type"):
        validate_dense_floor_artifact(payload)


def test_an_unknown_question_type_is_refused_rather_than_weighted_as_zero() -> None:
    """A type with no population weight must stop the run, not silently contribute nothing."""
    payload = _payload()
    payload["rows"] = [*payload["rows"], {**payload["rows"][0], "question_type": "invented"}]
    with pytest.raises(ValueError, match="invented"):
        validate_dense_floor_artifact(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.61", True])
def test_an_unusable_score_is_refused(bad: object) -> None:
    """NaN is the dangerous one, and the reason this is not merely tidiness.

    `abs(nan - x) > tol` is False, so a NaN made every comparison against its field vacuously true
    and an arbitrarily wrong summary passed. It also sorts as "not below the floor", so it could
    only ever understate demotion — the direction this artifact argues for.
    """
    payload = _payload()
    payload["rows"] = [
        {**payload["rows"][0], "max_returned_dense_score": bad}, *payload["rows"][1:]
    ]
    with pytest.raises(ValueError, match="max_returned_dense_score"):
        validate_dense_floor_artifact(payload)


def test_a_null_score_names_the_probe_not_an_empty_tenant() -> None:
    """Pinned on the MESSAGE, not just on the raise, because the raise is over-determined.

    Deleting the null branch still raises via the type check below it, so a test matching only the
    field name cannot tell the two apart. What the branch buys is the diagnosis, and the previous
    diagnosis was WRONG: it told the operator a null means the tenant held no rows. `query_dense`
    applies the tenant as a POST-filter over an HNSW index built across the whole table, so on a
    multi-tenant table the walk can come back empty for one query and not another. This repo
    measured 10 of 26 queries empty on one tenant while the rest returned hits. An operator sent to
    check `--table` when the fix is `hnsw.ef_search` loses the afternoon.
    """
    payload = _payload()
    payload["rows"] = [
        {**payload["rows"][0], "max_returned_dense_score": None}, *payload["rows"][1:]
    ]
    with pytest.raises(ValueError, match="probe depth"):
        validate_dense_floor_artifact(payload)


def test_the_runner_refuses_a_null_on_the_first_question_not_at_the_write() -> None:
    """Fail fast, asserted against the runner's source.

    The validator alone would catch an all-null run, but only at `write_dense_floor_artifact`,
    after an embedding call for every remaining question. Since a null is a property of the table
    rather than the question, question 1 already carries the whole signal, so the runner raises
    there and the operator pays one call instead of five hundred.
    """
    src = (ROOT / "benchmarks" / "enterprise_rag.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "retrieval_calibration"
    )
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert raises, "retrieval_calibration must refuse a null score where it is produced"
    assert "no dense hit for" in src


def test_a_hand_added_figure_beside_the_body_is_refused() -> None:
    """The defect class this module exists for, in its general form.

    An earlier version carried `floor_catches_no_unanswerable` as a hand-written block. The
    validator iterated the DERIVED keys, so that block was never checked and could be inverted to
    claim the floor catches every unanswerable question with nothing in the file to contradict it.
    It is derived now, and an unknown key is refused rather than ignored.
    """
    payload = _payload()
    payload["dense_floor_summary"]["floor_catches_everything"] = {"rate": 1.0}
    with pytest.raises(ValueError, match="unvalidated keys"):
        validate_dense_floor_artifact(payload)


def test_the_unanswerable_count_is_derived_not_asserted() -> None:
    payload = _payload()
    payload["dense_floor_summary"]["unanswerable_below_floor"]["below_0_50"] = 10
    with pytest.raises(ValueError, match="unanswerable_below_floor"):
        validate_dense_floor_artifact(payload)


def test_a_rounded_value_is_accepted_and_a_wrong_last_place_is_not() -> None:
    """Pins `_TOL` from both sides.

    Untested, it could be set to 0.3 or to 0.0 with the whole suite still green: every other
    rejection here uses a delta of 0.3 or more, which no plausible tolerance would accept.
    """
    payload = _payload()
    true_median = payload["dense_floor_summary"]["by_question_type"]["basic"]["median"]

    payload["dense_floor_summary"]["by_question_type"]["basic"]["median"] = true_median + 4e-5
    validate_dense_floor_artifact(payload)  # full-precision write, must be accepted

    payload["dense_floor_summary"]["by_question_type"]["basic"]["median"] = true_median + 1.5e-4
    with pytest.raises(ValueError, match=r"by_question_type\[basic\]\.median"):
        validate_dense_floor_artifact(payload)


def test_the_population_weights_match_the_benchmarks_own_counts() -> None:
    """All ten, against a committed file.

    Checking the sum and two entries let any same-sum swap through: `conflicting_info` 20 with
    `high_level` 10 keeps sum 500, `basic` 175 and `semantic` 125, and a wrong weight silently
    moves the headline number.
    """
    stats = json.loads(JUDGE_SUMMARY.read_text(encoding="utf-8"))["question_type_stats"]
    assert POPULATION == {name: block["count"] for name, block in stats.items()}
    assert sum(POPULATION.values()) == 500


def test_the_population_figure_is_named_an_estimate_and_the_census_is_separated() -> None:
    """⛔ This test replaces one that asserted the OPPOSITE and could not have caught being wrong.

    The old version asserted that two keys containing "lower_bound" existed and that the literal
    string "LOWER BOUND" appeared in the provenance note. It would have passed against a summary
    computing the opposite direction, because it never touched a number.

    The per-question measure IS exact now: the floor is applied per returned hit, and the summary
    scores the maximum over the returned hits. What is NOT a bound is the population figure, and
    the reason is sampling, not the instrument. Nine of the ten strata are ten-question samples of
    populations from 20 to 175, so the reweighted rate carries error in both directions. Only a
    stratum sampled to its full size is a census, and those are named separately so a reader has
    at least one figure with no sampling error in it.
    """
    summary = summarize(_payload()["rows"])

    assert "population_weighted_estimate_below_0_50" in summary
    assert "population_rate_estimate_below_0_50" in summary
    assert "lower_bound" not in json.dumps(summary), (
        "a stratified estimate must not be named a bound anywhere in the summary"
    )
    census = summary["census_strata"]
    assert all(
        summary["by_question_type"][name]["sampled"] == POPULATION[name] for name in census
    ), "a stratum is a census only when it was sampled to its full population"


def test_a_stratum_missing_from_the_sample_is_refused() -> None:
    """🔑 The numerator sums over strata PRESENT while the denominator is the whole population, so
    a truncated run publishes a silently deflated rate. Dropping two strata from the committed
    artifact took its headline from 0.056 to 0.022 with every other check still green, and
    `--limit-questions 10` produces exactly that shape.
    """
    payload = _payload()
    dropped = {"miscellaneous", "high_level"}
    payload["rows"] = [r for r in payload["rows"] if r["question_type"] not in dropped]

    with pytest.raises(ValueError, match="strata"):
        summarize(payload["rows"])


def test_a_failing_payload_leaves_no_file_behind(tmp_path: Path) -> None:
    """Validate-then-write, not write-then-regret."""
    payload = _payload()
    payload["dense_floor_summary"]["sample_below_0_50"] = 999
    target = tmp_path / "nested" / "out.json"
    with pytest.raises(ValueError):
        write_dense_floor_artifact(target, payload)
    assert not target.exists()


class TestGuardsTheAuditFoundUnpinned:
    """🔑 Mutation-driven. Two guards added with this change were untested: deleting the invariant
    check, and deleting `require_provenance` from the write site, both left the suite green. A
    guard nothing pins is a guard that can be removed by accident, which is this project's
    recorded recurring failure.
    """

    def test_a_returned_hit_above_the_corpus_top_1_is_refused(self) -> None:
        """The two are dense cosines of one query over one table, so this cannot happen unless the
        legs used different query vectors or different hnsw.ef_search. Either way nothing derived
        from the run can be interpreted."""
        payload = _payload()
        row = dict(payload["rows"][0])
        row["max_returned_dense_score"] = row["best_dense_score"] + 0.01
        payload["rows"] = [row, *payload["rows"][1:]]

        with pytest.raises(ValueError, match="exceeds"):
            validate_dense_floor_artifact(payload)

    def test_the_row_tolerance_is_pinned_from_both_sides(self) -> None:
        """Rows are full precision, so the identity needs float noise and nothing more. Reusing
        the summary's 5e-5 rounding tolerance let a genuine 4e-5 violation through the contract
        while the runner, which compares exactly, would have refused the same run."""
        payload = _payload()
        base = payload["rows"][0]["best_dense_score"]

        ok = dict(payload["rows"][0], max_returned_dense_score=base + 1e-12)
        validate_dense_floor_artifact({**payload, "rows": [ok, *payload["rows"][1:]]})

        bad = dict(payload["rows"][0], max_returned_dense_score=base + 4e-5)
        with pytest.raises(ValueError, match="exceeds"):
            validate_dense_floor_artifact({**payload, "rows": [bad, *payload["rows"][1:]]})

    def test_the_write_site_refuses_a_payload_with_no_provenance(self, tmp_path: Path) -> None:
        payload = _payload()
        del payload["_provenance"]

        with pytest.raises(ValueError, match="_provenance"):
            write_dense_floor_artifact(tmp_path / "out.json", payload)
        assert not (tmp_path / "out.json").exists(), "a refused payload must leave no file"

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("generation", "post-81/84", "generation"),
            ("status", "live", "status"),
            ("note", "   ", "note"),
        ],
    )
    def test_a_provenance_value_the_repo_suite_rejects_is_refused_at_the_write(
        self, tmp_path: Path, field: str, value: str, match: str
    ) -> None:
        """Truthiness was not enough. `tests/test_results_artifact_provenance.py` requires
        MEMBERSHIP for two of these, so a free-form `--calibrate-generation` typo wrote a file that
        passed this guard and failed CI on the commit, which is the failure it exists to prevent
        rather than relocate."""
        payload = _payload()
        payload["_provenance"] = {**payload["_provenance"], field: value}

        with pytest.raises(ValueError, match=match):
            write_dense_floor_artifact(tmp_path / "out.json", payload)
        assert not (tmp_path / "out.json").exists()

    def test_a_live_artifact_naming_a_successor_is_refused(self) -> None:
        payload = _payload()
        payload["_provenance"] = {**payload["_provenance"], "superseded_by": "something.json"}

        with pytest.raises(ValueError, match="successor"):
            require_provenance(payload)

    def test_the_summary_must_name_the_depth_and_arm_it_was_scored_under(self) -> None:
        """The scored quantity is the max over the RETURNED hits, so it depends on how many came
        back and under which arm. Neither is recoverable from the scores, so a summary without
        them cannot be compared against any other run."""
        payload = _payload()
        summary = dict(payload["dense_floor_summary"])
        del summary["scored_arm"]

        with pytest.raises(ValueError, match="scored_arm"):
            validate_dense_floor_artifact({**payload, "dense_floor_summary": summary})
