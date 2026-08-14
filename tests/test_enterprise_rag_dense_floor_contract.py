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
    POPULATION,
    summarize,
    validate_dense_floor_artifact,
    write_dense_floor_artifact,
)

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
        by_type.setdefault(row["question_type"], []).append(row["best_dense_score"])

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
    payload["dense_floor_summary"]["population_weighted_lower_bound_below_0_50"] = 1.0
    with pytest.raises(ValueError, match="population_weighted_lower_bound_below_0_50"):
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
    payload["rows"] = [{**payload["rows"][0], "best_dense_score": bad}, *payload["rows"][1:]]
    with pytest.raises(ValueError, match="best_dense_score"):
        validate_dense_floor_artifact(payload)


def test_a_null_score_is_named_as_an_empty_index_not_a_hard_question() -> None:
    """Pinned on the MESSAGE, not just on the raise, because the raise is over-determined.

    Deleting the null branch still raises via the type check below it, so a test matching only
    "best_dense_score" cannot tell the two apart — mutation confirmed it survives. What the branch
    buys is the diagnosis. `query_dense(k=1)` filters on tenant alone with no per-question
    predicate, so an empty result is a property of the table: a null never means "hard question",
    it means the tenant is empty or misnamed and the run measured no index at all. An operator told
    "null score" goes looking at the question; one told "the tenant held no rows" checks `--table`.
    """
    payload = _payload()
    payload["rows"] = [{**payload["rows"][0], "best_dense_score": None}, *payload["rows"][1:]]
    with pytest.raises(ValueError, match="tenant held no rows"):
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
    assert raises, "retrieval_calibration must refuse a null best_dense_score where it is produced"
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


def test_the_summary_reports_a_lower_bound_not_an_estimate() -> None:
    """`best_dense_score` is the corpus-wide dense top-1, not the best among the returned hits.

    The trust floor is applied per RETURNED hit over the post-fusion, post-rerank top-k, and
    max(dense over the returned k) <= the corpus top-1. So a question counted safe here can still
    have no returned hit above the floor: the real demotion rate is at least this one. The key
    names say so, because a reader quoting "5.6%" as the expected rate would understate the risk.
    """
    summary = summarize(_payload()["rows"])
    assert "population_weighted_lower_bound_below_0_50" in summary
    assert "population_rate_lower_bound_below_0_50" in summary
    assert "LOWER BOUND" in _payload()["_provenance"]["note"]


def test_a_failing_payload_leaves_no_file_behind(tmp_path: Path) -> None:
    """Validate-then-write, not write-then-regret."""
    payload = _payload()
    payload["dense_floor_summary"]["sample_below_0_50"] = 999
    target = tmp_path / "nested" / "out.json"
    with pytest.raises(ValueError):
        write_dense_floor_artifact(target, payload)
    assert not target.exists()
