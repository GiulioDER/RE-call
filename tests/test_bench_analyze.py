"""Offline tests for the benchmark's statistical layer.

Everything here is hand-built: no results file is read, no database is touched, no model is
called. The numbers this module produces are going to be published against a competitor, so the
p-values are pinned against values computed by hand rather than against whatever the code
returned the first time it ran.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.analyze import (
    EXACT_MAX_DISCORDANT,
    curve_points,
    discrimination,
    exact_binomial_two_sided,
    main,
    paired_abstention_mcnemar,
    paired_mcnemar,
    plot_curve,
)

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def _outcome(
    question_id: str,
    *,
    adversarial: bool = False,
    correct: bool | None = None,
    abstained: bool = False,
    category: str = "cat1",
) -> dict[str, Any]:
    """One entry of a results artifact's ``outcomes`` list, with the fields analysis reads."""
    return {
        "question_id": question_id,
        "category": category,
        "is_adversarial": adversarial,
        "context": "",
        "answer": "",
        "abstained": abstained,
        "correct": correct,
    }


def _accuracy_arms(
    both_correct: int, both_wrong: int, a_only: int, b_only: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Two arms over one question set with exactly the requested 2x2 accuracy table."""
    a_outcomes: list[dict[str, Any]] = []
    b_outcomes: list[dict[str, Any]] = []
    cells = (
        ("bc", both_correct, True, True),
        ("bw", both_wrong, False, False),
        ("ao", a_only, True, False),
        ("bo", b_only, False, True),
    )
    for tag, count, a_correct, b_correct in cells:
        for i in range(count):
            question_id = f"{tag}-{i}"
            a_outcomes.append(_outcome(question_id, correct=a_correct))
            b_outcomes.append(_outcome(question_id, correct=b_correct))
    return a_outcomes, b_outcomes


# -- the exact binomial ------------------------------------------------------------------------

def test_exact_binomial_matches_hand_computed_value() -> None:
    # b=8, c=2: two-sided p = 2 * P(X <= 2 | n=10, p=0.5) = 2 * (1 + 10 + 45) / 1024 = 0.109375
    assert exact_binomial_two_sided(8, 10) == pytest.approx(0.109375, abs=1e-12)


def test_exact_binomial_is_symmetric_in_the_two_discordant_cells() -> None:
    assert exact_binomial_two_sided(2, 10) == exact_binomial_two_sided(8, 10)


def test_exact_binomial_all_discordant_one_way() -> None:
    # b=10, c=0: 2 * (1/1024)
    assert exact_binomial_two_sided(10, 10) == pytest.approx(2 / 1024, abs=1e-12)


def test_exact_binomial_clamps_the_tied_case_to_one() -> None:
    assert exact_binomial_two_sided(5, 10) == 1.0


def test_exact_binomial_with_no_discordant_pairs_is_one() -> None:
    assert exact_binomial_two_sided(0, 0) == 1.0


# -- test selection ----------------------------------------------------------------------------

def test_below_threshold_uses_the_exact_binomial() -> None:
    a_outcomes, b_outcomes = _accuracy_arms(both_correct=5, both_wrong=5, a_only=8, b_only=2)
    result = paired_mcnemar(a_outcomes, b_outcomes)

    assert result["test"] == "exact_binomial"
    assert result["a_only"] + result["b_only"] == 10 < EXACT_MAX_DISCORDANT
    assert result["p_value"] == pytest.approx(0.109375, abs=1e-12)
    assert result["statistic"] == 8.0


def test_at_the_threshold_uses_chi_square_with_continuity_correction() -> None:
    a_outcomes, b_outcomes = _accuracy_arms(both_correct=5, both_wrong=5, a_only=20, b_only=5)
    result = paired_mcnemar(a_outcomes, b_outcomes)

    assert result["test"] == "chi2_continuity_corrected"
    assert result["a_only"] + result["b_only"] == EXACT_MAX_DISCORDANT
    # (|20-5| - 1)^2 / 25 = 196 / 25 = 7.84.
    # A chi-square on 1 df is a squared standard normal, so P(X > 7.84) = 2 * (1 - Phi(2.8)),
    # and a normal table gives 1 - Phi(2.8) = 0.0025551 -> p = 0.0051103.
    assert result["statistic"] == pytest.approx(7.84, abs=1e-12)
    assert result["p_value"] == pytest.approx(0.0051103, abs=1e-6)


def test_one_pair_below_the_threshold_still_switches_test() -> None:
    below = paired_mcnemar(*_accuracy_arms(0, 0, 20, 4))
    at = paired_mcnemar(*_accuracy_arms(0, 0, 20, 5))
    assert below["test"] == "exact_binomial"
    assert at["test"] == "chi2_continuity_corrected"


def test_full_table_is_reported_and_sums_to_n_paired() -> None:
    result = paired_mcnemar(*_accuracy_arms(both_correct=7, both_wrong=3, a_only=4, b_only=6))
    assert result["both_correct"] == 7
    assert result["both_wrong"] == 3
    assert result["a_only"] == 4
    assert result["b_only"] == 6
    assert result["n_paired"] == 20


def test_identical_arms_are_not_significant() -> None:
    result = paired_mcnemar(*_accuracy_arms(both_correct=10, both_wrong=10, a_only=0, b_only=0))
    assert result["a_only"] == result["b_only"] == 0
    assert result["p_value"] == 1.0
    assert result["n_paired"] == 20


# -- pairing hygiene ---------------------------------------------------------------------------

def test_mismatched_question_id_sets_raise() -> None:
    a_outcomes = [_outcome("q1", correct=True), _outcome("q2", correct=True)]
    b_outcomes = [_outcome("q1", correct=True), _outcome("q3", correct=True)]

    with pytest.raises(ValueError, match="same questions"):
        paired_mcnemar(a_outcomes, b_outcomes)


def test_extra_question_in_one_arm_raises_rather_than_intersecting() -> None:
    a_outcomes = [_outcome("q1", correct=True), _outcome("q2", correct=False)]
    b_outcomes = [_outcome("q1", correct=True)]

    with pytest.raises(ValueError, match="same questions"):
        paired_mcnemar(a_outcomes, b_outcomes)


def test_duplicate_question_id_raises() -> None:
    a_outcomes = [_outcome("q1", correct=True), _outcome("q1", correct=False)]
    b_outcomes = [_outcome("q1", correct=True), _outcome("q1", correct=False)]

    with pytest.raises(ValueError, match="duplicate question_id"):
        paired_mcnemar(a_outcomes, b_outcomes)


def test_disagreeing_adversarial_label_raises() -> None:
    a_outcomes = [_outcome("q1", correct=True)]
    b_outcomes = [_outcome("q1", adversarial=True, correct=None, abstained=True)]

    with pytest.raises(ValueError, match="adversarial in one arm"):
        paired_mcnemar(a_outcomes, b_outcomes)


# -- subset selection --------------------------------------------------------------------------

def test_accuracy_pairing_excludes_adversarial_questions() -> None:
    a_outcomes = [
        _outcome("ans-1", correct=True),
        _outcome("adv-1", adversarial=True, correct=None, abstained=True),
        _outcome("adv-2", adversarial=True, correct=None, abstained=False),
    ]
    b_outcomes = [
        _outcome("ans-1", correct=False),
        _outcome("adv-1", adversarial=True, correct=None, abstained=False),
        _outcome("adv-2", adversarial=True, correct=None, abstained=True),
    ]

    result = paired_mcnemar(a_outcomes, b_outcomes)
    assert result["n_paired"] == 1
    assert result["a_only"] == 1
    assert result["b_only"] == 0


def test_abstention_pairing_excludes_answerable_questions() -> None:
    a_outcomes = [
        _outcome("ans-1", correct=True, abstained=False),
        _outcome("ans-2", correct=False, abstained=True),
        _outcome("adv-1", adversarial=True, correct=None, abstained=True),
    ]
    b_outcomes = [
        _outcome("ans-1", correct=False, abstained=True),
        _outcome("ans-2", correct=True, abstained=False),
        _outcome("adv-1", adversarial=True, correct=None, abstained=False),
    ]

    result = paired_abstention_mcnemar(a_outcomes, b_outcomes)
    assert result["n_paired"] == 1
    assert result["a_only"] == 1
    assert result["b_only"] == 0
    assert result["both_correct"] == 0
    assert result["both_wrong"] == 0


def test_abstention_pairing_counts_abstaining_as_the_good_outcome() -> None:
    a_outcomes = [
        _outcome(f"adv-{i}", adversarial=True, correct=None, abstained=True) for i in range(3)
    ]
    b_outcomes = [
        _outcome(f"adv-{i}", adversarial=True, correct=None, abstained=False) for i in range(3)
    ]

    result = paired_abstention_mcnemar(a_outcomes, b_outcomes)
    assert result["a_only"] == 3
    assert result["b_only"] == 0
    # 2 * P(X <= 0 | n=3) = 2/8
    assert result["p_value"] == pytest.approx(0.25, abs=1e-12)


# -- discrimination ----------------------------------------------------------------------------

def _aggregate(abstention: float | None, false_abstain: float | None) -> dict[str, Any]:
    return {
        "adversarial_abstention": {"n": 47, "rate": abstention, "ci95": [None, None]},
        "answerable_false_abstain": {"n": 152, "rate": false_abstain, "ci95": [None, None]},
    }


def test_discrimination_is_the_difference_of_the_two_rates() -> None:
    assert discrimination(_aggregate(0.9362, 0.3618)) == pytest.approx(0.5744, abs=1e-9)


def test_discrimination_is_none_when_abstention_rate_is_none() -> None:
    assert discrimination(_aggregate(None, 0.3618)) is None


def test_discrimination_is_none_when_false_abstain_rate_is_none() -> None:
    assert discrimination(_aggregate(0.9362, None)) is None


def test_discrimination_is_none_when_the_block_is_missing_entirely() -> None:
    assert discrimination({"adversarial_abstention": {"n": 0, "rate": 1.0}}) is None


def test_blanket_conservatism_scores_perfect_abstention_but_zero_discrimination() -> None:
    """The case the metric exists for: refusing EVERYTHING looks perfect on abstention alone."""
    aggregate = _aggregate(1.0, 1.0)
    assert aggregate["adversarial_abstention"]["rate"] == 1.0
    assert discrimination(aggregate) == 0.0


def test_answering_everything_also_scores_zero_discrimination() -> None:
    assert discrimination(_aggregate(0.0, 0.0)) == 0.0


def test_perfect_discrimination_is_one() -> None:
    assert discrimination(_aggregate(1.0, 0.0)) == 1.0


# -- curve -------------------------------------------------------------------------------------

def _write_artifact(
    path: Path,
    *,
    arm: str,
    k: int,
    tokens_mean: float,
    accuracy: float,
    abstention: float,
    false_abstain: float,
) -> Path:
    doc = {
        "arm": arm,
        "model": "openai/gpt-4o-mini",
        "config": {"k": k, "arm": arm},
        "aggregate": {
            "answerable_accuracy": {"n": 152, "rate": accuracy, "ci95": [None, None]},
            "adversarial_abstention": {"n": 47, "rate": abstention, "ci95": [None, None]},
            "answerable_false_abstain": {
                "n": 152, "rate": false_abstain, "ci95": [None, None],
            },
            "retrieved_context": {
                "n": 199,
                "chars": {"mean": tokens_mean * 4, "median": tokens_mean * 4},
                "tokens_approx": {"mean": tokens_mean, "median": tokens_mean},
            },
            "by_category": {},
        },
        "outcomes": [],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_curve_points_reads_k_from_config_and_sorts_by_arm_then_tokens(tmp_path: Path) -> None:
    _write_artifact(
        tmp_path / "recall_k10.json", arm="recall", k=10, tokens_mean=536.7,
        accuracy=0.5, abstention=0.9, false_abstain=0.2,
    )
    _write_artifact(
        tmp_path / "recall_k5.json", arm="recall", k=5, tokens_mean=272.8,
        accuracy=0.4, abstention=0.95, false_abstain=0.3,
    )
    _write_artifact(
        tmp_path / "mem0_k10.json", arm="mem0", k=10, tokens_mean=254.8,
        accuracy=0.375, abstention=0.9362, false_abstain=0.3618,
    )

    points = curve_points(sorted(tmp_path.glob("*.json")))

    assert [(p["arm"], p["k"]) for p in points] == [
        ("mem0", 10), ("recall", 5), ("recall", 10),
    ]
    assert points[0]["ctx_tokens_mean"] == 254.8
    assert points[0]["accuracy"] == 0.375
    assert points[0]["discrimination"] == pytest.approx(0.5744, abs=1e-9)
    assert points[0]["n_answerable"] == 152
    assert points[0]["n_adversarial"] == 47
    assert points[0]["path"].endswith("mem0_k10.json")


def test_curve_points_prefers_a_top_level_k_when_present(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path / "run.json", arm="recall", k=5, tokens_mean=100.0,
        accuracy=0.4, abstention=0.9, false_abstain=0.2,
    )
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    doc["k"] = 20
    path.write_text(json.dumps(doc), encoding="utf-8")

    assert curve_points([path])[0]["k"] == 20


def test_curve_points_tolerates_a_missing_context_block(tmp_path: Path) -> None:
    path = tmp_path / "bare.json"
    path.write_text(json.dumps({"arm": "recall", "aggregate": {}}), encoding="utf-8")

    point = curve_points([path])[0]
    assert point["ctx_tokens_mean"] is None
    assert point["accuracy"] is None
    assert point["discrimination"] is None
    assert point["n_answerable"] == 0


def test_plot_curve_writes_a_figure_or_names_the_missing_extra(tmp_path: Path) -> None:
    """matplotlib lives in the `eval` extra, so BOTH environments are pinned, not just the happy
    one: with it the figure lands on disk, without it the caller is told which extra to install
    instead of getting a bare ImportError from three frames down."""
    _write_artifact(
        tmp_path / "recall.json", arm="recall", k=5, tokens_mean=272.8,
        accuracy=0.4, abstention=0.95, false_abstain=0.3,
    )
    points = curve_points(sorted(tmp_path.glob("*.json")))
    out_path = tmp_path / "figures" / "curve.png"

    if importlib.util.find_spec("matplotlib") is None:
        with pytest.raises(RuntimeError, match="matplotlib"):
            plot_curve(points, out_path)
    else:
        assert plot_curve(points, out_path) == out_path
        assert out_path.stat().st_size > 0


# -- CLI ---------------------------------------------------------------------------------------

def _artifact_with_outcomes(
    path: Path, arm: str, k: int, outcomes: list[dict[str, Any]]
) -> Path:
    doc = {
        "arm": arm,
        "config": {"k": k},
        "aggregate": {
            "answerable_accuracy": {"n": 2, "rate": 0.5, "ci95": [None, None]},
            "adversarial_abstention": {"n": 1, "rate": 1.0, "ci95": [None, None]},
            "answerable_false_abstain": {"n": 2, "rate": 0.25, "ci95": [None, None]},
            "retrieved_context": {"tokens_approx": {"mean": 300.0, "median": 300.0}},
            "by_category": {},
        },
        "outcomes": outcomes,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_cli_compare_prints_both_tests_and_both_discriminations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a_outcomes, b_outcomes = _accuracy_arms(both_correct=1, both_wrong=1, a_only=8, b_only=2)
    a_outcomes.append(_outcome("adv-0", adversarial=True, correct=None, abstained=True))
    b_outcomes.append(_outcome("adv-0", adversarial=True, correct=None, abstained=False))
    a_path = _artifact_with_outcomes(tmp_path / "a.json", "recall", 5, a_outcomes)
    b_path = _artifact_with_outcomes(tmp_path / "b.json", "mem0", 10, b_outcomes)

    assert main(["--compare", str(a_path), str(b_path)]) == 0

    out = capsys.readouterr().out
    assert "paired accuracy (McNemar" in out
    assert "paired abstention (McNemar" in out
    assert "exact_binomial" in out
    assert "0.109375" in out
    assert "J=0.7500" in out


def test_cli_curve_tabulates_every_matched_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_artifact(
        tmp_path / "recall.json", arm="recall", k=5, tokens_mean=272.8,
        accuracy=0.4, abstention=0.95, false_abstain=0.3,
    )
    _write_artifact(
        tmp_path / "mem0.json", arm="mem0", k=10, tokens_mean=254.8,
        accuracy=0.375, abstention=0.9362, false_abstain=0.3618,
    )

    assert main(["--curve", str(tmp_path / "*.json")]) == 0

    out = capsys.readouterr().out
    assert "recall" in out
    assert "mem0" in out
    assert "254.8" in out


def test_cli_rejects_a_glob_that_matches_nothing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--curve", str(tmp_path / "nothing-*.json")])


def test_cli_requires_something_to_do() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_rejects_plot_without_curve(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--plot", str(tmp_path / "out.png")])


def test_analyze_refuses_an_artifact_that_was_never_published(tmp_path: Path) -> None:
    """The in-band mark is a contract, not a comment.

    `benchmarks.run` quarantines a refused artifact outside the `results/*.json` glob AND marks
    it. Without a reader that honours the mark, a quarantined file reached directly, or by a
    `results/**/*.json` glob, is byte identical to a real measurement and gets tabulated as one.
    """
    from benchmarks.artifact_contract import load_published_artifact

    path = tmp_path / "refused.json"
    path.write_text(
        json.dumps(
            {
                "arm": "recall",
                "aggregate": {"answerable_accuracy": {"rate": 0.99, "n": 2}},
                "unpublished": True,
                "unpublished_reason": "benchmark cost claims require provider_metadata",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="REFUSED publication"):
        load_published_artifact(path)
