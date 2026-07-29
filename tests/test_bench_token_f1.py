"""Offline tests for the token-F1 re-scorer.

Same discipline as ``test_bench_analyze``: no results file is read, no model is called, and every
expected F1 is worked out by hand below rather than pinned to whatever the code happened to return
the first time. These numbers get compared against other people's published tables, so a scorer
that silently drifts from the textbook SQuAD definition would be worse than having no scorer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.token_f1 import (
    compare,
    main,
    normalize,
    paired_bootstrap,
    token_f1,
)


def _outcome(
    question_id: str,
    answer: str,
    gold: str,
    *,
    is_adversarial: bool = False,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "category": 5 if is_adversarial else 1,
        "is_adversarial": is_adversarial,
        "answer": answer,
        "gold": gold,
        "abstained": False,
        "correct": None,
        "question": f"q-{question_id}",
    }


def _artifact(tmp_path: Path, name: str, arm: str, outcomes: list[dict[str, Any]]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps({"arm": arm, "model": "openai/gpt-4o-mini", "outcomes": outcomes}),
        encoding="utf-8",
    )
    return path


class TestNormalize:
    def test_lowercases_and_strips_punctuation_and_articles(self) -> None:
        assert normalize("The CAT, sat.") == "cat sat"

    def test_collapses_whitespace(self) -> None:
        assert normalize("  a   spaced   out\tstring ") == "spaced out string"

    def test_article_removal_is_word_bounded(self) -> None:
        # "a" is an article; "australia" merely starts with one.
        assert normalize("a Australia") == "australia"


class TestTokenF1:
    def test_exact_match_is_one(self) -> None:
        assert token_f1("7 May 2023", "7 May 2023") == pytest.approx(1.0)

    def test_match_modulo_normalisation_is_one(self) -> None:
        assert token_f1("The CAT, sat.", "cat sat") == pytest.approx(1.0)

    def test_disjoint_is_zero(self) -> None:
        assert token_f1("cat", "dog") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # pred -> [red, car], gold -> [red, truck]; shared 1, P = R = 1/2, F1 = 0.5
        assert token_f1("the red car", "red truck") == pytest.approx(0.5)

    def test_overlap_is_a_multiset_not_a_set(self) -> None:
        # pred -> [x, x, y], gold -> [x, y]; shared 2, P = 2/3, R = 1, F1 = 0.8
        assert token_f1("x x y", "x y") == pytest.approx(0.8)

    def test_both_empty_is_a_match(self) -> None:
        assert token_f1("", "") == pytest.approx(1.0)

    @pytest.mark.parametrize(("pred", "gold"), [("", "something"), ("something", "")])
    def test_exactly_one_empty_is_a_miss(self, pred: str, gold: str) -> None:
        assert token_f1(pred, gold) == pytest.approx(0.0)

    def test_punctuation_only_prediction_normalises_to_empty(self) -> None:
        assert token_f1("...", "cat") == pytest.approx(0.0)

    def test_a_refusal_scores_near_zero_rather_than_being_special_cased(self) -> None:
        # The contract that matters: refusing an answerable question is scored as the failure it
        # is. If this ever returns None / raises / is skipped, a system could lift its mean by
        # abstaining on everything it finds hard.
        assert token_f1("I don't know.", "7 May 2023") == pytest.approx(0.0)


class TestPairedBootstrap:
    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        diffs = [0.1, -0.2, 0.3, 0.0, 0.5] * 20
        assert paired_bootstrap(diffs, resamples=500, seed=7) == paired_bootstrap(
            diffs, resamples=500, seed=7
        )

    def test_interval_brackets_the_observed_mean(self) -> None:
        diffs = [0.1, 0.2, 0.15, 0.25, 0.05] * 40
        low, high = paired_bootstrap(diffs, resamples=2_000, seed=0)
        assert low <= sum(diffs) / len(diffs) <= high

    def test_constant_differences_give_a_degenerate_interval(self) -> None:
        low, high = paired_bootstrap([0.25] * 50, resamples=500, seed=0)
        assert low == pytest.approx(0.25) and high == pytest.approx(0.25)


class TestCompare:
    def test_pairs_on_shared_questions_and_scores_by_hand(self, tmp_path: Path) -> None:
        a = _artifact(tmp_path, "a", "recall", [
            _outcome("q1", "7 May 2023", "7 May 2023"),   # 1.0
            _outcome("q2", "the red car", "red truck"),   # 0.5
        ])
        b = _artifact(tmp_path, "b", "mem0", [
            _outcome("q1", "cat", "7 May 2023"),          # 0.0
            _outcome("q2", "red truck", "red truck"),     # 1.0
        ])
        result = compare(a, b)

        assert result["n"] == 2
        assert result["a"]["token_f1"] == pytest.approx(0.75)   # (1.0 + 0.5) / 2
        assert result["b"]["token_f1"] == pytest.approx(0.5)    # (0.0 + 1.0) / 2
        assert result["delta"] == pytest.approx(0.25)
        assert result["a"]["arm"] == "recall" and result["b"]["arm"] == "mem0"

    def test_excludes_adversarial_questions(self, tmp_path: Path) -> None:
        # Adversarial rows have no gold to overlap with; including them would silently score the
        # abstention axis with a metric that cannot express it.
        outcomes = [
            _outcome("q1", "cat", "cat"),
            _outcome("q2", "anything", "", is_adversarial=True),
        ]
        a = _artifact(tmp_path, "a", "recall", outcomes)
        b = _artifact(tmp_path, "b", "mem0", outcomes)
        assert compare(a, b)["n"] == 1

    def test_reports_unshared_questions_rather_than_dropping_them_silently(
        self, tmp_path: Path
    ) -> None:
        a = _artifact(tmp_path, "a", "recall", [
            _outcome("q1", "cat", "cat"), _outcome("only-a", "cat", "cat"),
        ])
        b = _artifact(tmp_path, "b", "mem0", [
            _outcome("q1", "cat", "cat"), _outcome("only-b", "cat", "cat"),
        ])
        result = compare(a, b)
        assert result["n"] == 1
        assert result["dropped_unshared"] == {"a_only": 1, "b_only": 1}

    def test_refusals_are_counted_in_the_mean(self, tmp_path: Path) -> None:
        a = _artifact(tmp_path, "a", "recall", [
            _outcome("q1", "cat", "cat"), _outcome("q2", "I don't know.", "7 May 2023"),
        ])
        b = _artifact(tmp_path, "b", "mem0", [
            _outcome("q1", "cat", "cat"), _outcome("q2", "7 May 2023", "7 May 2023"),
        ])
        result = compare(a, b)
        assert result["a"]["token_f1"] == pytest.approx(0.5)  # not 1.0 with the refusal dropped
        assert result["delta"] == pytest.approx(-0.5)

    def test_no_shared_questions_is_an_error_not_an_empty_mean(self, tmp_path: Path) -> None:
        a = _artifact(tmp_path, "a", "recall", [_outcome("only-a", "cat", "cat")])
        b = _artifact(tmp_path, "b", "mem0", [_outcome("only-b", "cat", "cat")])
        with pytest.raises(SystemExit):
            compare(a, b)


class TestMain:
    def test_prints_a_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        a = _artifact(tmp_path, "a", "recall", [_outcome("q1", "cat", "cat")])
        b = _artifact(tmp_path, "b", "mem0", [_outcome("q1", "dog", "cat")])
        assert main(["--compare", str(a), str(b)]) == 0
        out = capsys.readouterr().out
        assert "token F1" in out and "recall" in out and "mem0" in out

    def test_json_mode_round_trips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = _artifact(tmp_path, "a", "recall", [_outcome("q1", "cat", "cat")])
        b = _artifact(tmp_path, "b", "mem0", [_outcome("q1", "dog", "cat")])
        assert main(["--compare", str(a), str(b), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n"] == 1 and payload["delta"] == pytest.approx(1.0)
