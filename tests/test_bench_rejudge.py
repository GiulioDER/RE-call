"""Offline tests for the judge-replay pass.

No network, no database, no model: the judge is a plain function. What is pinned here is the rule
that decides WHICH records go to the judge, because that rule sets the denominator of every
accuracy number the re-scored artifact reports — and a re-score that quietly judges the
abstentions produces a better-looking number that no reader can attribute to anything.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.rejudge import main, rejudge_document, rejudge_outcomes, to_outcome


def _record(
    question_id: str,
    *,
    adversarial: bool = False,
    abstained: bool = False,
    correct: bool | None = False,
    answer: str = "an answer",
    gold: str = "the gold",
    question: str = "a question?",
) -> dict[str, Any]:
    """One ``outcomes`` entry, in the shape ``benchmarks.run`` writes it."""
    return {
        "question_id": question_id,
        "category": "cat1",
        "is_adversarial": adversarial,
        "context": "some retrieved context",
        "answer": answer,
        "abstained": abstained,
        "correct": None if adversarial else correct,
        "question": question,
        "gold": gold,
    }


class RecordingJudge:
    """A judge that returns a scripted verdict per question_id and remembers what it was asked."""

    def __init__(self, verdicts: dict[str, bool], default: bool = False) -> None:
        self.verdicts = verdicts
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        for key, verdict in self.verdicts.items():
            if key in user:
                return "YES" if verdict else "NO"
        return "YES" if self.default else "NO"


def _always_yes(system: str, user: str) -> str:
    return "YES"


def test_only_judged_records_are_rejudged() -> None:
    """Adversarial and abstained records never reach the judge — the denominator rule."""
    outcomes = [
        _record("c:0", answer="answered", gold="G0"),
        _record("c:1", abstained=True, answer="NO_ANSWER", gold="G1"),
        _record("c:2", adversarial=True, abstained=True, answer="NO_ANSWER", gold=""),
        _record("c:3", adversarial=True, abstained=False, answer="a guess", gold=""),
    ]
    judge = RecordingJudge({}, default=True)

    records, report = rejudge_outcomes(outcomes, judge)

    assert report["n_judged"] == 1
    assert len(judge.calls) == 1
    # The one judged record is the answerable, non-abstained one, and it is the ONLY gold answer
    # the judge ever saw.
    assert "G0" in judge.calls[0][1]
    assert all("G1" not in user for _, user in judge.calls)


def test_adversarial_and_abstained_verdicts_are_untouched() -> None:
    """A judge that says YES to everything must not rescue an abstention or score an adversarial."""
    outcomes = [
        _record("c:0", answer="answered", correct=False),
        _record("c:1", abstained=True, answer="NO_ANSWER", correct=False),
        _record("c:2", adversarial=True, abstained=True, answer="NO_ANSWER"),
    ]

    records, report = rejudge_outcomes(outcomes, _always_yes)

    by_id = {r["question_id"]: r for r in records}
    assert by_id["c:0"]["correct"] is True  # re-judged, flipped
    assert by_id["c:1"]["correct"] is False  # abstained: still wrong, judge never called
    assert by_id["c:2"]["correct"] is None  # adversarial: still unscored
    assert report["flipped_to_correct"] == 1


def test_agreement_math() -> None:
    """Agreement counts, rate and both flip directions, over a hand-built set."""
    outcomes = [
        _record("c:0", correct=True, gold="AGREE_RIGHT"),  # judge says YES -> agree
        _record("c:1", correct=False, gold="AGREE_WRONG"),  # judge says NO  -> agree
        _record("c:2", correct=False, gold="FLIP_UP"),  # judge says YES -> flip to correct
        _record("c:3", correct=True, gold="FLIP_DOWN"),  # judge says NO  -> flip to incorrect
        _record("c:4", abstained=True, answer="NO_ANSWER", gold="NEVER_JUDGED"),
    ]
    judge = RecordingJudge(
        {"AGREE_RIGHT": True, "AGREE_WRONG": False, "FLIP_UP": True, "FLIP_DOWN": False}
    )

    _, report = rejudge_outcomes(outcomes, judge)

    assert report["n_judged"] == 4
    assert report["n_agree"] == 2
    assert report["n_disagree"] == 2
    assert report["agreement_rate"] == 0.5
    assert report["flipped_to_correct"] == 1
    assert report["flipped_to_incorrect"] == 1
    assert {d["question_id"] for d in report["disagreements"]} == {"c:2", "c:3"}
    flip_up = next(d for d in report["disagreements"] if d["question_id"] == "c:2")
    assert flip_up["old_verdict"] is False
    assert flip_up["new_verdict"] is True
    assert flip_up["gold"] == "FLIP_UP"


def test_agreement_rate_is_none_when_nothing_was_judged() -> None:
    """An artifact of nothing but adversarials reports no agreement, not a vacuous 1.0."""
    _, report = rejudge_outcomes([_record("c:0", adversarial=True, abstained=True)], _always_yes)

    assert report["n_judged"] == 0
    assert report["agreement_rate"] is None
    assert report["disagreements"] == []


def test_input_records_are_not_mutated() -> None:
    """The in-memory records handed in come back unchanged, whatever the new judge decided."""
    outcomes = [_record("c:0", correct=False), _record("c:1", correct=True)]
    before = json.dumps(outcomes, sort_keys=True)

    records, _ = rejudge_outcomes(outcomes, _always_yes)

    assert json.dumps(outcomes, sort_keys=True) == before
    assert [r["correct"] for r in records] == [True, True]


def test_corrupt_record_is_rejected() -> None:
    """A record breaking the `correct is None iff adversarial` invariant fails loudly."""
    broken = _record("c:0")
    broken["correct"] = None  # answerable, but unscored

    with pytest.raises(ValueError, match="must be None iff"):
        rejudge_outcomes([broken], _always_yes)


def test_rejudge_document_records_provenance_and_reaggregates() -> None:
    doc = {
        "arm": "recall",
        "model": "openai/gpt-4o-mini",
        "config": {"k": 5, "model": "openai/gpt-4o-mini"},
        "aggregate": {"answerable_accuracy": {"n": 2, "rate": 0.0, "ci95": [0.0, 0.0]}},
        "outcomes": [_record("c:0", correct=False), _record("c:1", correct=False)],
    }

    payload = rejudge_document(doc, _always_yes, "big/judge", "results/a.json")

    assert payload["rejudge"]["judge_model"] == "big/judge"
    assert payload["rejudge"]["source"] == "results/a.json"
    assert payload["rejudge"]["original_judge_model"] == "openai/gpt-4o-mini"
    assert payload["rejudge"]["original_aggregate"] == doc["aggregate"]
    # Recomputed by the shipped `aggregate`, so the re-scored file is the same schema as a run.
    assert payload["aggregate"]["answerable_accuracy"]["n"] == 2
    assert payload["aggregate"]["answerable_accuracy"]["rate"] == 1.0
    assert payload["arm"] == "recall"
    assert doc["aggregate"]["answerable_accuracy"]["rate"] == 0.0  # source untouched


def test_to_outcome_round_trips_the_scored_fields() -> None:
    outcome = to_outcome(_record("c:9", correct=True, answer="x"))

    assert outcome.question_id == "c:9"
    assert outcome.correct is True
    assert outcome.answer == "x"


def _write_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "arm": "recall",
                "config": {"k": 5, "model": "openai/gpt-4o-mini"},
                "aggregate": {},
                "outcomes": [_record("c:0", correct=False)],
            }
        ),
        encoding="utf-8",
    )


def test_cli_refuses_to_overwrite_the_input(tmp_path: Path, capsys: Any) -> None:
    """Even spelled differently, --out pointing at --input is rejected before any work happens."""
    source = tmp_path / "a.json"
    _write_artifact(source)

    spelled_differently = str(tmp_path / "." / "a.json")
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--model", "big/judge", "--out", spelled_differently])
    assert "never modified" in capsys.readouterr().err


def test_cli_refuses_to_clobber_an_existing_output(tmp_path: Path, capsys: Any) -> None:
    source = tmp_path / "a.json"
    _write_artifact(source)
    existing = tmp_path / "b.json"
    existing.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["--input", str(source), "--model", "big/judge", "--out", str(existing)])
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_validates_the_api_key_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The key is checked before the file is read, so a placeholder fails free — not mid-run."""
    source = tmp_path / "a.json"
    _write_artifact(source)
    monkeypatch.setenv("OPENROUTER_API_KEY", "...")

    with pytest.raises(SystemExit):
        main(["--input", str(source), "--model", "big/judge", "--out", str(tmp_path / "b.json")])
    assert "OpenRouter key" in capsys.readouterr().err
