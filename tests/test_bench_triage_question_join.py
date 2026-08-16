"""The question text a triage feature reads must come from a join that fails loudly.

Two registered features, `conjunctions` and `question_words`, scored exactly 0.500 on all 500
questions of the first run and were published as measured nulls. They were not measured at all:
the fixture carried no `question` key, the scorer read it with `.get(..., "")`, and a constant
feature is 0.500 by construction in this AUC implementation. A dead feature and a clean null are
indistinguishable in the output, which is what makes the failure worth a test rather than a fix.

So the join is a function with one behaviour under test: an id it cannot resolve is an error, not
an empty string. A PARTIAL join is the worse case and is covered separately, because it leaves
some rows with real text and some with none, manufacturing a split in every text feature.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.analyse_triage import load_question_text


def _questions_file(tmp_path, rows: list[tuple[str, str]]):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "".join(json.dumps({"question_id": qid, "question": q}) + "\n" for qid, q in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_returns_the_text_for_every_fixture_id(tmp_path) -> None:
    path = _questions_file(tmp_path, [("q1", "What is alpha?"), ("q2", "Who signed beta?")])

    assert load_question_text(path, {"q1", "q2"}) == {
        "q1": "What is alpha?",
        "q2": "Who signed beta?",
    }


def test_an_id_missing_from_the_questions_file_raises_rather_than_defaulting(tmp_path) -> None:
    path = _questions_file(tmp_path, [("q1", "What is alpha?")])

    with pytest.raises(SystemExit, match="q2"):
        load_question_text(path, {"q1", "q2"})


def test_a_partial_join_names_how_many_are_missing_not_just_that_some_are(tmp_path) -> None:
    """The count is the difference between "one id is stale" and "the wrong file was passed"."""
    path = _questions_file(tmp_path, [("q1", "a"), ("q2", "b")])

    with pytest.raises(SystemExit, match="3 "):
        load_question_text(path, {"q1", "q2", "q3", "q4", "q5"})


def test_extra_questions_in_the_file_are_not_an_error(tmp_path) -> None:
    """The questions file is the full benchmark; a fixture may legitimately be a `--limit` pilot."""
    path = _questions_file(tmp_path, [("q1", "a"), ("q2", "b"), ("q3", "c")])

    assert set(load_question_text(path, {"q1"})) == {"q1"}


def test_a_blank_question_is_refused_because_it_is_the_failure_being_guarded_against(
    tmp_path,
) -> None:
    """An empty string in the FILE reproduces exactly the bug the join exists to prevent, and it
    would otherwise pass the id check and go on to score 0.500."""
    path = _questions_file(tmp_path, [("q1", "a"), ("q2", "   ")])

    with pytest.raises(SystemExit, match="blank"):
        load_question_text(path, {"q1", "q2"})
