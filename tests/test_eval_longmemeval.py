"""LongMemEval → recall.eval.labelled conversion.

The fixtures here are hand-written to the schema documented in the LongMemEval repository
(MIT, github.com/xiaowu0162/LongMemEval), not sampled from the dataset itself: the dataset is
not vendored, and a converter test that needs a 115k-token download to run is a test nobody
runs. Every field asserted below is one the real file carries.
"""
from __future__ import annotations

import json

import pytest

from recall.eval.longmemeval import ConversionError, convert


def _instance(**over) -> dict:
    """A minimal well-formed LongMemEval instance."""
    base = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "what did I say my deploy target was?",
        "answer": "staging",
        "question_date": "2026/05/02 (Sat) 10:11",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2026/04/01 (Wed) 09:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "we deploy to staging first", "has_answer": True},
                {"role": "assistant", "content": "noted"},
            ]
        ],
        "answer_session_ids": ["s1"],
    }
    base.update(over)
    return base


def test_a_session_becomes_one_markdown_file_named_by_its_session_id(tmp_path):
    convert([_instance()], tmp_path)

    assert (tmp_path / "corpus" / "s1.md").is_file()


def test_turns_are_rendered_in_order_and_labelled_by_role(tmp_path):
    convert([_instance()], tmp_path)

    body = (tmp_path / "corpus" / "s1.md").read_text(encoding="utf-8")
    user_at = body.index("we deploy to staging first")
    assistant_at = body.index("noted")
    assert user_at < assistant_at
    # each turn's content is preceded by its own role label, not by the other one
    assert "user" in body[:user_at]
    assert "assistant" in body[user_at:assistant_at]


def test_the_session_date_is_carried_into_frontmatter(tmp_path):
    convert([_instance()], tmp_path)

    body = (tmp_path / "corpus" / "s1.md").read_text(encoding="utf-8")
    assert body.startswith("---\n")
    assert "2026/04/01 (Wed) 09:00" in body.split("---", 2)[1]


def test_answer_sessions_become_the_questions_relevant_files(tmp_path):
    convert([_instance(haystack_session_ids=["s1", "s2"],
                       haystack_dates=["d1", "d2"],
                       haystack_sessions=[[{"role": "user", "content": "a"}],
                                          [{"role": "user", "content": "b"}]],
                       answer_session_ids=["s2"])], tmp_path)

    q = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))[0]
    assert q["relevant_files"] == ["s2.md"]
    assert q["answerable"] is True
    assert q["query"] == "what did I say my deploy target was?"


def test_an_abstention_instance_becomes_an_unanswerable_question(tmp_path):
    # The dataset marks these by suffix and gives them no answer location; the retrieval
    # protocol everyone else publishes SKIPS them, which is exactly why they are worth keeping.
    inst = _instance(question_id="q1_abs", answer_session_ids=[])

    convert([inst], tmp_path)

    q = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))[0]
    assert q["answerable"] is False
    assert "relevant_files" not in q


def test_the_question_type_is_preserved_for_per_category_analysis(tmp_path):
    convert([_instance(question_type="knowledge-update")], tmp_path)

    q = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))[0]
    assert q["question_type"] == "knowledge-update"


def test_a_session_shared_by_two_instances_is_written_once(tmp_path):
    first = _instance(question_id="q1")
    second = _instance(question_id="q2", question="and what about the database?")

    report = convert([first, second], tmp_path)

    assert report.sessions_written == 1
    assert report.sessions_deduplicated == 1
    assert len(list((tmp_path / "corpus").glob("*.md"))) == 1


def test_a_session_reused_at_a_different_date_is_deduplicated_not_rejected(tmp_path):
    # A distractor session is placed into many haystacks, and the benchmark timestamps it
    # per-haystack. Same conversation, different date, and treating that as a content conflict
    # would refuse to convert the real dataset at all.
    first = _instance(question_id="q1", haystack_dates=["2026/04/01 (Wed) 09:00"])
    second = _instance(question_id="q2", haystack_dates=["2026/06/15 (Mon) 14:30"])

    report = convert([first, second], tmp_path)

    assert report.sessions_written == 1
    assert report.sessions_at_multiple_dates == 1


def test_every_date_a_reused_session_was_seen_at_is_recorded(tmp_path):
    # The merged corpus can only hold one document per session, so the discarded timestamps
    # have to be visible somewhere or the loss is silent.
    first = _instance(question_id="q1", haystack_dates=["2026/04/01 (Wed) 09:00"])
    second = _instance(question_id="q2", haystack_dates=["2026/06/15 (Mon) 14:30"])

    convert([first, second], tmp_path)

    body = (tmp_path / "corpus" / "s1.md").read_text(encoding="utf-8")
    assert "2026/04/01 (Wed) 09:00" in body
    assert "2026/06/15 (Mon) 14:30" in body


def test_the_same_session_id_carrying_different_text_is_an_error(tmp_path):
    # Silently keeping one of the two would put a document in the corpus that no longer matches
    # the haystack the question was written against, and the run would still report a number.
    first = _instance(question_id="q1")
    second = _instance(
        question_id="q2",
        haystack_sessions=[[{"role": "user", "content": "something else entirely"}]],
    )

    with pytest.raises(ConversionError, match="s1"):
        convert([first, second], tmp_path)


def test_an_answerable_instance_with_no_answer_session_is_an_error(tmp_path):
    with pytest.raises(ConversionError, match="q1"):
        convert([_instance(answer_session_ids=[])], tmp_path)


def test_an_answer_session_missing_from_the_haystack_is_an_error(tmp_path):
    # Gold that names a document the corpus does not contain scores 0 for a reason that has
    # nothing to do with retrieval.
    with pytest.raises(ConversionError, match="s9"):
        convert([_instance(answer_session_ids=["s9"])], tmp_path)


def test_a_haystack_whose_ids_and_sessions_disagree_in_length_is_an_error(tmp_path):
    with pytest.raises(ConversionError, match="haystack"):
        convert([_instance(haystack_session_ids=["s1", "s2"])], tmp_path)


def test_a_session_id_that_would_escape_the_corpus_directory_is_rejected(tmp_path):
    # The ids are data from a downloaded file. A converter that joins them onto a path without
    # checking is an arbitrary-write primitive.
    with pytest.raises(ConversionError, match="session id"):
        convert([_instance(haystack_session_ids=["../escaped"],
                           answer_session_ids=["../escaped"])], tmp_path)


def test_the_report_counts_questions_by_category(tmp_path):
    instances = [
        _instance(question_id="q1", question_type="knowledge-update"),
        _instance(question_id="q2", question_type="knowledge-update"),
        _instance(question_id="q3_abs", question_type="single-session-user",
                  answer_session_ids=[]),
    ]

    report = convert(instances, tmp_path)

    assert report.by_type["knowledge-update"] == 2
    assert report.abstention == 1
    assert report.answerable == 2
