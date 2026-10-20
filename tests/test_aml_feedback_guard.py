"""ST-1a: failed answer-feedback records are not stored at Add, under RECALL_AML_FEEDBACK_GUARD=1.

The record shapes follow AgentMemoryBench's public code (``solomoon313/AgentMemoryBench`` at
``9dfda8de``: the ``result`` dict and the ``llm_judge`` tool message) and one public participant's
flat form; every question and answer below is invented.

Red proofs, run 2026-10-10 on branch ``claude/st1-feedback-guard`` (off ``225e7509``), after the
asked-question gate, with ``PYTHONDONTWRITEBYTECODE=1``: each row is one plausible mutation of the
named production line with this file unchanged, each named test run alone, the ``E`` line it failed
with (an assertion, never an import, collection or fixture error), then the line restored and the
whole file run green (26 passed). Node ids are ``tests/test_aml_feedback_guard.py::<name>``. The
first version's 14 mutations (``6e1443b6``) were re-run against the gated code, with the anchors that
moved.

- ``test_a_failed_record_that_carries_the_correct_answer_is_kept``: ``and not record.has_reference``
  deleted from ``filter_feedback``. Failed: ``([], 1) == ([Message(...)], 0)``.
- ``test_the_guard_drops_the_failed_record_and_stores_the_rest`` and
  ``test_an_add_of_only_failed_records_is_durable_and_empty``: ``request = request.model_copy(...)``
  in ``HostedService._add_once`` replaced by ``pass``. Failed: ``where False = all(...)`` (the record
  stored in the window) and ``(1, 0) == (0, 0)``.
- ``test_the_guard_is_off_unless_set``: ``HostedService.feedback_guard`` defaulting to True. Failed:
  ``assert True is False``.
- ``test_a_flat_record_with_a_failing_score_is_dropped``: ``value > 0`` in ``_outcome`` made
  ``value >= 0``. Failed: ``([Message(...)], 0) == ([], 1)``.
- ``test_a_wrong_judge_verdict_drops_the_whole_feedback_unit``: the judge-unit branch of
  ``filter_feedback`` disabled (``if False and ...``). Failed: ``[Message(...)] == []``.
- ``test_a_lone_judge_message_is_not_a_unit``: ``len(messages) > 1`` made ``>= 1``. Failed: ``[] ==
  [Message(...)]``.
- ``test_anything_short_of_a_whole_readable_record_is_stored_unchanged``, four mutations: only the
  first outcome key read failed ``[contradicts]``; ``_LABEL_LINE`` allowed to span lines failed
  ``[two-lines]``; the answer-key check made ``if False:`` failed ``[no-answer]``; an unreadable
  outcome string read as ``"failed"`` failed ``[unreadable]``; each ``[] == [Message(...)]``.
- ``test_a_record_beside_an_image_is_not_read``: ``return None`` for a non-text part in ``_text``
  made ``continue``. Failed: ``FeedbackRecord(verdict='failed', ...) is None``.
- ``test_a_bad_value_stops_service_startup``: ``self.feedback_guard`` removed from the constructor's
  startup reads. Failed: ``DID NOT RAISE ValueError``.
- ``test_the_benchmark_result_shape_is_read_with_its_outer_reward``: the merge of outer outcome keys
  in ``record_verdict`` deleted. Failed: ``assert (None is not None)``.
- ``test_a_flat_record_with_a_passing_score_is_kept_and_counted``: a passing record skipped
  (``continue``) after it is counted. Failed: ``([], 0, 1) == ([Message(...)], 0, 1)``.
- ``test_a_failed_record_about_a_question_never_searched_is_stored``,
  ``test_a_restart_forgets_what_was_asked_and_stores_everything`` and
  ``test_with_no_feedback_records_the_guard_stores_exactly_what_the_off_path_stores``: the gate
  bypassed in ``_add_once`` (``asked=None``). Failed: ``where False = any(...)`` twice, and the stored
  rows differ.
- ``test_the_guard_drops_the_failed_record_and_stores_the_rest`` and
  ``test_a_query_that_carries_the_question_counts_as_asking_it``: ``self._asked_questions.remember``
  in ``search`` replaced by ``pass``. Failed: ``where False = all(...)`` both.
- ``test_searches_made_while_the_guard_is_off_are_not_remembered``: ``if self.feedback_guard:``
  before ``remember`` made ``if True:``. Failed: ``where False = any(...)``.
- ``test_a_query_that_carries_the_question_counts_as_asking_it``: the containment ``return any(...)``
  in ``AskedQuestions.asked`` made ``return False``. Failed: ``where False = all(...)``.
- ``test_short_questions_must_match_exactly``, three mutations: the ``MIN_CONTAINED_CHARS`` exact
  rule deleted failed ``assert not True`` on ``asked('t', 'Mara now')``; tenants not separated
  (``seen`` taken from any tenant) failed ``assert not True`` on ``asked('other', ...)``; no case
  folding in ``normalize_question`` failed ``assert False`` on the upper-case question.
- ``test_a_judge_unit_is_dropped_only_when_its_question_was_asked``: ``and
  was_asked(unit_question(messages))`` deleted. Failed: ``[] == [Message(...)]``.

Red proofs, run 2026-10-20 on the remote test host (``sess-c9-fg-fix``, branch ``claude/c9-feedback-guard-fix-9bee2c``
off ``43693e37``), same method, for the fix of three false drops found by the CCA hunt of the C9
Full 2 build: U-45 (mem0's gold read as the model's answer), U-46 (a judge unit took the whole Add
and its question ran to the end of the message), U-44 (a deleted user kept its asked questions).

Baseline: the new tests run against ``43693e37``'s ``feedback_guard.py`` and ``service.py``, this file
unchanged. Red there, each in its assertion:

- ``test_a_mem0_record_with_the_gold_under_answer_is_kept``: ``([], 1) == ([Message(...)], 0)``.
- ``test_the_feedback_shapes_the_guard_drops_stay_dropped[streamicl-memories]``: ``([Message(...)],
  0) == ([], 3)`` (the old code read the first ``Question:`` line, a memory).
- ``test_dialogue_sent_beside_a_wrong_judge_unit_is_stored``: ``([], 5) == ([Message(...)], 3)``.
- ``test_a_question_line_followed_by_dialogue_is_not_an_answer_turn``: ``([], 3) == ([Message(...)],
  0)``.
- ``test_a_reference_sent_after_a_wrong_judge_unit_is_stored``: ``([], 4) == ([Message(...)], 3)``.
- ``test_a_judge_unit_that_holds_the_reference_is_kept``: ``([], 4) == ([Message(...)], 0)``.
- ``test_the_question_of_a_turn_is_its_last_not_a_memory_before_it``: ``([], 3) == ([Message(...)],
  0)``.
- ``test_deleting_a_user_forgets_what_it_was_asked``: ``assert False`` (the record was dropped).

Green on the baseline, because the old code drops these shapes too, and each red under a mutation
of the fixed code (the ``[id]`` is a case of
``test_the_feedback_shapes_the_guard_drops_stay_dropped``):

- ``[st1c-flat]``, ST-1c's measured shape: ``value > 0`` in ``_outcome`` made ``>= 0``. Failed:
  ``([Message(...)], 0) == ([], 1)``.
- ``[flat-answer-only]``: the ``answer`` key no longer read as an answer (``if not model_answered:``),
  and ``answer`` always read as the reference (``model_answered and`` deleted). Both failed
  ``([Message(...)], 0) == ([], 1)``.
- ``[amb-result]``: the merge of outer outcome keys in ``record_verdict`` deleted. Failed:
  ``([Message(...)], 0) == ([], 1)``.
- ``[amb-judge-unit]``: ``drop.update(span)`` in ``filter_feedback`` made ``pass``. Failed:
  ``([Message(...)], 0) == ([], 3)``.
- ``[amb-system-and-memories]``: the system message before the turn left out (``if False:`` in
  ``_judge_units``). Failed: ``([Message(...)], 3) == ([], 4)``.
- ``[answer-inline]``: ``_FRAMED_QUESTION`` made to require ``Answer:`` at the end of the message.
  Failed: ``([Message(...)], 0) == ([], 2)``.
- ``[question-on-next-line]``: the whitespace ``_FRAMED_QUESTION`` allows before the question
  narrowed to spaces and tabs. Failed: ``([Message(...)], 0) == ([], 3)``.
- ``test_judge_verdicts_that_disagree_in_one_add_drop_nothing``: ``if len(verdicts) == 1`` made ``if
  verdicts``. Failed: ``([], 6) == ([Message(...)], 0)``.

Mutations of the fixed code that also turn the baseline-red tests red: ``starts[-1]`` made
``starts[0]`` in ``turn_question`` (``[streamicl-memories]`` and
``test_the_question_of_a_turn_is_its_last_not_a_memory_before_it``); the one-line rule at the end of
``turn_question`` made ``return line or None`` (``test_a_question_line_followed_by_dialogue_is_not_an_answer_turn``);
the reference check removed from the unit branch (``test_a_judge_unit_that_holds_the_reference_is_kept``,
``([], 4) == ([Message(...)], 0)``); every unit made the whole Add
(``test_dialogue_sent_beside_a_wrong_judge_unit_is_stored``, ``([], 5) == ...``, and
``test_a_reference_sent_after_a_wrong_judge_unit_is_stored``, ``([Message(...)], 0) == ...``);
``self._asked_questions.forget(tenant)`` in ``HostedService.delete_user`` made ``pass``
(``test_deleting_a_user_forgets_what_it_was_asked``, ``assert False``).

Re-anchored, since the fix removed the lines their 2026-10-10 mutations changed:
``test_a_lone_judge_message_is_not_a_unit`` (a judge with no question turn given a unit of its own,
failed ``[] == [Message(...)]``); ``test_a_judge_unit_is_dropped_only_when_its_question_was_asked``
(``and was_asked(question)`` deleted, failed ``[] == [Message(...)]``);
``test_a_wrong_judge_verdict_drops_the_whole_feedback_unit`` (``drop.update(span)`` made ``pass``,
failed ``[Message(...)] == []``). Every case restored, the whole file: 41 passed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from recall_aml.feedback_guard import (
    FEEDBACK_GUARD_ENV,
    AskedQuestions,
    filter_feedback,
    judge_verdict,
    message_verdict,
)
from recall_aml.models import AddRequest, Message, TextContentPart
from tests.test_aml_forget import _Repository as _ServiceRepository, _search, _service, _text

PREFIX = "Streaming online QA feedback:\n"
QUESTION = "Which city did Mara move to after the conference?"
WRONG = "Lisbon"


def _record(**fields) -> str:
    return PREFIX + json.dumps(fields)


def _flat(score, **extra) -> Message:
    return Message(role="user", content=_record(question=QUESTION, predicted_answer=WRONG, score=score, **extra))


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch) -> None:
    monkeypatch.delenv(FEEDBACK_GUARD_ENV, raising=False)


# ---------------------------------------------------------------- the detector


def test_a_flat_record_with_a_failing_score_is_dropped() -> None:
    result = filter_feedback([_flat(0)])
    assert (result.kept, result.dropped) == ([], 1)


def test_a_flat_record_with_a_passing_score_is_kept_and_counted() -> None:
    message = _flat(1)
    result = filter_feedback([message])
    assert (result.kept, result.dropped, result.passed) == ([message], 0, 1)


def test_the_benchmark_result_shape_is_read_with_its_outer_reward() -> None:
    record = {"status": "completed", "reward": 0, "result": {"question": QUESTION, "predicted_answer": WRONG}}
    found = message_verdict(Message(role="user", content=json.dumps(record)))
    assert found is not None and (found.verdict, found.has_reference, found.question) == ("failed", False, QUESTION)
    record["reward"] = 1
    found = message_verdict(Message(role="user", content=json.dumps(record)))
    assert found is not None and found.verdict == "passed"


def test_a_failed_record_that_carries_the_correct_answer_is_kept() -> None:
    """The reference answer is evidence; only a wrong answer with nothing to correct it goes."""
    message = _flat(0, gold_answer="Porto")
    result = filter_feedback([message])
    assert (result.kept, result.dropped) == ([message], 0)


def test_a_wrong_judge_verdict_drops_the_whole_feedback_unit() -> None:
    unit = [
        Message(role="user", content=f"Question: {QUESTION}\nAnswer:"),
        Message(role="assistant", content=WRONG),
        Message(role="tool", content='response: {"label": "WRONG"}\nreasoning: the city differs'),
    ]
    assert judge_verdict(unit[2]) == "failed"
    assert filter_feedback(unit).kept == []
    correct = unit[:2] + [Message(role="tool", content='response: {"label": "CORRECT"}\nreasoning: matches')]
    assert filter_feedback(correct).kept == correct


def test_a_lone_judge_message_is_not_a_unit() -> None:
    lone = [Message(role="tool", content='response: {"label": "WRONG"}')]
    assert filter_feedback(lone).kept == lone


@pytest.mark.parametrize(
    "content",
    [
        "Mara said the conference was in Lisbon and she scored 0 goals in the match.",
        json.dumps({"question": QUESTION, "predicted_answer": WRONG}),
        json.dumps({"question": QUESTION, "score": 0}),
        json.dumps({"predicted_answer": WRONG, "score": 0}),
        json.dumps({"question": QUESTION, "predicted_answer": WRONG, "score": "maybe"}),
        json.dumps({"question": QUESTION, "predicted_answer": WRONG, "score": 0, "correct": True}),
        "Here is my log.\n" + _record(question=QUESTION, predicted_answer=WRONG, score=0) + "\nThanks.",
        "Line one of a note:\nline two of the note:\n" + json.dumps({"question": QUESTION, "answer": WRONG, "score": 0}),
    ],
    ids=["prose", "no-outcome", "no-answer", "no-question", "unreadable", "contradicts", "embedded", "two-lines"],
)
def test_anything_short_of_a_whole_readable_record_is_stored_unchanged(content) -> None:
    message = Message(role="user", content=content)
    assert filter_feedback([message]).kept == [message]


def test_a_record_beside_an_image_is_not_read() -> None:
    message = Message(
        role="user",
        content=[TextContentPart(type="text", text=_record(question=QUESTION, predicted_answer=WRONG, score=0))],
    )
    found = message_verdict(message)
    assert found is not None and found.verdict == "failed"
    with_image = Message.model_validate({
        "role": "user",
        "content": [
            {"type": "text", "text": _record(question=QUESTION, predicted_answer=WRONG, score=0)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _PNG}},
        ],
    })
    assert message_verdict(with_image) is None


_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


# ---------------------------------------------------------------- the service, C9 as served

ORDINARY = "Mara told me the fountain in the old square was restored this spring."


def _add(service, user: str, messages: list[Message], request_id: str = "r1"):
    return asyncio.run(service.add(AddRequest(request_id=request_id, user_id=user, session_id=f"{user}:s1",
                                              messages=messages)))


def _stored(repository) -> list[str]:
    return sorted(chunk.text for rows in repository.chunks.values() for chunk in rows.values())


def test_the_guard_is_off_unless_set(monkeypatch) -> None:
    service, repository = _service()
    assert service.feedback_guard is False
    _add(service, "fg-off", [_flat(0)])
    assert any(WRONG in text for text in _stored(repository))


def _asked(service, user: str, query: str = QUESTION) -> None:
    """A tenant with something in it, searched with ``query``, as the platform would before feedback."""
    _add(service, user, [Message(role="user", content=ORDINARY)], request_id="seed")
    _search(service, user, query)


def test_the_guard_drops_the_failed_record_and_stores_the_rest(monkeypatch) -> None:
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    service, repository = _service()
    _asked(service, "fg-on")
    _add(service, "fg-on", [Message(role="user", content="Good news about the square."), _flat(0)])
    stored = _stored(repository)
    assert stored and all(WRONG not in text for text in stored)
    assert any("Good news about the square." in text for text in stored)
    assert all(WRONG not in _text(item) for item in _search(service, "fg-on", QUESTION).data)


def test_an_add_of_only_failed_records_is_durable_and_empty(monkeypatch) -> None:
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    service, repository = _service()
    _asked(service, "fg-empty")
    before = _stored(repository)
    first = _add(service, "fg-empty", [_flat(0)])
    again = _add(service, "fg-empty", [_flat(0)])
    assert (first.raw_count, first.compiled_count) == (0, 0)
    assert again == first
    assert _stored(repository) == before


def test_a_failed_record_about_a_question_never_searched_is_stored(monkeypatch) -> None:
    """Outside Streaming a tenant is written before it is searched, so the guard cannot act there."""
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    service, repository = _service()
    _asked(service, "fg-unasked", "What did Mara order at the cafe on Friday?")
    _add(service, "fg-unasked", [_flat(0)])
    assert any(WRONG in text for text in _stored(repository))


def test_a_query_that_carries_the_question_counts_as_asking_it(monkeypatch) -> None:
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    service, repository = _service()
    _asked(service, "fg-options", QUESTION + "\nOptions: (a) Lisbon (b) Porto (c) Madrid")
    _add(service, "fg-options", [_flat(0)])
    assert all(WRONG not in text for text in _stored(repository))


def test_a_restart_forgets_what_was_asked_and_stores_everything(monkeypatch) -> None:
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    service, repository = _service()
    _asked(service, "fg-restart")
    restarted, _ = _service(repository)
    _add(restarted, "fg-restart", [_flat(0)])
    assert any(WRONG in text for text in _stored(repository))


def test_searches_made_while_the_guard_is_off_are_not_remembered(monkeypatch) -> None:
    service, repository = _service()
    _asked(service, "fg-late")
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    _add(service, "fg-late", [_flat(0)])
    assert any(WRONG in text for text in _stored(repository))


def test_a_judge_unit_is_dropped_only_when_its_question_was_asked() -> None:
    unit = [
        Message(role="user", content=f"Question: {QUESTION}\nAnswer:"),
        Message(role="assistant", content=WRONG),
        Message(role="tool", content='response: {"label": "WRONG"}'),
    ]
    assert filter_feedback(unit, asked=lambda q: q == QUESTION).kept == []
    assert filter_feedback(unit, asked=lambda q: False).kept == unit


def test_short_questions_must_match_exactly() -> None:
    asked = AskedQuestions()
    asked.remember("t", "Where is Mara now and who is she with?")
    assert asked.asked("t", "  where is MARA now and who is she with? ")
    assert not asked.asked("t", "Mara now")
    assert not asked.asked("other", "Where is Mara now and who is she with?")


def test_with_no_feedback_records_the_guard_stores_exactly_what_the_off_path_stores(monkeypatch) -> None:
    """Near-records included, and a real failed record about a question the tenant was never asked."""
    messages = [
        Message(role="user", content=ORDINARY),
        Message(role="assistant", content="Good to hear."),
        Message(role="user", content=json.dumps({"question": QUESTION, "predicted_answer": WRONG, "score": "n/a"})),
        Message(role="user", content="Here is my log.\n" + _record(question=QUESTION, predicted_answer=WRONG, score=0)),
        _flat(0),
    ]
    off_service, off_repository = _service()
    off = _add(off_service, "fg-same", messages)
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    on_service, on_repository = _service()
    on = _add(on_service, "fg-same", messages)
    assert on == off
    assert {t: {i: c.text for i, c in rows.items()} for t, rows in on_repository.chunks.items()} == {
        t: {i: c.text for i, c in rows.items()} for t, rows in off_repository.chunks.items()
    }


def test_a_bad_value_stops_service_startup(monkeypatch) -> None:
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "yes")
    with pytest.raises(ValueError, match=FEEDBACK_GUARD_ENV):
        _service()


# ---------------------------------------------------------------- 2026-10-20: false drops (U-44, U-45, U-46)

OTHER = "What did Mara order at the cafe on Friday afternoon?"
JUDGE_WRONG = 'response: {"label": "WRONG"}\nreasoning: the city differs'
JUDGE_RIGHT = 'response: {"label": "CORRECT"}\nreasoning: matches'
AMB_SYSTEM = "You are an intelligent memory assistant. Prioritize the most recent memory."


def _asked_only(*questions: str):
    """The service's own gate, for one tenant searched with exactly ``questions``."""
    asked = AskedQuestions()
    for question in questions:
        asked.remember("t", question)
    return lambda question: asked.asked("t", question)


def _turn(question: str = QUESTION, memories: str = "") -> Message:
    return Message(role="user", content=f"{memories}Question: {question}\nAnswer:")


def test_a_mem0_record_with_the_gold_under_answer_is_kept() -> None:
    """U-45: beside ``response``, ``answer`` is mem0's gold answer, so the record is evidence."""
    message = Message(role="user", content=json.dumps(
        {"question": QUESTION, "answer": "Porto", "response": WRONG, "category": 2, "llm_score": 0}
    ))
    result = filter_feedback([message], asked=lambda q: True)
    assert (result.kept, result.dropped) == ([message], 0)


@pytest.mark.parametrize(
    "unit",
    [
        [_flat(0)],
        [Message(role="user", content=_record(question=QUESTION, answer=WRONG, score=0))],
        [Message(role="user", content=json.dumps(
            {"status": "completed", "reward": 0, "result": {"question": QUESTION, "predicted_answer": WRONG}}
        ))],
        [_turn(), Message(role="assistant", content=WRONG), Message(role="tool", content=JUDGE_WRONG)],
        [
            Message(role="system", content=AMB_SYSTEM),
            _turn(memories="Memories:\n\n--- Session 3 started at 1:56 pm on 8 May, 2023 ---\nMara: I moved.\n\n"),
            Message(role="assistant", content=WRONG),
            Message(role="tool", content=JUDGE_WRONG),
        ],
        [
            _turn(memories=f"Memories:\n\nQuestion: {OTHER}\nAnswer: a flat white\n\n"),
            Message(role="assistant", content=WRONG),
            Message(role="tool", content=JUDGE_WRONG),
        ],
        [
            Message(role="user", content=f"Question: {QUESTION}\nAnswer: {WRONG}"),
            Message(role="tool", content=JUDGE_WRONG),
        ],
        [
            Message(role="user", content=f"Question:\n{QUESTION}\nAnswer:"),
            Message(role="assistant", content=WRONG),
            Message(role="tool", content=JUDGE_WRONG),
        ],
    ],
    ids=["st1c-flat", "flat-answer-only", "amb-result", "amb-judge-unit", "amb-system-and-memories",
         "streamicl-memories", "answer-inline", "question-on-next-line"],
)
def test_the_feedback_shapes_the_guard_drops_stay_dropped(unit) -> None:
    """Every shape the guard was built for, about a question asked earlier, is still dropped whole."""
    result = filter_feedback(unit, asked=_asked_only(QUESTION))
    assert (result.kept, result.dropped) == ([], len(unit))


def test_dialogue_sent_beside_a_wrong_judge_unit_is_stored() -> None:
    """U-46: the unit is the span from its question turn to its judge, not the whole Add."""
    dialogue = [
        Message(role="user", content="Mara: I finally booked the flights to Porto."),
        Message(role="assistant", content="Lovely, when do you leave?"),
    ]
    unit = [_turn(), Message(role="assistant", content=WRONG), Message(role="tool", content=JUDGE_WRONG)]
    result = filter_feedback(dialogue + unit, asked=_asked_only(QUESTION))
    assert (result.kept, result.dropped) == (dialogue, 3)


def test_a_question_line_followed_by_dialogue_is_not_an_answer_turn() -> None:
    """U-46: the question was read to the end of the message, and then contained the asked one."""
    messages = [
        Message(role="user", content=f"Question: {QUESTION}\nMara: we should ask Leo about the venue."),
        Message(role="assistant", content="Sure, I will."),
        Message(role="tool", content=JUDGE_WRONG),
    ]
    result = filter_feedback(messages, asked=_asked_only(QUESTION))
    assert (result.kept, result.dropped) == (messages, 0)


_GOLD_RECORD = json.dumps(
    {"reward": 0, "result": {"question": QUESTION, "gold_answer": ["Porto"], "predicted_answer": WRONG}}
)


def test_a_reference_sent_after_a_wrong_judge_unit_is_stored() -> None:
    """U-46: the reference was dropped with the unit; now the unit goes and the reference stays."""
    gold = Message(role="user", content=_GOLD_RECORD)
    unit = [_turn(), Message(role="assistant", content=WRONG), Message(role="tool", content=JUDGE_WRONG)]
    result = filter_feedback(unit + [gold], asked=_asked_only(QUESTION))
    assert (result.kept, result.dropped) == ([gold], 3)


def test_a_judge_unit_that_holds_the_reference_is_kept() -> None:
    """U-46: the reference exemption holds for a judge unit as it does for a lone record."""
    unit = [
        _turn(),
        Message(role="assistant", content=WRONG),
        Message(role="user", content=_GOLD_RECORD),
        Message(role="tool", content=JUDGE_WRONG),
    ]
    result = filter_feedback(unit, asked=_asked_only(QUESTION))
    assert (result.kept, result.dropped) == (unit, 0)


def test_the_question_of_a_turn_is_its_last_not_a_memory_before_it() -> None:
    """A memory that quotes an asked question does not make a new question asked."""
    unit = [
        _turn(OTHER, memories=f"Memories:\n\nQuestion: {QUESTION}\nAnswer: Porto\n\n"),
        Message(role="assistant", content="a cappuccino"),
        Message(role="tool", content=JUDGE_WRONG),
    ]
    result = filter_feedback(unit, asked=_asked_only(QUESTION))
    assert (result.kept, result.dropped) == (unit, 0)


def test_judge_verdicts_that_disagree_in_one_add_drop_nothing() -> None:
    messages = [
        _turn(), Message(role="assistant", content=WRONG), Message(role="tool", content=JUDGE_WRONG),
        _turn(OTHER), Message(role="assistant", content="a flat white"), Message(role="tool", content=JUDGE_RIGHT),
    ]
    result = filter_feedback(messages, asked=_asked_only(QUESTION, OTHER))
    assert (result.kept, result.dropped) == (messages, 0)


class _DeletableRepository(_ServiceRepository):
    def delete_tenant(self, tenant):
        count = sum(len(rows) for key, rows in self.chunks.items() if key.startswith(tenant))
        for key in [key for key in self.chunks if key.startswith(tenant)]:
            self.chunks.pop(key)
        for key in [key for key in self.receipts if key[0] == tenant]:
            self.receipts.pop(key)
        return count


def test_deleting_a_user_forgets_what_it_was_asked(monkeypatch) -> None:
    """U-44: a user created again under a deleted id was never searched, so nothing is dropped."""
    monkeypatch.setenv(FEEDBACK_GUARD_ENV, "1")
    repository = _DeletableRepository()
    service, _ = _service(repository)
    _asked(service, "fg-deleted")
    asyncio.run(service.delete_user("fg-deleted"))
    _add(service, "fg-deleted", [_flat(0)], request_id="after-delete")
    assert any(WRONG in text for text in _stored(repository))
