"""The two preregistered answer arms, and the property that makes them safe to run.

H17 and H21 are registered in `docs/preregistrations/2026-08-20-atm-evidence-allocation-and-selection.md`
with two-sided predictions: each can lose points as well as win them, so `baseline` has to stay a
real control and the envelope has to fail closed rather than throw away a completion.
"""

from __future__ import annotations

import pytest

from benchmarks.atm_full_run import (
    ANSWER_POLICIES,
    BASELINE_SYSTEM,
    DISPOSITION_SYSTEM,
    SELECTION_SYSTEM,
    is_refusal,
    parse_selection,
    system_prompt,
)


def test_baseline_is_the_official_prompt_untouched() -> None:
    """The control must be the official oracle text verbatim, or it is not a control."""
    assert system_prompt("baseline") == BASELINE_SYSTEM
    assert "If the evidence is insufficient, answer 'Unknown'." in BASELINE_SYSTEM
    assert "comma-separated" in BASELINE_SYSTEM


def test_the_arms_are_additive_over_the_same_baseline() -> None:
    assert system_prompt("disposition") == BASELINE_SYSTEM + DISPOSITION_SYSTEM
    assert system_prompt("selection") == BASELINE_SYSTEM + SELECTION_SYSTEM
    assert system_prompt("both") == BASELINE_SYSTEM + DISPOSITION_SYSTEM + SELECTION_SYSTEM


def test_disposition_does_not_forbid_refusing() -> None:
    """H17 shifts the threshold; it must not remove the option, because 14 of 58 measured refusals
    were correct and 23 gold answers are abstentions."""
    assert "Unknown" in DISPOSITION_SYSTEM
    assert system_prompt("disposition").count("Unknown") >= 2


def test_an_unknown_policy_is_refused_rather_than_silently_ignored() -> None:
    with pytest.raises(ValueError, match="unknown answer policy"):
        system_prompt("whatever")
    assert set(ANSWER_POLICIES) == {"baseline", "disposition", "selection", "both", "coverage"}


def test_a_well_formed_envelope_yields_the_answer_and_its_marks() -> None:
    raw = """Here it is:
    {"qualifiers": ["Springfield, Illinois"],
     "items": [{"id": "a", "matches": "yes", "failing_qualifier": null},
               {"id": "b", "matches": "no", "failing_qualifier": "Springfield, Illinois"}],
     "answer": "Temperance Hall"}"""
    answer, diagnostics = parse_selection(raw)
    assert answer == "Temperance Hall"
    assert diagnostics["parse_failed"] is False
    assert diagnostics["matched"] == 1
    assert diagnostics["rejected"] == 1
    assert diagnostics["qualifiers"] == ["Springfield, Illinois"]
    assert diagnostics["items"][1]["failing_qualifier"] == "Springfield, Illinois"


@pytest.mark.parametrize(
    "raw",
    [
        "just prose, no object at all",
        '{"qualifiers": [], "items": [], "answer": ""}',
        '{"qualifiers": [], "items": []}',
        '{"answer": {"nested": "wrong type"}}',
        "[1, 2, 3]",
    ],
)
def test_a_broken_envelope_fails_closed_to_the_raw_completion(raw: str) -> None:
    """Never raise, never return empty: the completion was already paid for, and a malformed
    envelope must cost the diagnostics for one question rather than its answer."""
    answer, diagnostics = parse_selection(raw)
    assert answer == raw
    assert diagnostics["parse_failed"] is True
    assert diagnostics["reason"]


def test_the_parse_failure_carries_a_reason_that_names_the_shape() -> None:
    _, diagnostics = parse_selection("no json here")
    assert "no JSON object" in diagnostics["reason"]
    _, diagnostics = parse_selection('{"answer": ""}')
    assert "answer field" in diagnostics["reason"]


@pytest.mark.parametrize(
    "answer",
    [
        "Unknown",
        "unknown",
        "The available memory does not contain enough information.",
        "There is no information about that.",
        "insufficient information to answer",
    ],
)
def test_refusals_are_recognised_however_they_are_worded(answer: str) -> None:
    assert is_refusal(answer)


@pytest.mark.parametrize("answer", ["37", "£512.30", "Temperance Hall, Union Street"])
def test_a_real_answer_is_not_counted_as_a_refusal(answer: str) -> None:
    assert not is_refusal(answer)


def test_the_refusal_markers_never_reach_the_prompt() -> None:
    """The marker list mirrors the scorer's vocabulary for MEASUREMENT. Feeding it back into the
    system under test would be tuning to the metric, so this pins the boundary."""
    for policy in ANSWER_POLICIES:
        prompt = system_prompt(policy)
        assert "insufficient information" not in prompt
        assert "no evidence" not in prompt


def test_the_output_ceiling_is_not_retried() -> None:
    """Found while wiring H21: the envelope adds output tokens, so the selection arm meets this
    error often, and the loop used to pay for the identical request four times."""
    import benchmarks.atm_full_run as runner

    calls = {"n": 0}

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"finish_reason": "length", "message": {"content": "cut off"}}]}

    def _post(*args: object, **kwargs: object) -> _Response:
        calls["n"] += 1
        return _Response()

    original = runner.requests.post
    runner.requests.post = _post
    try:
        with pytest.raises(runner.CompletionTruncated, match="token output ceiling"):
            runner.generate_answer(
                question="q",
                qtype=None,
                evidence="e",
                model="m",
                base_url="http://example.invalid",
                api_key="k",
                reasoning_effort="medium",
                max_output_tokens=1024,
                max_attempts=4,
            )
    finally:
        runner.requests.post = original
    assert calls["n"] == 1, f"paid for the same doomed request {calls['n']} times"


def test_a_partial_answer_is_not_mistaken_for_a_refusal() -> None:
    """A bare 'does not contain' marker would flag this as a refusal. Over-counting refusals
    corrupts the exact number H17 and H21 must be judged against."""
    assert not is_refusal("The email does not contain a price, but the total is 50 pounds.")
    assert is_refusal("The available memory does not contain enough information.")


def test_a_truncated_envelope_rescues_the_answer_rather_than_submitting_the_blob() -> None:
    """The worst case this guards: on a list question the scorer harvests ids from free text, so
    submitting the raw blob would predict every id in `items` against a usually singleton gold."""
    raw = (
        '{"qualifiers": ["Springfield, Illinois"], '
        '"items": [{"id": "20241116_120000", "matches": "no"}, '
        '{"id": "20241117_100000", "matches": "no"}], '
        '"answer": "Temperance Hall"'
    )
    answer, diagnostics = parse_selection(raw)
    assert answer == "Temperance Hall"
    assert diagnostics["parse_failed"] is True
    assert diagnostics["rescued_answer"] is True
    assert "20241116_120000" not in answer


def test_the_rescue_gives_up_honestly_when_there_is_no_answer_field() -> None:
    raw = '{"qualifiers": ["a"], "items": [{"id": "x", "matches": "no"}'
    answer, diagnostics = parse_selection(raw)
    assert answer == raw
    assert diagnostics["rescued_answer"] is False


def test_no_index_is_used_when_every_retrieval_row_is_checkpointed(tmp_path) -> None:
    """An answer-arm comparison replays one run's retrieval into every arm. Building the store,
    the Voyage embedder and the reranker for that is waste, and it couples the replay to the
    schema version of a shared database: the first attempt died on SchemaTooNew for a migration a
    newer branch had applied, on a run that was never going to issue a query."""
    import benchmarks.atm_full_run as runner

    with runner._no_index() as (retriever, embedder, chunks, index_ms):
        assert retriever is None, "a stub that answers queries would hide the mistake"
        assert embedder is None
        assert chunks == []
        assert index_ms == 0


def test_a_truncated_question_does_not_kill_the_run(tmp_path, monkeypatch) -> None:
    """Arm C died on its LAST question and took arm D with it. One question that will not
    terminate must cost one question, not the other 299 and not the arms queued behind it."""
    import benchmarks.atm_full_run as runner

    calls = {"n": 0}

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            calls["n"] += 1
            # The second question is the one that will not terminate.
            if calls["n"] == 2:
                return {"choices": [{"finish_reason": "length", "message": {"content": "..."}}]}
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": "an answer"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "model": "m",
            }

    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: _Response())
    ok = 0
    truncated = 0
    for _ in range(3):
        try:
            runner.generate_answer(
                question="q", qtype=None, evidence="e", model="m",
                base_url="http://example.invalid", api_key="k",
                reasoning_effort="medium", max_output_tokens=16384, max_attempts=4,
            )
            ok += 1
        except runner.CompletionTruncated:
            truncated += 1
    assert ok == 2 and truncated == 1, f"expected 2 answers and 1 truncation, got {ok} and {truncated}"


def test_coverage_is_additive_over_the_baseline_like_every_other_arm() -> None:
    from benchmarks.atm_full_run import COVERAGE_SYSTEM

    assert system_prompt("coverage") == BASELINE_SYSTEM + COVERAGE_SYSTEM
    assert "coverage" in ANSWER_POLICIES


def test_coverage_scopes_itself_away_from_the_types_it_must_not_touch() -> None:
    """number is exact multiset equality and list_recall is Jaccard over ids. Both are scored
    without a judge, so they are the control: if this instruction reaches them the arm is invalid
    whatever QS does."""
    from benchmarks.atm_full_run import COVERAGE_SYSTEM

    assert "single value" in COVERAGE_SYSTEM
    assert "recall or list" in COVERAGE_SYSTEM


def test_coverage_never_quotes_the_rubric_it_was_derived_from() -> None:
    """The diagnosis came from the judge's published criterion. The INSTRUCTION must name only the
    question and the evidence, or this stops being answer quality and becomes metric fitting."""
    from benchmarks.atm_full_run import COVERAGE_SYSTEM

    lowered = COVERAGE_SYSTEM.lower()
    for forbidden in ("ground truth", "accuracy", "judge", "rubric", "score", "unknown"):
        assert forbidden not in lowered, f"{forbidden!r} leaked into the serving prompt"
