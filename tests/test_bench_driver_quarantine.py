"""One unusable completion must cost one unit of work, not a whole run's paid work.

`benchmarks/llm.py` now RAISES where it used to return `""` (ddfc618, 8021aaf), which is right:
an empty answer scored as the system's answer is a wrong published number. But raising only helps
where somebody catches it, and the bug review of that commit found the quarantine existed on
exactly one of five paths:

    benchmarks/beam/run.py:_run_pool      per question, with a terminal breaker   ✅ already
    benchmarks/beam/run.py:judge_answer   per nugget                              ⛔ bypassed
    benchmarks/run.py:run_arm             per question                            ⛔ none
    benchmarks/rejudge.py:rejudge_outcomes  per record                            ⛔ none
    benchmarks/judge_quality.py:run_probe   per item                              ⛔ none

Each unguarded one loses more than the item that failed, because all three write their output only
after their whole loop finishes. On LOCOMO the loss window is a whole conversation (up to ~200
already-billed questions, since `_append_records` runs after `run_arm` returns).

⚠️ **QUARANTINE ALONE WOULD BE WORSE THAN THE DISEASE, AND THAT IS MOST OF THIS FILE.**
`benchmarks/beam/run.py`'s own docstring records what a blanket per-item catch produces: an empty
OpenRouter balance became "a tidy `n: 599` summary and exit 0", 101 questions "failed", the
artifact looking publishable and nothing saying the account was empty. "Infrastructure wearing a
verdict's clothes." `benchmarks/run.py` has NO terminal detection at all, so adding a bare
`except Exception` there would have introduced exactly that failure into LOCOMO. Every driver
below therefore pairs the quarantine with a terminal check, and every one of those is tested.

🔑 The terminal check itself moved to `benchmarks/llm.py:is_terminal` and is now TYPE-and-STATUS
based rather than a substring match. Two reasons, both measured:

  1. It is needed in four places now, and beam's copy was private to beam.
  2. The substring version had false positives on OUR OWN messages, in both directions of the same
     bug 8021aaf fixed for `_is_transient`: `max_tokens=1402` and a provider-chosen
     `finish_reason` both made `_is_terminal` return True, aborting a whole run over a ceiling or a
     string the provider picked. A numeric status is now decisive, our own exception types are
     never terminal, and the text markers survive only as the last resort for a transport that
     carried no status at all. That is the same precedence `recall.embeddings._is_transient`
     already uses, for the same reason.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from benchmarks import run as run_module
from benchmarks.beam import run as beam
from benchmarks.judge_quality import (
    REWORDED,
    SWAPPED,
    VERBATIM,
    ProbeItem,
    measure_model,
    run_probe,
)
from benchmarks.llm import (
    CompletionTruncated,
    EmptyCompletion,
    NoCompletionChoices,
    is_terminal,
)
from benchmarks.rejudge import rejudge_document, rejudge_outcomes
from benchmarks.run import run_arm


class _Status(Exception):
    """A provider error carrying a numeric status, the way the OpenAI SDK's errors do.

    ⚠️ The SHAPE here is not guessed, because a guard built on an unverified shape is a guard that
    cannot fire. Measured against openai 2.48.0:

        AuthenticationError.status_code      401   (a CLASS attribute)
        PermissionDeniedError.status_code    403   (class attribute)
        RateLimitError.status_code           429   (class attribute)
        APIStatusError                       no class attribute; the INSTANCE carries
                                             `status_code` from the response
        APIConnectionError / APITimeoutError no status at all

    That last pair is why the text markers survive as a fallback, and `APIStatusError` is the one
    that matters most here: OpenRouter's 402 has no dedicated class, so out-of-credits arrives as a
    generic `APIStatusError` whose status lives only on the instance. An instance attribute is
    exactly what this double models. Verified end to end in `test_real_openai_errors_classify` for
    anyone running with the `bench` extra installed.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


#: The run stamp `main` bakes into the artifact filename. Matches `tests/test_bench_run.py`, whose
#: `_STAMP_1CONV` this arm imports rather than restates.
_ARTIFACT_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _reject_constant(name: str) -> object:
    """Make `json.loads` refuse NaN/Infinity instead of silently accepting Python's extension."""
    raise ValueError(f"non-JSON constant in output: {name}")


def _out_of_credits() -> _Status:
    return _Status("Error code: 402 - requires more credits", 402)


# --------------------------------------------------------------------------------------------
# 1. The shared terminal classifier.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 402])
def test_an_auth_or_billing_status_is_terminal(status: int) -> None:
    """These fail every remaining request identically. Dropping them one at a time is how a dead
    key becomes a publishable-looking artifact."""
    assert is_terminal(_Status("nope", status))


def test_a_rate_limit_is_not_terminal() -> None:
    """Guards the guard. 429 is the single most common failure in a long run, and calling it
    terminal would abort on the one thing the retry layer exists to absorb."""
    assert not is_terminal(_Status("slow down", 429))


def test_a_numeric_status_beats_a_marker_in_the_text() -> None:
    """The precedence that kills the false positives. A 429 whose body happens to quote "402"
    stays a rate limit, because the transport STATED the status and a substring did not."""
    assert not is_terminal(_Status("retry later; you used 402 tokens", 429))


def test_a_marker_still_fires_when_no_status_was_carried() -> None:
    """Kept as a last resort, not deleted: `openai.APIConnectionError` carries no status at all,
    and there the text is the only evidence available."""
    assert is_terminal(RuntimeError("insufficient_quota: add credits"))


def test_a_bare_status_digit_is_read_only_when_no_status_was_carried() -> None:
    """The `_AMBIGUOUS_MARKERS` fallback, which nothing exercised until a mutation said so.

    The arm above uses an ACCOUNT phrase, which now short-circuits ahead of the status, so deleting
    the digit fallback entirely left the suite green. These digits are the whole reason the
    fallback is split in two: they are the ones that produced the false positives on our own
    messages, so they may only speak when the transport said nothing at all.
    """
    assert is_terminal(RuntimeError("Error code: 402 - payment required")), (
        "no status was carried, so the digits are the only evidence there is"
    )
    assert is_terminal(RuntimeError("HTTP 401 unauthorized"))
    assert not is_terminal(RuntimeError("Error code: 500 - upstream blew up"))


def test_exhausted_quota_is_terminal_even_though_it_arrives_as_a_rate_limit() -> None:
    """⛔ THE REGRESSION A REVIEW CAUGHT, and the most expensive one in this change.

    OpenAI signals exhausted credit as **HTTP 429 with `code: insufficient_quota`**. The first
    version of `is_terminal` made a numeric status decisive over ALL markers, so this returned
    False and the marker became dead code for its own canonical carrier. It is worse than a plain
    miss: 429 is TRANSIENT under `_is_transient`, so every question would pay four retries and
    then be dropped one at a time — the exact "tidy `n: 599`, exit 0" artifact the quarantine was
    added to prevent, produced by the guard meant to prevent it.

    Measured: the old substring version returned True here, so this was a regression and not a
    pre-existing gap.
    """
    assert is_terminal(_Status("Error code: 429 - You exceeded your quota (insufficient_quota)", 429))


def test_a_bare_rate_limit_is_still_not_terminal() -> None:
    """Guards the guard above. The account phrase is what promotes a 429, not the 429 itself."""
    assert not is_terminal(_Status("Error code: 429 - Rate limit reached, please retry", 429))


@pytest.mark.parametrize(
    "exc",
    [
        EmptyCompletion("no text (finish_reason=unrecognised).", finish_reason="upstream 402"),
        CompletionTruncated("completion hit max_tokens=1402 and was cut off."),
        NoCompletionChoices("the provider returned a 200 with an empty `choices` list"),
    ],
)
def test_our_own_failures_are_never_terminal(exc: Exception) -> None:
    """⛔ The bug this classifier was rewritten to kill. Under the substring version,
    `max_tokens=1402` and a provider-chosen `finish_reason` of "upstream 402" both returned True,
    so one truncation at an unlucky ceiling, or one string the PROVIDER picked, aborted an entire
    run. None of these three says anything about the account."""
    assert not is_terminal(exc)


def test_real_openai_errors_classify() -> None:
    """The stub above against the genuine article, for whoever has the `bench` extra.

    ⚠️ SUPPLEMENT, NOT THE GUARD. `openai` ships in the `bench` extra, and pyproject says that
    extra is "deliberately excluded from `dev`", so this arm SKIPS in CI and the `_Status` arms are
    what actually run there. It is here because the one thing a hand-rolled double cannot prove is
    that it resembles the SDK, and a silent skip must not be mistaken for coverage.

    The 402 case is the one worth the import: it has no dedicated exception class, so it arrives as
    a bare `APIStatusError` carrying its status only on the instance. If that ever stops being
    true, out-of-credits stops being terminal and a dead balance goes back to being dropped one
    question at a time.
    """
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    def status_error(code: int) -> Exception:
        return openai.APIStatusError(
            f"Error code: {code}", response=httpx.Response(code, request=request), body=None
        )

    assert is_terminal(status_error(402)), "out of credits must stop the run"
    assert is_terminal(status_error(401))
    assert not is_terminal(status_error(429)), "a rate limit is what the retry layer absorbs"
    assert not is_terminal(status_error(500))
    assert is_terminal(
        openai.AuthenticationError(
            "bad key", response=httpx.Response(401, request=request), body=None
        )
    )
    # No status at all, and no marker in the text: not terminal, and it must not crash reaching
    # for an attribute that is absent.
    assert not is_terminal(openai.APIConnectionError(request=request))


def test_beam_still_exposes_its_own_name_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`benchmarks/beam/run.py:_is_terminal` is called from `_run_pool` in two places. It must be
    the SAME classifier, not a second copy that drifts."""
    assert beam._is_terminal(_out_of_credits())
    assert not beam._is_terminal(CompletionTruncated("hit max_tokens=1402"))


# --------------------------------------------------------------------------------------------
# 2. BEAM's judge: one nugget, not one question.
# --------------------------------------------------------------------------------------------


def _question() -> Any:
    return beam.Question(
        question_id="1M_0_q0_test",
        chat_size="1M",
        conversation_idx=0,
        question_type="information_extraction",
        question="Where did the user study psychology?",
        rubric=["States the user studied at Leiden", "Mentions the year 2019"],
        difficulty="easy",
    )


def test_a_failed_nugget_judgment_does_not_discard_the_question() -> None:
    """`judge_answer` already degrades an UNPARSEABLE reply to NaN, counts it in `judge_errors`,
    and averages over the nuggets that did score (`_parse_judge("")` returns NaN, verified). An
    empty completion used to land there and now raises, so it skipped that mechanism entirely and
    took the question with it — including the generated answer, which the module calls the
    expensive half."""
    calls = {"n": 0}

    def judge(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmptyCompletion("no text (finish_reason=content_filter).")
        return '{"score": 1.0, "reason": "yes"}'

    mean, nuggets, errors = beam.judge_answer(_question(), "an answer", judge)

    assert errors == 1
    assert len(nuggets) == 2, "both nuggets are still reported, one of them as a failure"
    assert nuggets[0]["score"] is None, "None, not NaN: the row has to stay valid JSON"
    assert mean == 1.0, "the mean is over the nugget that scored, not over a zero"


def test_a_failed_nugget_names_the_cause_in_its_reason() -> None:
    """The NaN row is what an operator reads to tell a broken judge from a stingy one."""

    def judge(system: str, user: str) -> str:
        raise EmptyCompletion("no text (finish_reason=content_filter).")

    _, nuggets, errors = beam.judge_answer(_question(), "an answer", judge)

    assert errors == 2
    assert "EmptyCompletion" in nuggets[0]["reason"]


def test_a_degraded_nugget_row_is_still_valid_json() -> None:
    """⛔ `json.dumps` emits a BARE `NaN` token, which is not valid JSON: jq, `JSON.parse` and
    Postgres jsonb all reject it. This module already knows that — the `--no-judge` path at
    `score_question` uses None for exactly this reason and says so — but `nugget_scores` carried
    the raw NaN, and one failed nugget out of two is enough to poison the row.

    The hazard pre-dates this work via `_parse_judge`, but the quarantine created a far more
    likely route: before it, an exception discarded the question and NO row was written, so
    keeping the paid work meant nothing if what got kept was unparseable.
    """
    def judge(system: str, user: str) -> str:
        raise EmptyCompletion("no text.")

    row = beam.score_question(_question(), [{"text": "m"}], lambda s, u: "an answer", judge,
                              cutoff=5)
    line = json.dumps(row, ensure_ascii=False)

    # `allow_nan=False` is the check the module's own comment names as the proof.
    json.dumps(row, allow_nan=False)
    json.loads(line, parse_constant=_reject_constant)
    assert row["judgment"] == "ERROR", "the state is still carried, just not as a NaN token"


def test_a_partly_degraded_question_still_averages_over_what_scored() -> None:
    """Guards the fix above. Swapping NaN for None must not break `usable`, which filters on
    `score == score` — and `None == None` is True, so a naive swap would feed None to
    `statistics.mean` and crash. The filter has to test for a real number, not just for non-NaN."""
    calls = {"n": 0}

    def judge(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmptyCompletion("no text.")
        return '{"score": 1.0, "reason": "yes"}'

    mean, nuggets, errors = beam.judge_answer(_question(), "an answer", judge)

    assert mean == 1.0 and errors == 1
    assert nuggets[0]["score"] is None
    json.dumps(nuggets, allow_nan=False)


def test_a_terminal_error_still_aborts_from_inside_the_nugget_loop() -> None:
    """The carve-out. Degrading a dead account to NaN would fill an artifact with unscored rows
    and report it as a completed run."""

    def judge(system: str, user: str) -> str:
        raise _out_of_credits()

    with pytest.raises(_Status):
        beam.judge_answer(_question(), "an answer", judge)


# --------------------------------------------------------------------------------------------
# 3. LOCOMO: one question, not one conversation.
# --------------------------------------------------------------------------------------------


class _Sys:
    name = "fake"

    def ingest(self, conversation: dict[str, Any]) -> None:
        pass

    def retrieve(self, question: str) -> str:
        return "a memory"


def _questions() -> list[dict[str, Any]]:
    return [
        {"question_id": "a", "category": "c", "adversarial": False,
         "question": "poison?", "answer": "x"},
        {"question_id": "b", "category": "c", "adversarial": False,
         "question": "fine?", "answer": "500"},
    ]


def _completer_failing_on(needle: str, exc: Exception):
    def completer(system: str, user: str) -> str:
        if needle in user:
            raise exc
        return "YES" if "Correct?" in user else "500"
    return completer


def test_one_failed_question_does_not_discard_its_conversation() -> None:
    """`run_arm` built the whole conversation's outcomes in one list comprehension, and
    `_append_records` runs only AFTER it returns, so a refusal on question 150 of 200 threw away
    149 already-billed answers and wrote nothing."""
    completer = _completer_failing_on("poison", EmptyCompletion("no text."))

    outcomes, _agg, dropped = run_arm(_Sys(), completer, _questions())

    assert [o.question_id for o in outcomes] == ["b"]
    assert dropped == ["a"]


def test_a_clean_run_reports_nothing_dropped() -> None:
    """Guards the guard: the new keys must be present and empty on the happy path, or a consumer
    cannot tell "no failures" from "this field did not exist yet"."""
    outcomes, _agg, dropped = run_arm(
        _Sys(), lambda s, u: "YES" if "Correct?" in u else "500", _questions()
    )

    assert len(outcomes) == 2
    assert dropped == []


def test_a_systematic_non_terminal_fault_aborts_instead_of_publishing_a_shell() -> None:
    """⛔ The second regression a review caught. A wrong `--model` (404), or a 400 on every
    request, is not terminal by ANY definition of the word — so with only the terminal check every
    question was dropped, `outcomes` came back empty, `aggregate([])` produced rate-None blocks,
    the cost contract passed and the artifact was written with exit 0. Before the quarantine
    existed, all three of those aborted the run and published nothing.

    `CONSECUTIVE_FAILURE_LIMIT` is the floor, and the argument is `benchmarks/mtrag/generation.py`'s
    verbatim: one question can fail on its own merits, five in a row is the run being broken.
    """
    calls = {"n": 0}

    def completer(system: str, user: str) -> str:
        calls["n"] += 1
        raise _Status("Error code: 404 - no such model", 404)

    questions = [
        {"question_id": str(i), "category": "c", "adversarial": False,
         "question": "q?", "answer": "a"}
        for i in range(20)
    ]

    with pytest.raises(run_module.RunAborted, match="in a row"):
        run_arm(_Sys(), completer, questions)

    assert calls["n"] == run_module.CONSECUTIVE_FAILURE_LIMIT, (
        "it must stop AT the limit, not run the remaining 15 questions first"
    )


def test_scattered_failures_do_not_trip_the_breaker() -> None:
    """Guards the guard: the breaker counts CONSECUTIVE failures, so an ordinary flaky question
    between successes must not end a run that is working.

    ⚠️ More failures than `CONSECUTIVE_FAILURE_LIMIT`, deliberately. An earlier version of this arm
    interleaved only four, which is fewer than the limit — so deleting the `consecutive = 0` reset
    (making the counter cumulative, and the breaker fire on the 5th failure ANYWHERE in a run) left
    the suite green. The alternation has to outlast the limit for the reset to be observable.
    """

    def completer(system: str, user: str) -> str:
        if "poison" in user:
            raise EmptyCompletion("no text.")
        return "YES" if "Correct?" in user else "500"

    questions = [
        {"question_id": f"bad{i}", "category": "c", "adversarial": False, "question": "poison?",
         "answer": "x"}
        if i % 2 == 0 else
        {"question_id": f"ok{i}", "category": "c", "adversarial": False, "question": "ok?",
         "answer": "500"}
        for i in range(2 * (run_module.CONSECUTIVE_FAILURE_LIMIT + 3))
    ]

    outcomes, _agg, dropped = run_arm(_Sys(), completer, questions)

    assert len(dropped) > run_module.CONSECUTIVE_FAILURE_LIMIT, (
        "the run must survive more total failures than the limit, as long as none are consecutive"
    )
    assert len(outcomes) == len(dropped)


def test_the_published_artifact_names_the_questions_it_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⛔ END TO END, THROUGH `main`, AND THAT IS THE POINT OF THIS ARM.

    The drop list first rode `run_arm`'s aggregate dict — and both production callers unpack that
    as `_`, and `main` recomputes `aggregate(outcomes)` from the pooled outcomes anyway, so the
    field reached no artifact while `run_arm`'s docstring claimed the opposite. Every test at the
    time read the returned value directly, so all of them were green.

    A quarantine that shrinks `n` without recording why is a quarantine that publishes a short run
    as a small one. This asserts the field in the FILE, which is the only place a reader looks.
    """
    from benchmarks.systems import RecallSystem
    from tests.test_bench_run import _FAKE_KEY, _STAMP_1CONV, _write_fixture

    def _retrieve(self: RecallSystem, question: str) -> str:
        return "ctx"

    def _ingest(self: RecallSystem, conversation: dict[str, Any]) -> None:
        return None

    def _no_preflight(self: RecallSystem, questions: list[str], *, sample: int,
                      metric_class: str, allow_inert: bool) -> list[dict[str, Any]]:
        return []

    def _build(arm: str, model: str, openrouter_key: str, k: int, run_id: str,
               embedder: str = "fastembed", **_extra: object) -> Any:
        return RecallSystem("postgresql://x/y", embedder_name="hashing", k=k)

    def _complete(self: Any, system: str, user: str) -> str:
        if "adoption" in user:
            raise EmptyCompletion("no text (finish_reason=content_filter).")
        return "YES" if "Correct?" in user else "an answer"

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(RecallSystem, "ingest", _ingest)
    monkeypatch.setattr(RecallSystem, "retrieve", _retrieve)
    monkeypatch.setattr(RecallSystem, "ablation_preflight", _no_preflight)
    monkeypatch.setattr(run_module, "_build_system", _build)
    monkeypatch.setattr(run_module.OpenRouterLLM, "complete", _complete)

    out = tmp_path / "results"
    code = run_module.main(
        ["--arm", "recall", "--data", str(_write_fixture(tmp_path)),
         "--conversations", "1", "--out", str(out)],
        now=_ARTIFACT_NOW,
    )
    assert code == 0

    payload = json.loads((out / f"{_STAMP_1CONV}.json").read_text(encoding="utf-8"))
    assert payload["dropped"]["n"] == 1
    assert payload["dropped"]["question_ids"] == ["conv-a:0"]
    assert "conv-a:0" not in {o["question_id"] for o in payload["outcomes"]}, (
        "the dropped question is genuinely absent from n, which is why it must be named"
    )


def test_a_terminal_error_aborts_the_locomo_run() -> None:
    """⚠️ LOCOMO had NO terminal detection before this change, so a bare per-question catch would
    have turned an empty balance into a short artifact and exit 0 — the exact failure BEAM's
    `_run_pool` docstring was written about."""
    completer = _completer_failing_on("poison", _out_of_credits())

    with pytest.raises(run_module.RunAborted):
        run_arm(_Sys(), completer, _questions())


# --------------------------------------------------------------------------------------------
# 4. rejudge: one record, and never a silent agreement.
# --------------------------------------------------------------------------------------------


def _record(qid: str, correct: bool = True) -> dict[str, Any]:
    return {
        "question_id": qid, "category": "c", "is_adversarial": False, "adversarial": False,
        "context": "", "answer": "an answer", "abstained": False, "correct": correct,
        "question": "q?", "gold": "g",
    }


def test_a_failed_rejudgment_keeps_the_old_verdict_and_is_not_counted_as_agreement() -> None:
    """⛔ The trap specific to this driver. Falling back to the old verdict and saying nothing
    would count as agreement, and `agreement_rate` is the number this tool exists to produce: a
    broken judge would read as a judge that agrees with the baseline."""
    calls = {"n": 0}

    def judge(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmptyCompletion("no text.")
        return "YES"

    records, report = rejudge_outcomes([_record("a"), _record("b")], judge)

    assert report["n_failed"] == 1
    assert report["failed_question_ids"] == ["a"]
    assert report["n_judged"] == 1, "the failed record is not judged, so it is not in the base"
    assert records[0]["correct"] is True, "untouched, not flipped"
    assert records[0]["rejudge_failed"] is True


def test_a_mixed_rejudge_aggregate_says_so_beside_itself() -> None:
    """⛔ Keeping the baseline verdict is right, but it makes `payload["aggregate"]` a MIXTURE of
    two judges — and that block sits beside `original_aggregate`, which is the pair a reader diffs
    to attribute a change to the judge. `n_failed` lives two levels down under
    `rejudge.agreement`, so nothing stopped the headline being read as the new judge's alone.
    """
    calls = {"n": 0}

    def judge(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmptyCompletion("no text.")
        return "NO"

    doc = {"outcomes": [_record("a", correct=True), _record("b", correct=True)],
           "aggregate": {}, "config": {"model": "old/judge"}}
    payload = rejudge_document(doc, judge, "new/judge", "src.json")

    assert payload["aggregate_mixed_verdicts"]["n"] == 1
    assert payload["aggregate_mixed_verdicts"]["question_ids"] == ["a"]
    # The mixture itself: 'a' kept the baseline True, 'b' was re-judged to False.
    assert payload["aggregate"]["answerable_accuracy"]["n"] == 2
    assert payload["rejudge"]["agreement"]["n_judged"] == 1


def test_a_clean_rejudge_reports_an_unmixed_aggregate() -> None:
    """Guards the guard: present and zero on the happy path, so a consumer can tell "not mixed"
    from "this field predates the guard"."""
    doc = {"outcomes": [_record("a")], "aggregate": {}, "config": {"model": "old/judge"}}
    payload = rejudge_document(doc, lambda s, u: "YES", "new/judge", "src.json")

    assert payload["aggregate_mixed_verdicts"]["n"] == 0
    assert payload["aggregate_mixed_verdicts"]["question_ids"] == []


def test_a_terminal_error_aborts_the_rejudge_run() -> None:
    def judge(system: str, user: str) -> str:
        raise _out_of_credits()

    with pytest.raises(_Status):
        rejudge_outcomes([_record("a")], judge)


# --------------------------------------------------------------------------------------------
# 5. judge_quality: one probe item, and never a fabricated verdict.
# --------------------------------------------------------------------------------------------


def _item(item_id: str) -> ProbeItem:
    return ProbeItem(
        item_id=item_id, probe="verbatim", question_id=item_id, construction="verbatim",
        question="q?", gold="g", prediction="p", expected=True,
    )


def test_a_failed_probe_item_is_excluded_rather_than_scored_false() -> None:
    """⛔ `measure_model` builds its accept rate with `_rate([bool(v["accepted"]) ...])`, so a
    failed item recorded as `accepted: False` would not be an omission — it would be a fabricated
    REJECTION, moving the very false-accept rate this module exists to measure."""
    calls = {"n": 0}

    def judge(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmptyCompletion("no text.")
        return "YES"

    verdicts = run_probe([_item("a"), _item("b")], judge)

    assert len(verdicts) == 2
    assert verdicts[0]["accepted"] is None
    assert verdicts[0]["judge_error"] is None, "a failed call is not evidence about the judge"
    assert verdicts[0]["failed"] is True
    assert verdicts[1]["accepted"] is True


def test_a_terminal_error_aborts_the_probe() -> None:
    def judge(system: str, user: str) -> str:
        raise _out_of_credits()

    with pytest.raises(_Status):
        run_probe([_item("a")], judge)


def test_a_failed_item_is_excluded_from_the_published_accept_rate() -> None:
    """⛔ THE ARM THAT MATTERS, and the one this file was missing.

    `run_probe` recording `accepted: None` achieves nothing on its own, because `measure_model`
    computes `_rate([bool(v["accepted"]) ...])` and `bool(None)` is `False`. Without the filter
    there, a judge call that never returned lands in the published rate as a REJECTION the judge
    never made — n stays 2 and a perfect judge scores 0.5.

    Mutation-driven: dropping `if v["accepted"] is not None` from `measure_model` left the whole
    suite green, because every other arm here checks the verdict SHAPE and none of them checked
    the number built from it. A guard nothing measures is a guard that can be deleted silently.
    """
    calls = {"n": 0}

    def judge(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmptyCompletion("no text.")
        return "YES"

    scorecard = measure_model(
        {VERBATIM: [_item("a"), _item("b")], REWORDED: [], SWAPPED: []}, judge
    )

    assert scorecard["verbatim_accept_rate"]["n"] == 1, "the failed item leaves the denominator"
    assert scorecard["verbatim_accept_rate"]["rate"] == 1.0, (
        "the judge accepted every item it actually answered; 0.5 here would be a fabricated "
        "rejection carried into a published number"
    )
    assert scorecard["judge_failures"][VERBATIM] == 1, "and the shrunken n has to say why"
    assert scorecard["judge_errors"][VERBATIM] == 0, "a call that never returned is not an error"
