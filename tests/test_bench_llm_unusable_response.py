"""A 200 whose body carries no answer must not become a benchmark result.

The same two holes `tests/test_mtrag_generation_unusable_response.py` closed in
`benchmarks/mtrag/generation.py`, here in `benchmarks/llm.py`, which the commit that fixed the
other one names as carrying them verbatim. Both are reached by a SUCCESSFUL call, and both are
independent of the `finish_reason == "length"` guard they sit beside:

  1. **`choices[0].message.content` is `None`.** What an OpenAI-compatible provider returns for
     `finish_reason == "content_filter"` and for `"tool_calls"`. `_complete_once` ended
     `return content or ""`, so every caller of the `Completer` seam — the BEAM answerer, the
     judge — received an empty string that is indistinguishable from a system with nothing to say,
     and scored it as one.
  2. **`choices` is empty.** OpenRouter answers 200 with `{"choices": []}` when the upstream it
     routed to faults. `resp.choices[0]` was indexed blind, so this surfaced as
     `IndexError: list index out of range`, which names a Python operation rather than the provider
     and sends whoever reads it hunting for a bug in the harness.

⚠️ **THE CLASSIFICATION MECHANISM IS THE DIFFERENCE FROM THE MTRAG FILE, AND IT INVERTS THE
DEFAULT.** `generate_one` runs its own loop over a `PERMANENT_ERROR_NAMES` set, so anything not
named there is retried: transient is the default and only permanence must be declared. This module
retries through `recall.embeddings.retry_with_backoff`, whose `_is_transient` returns False for an
exception carrying no status and no marker word — so here NON-transient is the default and
**both** decisions have to be stated, or `NoCompletionChoices` would silently fail fast.

⛔ Stating them by TYPE rather than leaving them to the message is not tidiness. `_is_transient`'s
last resort is substring-matching the exception's rendered text, and `EmptyCompletion`'s message
interpolates `finish_reason`, which is a **provider-controlled string**. A provider that spelled it
`"timeout"` or `"unavailable"` would flip our own permanent error to transient and buy four
refusals at full price. The same coincidence already bites `CompletionTruncated` in this tree,
which is why it joins the tuple: see `test_a_truncated_completion_fails_fast_whatever_the_ceiling_spells`.

Neither fault is truncation and neither may be folded into that guard. `CompletionTruncated`'s
message tells the operator to raise `DEFAULT_MAX_TOKENS`. That is right advice for exactly one
cause and wrong advice for these two: no ceiling produces a content filter, and no ceiling produces
an empty `choices` array.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from benchmarks.llm import (
    CompletionTruncated,
    EmptyCompletion,
    NoCompletionChoices,
    OpenRouterLLM,
    PERMANENT_ERRORS,
    TRANSIENT_ERRORS,
)

#: A ceiling that cannot match anything by luck, and that `_is_transient` reads as NON-transient on
#: its own (it contains no marker). Asserting it is ABSENT from the two new messages is evidence
#: they do not hand back truncation's advice.
_CEILING = 137


class _RateLimited(Exception):
    """A genuine 429, for the arm that proves the classifier still DELEGATES.

    `status_code` is the attribute `_is_transient` reads first, so this is transient by the shared
    heuristic and by nothing this module added.
    """

    status_code = 429


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    choices: list[object] | None = None,
    raises: Exception | None = None,
    usage: object | None = None,
) -> list[dict[str, Any]]:
    """An `openai` module whose `create` replays a scripted response and records every call.

    `choices` is the whole LIST rather than one choice, because an empty list is one of the two
    faults under test. `raises` takes precedence, so one stub serves the response arms and the
    delegation arm.

    The SDK is faked into `sys.modules` rather than skipped around, following
    `test_bench_llm_max_tokens.py`: `openai` lives in the `bench` extra, so an `importorskip` would
    make these guards silently absent wherever the extra is not installed.
    """
    calls: list[dict[str, Any]] = []

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            calls.append(kwargs)
            if raises is not None:
                raise raises
            return types.SimpleNamespace(
                # NOT `list(...)`: one arm scripts `choices=None`, the shape the SDK surfaces
                # when the body omits the field, and coercing it would hide that arm.
                choices=choices, usage=usage
            )

    class _FakeOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


def _choice(
    content: object, finish_reason: str | None = "stop", *, carry_reason: bool = True
) -> object:
    """One `choices[0]` in the SDK's shape.

    `content=None` is the real wire shape for a filtered completion: the key is present and null,
    not absent. `carry_reason=False` omits `finish_reason` entirely, which is how a provider that
    does not send the field looks, and how the fakes in `test_bench_llm.py` are built.

    `content` is `object` rather than `str | None` because the SDK does not validate the field and
    one arm deliberately scripts a list of content parts, which is the shape that used to crash.
    """
    attrs: dict[str, object] = {"message": types.SimpleNamespace(content=content)}
    if carry_reason:
        attrs["finish_reason"] = finish_reason
    return types.SimpleNamespace(**attrs)


def _llm(**kwargs: Any) -> OpenRouterLLM:
    """`sleep` is swallowed so the retry arms run at test speed rather than paying real backoff."""
    kwargs.setdefault("max_tokens", _CEILING)
    kwargs.setdefault("max_attempts", 4)
    return OpenRouterLLM(model="test/model", api_key="k", sleep=lambda _: None, **kwargs)


# --------------------------------------------------------------------------------------------
# 1. `content` is None or blank: nothing to score, and no attempt can change that.
# --------------------------------------------------------------------------------------------


def test_a_completion_with_no_text_is_never_returned_as_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect, pinned. `None or ""` is `""`, and `complete` is the `Completer` seam: every
    consumer downstream takes what it returns as the system's answer. Nothing past this point
    looks at the string again, so this is the only place it can be caught."""
    _install_fake_openai(monkeypatch, choices=[_choice(None, "content_filter")])

    with pytest.raises(EmptyCompletion) as caught:
        _llm().complete("s", "u")

    assert "content_filter" in str(caught.value), (
        "the operator has to tell a filter refusal from a tool-call stub from a provider bug, and "
        "`finish_reason` is the only field that distinguishes them"
    )


def test_a_completion_with_no_text_costs_exactly_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A content filter fires on the PROMPT, and the prompt is byte-identical on every attempt, so
    the three extra calls buy the same refusal at the same price. Refusals also land on particular
    questions rather than on whole runs, so nothing upstream caps them."""
    calls = _install_fake_openai(monkeypatch, choices=[_choice(None, "content_filter")])

    with pytest.raises(EmptyCompletion):
        _llm().complete("s", "u")

    assert len(calls) == 1, "a refusal is deterministic in the request; it must not be retried"


def test_a_completion_of_only_whitespace_is_refused_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`"  \\n "` is as unscorable as `""` and reaches a judge the same way. The emptiness test has
    to be `.strip()`-based even though the RETURN is not — see the verbatim arm below."""
    _install_fake_openai(monkeypatch, choices=[_choice("  \n \t ", "stop")])

    with pytest.raises(EmptyCompletion):
        _llm().complete("s", "u")


def test_an_empty_answer_is_not_reported_as_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚠️ Why this is its own type and not a wider `finish_reason` check. `CompletionTruncated`
    means the completion hit OUR ceiling and its message says to raise `DEFAULT_MAX_TOKENS`. No
    value of that constant produces a filter refusal, so folding the two together would hand the
    operator advice that cannot work, for a cause it does not name."""
    _install_fake_openai(monkeypatch, choices=[_choice(None, "content_filter")])

    with pytest.raises(EmptyCompletion) as caught:
        _llm().complete("s", "u")

    assert not isinstance(caught.value, CompletionTruncated)
    assert "max_tokens" not in str(caught.value) and str(_CEILING) not in str(caught.value), (
        "the ceiling is irrelevant to a refusal; naming it sends the operator to the wrong flag"
    )


def test_the_empty_answer_is_classified_permanent_by_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the classification to the CLASS, not to the wording. `EmptyCompletion`'s message
    interpolates `finish_reason`, which the provider controls; a provider spelling it `"timeout"`
    would make `_is_transient`'s substring fallback read our permanent error as retryable."""
    assert EmptyCompletion in PERMANENT_ERRORS

    calls = _install_fake_openai(monkeypatch, choices=[_choice(None, "timeout")])

    with pytest.raises(EmptyCompletion):
        _llm().complete("s", "u")

    assert len(calls) == 1, "a provider-supplied marker word must not reclassify our own error"


def test_an_unrecognised_finish_reason_cannot_reach_a_substring_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⛔ The first version of this fix interpolated `finish_reason` raw, and that was the very
    mistake the module docstring accuses the message-based route of.

    `finish_reason` is chosen by the PROVIDER, and `benchmarks/beam/run.py:_is_terminal` substring
    matches the rendered exception against `_TERMINAL_MARKERS` ("402", "401", "insufficient_quota",
    "requires more credits"). A terminal verdict is not one dropped question: `_run_pool` raises
    `RunAborted` and cancels every remaining one. So a provider echoing an upstream `402` into
    `finish_reason` could end a 1,986-question run, and classifying by TYPE one layer down bought
    nothing while the same string still fed a second classifier one layer up.
    """
    from benchmarks.beam.run import _is_terminal

    _install_fake_openai(monkeypatch, choices=[_choice(None, "upstream 402 refused")])

    with pytest.raises(EmptyCompletion) as caught:
        _llm().complete("s", "u")

    assert not _is_terminal(caught.value), "a provider string must not be able to abort a run"
    assert "402" not in str(caught.value)


def test_the_raw_finish_reason_survives_on_the_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Withheld from the MESSAGE, not thrown away. The unrecognised value is the one an operator
    most needs, so it moves to an attribute, where nothing substring-matches it."""
    _install_fake_openai(monkeypatch, choices=[_choice(None, "upstream 402 refused")])

    with pytest.raises(EmptyCompletion) as caught:
        _llm().complete("s", "u")

    assert caught.value.finish_reason == "upstream 402 refused"


def test_non_str_content_is_refused_by_type_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`message.content` is typed `Optional[str]` but openai-python builds response models without
    validating, so a non-conforming body passes straight through. `not content.strip()` then raised
    `AttributeError: 'list' object has no attribute 'strip'` from inside `_complete_once` — exactly
    the "names a Python operation rather than the provider" failure `NoCompletionChoices` exists to
    abolish, reintroduced by the guard meant to remove it.

    `content or ""` used to hand the value back instead, which is no better: the seam is declared
    to return `str`, so it would have been scored as an answer or blown up further downstream.

    ⚠️ NARROWED. This arm used to script a LIST of text blocks, and a list is no longer refused:
    it is READ, joined, and returned, because a gateway serialising content as blocks is sending a
    well formed answer and failing it would blame the system under test for its gateway's
    encoding. See `tests/test_bench_llm_block_content.py`. A dict exercises the identical
    `AttributeError` path this arm was written for, so the guard it pins is unchanged.
    """
    _install_fake_openai(monkeypatch, choices=[_choice({"text": "hi"}, "stop")])

    with pytest.raises(EmptyCompletion) as caught:
        _llm().complete("s", "u")

    assert "dict" in str(caught.value), "the operator needs the SHAPE that actually came back"


# --------------------------------------------------------------------------------------------
# 2. `choices` is empty: a fault on the provider's side, legible and retried.
# --------------------------------------------------------------------------------------------


def test_a_response_with_no_choices_names_the_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """`resp.choices[0]` raised `IndexError: list index out of range`, which names a Python
    operation rather than the provider returning an answerless 200."""
    _install_fake_openai(monkeypatch, choices=[])

    with pytest.raises(NoCompletionChoices) as caught:
        _llm().complete("s", "u")

    message = str(caught.value)
    assert "IndexError" not in message, "the raw indexing failure must not be what surfaces"
    assert "choices" in message, "the message has to say WHAT the provider returned"


def test_a_response_with_no_choices_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberately transient, unlike the arms above, and it has to be stated: an exception with no
    `status_code` and no marker word is NON-transient by default under `_is_transient`, which is the
    opposite of the mtrag loop's default.

    An empty `choices` on a 200 is OpenRouter reporting that the upstream it routed to faulted. The
    request is well-formed and a second attempt can be served by a healthy upstream, so classifying
    it permanent would fail a call that would have succeeded.

    This arm is also the liveness proof for the two `len(calls) == 1` assertions above: same stub,
    same loop, and it watches the count reach four."""
    calls = _install_fake_openai(monkeypatch, choices=[])

    with pytest.raises(NoCompletionChoices):
        _llm().complete("s", "u")

    assert len(calls) == 4, "an upstream fault is worth another route"


def test_the_no_choices_error_is_classified_transient() -> None:
    """Pins the decision itself, so flipping it is an edit to a stated choice rather than a silent
    consequence of leaving a type out of a tuple."""
    assert NoCompletionChoices in TRANSIENT_ERRORS
    assert NoCompletionChoices not in PERMANENT_ERRORS


# --------------------------------------------------------------------------------------------
# 3. The classification is an OVERRIDE, not a replacement.
# --------------------------------------------------------------------------------------------


def test_a_rate_limit_is_still_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness for the delegation. A classifier that answered only about this module's own types
    and returned False for everything else would pass every arm above while destroying the retry
    the class docstring exists for: one 429 on question 900 of 1,986 would end the run."""
    calls = _install_fake_openai(monkeypatch, raises=_RateLimited("429 slow down"))

    with pytest.raises(_RateLimited):
        _llm().complete("s", "u")

    assert len(calls) == 4, "the shared transient heuristic must still decide everything else"


def test_a_truncated_completion_fails_fast_whatever_the_ceiling_spells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CompletionTruncated` documents itself as deliberately not transient, but until this tuple
    existed nothing enforced it: `_is_transient` fell through to substring markers that include the
    literal `"429"`, and the message interpolates the ceiling. Measured on this tree, 16384 is
    correctly False while 4290, 429 and 1429 are all True — so a `finish_reason="length"` at
    `max_tokens=4290` paid four billed requests for a failure guaranteed to repeat, and
    `DEFAULT_MAX_TOKENS`'s own docstring invites the edit that lands on such a ceiling."""
    calls = _install_fake_openai(monkeypatch, choices=[_choice("cut off ri", "length")])

    with pytest.raises(CompletionTruncated):
        _llm(max_tokens=4290).complete("s", "u")

    assert len(calls) == 1, "truncation must fail fast whatever the ceiling happens to spell"


# --------------------------------------------------------------------------------------------
# 4. Guards the guards: every ordinary answer still comes back, unchanged.
# --------------------------------------------------------------------------------------------


def test_an_ordinary_answer_is_still_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both new checks sit on the success path, so an over-eager one turns every completed call
    into a failure. This is the arm that goes red if the emptiness test is ever widened."""
    calls = _install_fake_openai(monkeypatch, choices=[_choice("an answer", "stop")])

    assert _llm().complete("s", "u") == "an answer"
    assert len(calls) == 1


def test_an_answer_is_returned_verbatim_and_not_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚠️ The deliberate difference from `generate_one`, which returns `.strip()`ed text because a
    stripped string is what it writes to the submission. This seam returns the completion to
    arbitrary consumers, so `.strip()` here would silently change what every existing caller
    receives. The emptiness check strips only to DECIDE; the value handed back is untouched."""
    _install_fake_openai(monkeypatch, choices=[_choice("  an answer  ", "stop")])

    assert _llm().complete("s", "u") == "  an answer  "


def test_an_answer_from_a_response_carrying_no_finish_reason_is_still_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the choice defensively must not make the SHAPE stricter: absence of `finish_reason`
    is not evidence of anything, and any hand-rolled double omits it."""
    _install_fake_openai(monkeypatch, choices=[_choice("fine", carry_reason=False)])

    assert _llm().complete("s", "u") == "fine"


def test_an_empty_completion_is_still_counted_as_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The empty-text guard belongs AFTER the usage block, exactly as truncation's does. The tokens
    were spent whatever the body turned out to carry, and `benchmarks.usage` subtracts this figure
    to publish `memory_layer`: an undercount there invents cost that was never incurred."""
    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=0)
    _install_fake_openai(monkeypatch, choices=[_choice(None, "content_filter")], usage=usage)
    llm = _llm(max_attempts=1)

    with pytest.raises(EmptyCompletion):
        llm.complete("s", "u")

    assert llm.usage() == {"calls": 1, "prompt_tokens": 11, "completion_tokens": 0}


def test_a_no_choices_response_is_still_counted_as_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its OWN arm, and it exists because the arm above does not cover it.

    Hoisting the no-choices guard above the usage block was mutated in and survived the whole
    suite: every other empty-answer arm scripts a NON-empty `choices`, so none of them reaches the
    hoisted branch. An answerless 200 is still a billed request, and losing its prompt tokens
    understates `harness`, which is the subtrahend in `total - harness`.
    """
    usage = types.SimpleNamespace(prompt_tokens=9, completion_tokens=0)
    _install_fake_openai(monkeypatch, choices=[], usage=usage)
    llm = _llm(max_attempts=1)

    with pytest.raises(NoCompletionChoices):
        llm.complete("s", "u")

    assert llm.usage() == {"calls": 1, "prompt_tokens": 9, "completion_tokens": 0}
    assert llm.provider_metadata().latency_ms is not None, "the call took time and it was billed"


def test_an_absent_choices_field_is_named_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    """⛔ The `llm.py` HALF of a wording a comment claims is "kept word-for-word in step" with
    `benchmarks/mtrag/generation.py`. Only the mtrag half was pinned, so reverting this side alone
    left the suite green: a claimed invariant with a test on one of its two sides.

    The shape matters as well as the wording. `not choices` takes the branch for `None`, which is
    what the SDK surfaces when the body omits `choices` entirely — OpenRouter's documented failure
    shape, and the one this stub had no arm for.
    """
    _install_fake_openai(monkeypatch, choices=None)

    with pytest.raises(NoCompletionChoices) as caught:
        _llm().complete("s", "u")

    assert "absent" in str(caught.value), (
        "the message must name the shape that arrived, and stay in step with the mtrag raise site"
    )
