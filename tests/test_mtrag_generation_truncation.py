"""A completion cut off by our own ceiling must never be scored as an answer.

`benchmarks/llm.py` has guarded this since `CompletionTruncated` was added: a 200 that stopped for
`length` is a healthy provider handing back half an answer, and returning it charges OUR
configuration to the system under test. `benchmarks/mtrag/generation.py` sends `max_tokens` on
every call and never looked at `finish_reason`, so the same half-answer became a `predictions`
entry in a submission file, indistinguishable from a genuine short answer or a genuine refusal
once a judge has scored it.

⚠️ This path is the one where the ceiling is tight. `benchmarks/llm.py` reserves 16,384 tokens;
`--max-tokens` here defaults to **512**, a 32x smaller ceiling on the seam that had no guard at
all. How often 512 actually bites on MTRAG's short-form answers is NOT measured here, and should
not be read out of `DEFAULT_MAX_TOKENS`'s numbers: its ~850-token mean is the BEAM answerer with
reasoning tokens counted, a different workload. The point is only that this is the cheaper ceiling
to cross, and the one that was returning the crossing as an answer.

Two properties, and both are about cost:

  1. The truncated text never comes back as an answer. This is the correctness half.
  2. It costs exactly ONE billed request. `generate_one` retries four times by default, and a
     completion that hit the ceiling hits it again on every attempt, so a guard that raises
     WITHOUT being classified permanent turns one wasted call into four across every task.

Counting `create` calls on a local stub is the honest measure of the bill here, unlike the real
HTTP server that `tests/test_bench_llm_truncation_is_not_transient.py` needs: truncation arrives
as a **200**, so the OpenAI SDK's own transport-level retries never fire on it and there is no
request the in-process count would be blind to. `generate_one`'s own loop is the only multiplier.
"""

from __future__ import annotations

import pytest

from benchmarks.llm import CompletionTruncated
from benchmarks.mtrag import generation as gen

#: A ceiling that cannot match anything by luck, so an assertion that it reaches the operator is
#: evidence the RAISED message names the real number rather than some constant from another module.
_CEILING = 137

#: What a cut-off answer actually looks like: fluent, on-topic, and stopped mid-word. This is the
#: reason the guard exists — nothing about the STRING says it is broken, so only `finish_reason`
#: can tell a judge's input apart from a genuine answer.
_CUT_OFF = "The Cardinals play home games at Busch Stadium, and their away schedule inclu"


def _choice(text: str, finish_reason: str | None = "stop", *, carry_reason: bool = True) -> object:
    """One `choices[0]` in the SDK's shape.

    `carry_reason=False` omits the attribute ENTIRELY rather than setting it to `None`. That is
    not a hypothetical shape: the fakes in `tests/test_mtrag_generation.py` are built that way,
    and so is any hand-rolled double, so the guard has to read it defensively or it turns every
    such response into a spurious truncation.
    """
    attrs: dict[str, object] = {"message": type("M", (), {"content": text})()}
    if carry_reason:
        attrs["finish_reason"] = finish_reason
    return type("C", (), attrs)()


class _Client:
    """An OpenAI-shaped client that counts billed calls and replays a scripted response.

    `raises` takes precedence over `choice`, so one stub serves both the truncation arms and the
    liveness arm below.
    """

    def __init__(self, choice: object | None = None, raises: Exception | None = None) -> None:
        self.calls = 0
        self._choice = choice
        self._raises = raises
        self.chat = self

    @property
    def completions(self) -> "_Client":
        return self

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return type("R", (), {"choices": [self._choice]})()


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """The retry path runs at test speed. `GENERATION_BACKOFF_S` is 2.0 with exponential growth,
    so the liveness arm's four attempts would otherwise sleep 14 seconds to prove a count."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)


def test_a_truncated_completion_is_never_returned_as_an_answer() -> None:
    """The defect, pinned. `generate_one` returned `content.strip()` whatever stopped the
    completion, so this half-sentence became a `predictions` entry and was scored."""
    client = _Client(_choice(_CUT_OFF, "length"))

    with pytest.raises(RuntimeError) as caught:
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], _CEILING)

    assert _CUT_OFF not in str(caught.value), (
        "the cut-off text must not be handed back in any form: a caller that logs the error and "
        "keeps going would otherwise still have a scorable answer in its hands"
    )
    assert str(_CEILING) in str(caught.value), (
        "the operator's fix is a bigger ceiling, so the message must name the ceiling it hit"
    )


def test_a_truncated_completion_costs_exactly_one_request() -> None:
    """Retrying an over-long request buys the same truncation again at four times the price. The
    ceiling is a property of the REQUEST, so no number of attempts can make the answer fit."""
    client = _Client(_choice(_CUT_OFF, "length"))

    with pytest.raises(RuntimeError):
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], _CEILING)

    assert client.calls == 1, "truncation is guaranteed to repeat; it must not be retried"


def test_the_stub_can_see_a_retry_so_a_count_of_one_means_something() -> None:
    """Liveness for the arm above. `calls == 1` is only evidence if this same stub WOULD have
    counted more, so drive a genuinely transient failure through it and watch it reach four."""
    client = _Client(raises=ConnectionError("transient"))

    with pytest.raises(RuntimeError):
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], _CEILING)

    assert client.calls == gen.GENERATION_ATTEMPTS


def test_the_permanent_marker_is_wired_to_the_class_and_not_to_a_stale_string() -> None:
    """`PERMANENT_ERROR_NAMES` classifies by `type(exc).__name__`, so a literal entry would be a
    string coupling to a class in another module: renaming `CompletionTruncated` would leave the
    guard raising correctly while silently restoring the 4x retry, and no arm above would catch
    it — they raise the class, so they follow the rename. The entry is therefore DERIVED from the
    class, which designs that hazard out; this pins that the wiring exists at all, so deleting it
    fails here as well as in the cost arm."""
    assert CompletionTruncated.__name__ in gen.PERMANENT_ERROR_NAMES


def test_a_completion_that_stopped_normally_is_still_returned() -> None:
    """Guards the guard: only `length` is truncation. `stop` is the ordinary case and must be
    untouched, or the guard converts every successful task into a failure."""
    client = _Client(_choice("  an answer  ", "stop"))

    assert gen.generate_one(
        client, "openai/gpt-4o", [{"role": "user", "content": "x"}], _CEILING
    ) == "an answer"
    assert client.calls == 1


def test_a_response_carrying_no_finish_reason_at_all_is_still_returned() -> None:
    """A response with the attribute ABSENT is not truncated, it is unstated. Reading it with
    plain attribute access would raise `AttributeError` here, and that error is not in
    `PERMANENT_ERROR_NAMES`, so a missing field would be retried four times and then reported as
    a generation failure — a guard against scoring bad answers turning into a bill for good ones."""
    client = _Client(_choice("an answer", carry_reason=False))

    assert gen.generate_one(
        client, "openai/gpt-4o", [{"role": "user", "content": "x"}], _CEILING
    ) == "an answer"
    assert client.calls == 1
