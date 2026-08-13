"""`generate_one` must spend `GENERATION_ATTEMPTS` requests, not that number times the SDK's.

`generate_one` carries its own retry loop and a warning above it: "Every attempt is a billed
call. Raising `GENERATION_ATTEMPTS` raises the worst-case bill by the same factor." That sentence
is only true if the SDK's own retry layer is switched off underneath. `openai` defaults
`max_retries` to 2, so with it left on the real worst case for the shipped `GENERATION_ATTEMPTS
= 4` is 12 requests, and the warning understates the bill it exists to give by 3x.

The multiplication is worst exactly where it hurts most. An MT-RAG generation run makes one
billed call per task back to back, which is the traffic shape a provider rate-limits; answering a
429 with triple the requests aims the multiplication at a provider that has just said it is
overloaded, and triples the bill for it.

These tests count real HTTP requests against a local stub rather than asserting on constructor
keywords, because the keyword is the mechanism and the request count is the behaviour. A fake
`chat.completions.create` would be blind to the layer under test: the SDK's retries live below
that method, inside its transport.
"""

from __future__ import annotations

import pytest

from benchmarks.mtrag import generation
from tests.provider_stub_helpers import CHAT_OK, ProviderStub, provider_stub

openai = pytest.importorskip("openai")

_MESSAGES = [{"role": "user", "content": "q"}]


@pytest.fixture
def instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse `generate_one`'s own backoff so the retry path runs at test speed.

    Zeroing the base rather than patching `time.sleep` keeps the sleep call itself in place, and
    keeps the patch inside the module under test instead of mutating the stdlib for the session.
    The shipped 2.0s base would otherwise cost 2 + 4 + 8 = 14s per exhausted retry.
    """
    monkeypatch.setattr(generation, "GENERATION_BACKOFF_S", 0.0)


def _client(stub: ProviderStub, monkeypatch: pytest.MonkeyPatch) -> object:
    """`openrouter_client` pointed at the stub, registered so teardown closes its connection.

    Built through the real factory rather than by constructing `OpenAI` here: the factory is
    where the retry policy is configured, so a test that bypassed it would pass whatever the
    factory did.
    """
    monkeypatch.setattr(generation, "OPENROUTER_BASE_URL", stub.base_url)
    return stub.track(generation.openrouter_client(api_key="sk-test"))


def test_a_rate_limit_costs_generation_attempts_and_no_more(
    instant_backoff: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One 429 must cost 4 requests, which is what `GENERATION_ATTEMPTS = 4` means.

    With the SDK's own 2 retries left on, the layers multiply to 12.
    """
    with provider_stub(CHAT_OK) as stub:
        client = _client(stub, monkeypatch)
        stub.arm(429, "please slow down")

        with pytest.raises(RuntimeError, match="gave up after 4 attempts"):
            generation.generate_one(client, "stub/model", _MESSAGES, max_tokens=16)

        assert stub.count == 4


def test_the_shipped_attempt_budget_is_the_one_the_warning_prices(
    instant_backoff: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warning above `generate_one` prices the bill in units of `GENERATION_ATTEMPTS`.

    Pinning the constant here is what makes the count above mean "the whole budget" rather than
    "four". If someone raises it, this test says so at the same moment the bill changes.
    """
    assert generation.GENERATION_ATTEMPTS == 4, "this test exists to pin the default; it moved"

    with provider_stub(CHAT_OK) as stub:
        client = _client(stub, monkeypatch)
        monkeypatch.setattr(generation, "GENERATION_ATTEMPTS", 2)
        stub.arm(503, "upstream is not accepting work")

        with pytest.raises(RuntimeError, match="gave up after 2 attempts"):
            generation.generate_one(client, "stub/model", _MESSAGES, max_tokens=16)

        assert stub.count == 2


def test_a_permanent_error_is_not_retried(
    instant_backoff: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 400 must still fail fast, at one request.

    Worth pinning on this path specifically: a generation run's 400 is usually a context-length
    overflow, where resending the same over-long payload cannot succeed and is billed anyway.
    The SDK does not retry 400 either, so this arm passes with the defect present; it is here to
    prove the fix removed a multiplication rather than the retry itself.
    """
    with provider_stub(CHAT_OK) as stub:
        client = _client(stub, monkeypatch)
        stub.arm(400, "input is not valid for this model")

        with pytest.raises(RuntimeError, match="BadRequestError"):
            generation.generate_one(client, "stub/model", _MESSAGES, max_tokens=16)

        assert stub.count == 1


def test_a_permanent_error_reports_the_attempt_it_actually_cost(
    instant_backoff: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚠️ The report must not price a failure at four times what it cost.

    The arm above proves a 400 is billed once. This one is about what the operator is TOLD, which
    is a separate claim and was wrong: the give-up message interpolated `GENERATION_ATTEMPTS`
    rather than the attempt the loop actually reached, so a failure that cost one request still
    read "gave up after 4 attempts".

    That string is not cosmetic. `main` copies it verbatim into `.failed.jsonl` and into the
    `task_failed` event, and the `error_type` beside it is `RuntimeError` for every generation
    failure, so the count inside this sentence is the ONLY attempt information an operator has.
    Reading it back to size a bill, or to judge whether the retry policy is behaving, gives 4x the
    real figure for every permanent cause.
    """
    with provider_stub(CHAT_OK) as stub:
        client = _client(stub, monkeypatch)
        stub.arm(400, "input is not valid for this model")

        with pytest.raises(RuntimeError, match=r"gave up after 1 attempt\b"):
            generation.generate_one(client, "stub/model", _MESSAGES, max_tokens=16)

        assert stub.count == 1


def test_a_success_costs_exactly_one_request(
    instant_backoff: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path must not be perturbed: one call, one request, the content returned."""
    with provider_stub(CHAT_OK) as stub:
        client = _client(stub, monkeypatch)

        assert generation.generate_one(client, "stub/model", _MESSAGES, max_tokens=16) == "ok"

        assert stub.count == 1


def test_a_retry_after_header_paces_this_loop_too(
    monkeypatch: pytest.MonkeyPatch, instant_backoff: None
) -> None:
    """Switching the SDK's retries off took this loop's `Retry-After` compliance with them.

    `openrouter_client` passes `max_retries=0`, so the transport layer that obeyed the header is
    gone, and nothing in `generate_one` read it. Its fixed 2/4/8 ladder spends all four attempts
    inside 14s, so a per-minute rate limit that the stock client would simply have waited out
    fails the task — and `CONSECUTIVE_FAILURE_LIMIT` turns five such tasks into an aborted run.
    Cheaper per failure and strictly less able to survive one is not the trade this change was
    making, so the loop honours the header itself now.

    `GENERATION_BACKOFF_S` is zeroed by the fixture, so anything slept here is the provider's ask
    and nothing else.
    """
    slept: list[float] = []
    monkeypatch.setattr(generation.time, "sleep", slept.append)

    with provider_stub(CHAT_OK) as stub:
        stub.arm(429, "please slow down")
        stub.headers = {"Retry-After": "20"}
        client = stub.track(
            openai.OpenAI(api_key="k", base_url=stub.base_url, max_retries=0, timeout=5.0)
        )
        with pytest.raises(RuntimeError):
            generation.generate_one(client, "m", _MESSAGES, max_tokens=16)

    assert slept, "the loop never slept, so it cannot have paced anything"
    assert all(d >= 20.0 for d in slept), f"a 20s cooldown was not waited out: {slept}"
