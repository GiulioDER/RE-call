"""`_is_transient` must let a numeric status have the last word.

The classifier used to read a numeric `status_code`/`status` and then ALSO match markers in the
exception text. When both fired, the text won: a permanent 400 whose message merely contained
the digits "429" was classified transient. The markers are a fallback for errors that
carry no status at all (voyageai spells it `http_status`, and connection/timeout errors carry
nothing), so they must not be consulted once the transport has already stated the status.

The cost of the false positive is not one wasted attempt. `retry_with_backoff` resends the whole
payload, and a caller whose payload is a prompt carrying a whole document body pays for the
refusal on every attempt. `benchmarks/llm.py` is that shape in this repository, sending the
retrieved context `benchmarks/pipeline.py` builds; the case this was found on is an extraction
engine on an unlanded branch, which is not in this tree.
"""

import pytest

from recall.embeddings import _is_transient, retry_with_backoff


class _StatusError(Exception):
    """An SDK-shaped error: a transport status alongside a human message.

    Both are supplied per-case, because the whole point of these tests is the disagreement
    between them.
    """

    def __init__(self, message: str, status_code: int | None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _StatusOnlyError(Exception):
    """An error that spells the status `status`, with no `status_code` at all.

    A separate class rather than a `None` on `_StatusError`, because `getattr(exc, "status_code",
    None)` cannot tell an absent attribute from one set to None — so the attribute has to be
    genuinely missing for these tests to reach the second lookup.
    """

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def _attempts_used(exc: Exception, attempts: int = 3) -> int:
    """Run `retry_with_backoff` over a function that always raises `exc`, and count the calls.

    Counting real attempts rather than asserting on `_is_transient` directly: the classifier is
    private, and "was this retried" is the behaviour anyone actually cares about.
    """
    calls = {"n": 0}

    def always() -> object:
        calls["n"] += 1
        raise exc

    with pytest.raises(type(exc)):
        retry_with_backoff(always, attempts=attempts, base_delay=0.0, sleep=lambda _s: None)
    return calls["n"]


#: The reported failure, verbatim. "10429 tokens" is not a rate limit, and no number of retries
#: will make an over-long prompt fit.
_OVERFLOW = (
    "This model's maximum context length is 8192 tokens, however your messages "
    "resulted in 10429 tokens"
)


def test_a_400_whose_message_contains_429_is_not_retried() -> None:
    """The reported case, on a stub, so that CI actually runs it.

    The real-SDK twin below is gated behind the `bench` extra, and CI installs `.[dev]` — which
    means the one test reproducing the exact bug this fix exists for would never execute there.
    A stub-only test proves less, and a skipped test proves nothing at all, so both exist.
    """
    exc = _StatusError(f"Error code: 400 - {_OVERFLOW}", status_code=400)
    assert _is_transient(exc) is False
    assert _attempts_used(exc) == 1


def test_the_real_sdk_400_that_was_reported_is_not_retried() -> None:
    """The same case built from the real SDK rather than a look-alike.

    What this guards is a rename BY openai, not by us: under every mutation of our own code this
    test dies exactly when the stub twin above dies, so it adds nothing there. Its value is that
    the stub's `status_code` is an assumption about someone else's library, and this is the only
    thing that would notice if that assumption stopped holding.
    """
    openai = pytest.importorskip("openai", reason="needs the bench extra (pip install recall[bench])")
    httpx = pytest.importorskip("httpx", reason="openai's transport dep; only missing if it drops it")

    body = {
        "error": {
            "message": _OVERFLOW,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json=body,
    )
    real = openai.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)

    assert real.status_code == 400  # the attribute the fix decides on; guards a rename
    assert "429" in str(real)  # the marker that used to hijack the decision
    assert _is_transient(real) is False
    assert _attempts_used(real) == 1


def test_a_real_429_is_still_retried() -> None:
    """The status the classifier exists for. Deciding on the number must not lose this.

    The message is deliberately marker-free — no digits, no "rate limit". Spelling it the way a
    provider would, `"Error code: 429 - slow down"`, made the text fallback able to satisfy this
    test on its own, so deleting the whole numeric branch left it green: a guard for the numeric
    branch that did not need the numeric branch.
    """
    exc = _StatusError("slow down", status_code=429)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


@pytest.mark.parametrize("status", [500, 503, 599])
def test_the_whole_5xx_band_is_still_retried(status: int) -> None:
    """5xx is the other transient family. Marker-free for the same reason as the 429 above.

    `"Error code: 503 - ..."` contains `" 503"`, leading space and all, which IS one of the
    markers — so the obvious phrasing hid the same hole here.

    Both ends of the band, not one point in the middle. With 503 as the only case, moving the
    lower bound to `501 <= status` left every test green while making a plain HTTP 500 fail on
    the first attempt.
    """
    exc = _StatusError("upstream is having a bad day", status_code=status)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_a_401_is_not_rescued_by_a_marker_word_in_its_message() -> None:
    """Generalises the bug past the "429"-in-a-number coincidence.

    A bad key is permanent however the provider words the failure, so any marker appearing in a
    401's prose must be ignored once the status has spoken.
    """
    exc = _StatusError("401 unauthorized: connection to the tenant timed out", status_code=401)
    assert _is_transient(exc) is False
    assert _attempts_used(exc) == 1


def test_a_status_spelled_status_also_decides_alone() -> None:
    """The second lookup is now decisive too, and that is a bigger change than the first.

    Before the fix a wrong `status` could only ever ADD a spurious True; now it can force a
    permanent False and suppress a real retry. Nothing covered that limb, so deleting the
    `getattr(exc, "status", None)` line entirely left the whole file green.
    """
    exc = _StatusOnlyError(f"Error code: 400 - {_OVERFLOW}", status=400)
    assert not hasattr(exc, "status_code")  # the first lookup must genuinely miss
    assert _is_transient(exc) is False
    assert _attempts_used(exc) == 1


def test_a_transient_status_spelled_status_is_still_retried() -> None:
    """The other direction of the same limb, and the one that costs a real outage if lost.

    Asserting only that `status` can refuse a retry leaves an implementation in which the second
    lookup may ONLY refuse: under it a rate limit spelled `status = 429` falls through to the
    text, and with marker-free prose it is classified permanent. The message here is marker-free
    precisely so nothing but the numeric limb can produce True.
    """
    exc = _StatusOnlyError("slow down", status=429)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_status_code_is_read_before_status() -> None:
    """The two lookups are ordered, and the order is load-bearing once each one is decisive.

    Both directions, because pinning only one leaves an implementation in which EITHER attribute
    may declare the error transient. That is not a hypothetical weakening: it re-opens this
    fix's own bug, since a permanent 400 sitting beside a stale 503 would be retried again.
    """
    transient_first = _StatusError("slow down", status_code=429)
    transient_first.status = 400  # type: ignore[attr-defined]
    assert _is_transient(transient_first) is True
    assert _attempts_used(transient_first) == 3

    permanent_first = _StatusError("upstream is having a bad day", status_code=400)
    permanent_first.status = 503  # type: ignore[attr-defined]
    assert _is_transient(permanent_first) is False
    assert _attempts_used(permanent_first) == 1


def test_an_error_with_no_status_still_falls_back_to_text_markers() -> None:
    """The fallback must survive the fix.

    Nothing on the voyage path sets `status_code`/`status`, and `openai.APIConnectionError`
    carries no status either — for those, the text is the only evidence there is. Deleting the
    fallback would make every network blip fail on the first attempt.
    """
    exc = RuntimeError("connection reset by peer")
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_the_fallback_matches_case_insensitively() -> None:
    """Pins the `.lower()` in the fallback, using the exact string the docstring leans on.

    Every marker is written lowercase, and `openai.APIConnectionError` stringifies to
    `"Connection error."` with a capital C. Drop the `.lower()` and that error stops being
    transient while the rest of the suite stays green — which would silently retract the
    fallback from the one shape `_is_transient`'s docstring names as depending on it.
    """
    exc = RuntimeError("Connection error.")
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_an_unset_status_falls_back_to_text_markers() -> None:
    """`voyageai.error.RateLimitError` sets `http_status = None` — the attribute exists, empty.

    An error that carries the attribute but never filled it in has told us nothing, so it must
    be treated as a no-status error rather than as a status that is not 429.
    """
    exc = _StatusError("rate limit exceeded", status_code=None)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_a_status_that_is_not_an_http_number_falls_back_to_text_markers() -> None:
    """`status` is not a reserved word: plenty of errors use it for a job state, not a status code.

    This is what `isinstance(status, int)` buys over `status is not None`, and it is the
    difference between falling back and crashing — `500 <= "error"` raises `TypeError` from
    inside the classifier, turning a retryable blip into a failure in the retry logic itself.
    """
    exc = _StatusError("connection reset by peer", status_code=None)
    exc.status = "error"  # type: ignore[attr-defined]
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3
