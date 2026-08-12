"""`_is_transient` must let a numeric status have the last word.

The classifier reads a numeric `status_code`/`status` first and then ALSO matches markers in the
exception text. When both fire, the text used to win: a permanent 400 whose message merely
contained the digits "429" was classified transient. The markers are a fallback for errors that
carry no status at all (voyageai spells it `http_status`, and connection/timeout errors carry
nothing), so they must not be consulted once the transport has already stated the status.

The cost of the false positive is not one wasted attempt. `retry_with_backoff` resends the whole
payload, and the payload on the extraction path is a prompt with an entire memo body embedded in
it — so a context-length overflow, which is permanent by construction, is paid for three times.
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


def test_a_400_whose_message_contains_429_is_not_retried() -> None:
    """The verified failing input, built from the real SDK rather than a look-alike.

    A stub would prove only that the stub matches itself; this pins the fix to the attribute
    `openai` actually sets. "10429 tokens" is not a rate limit, and no number of retries will
    make an over-long prompt fit.
    """
    openai = pytest.importorskip("openai", reason="needs the extract/bench extra")
    httpx = pytest.importorskip("httpx", reason="openai's transport dep; only missing if it drops it")

    body = {
        "error": {
            "message": (
                "This model's maximum context length is 8192 tokens, however your messages "
                "resulted in 10429 tokens"
            ),
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
    """The status the classifier exists for. Deciding on the number must not lose this."""
    exc = _StatusError("Error code: 429 - slow down", status_code=429)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_a_503_is_still_retried() -> None:
    """5xx is the other transient family, and its text carries no marker of its own here."""
    exc = _StatusError("Error code: 503 - upstream is having a bad day", status_code=503)
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


def test_an_error_with_no_status_still_falls_back_to_text_markers() -> None:
    """The fallback must survive the fix.

    Nothing on the voyage path sets `status_code`/`status`, and `openai.APIConnectionError`
    carries no status either — for those, the text is the only evidence there is. Deleting the
    fallback would make every network blip fail on the first attempt.
    """
    exc = RuntimeError("connection reset by peer")
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
