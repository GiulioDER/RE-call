"""Guards for `_is_transient`, the predicate every cloud embedder's retry rests on.

`_is_transient` decides on a numeric status when the exception carries one, and otherwise falls
back to matching well known markers in the exception text. Until the change these tests arrived
with, that fallback was not a fallback at all for `VoyageEmbedder`: voyageai spells its status
`http_status`, which the classifier did not read, so every Voyage retry decision rested on the
marker tuple alone.

Two things are pinned here that the rest of the suite did not pin. Every marker in
`_TRANSIENT_MARKERS` gets its own case, so no marker can be deleted silently: measured against
the suite as it stood before this file, deleting "429", " 503", "connection", "reset by peer",
"timed out", or all four of " 500"/" 502"/" 503"/" 504" at once each left it green. And the two
normalising steps around the match, the `type(exc).__name__` prefix and the `.lower()`, each get
a case that only they can satisfy.

Behaviour is asserted by counting real calls through `retry_with_backoff` rather than by reading
the private predicate's return value, so a classifier that is right in isolation but wired up
wrongly still fails.
"""

from __future__ import annotations

import pytest

from recall.embeddings import _TRANSIENT_MARKERS, retry_with_backoff

# Imported at MODULE scope, not with `importorskip` inside the tests that need it. Importing
# voyageai costs ~90 s on a machine that has the extra, because it pulls langchain_text_splitters
# -> sentence_transformers -> transformers, and `pytest-timeout` clocks only the test protocol.
# Inside a test body that 90 s is billed to one item against the suite's `timeout = 120`, and
# `timeout_method = "thread"` does not redden the item when it overruns: it `os._exit`s the whole
# session. Out here the cost lands in collection, where nothing is timing it.
#
# `except Exception`, not `except ImportError`, and the width is the point. At module scope a
# failure that is not caught aborts COLLECTION, so the whole session reports zero tests instead
# of three red ones. That is not hypothetical on this chain: a CUDA torchvision beside a CPU
# torch raises `RuntimeError: operator torchvision::nms does not exist` from transformers'
# eager torchvision import, not an ImportError. The cause is kept in the skip reason rather
# than swallowed, so a broken extra is legible under `-rs` instead of looking uninstalled.
_voyage_import_error: str | None = None
try:
    import voyageai.error as voyage_error
except Exception as exc:  # pragma: no cover - the extra is absent in CI by design
    voyage_error = None
    # `exc.name`, NOT a substring of the message. Only one shape of failure means "the extra is
    # not installed", and the text cannot identify it: a chain that dies on a missing
    # sub-dependency says `No module named 'torch'`, and a half-installed package says
    # `No module named 'voyageai.error'`. Both would read as absence and send the reader to
    # install what is already installed. `exc.name` is 'voyageai' for absence alone.
    if not (isinstance(exc, ModuleNotFoundError) and exc.name == "voyageai"):
        _voyage_import_error = repr(exc)

requires_voyage = pytest.mark.skipif(
    voyage_error is None,
    reason=(
        f"voyageai present but unimportable: {_voyage_import_error}"
        if _voyage_import_error
        else "needs the voyage extra (pip install recall-rag[voyage])"
    ),
)


def _attempts(exc: Exception, *, attempts: int = 2) -> int:
    """Return how many times `retry_with_backoff` actually calls a function raising ``exc``.

    2 means the error was classified transient and retried, 1 means it was re-raised on the
    spot. Sleeping is stubbed out, so this costs nothing.
    """
    calls = {"n": 0}

    def fn() -> object:
        calls["n"] += 1
        raise exc

    with pytest.raises(type(exc)):
        retry_with_backoff(fn, attempts=attempts, base_delay=0.0, sleep=lambda _s: None)
    return calls["n"]


class _StatusError(Exception):
    """A provider error spelling its HTTP status `status_code`, as the current SDKs do."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _HttpStatusError(Exception):
    """A provider error spelling its HTTP status `http_status`, as voyageai does.

    Deliberately named so that its own class name carries no marker. `voyageai`'s real
    `ServiceUnavailableError` is classified transient today only because the word "unavailable"
    happens to be in its class name, which would make a guard built on it pass for the wrong
    reason and stay green if the numeric read were removed again.
    """

    def __init__(self, message: str, http_status: int) -> None:
        super().__init__(message)
        self.http_status = http_status


#: One message per marker, each carrying THAT marker and no other. The isolation is not asserted
#: by eye: `test_each_marker_case_matches_exactly_one_marker` proves it mechanically, so deleting
#: a marker from `_TRANSIENT_MARKERS` reddens exactly the case that pins it.
MARKER_CASES: tuple[tuple[str, str], ...] = (
    ("429", "http 429 returned by the api"),
    (" 500", "http 500 returned by the api"),
    (" 502", "http 502 returned by the api"),
    (" 503", "http 503 returned by the api"),
    (" 504", "http 504 returned by the api"),
    ("rate limit", "rate limit exceeded for this key"),
    ("too many requests", "too many requests"),
    ("timeout", "read timeout while awaiting a response"),
    ("timed out", "the request timed out"),
    ("temporarily", "the service is temporarily degraded"),
    ("connection", "connection aborted by the client"),
    ("reset by peer", "socket forcibly reset by peer"),
    ("unavailable", "backend is currently unavailable"),
)


def test_every_marker_has_a_case() -> None:
    """A marker added to the classifier without a case here is a marker nothing pins.

    Equality rather than a subset check, and ordered, so the two lists have to be maintained
    together in both directions: a marker deleted from the classifier fails here too.
    """
    assert tuple(marker for marker, _ in MARKER_CASES) == _TRANSIENT_MARKERS


def test_each_marker_case_matches_exactly_one_marker() -> None:
    """Each case's message must isolate its own marker, or the case pins nothing.

    A message matching two markers stays green when either one is deleted, which is precisely
    the failure this file exists to prevent: "connection reset by peer" carries both
    "connection" and "reset by peer", so it cannot pin either.
    """
    for marker, message in MARKER_CASES:
        text = f"RuntimeError {message}".lower()
        matched = [m for m in _TRANSIENT_MARKERS if m in text]
        assert matched == [marker], f"{message!r} matched {matched}, expected only {marker!r}"


@pytest.mark.parametrize(
    ("marker", "message"), MARKER_CASES, ids=[marker.strip() for marker, _ in MARKER_CASES]
)
def test_a_status_less_error_is_retried_on_each_marker(marker: str, message: str) -> None:
    """Every marker is load bearing on its own, because for some callers it is all there is.

    A `VoyageEmbedder` failure carries no attribute this classifier reads, so the marker is the
    entire retry decision on that path.
    """
    assert _attempts(RuntimeError(message)) == 2, f"marker {marker!r} did not trigger a retry"


@pytest.mark.parametrize(
    "message",
    [
        "Connection Reset By Peer",
        "Service Temporarily Unavailable",
        "Read Timeout",
        "Too Many Requests",
    ],
)
def test_the_match_is_case_insensitive(message: str) -> None:
    """Pins the `.lower()` on the exception text.

    Every marker these messages match is capitalised in all of its words, which is what makes
    the case bite: `RuntimeError("Connection reset by peer")` would NOT pin the normalisation,
    since "reset by peer" is still lowercase in it and matches with or without the `.lower()`.
    Providers do return title case reason phrases, "Service Temporarily Unavailable" being the
    literal HTTP 503 phrase, so this is the shape a real 503 body arrives in.
    """
    assert _attempts(RuntimeError(message)) == 2


@pytest.mark.parametrize("exc", [TimeoutError("boom"), ConnectionError("boom")])
def test_the_exception_type_name_is_part_of_the_matched_text(exc: Exception) -> None:
    """Pins the `type(exc).__name__` prefix.

    `TimeoutError("boom")` and `ConnectionError("boom")` say nothing transient in their message;
    the evidence is entirely in the class name, and both are raised with an empty or unhelpful
    message by real client stacks. Dropping the prefix from the matched text loses them.
    """
    assert _attempts(exc) == 2


@pytest.mark.parametrize("status", [429, 500, 503, 599])
def test_a_numeric_status_in_the_retryable_band_is_retried(status: int) -> None:
    """429 and the whole 5xx band retry on the number alone, with no marker in the text."""
    assert _attempts(_StatusError("boom", status_code=status)) == 2


@pytest.mark.parametrize("status", [600, 700, 401, 400, 404])
def test_a_numeric_status_outside_the_retryable_band_is_not_retried(status: int) -> None:
    """Pins both ends of the numeric band, the `< 600` in particular.

    600 is not an HTTP status, so a carrier reporting one is reporting something the classifier
    has no basis to call transient. Without the upper bound the condition degrades to
    `500 <= status`, which retries every out of range number a broken or non HTTP carrier
    reports. 401 and 400 pin the lower end: a bad key or a bad request is permanent, and
    retrying it pays for the same refusal three times.
    """
    assert _attempts(_StatusError("boom", status_code=status)) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_status_spelled_http_status_is_read_as_a_status(status: int) -> None:
    """voyageai's spelling has to reach the numeric branch, or Voyage retries on prose alone.

    `voyageai.error.VoyageError.__init__` takes `http_status` as its third positional argument
    and the request layer passes the real response code into it, so every Voyage error carries
    the number. The classifier read only `status_code` and `status`, so it never saw it, and
    the message is what decided. Those messages are fixed strings that contain no marker at all:
    a real 500 arrives as "The server failed to process the request." and was NOT retried.
    """
    assert _attempts(_HttpStatusError("boom", http_status=status)) == 2


@pytest.mark.parametrize("status", [400, 401, 404, 422, 600])
def test_a_non_retryable_http_status_is_still_not_retried(status: int) -> None:
    """Reading `http_status` must not degrade into treating its presence as transient.

    An implementation that retried whenever `http_status` was set would pass the test above and
    turn a bad Voyage key into three paid refusals, which is the expensive direction.
    """
    assert _attempts(_HttpStatusError("boom", http_status=status)) == 1


@requires_voyage
def test_a_real_voyage_server_error_is_retried() -> None:
    """The same guard against the installed SDK, so the attribute name cannot drift unnoticed.

    Constructed with voyageai's own positional signature (message, http_body, http_status), the
    way `api_requestor._interpret_response_line` raises it on a real 500.
    """
    exc = voyage_error.ServerError("The server failed to process the request.", None, 500)
    assert getattr(exc, "status_code", None) is None  # the reason the numeric read used to miss
    assert getattr(exc, "status", None) is None
    assert exc.http_status == 500
    assert _attempts(exc) == 2


@requires_voyage
def test_a_real_voyage_rate_limit_is_retried_without_a_marker_in_its_message() -> None:
    """A real 429 whose body says nothing marker shaped still has to retry.

    The message on a Voyage 429 is the server's `detail` string, not a fixed phrase, and the
    class name "RateLimitError" does not contain the marker "rate limit" because the marker has
    a space in it. On the text alone this error is indistinguishable from a permanent one.
    """
    exc = voyage_error.RateLimitError("You have exceeded your quota.", None, 429)
    text = f"{type(exc).__name__} {exc}".lower()
    assert not [m for m in _TRANSIENT_MARKERS if m in text], "message accidentally carries a marker"
    assert _attempts(exc) == 2


@requires_voyage
def test_a_real_voyage_auth_error_fails_fast() -> None:
    """The other direction on the same SDK: a 401 is permanent and must not be paid for twice."""
    exc = voyage_error.AuthenticationError("Provided API key is invalid.", None, 401)
    assert _attempts(exc) == 1
