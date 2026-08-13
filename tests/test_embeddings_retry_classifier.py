"""`_is_transient` must let a numeric status have the last word.

The classifier used to read a numeric `status_code`/`status` and then ALSO match markers in the
exception text. When both fired, the text won: a permanent 400 whose message merely contained
the digits "429" was classified transient. The markers are a fallback for errors that
carry no status at all (connection/timeout errors carry nothing), so they must not be consulted
once the transport has already stated the status.

The second half of this file, from `MARKER_CASES` down, covers the other side of that split: the
marker tuple itself, and the two normalising steps around the match. It was written because the
tuple turned out to be almost entirely unpinned. Deleting "429", " 503", "connection", "reset by
peer", "timed out", or all four of " 500"/" 502"/" 503"/" 504" at once each left the suite green,
as did dropping the `type(exc).__name__` prefix. That mattered more than an ordinary coverage gap
because `http_status` was not read then, so the tuple was the ENTIRE retry decision for every
voyageai error. It is read now, which is what the `_HttpStatusError` cases pin.

The cost of the false positive is not one wasted attempt. `retry_with_backoff` resends the whole
payload, and a caller whose payload is a prompt carrying a whole document body pays for the
refusal on every attempt. `benchmarks/llm.py` is that shape in this repository, sending the
retrieved context `benchmarks/pipeline.py` builds; the case this was found on is an extraction
engine on an unlanded branch, which is not in this tree.
"""

import pytest

from recall.embeddings import _TRANSIENT_MARKERS, _is_transient, retry_with_backoff

# Imported at MODULE scope, not with `importorskip` inside the tests that need it. Importing
# voyageai costs 75 to 105 s on a machine that has the extra, varying with load, because it pulls
# langchain_text_splitters -> sentence_transformers -> transformers, and `pytest-timeout` clocks
# only the test protocol. Inside a test body that is billed to ONE item against the suite's
# `timeout = 120` (measured `90.88s call`), and `timeout_method = "thread"` does not redden the
# item when it overruns: it `os._exit`s the whole session. Out here the cost lands in collection,
# where nothing is timing it.
#
# `except Exception`, not `except ImportError`, and the width is the point. At module scope a
# failure that is not caught aborts COLLECTION, so the whole session reports zero tests instead of
# a few red ones. That is not hypothetical on this chain: a CUDA torchvision beside a CPU torch
# raises `RuntimeError: operator torchvision::nms does not exist` from transformers' eager
# torchvision import, not an ImportError.
_voyage_import_error: str | None = None
try:
    import voyageai.error as voyage_error
except Exception as exc:
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
    openai = pytest.importorskip(
        "openai", reason="needs the bench extra (pip install recall-rag[bench])"
    )
    httpx = pytest.importorskip(
        "httpx", reason="openai's transport dep; only missing if it drops it"
    )

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


@pytest.mark.parametrize("status", [402, 499])
def test_a_status_below_the_band_is_not_retried(status: int) -> None:
    """Pins the 5xx floor from BELOW, which the band above cannot do.

    Testing 500 stops the floor being RAISED; nothing stopped it being LOWERED, and `402 <=
    status` left every test green. Two cases, because one does not do it: with only 402, floors
    at 413, 422, 451 and 499 all still survived, since a floor set above the value under test
    cannot reclassify it. 499 is the value immediately below the band, and it is what actually
    closes the floor: every downward move dies on that case alone.

    402 therefore earns its place as the real one rather than the boundary one, plus the single
    point-widening `status in (402, 429)` that only it kills. `benchmarks/llm.py` records an
    actual OpenRouter refusal, `402 ... You requested up to 65536 tokens, but can only afford
    64714`, that killed a BEAM run mid-arm. A credit refusal is permanent, and retrying it four
    times from that caller is the exact failure class this whole change exists to prevent.
    """
    exc = _StatusError("insufficient credits", status_code=status)
    assert _is_transient(exc) is False
    assert _attempts_used(exc) == 1


@pytest.mark.parametrize("status", [408, 409])
def test_a_server_returned_timeout_or_conflict_is_not_retried_here(status: int) -> None:
    """The exclusion `_is_transient`'s docstring calls deliberate, now enforced.

    openai's own client retries both (409 with the comment "Retry on lock timeouts"), so this is
    a place where we knowingly differ from the SDK. Widening the branch to `status in (408, 409,
    429)` left all fourteen tests green, which made "deliberate" a claim no test could check —
    the same shape as the defects the earlier rounds here fixed.

    Prose is marker-free on purpose: the natural wording for a 408 contains "timeout", which
    would let the fallback answer and hide whichever way the numeric branch went.
    """
    exc = _StatusError("the gateway gave up waiting", status_code=status)
    assert _is_transient(exc) is False
    assert _attempts_used(exc) == 1


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

    `openai.APIConnectionError` carries no status, and neither do voyageai's CLIENT-side errors,
    `Timeout` and `APIConnectionError`, which are raised with a message and nothing else. For
    those the text is the only evidence there is, and deleting the fallback would make every
    network blip fail on the first attempt. Voyage's SERVER-side errors are no longer in that
    set: they carry the response code in `http_status`, which is now read.
    """
    exc = RuntimeError("connection reset by peer")
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_the_fallback_matches_case_insensitively() -> None:
    """Pins the `.lower()` in the fallback, using the exact string the docstring leans on.

    Every marker is written lowercase, and `openai.APIConnectionError` stringifies to
    `"Connection error."` with a capital C. Dropping the `.lower()` retracts the fallback from
    the one shape `_is_transient`'s docstring names as depending on it.

    Not subsumed by `test_the_fallback_normalises_every_word_not_just_the_first`, though it looks
    like it should be. On the `.lower()` mutation the wider test is stronger: that mutation
    reddens eight items with the voyage extra installed and seven without it, four of them from
    the wider test against this test's one. But this message matches exactly ONE marker,
    "connection", where every message in the wider test matches two or matches a different one,
    so deleting "connection" reddens this test and leaves all four of those green. Both measured.
    """
    exc = RuntimeError("Connection error.")
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_an_unset_status_falls_back_to_text_markers() -> None:
    """An attribute that exists but was never filled in has told us nothing.

    It must read as a no-status error rather than as a status that is merely not 429, or the
    fallback is skipped on the evidence of an empty slot.

    This used to cite `voyageai.error.RateLimitError` as the exemplar, on the grounds that it
    "sets `http_status = None`". That was measured on a hand-constructed instance: the default is
    None, but `api_requestor` fills it with the real code on every error the SDK raises, which is
    what `test_a_real_voyage_rate_limit_is_retried_without_a_marker_in_its_message` now depends
    on. The genuine empty-slot shapes are voyageai's client-side `Timeout` and
    `APIConnectionError`, pinned below.
    """
    exc = _StatusError("rate limit exceeded", status_code=None)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_a_status_that_is_not_an_http_number_falls_back_to_text_markers() -> None:
    """`status` is not a reserved word: plenty of errors use it for a job state, not a status code.

    This is what `isinstance(status, int)` buys over `status is not None`, and it is the
    difference between falling back and crashing — `500 <= "error"` raises `TypeError` from
    inside the classifier, turning a retryable blip into a failure in the retry logic itself.

    Knowingly not covered: widening the gate to `isinstance(status, (int, float))` survives every
    test here. It diverges only on a float-valued status, which no transport in this repository
    produces, and it diverges in the safe direction anyway. Left alive on the same reasoning as
    the `< 600` upper bound, and recorded here so it reads as a decision rather than an oversight.

    That reasoning still governs after the third status spelling was added. `http_status` widens
    what can reach the numeric branch but not what the branch decides, so it gives no new route
    to a status of 600 or above, and the survivors #298 recorded stay survivors.
    """
    exc = _StatusError("connection reset by peer", status_code=None)
    exc.status = "error"  # type: ignore[attr-defined]
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3
# ---------------------------------------------------------------------------------------------
# The marker fallback, and the third spelling of the status that used to reach it.
# ---------------------------------------------------------------------------------------------


class _HttpStatusError(Exception):
    """An error spelling its HTTP status `http_status`, as every `voyageai` error does.

    Deliberately named so its own class name carries no marker. `voyageai`'s real
    `ServiceUnavailableError` is classified transient even with the numeric read removed, because
    the word "unavailable" happens to be in its class name, so a guard built on it would pass for
    the wrong reason and stay green if the `http_status` read were deleted again.
    """

    def __init__(self, message: str, http_status: int | None) -> None:
        super().__init__(message)
        self.http_status = http_status


#: One message per marker in `_TRANSIENT_MARKERS`, each carrying THAT marker and no other. The
#: isolation is not asserted by eye: `test_each_marker_case_matches_exactly_one_marker` proves it,
#: so deleting a marker reddens exactly the case that pins it.
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

    Equality rather than a subset check, and ordered, so the two lists are maintained together in
    both directions: a marker deleted from the classifier fails here too.
    """
    assert tuple(marker for marker, _ in MARKER_CASES) == _TRANSIENT_MARKERS


def test_each_marker_case_matches_exactly_one_marker() -> None:
    """Each case's message must isolate its own marker, or the case pins nothing.

    A message matching two markers stays green when either is deleted, which is precisely the
    failure this half of the file exists to prevent: "connection reset by peer", the phrase the
    fallback test above uses, carries both "connection" and "reset by peer", so it cannot pin
    either one.
    """
    for marker, message in MARKER_CASES:
        text = f"RuntimeError {message}".lower()
        matched = [m for m in _TRANSIENT_MARKERS if m in text]
        assert matched == [marker], f"{message!r} matched {matched}, expected only {marker!r}"


@pytest.mark.parametrize(
    ("marker", "message"), MARKER_CASES, ids=[marker.strip() for marker, _ in MARKER_CASES]
)
def test_a_status_less_error_is_retried_on_each_marker(marker: str, message: str) -> None:
    """Every marker is load bearing on its own, because where it is reached it is all there is."""
    exc = RuntimeError(message)
    assert _is_transient(exc) is True, f"marker {marker!r} did not classify as transient"
    assert _attempts_used(exc) == 3


@pytest.mark.parametrize(
    "message",
    [
        "Connection Reset By Peer",
        "Service Temporarily Unavailable",
        "Read Timeout",
        "Too Many Requests",
    ],
)
def test_the_fallback_normalises_every_word_not_just_the_first(message: str) -> None:
    """Widens the `.lower()` guard above past the single shape it uses.

    Every marker these messages match is capitalised in ALL of its words, which is what makes
    them bite. `RuntimeError("Connection reset by peer")` would NOT pin the normalisation, since
    "reset by peer" is still lowercase in it and matches with or without the `.lower()`.
    "Service Temporarily Unavailable" is the literal HTTP 503 reason phrase, so this is the shape
    a real 503 body arrives in.
    """
    exc = RuntimeError(message)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


@pytest.mark.parametrize("exc", [TimeoutError("boom"), ConnectionError("boom")])
def test_the_exception_type_name_is_part_of_the_matched_text(exc: Exception) -> None:
    """Pins the `type(exc).__name__` prefix on the matched text.

    `TimeoutError("boom")` and `ConnectionError("boom")` say nothing transient in their message;
    the evidence is entirely in the class name, and both are raised with an empty or unhelpful
    message by real client stacks. Dropping the prefix loses them.
    """
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_status_spelled_http_status_is_read_as_a_status(status: int) -> None:
    """voyageai's spelling has to reach the numeric branch, or Voyage decides on prose alone.

    `voyageai.error.VoyageError.__init__` takes `http_status` as its third positional argument and
    the request layer passes the real response code into it, so every Voyage error carries the
    number. The classifier read only `status_code` and `status`, so it never saw it, and the
    message decided instead. Those messages are fixed strings containing no marker at all: a real
    500 arrives as "The server failed to process the request." and was NOT retried.
    """
    exc = _HttpStatusError("boom", http_status=status)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_a_non_retryable_http_status_is_still_not_retried(status: int) -> None:
    """Reading `http_status` must not degrade into treating its presence as transient.

    An implementation that retried whenever `http_status` was set would pass the test above and
    turn a bad Voyage key into three paid refusals, which is the expensive direction.

    Deliberately no case at 600 or above. That would pin the `< 600` upper bound, which #298
    recorded as a knowing survivor, and the third spelling is not a reason to reopen a decision
    taken about the branch it feeds. See the note in
    `test_a_status_that_is_not_an_http_number_falls_back_to_text_markers`.
    """
    exc = _HttpStatusError("boom", http_status=status)
    assert _is_transient(exc) is False
    assert _attempts_used(exc) == 1


def test_an_http_status_that_was_never_filled_in_falls_back_to_the_markers() -> None:
    """The attribute PRESENT and None is the shape voyageai's client-side errors have.

    `Timeout` and `APIConnectionError` are raised with a message and nothing else, so
    `http_status` sits at its constructor default. Of the errors `api_requestor` raises, they are
    the only two the markers still decide, and they are the two most worth retrying. (Message-only
    raises exist elsewhere in the SDK as well, in the keyless-client and local-model checks and in
    the response-object layer. None carries a marker, so they fail fast, which is right for them.)

    Reading the attribute's PRESENCE rather than its value would stop those retrying. Measured
    with the extra installed, that mutant is caught here and by the voyage-gated twin below, and
    by nothing else. CI installs `.[dev]`, which has no voyage extra, so without THIS case the
    mutant would ship green there.
    """
    exc = _HttpStatusError("connection reset by peer", http_status=None)
    assert _is_transient(exc) is True
    assert _attempts_used(exc) == 3


def test_http_status_is_read_after_status_code_not_before() -> None:
    """The third lookup is ordered too, for the reason the first two are.

    Each limb is decisive, so an implementation reading `http_status` FIRST lets a stale or
    secondary value overrule the one the transport actually stated. No SDK sets both today, which
    is why this cannot be provoked from a real error, and it is exactly why a test has to say so:
    the ordering is otherwise free to drift with nothing to catch it.

    Both directions, as `test_status_code_is_read_before_status` does for the first two lookups.
    Pinning only the first left an implementation in which `http_status` may REFUSE but never
    grant, which survived every other case in this file. That direction suppresses a retry the
    stated status asked for, which costs an outage rather than a wasted request.
    """
    permanent_first = _StatusError("this request will never fit", status_code=400)
    permanent_first.http_status = 503  # type: ignore[attr-defined]
    assert _is_transient(permanent_first) is False
    assert _attempts_used(permanent_first) == 1

    transient_first = _StatusError("slow down", status_code=429)
    transient_first.http_status = 400  # type: ignore[attr-defined]
    assert _is_transient(transient_first) is True
    assert _attempts_used(transient_first) == 3


@requires_voyage
def test_a_real_voyage_client_side_error_still_decides_on_its_text() -> None:
    """The empty-slot case above, against the SDK's own classes rather than a stand-in.

    `api_requestor` raises both of these with a message alone, so the number this change added is
    genuinely absent here and the markers are the whole decision.

    Where the evidence sits differs between the two, which is why both are here. `Timeout`
    carries "timeout" in its class name and "timed out" in its message, two different markers.
    `APIConnectionError`'s message carries no marker at all, so its class name is the only
    evidence, and this case therefore depends on the `type(exc).__name__` prefix as well.
    """
    for exc in (
        voyage_error.Timeout("Request timed out"),
        voyage_error.APIConnectionError("Error communicating with VoyageAI"),
    ):
        assert exc.http_status is None, "the SDK started filling this in; the case needs rewriting"
        assert _is_transient(exc) is True
        assert _attempts_used(exc) == 3


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
    assert _attempts_used(exc) == 3


@requires_voyage
def test_a_real_voyage_rate_limit_is_retried_without_a_marker_in_its_message() -> None:
    """A real 429 whose body says nothing marker shaped still has to retry.

    The message on a Voyage 429 is the server's `detail` string, not a fixed phrase, and the class
    name "RateLimitError" does not contain the marker "rate limit" because the marker has a space
    in it. On the text alone this error is indistinguishable from a permanent one.
    """
    exc = voyage_error.RateLimitError("You have exceeded your quota.", None, 429)
    text = f"{type(exc).__name__} {exc}".lower()
    assert not [m for m in _TRANSIENT_MARKERS if m in text], "message accidentally carries a marker"
    assert _attempts_used(exc) == 3


@requires_voyage
def test_a_real_voyage_auth_error_fails_fast() -> None:
    """The other direction on the same SDK: a 401 is permanent and must not be paid for twice."""
    exc = voyage_error.AuthenticationError("Provided API key is invalid.", None, 401)
    assert _attempts_used(exc) == 1
