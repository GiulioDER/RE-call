"""The layer that owns retries must also honour the pacing the provider asks for.

Switching the SDK's own retry layer off made `retry_with_backoff` the only retry policy in the
cloud embedding path. That layer did one thing this one did not: `openai`'s
`_calculate_retry_timeout` obeys a `Retry-After` header up to 60s in preference to its own
backoff. Without it, the whole budget for a 429 is three requests inside 1.5s worst case
(`uniform(0, 0.5)` then `uniform(0, 1.0)`), which for a per-minute rate limit means all three
land in the same closed window and `embed()` raises on a corpus indexing run that would
otherwise have recovered.

So the tests here are not about the header being parsed. They are about the budget spanning the
window the provider named.

The exceptions are REAL ones, produced by a real 429 over a real socket, because the thing under
test is where the header ends up on the SDK's exception object — which a hand-built stand-in
would simply assert into existence.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import pytest

from recall.embeddings import retry_with_backoff
from tests.provider_stub_helpers import EMBEDDINGS_OK, provider_stub

openai = pytest.importorskip("openai")

# `voyageai` is NOT imported here. It drags `transformers` and `torch` in behind it, and with the
# module-scope import this file took **45.33s to COLLECT on its own, against 1.04s now** (measured
# back to back, warm, 2026-08-23). That is what you paid every time you ran this one file while
# working on it. It comes from the session-scoped `voyageai_sdk` fixture in `tests/conftest.py`
# instead, which is why the two tests below take that fixture and carry a 300s timeout: whichever
# runs first is billed for the import. The fixture's docstring carries the whole-suite figure,
# which is much smaller, and the correction to an earlier claim that it was much larger.


def _rate_limited_with(headers: dict[str, str]) -> Exception:
    """A real `openai.RateLimitError` carrying `headers`, from an actual 429 response."""
    with provider_stub(EMBEDDINGS_OK) as stub:
        stub.arm(429, "please slow down")
        stub.headers = headers
        client = stub.track(
            openai.OpenAI(api_key="k", base_url=stub.base_url, max_retries=0, timeout=5.0)
        )
        try:
            client.embeddings.create(model="m", input=["x"], encoding_format="float")
        except openai.RateLimitError as exc:
            return exc
    raise AssertionError("the stub did not produce a rate-limit error")


def _delays(exc: Exception, *, attempts: int = 2) -> list[float]:
    """Every delay `retry_with_backoff` takes while exhausting `attempts` on `exc`.

    The recorder enforces `time.sleep`'s own contract instead of being a bare `list.append`.
    Every "fell back to the jittered draw" assertion in this file is one-sided (`<= 2.0`), and a
    NEGATIVE delay satisfies all of them — so dropping the `0.0 <` limb from `_capped` left the
    whole suite green while making the real `time.sleep` raise `ValueError` from inside
    `retry_with_backoff`'s except block, replacing the provider's error. One line here closes
    that for every case at once, which per-test bounds would not.
    """
    slept: list[float] = []

    def _record(delay: float) -> None:
        assert delay >= 0.0, f"time.sleep would reject this: {delay}"
        slept.append(delay)

    def _always_fails() -> None:
        raise exc

    with pytest.raises(type(exc)):
        retry_with_backoff(_always_fails, attempts=attempts, sleep=_record)
    return slept


def test_a_retry_after_header_paces_the_next_attempt() -> None:
    """A provider asking for 20s must not be re-asked in under a second.

    20 is deliberately longer than the entire unpaced budget: no combination of the jittered
    draws can produce this delay, so the assertion cannot pass by accident.
    """
    delays = _delays(_rate_limited_with({"Retry-After": "20"}))

    assert len(delays) == 1
    assert delays[0] >= 20.0


def test_retry_after_ms_is_honoured_too() -> None:
    """OpenAI and its compatible proxies send the millisecond spelling more often than not."""
    delays = _delays(_rate_limited_with({"retry-after-ms": "9000"}))

    assert len(delays) == 1
    assert 9.0 <= delays[0] < 10.0


def test_the_http_date_spelling_is_honoured_too() -> None:
    """`Retry-After` is defined as either a delay in seconds or an HTTP-date, and real providers
    send both. Reading only the integer form would leave exactly the gap this file exists to
    close, for the half of providers that chose the other spelling."""
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    started = time.monotonic()
    delays = _delays(_rate_limited_with({"Retry-After": format_datetime(when, usegmt=True)}))
    elapsed = time.monotonic() - started

    # The wait is recomputed against `now` when the error is classified, so it decays by however
    # long the stub, the client and the request took. Asserting a fixed 25.0 floor would leave
    # ~4s of slack on a loaded CI runner (`format_datetime` also floors to whole seconds), and a
    # smaller stall would weaken the test silently rather than fail it.
    assert len(delays) == 1
    assert delays[0] >= 30.0 - elapsed - 1.0
    assert delays[0] <= 30.5


def test_an_absurd_retry_after_is_ignored_rather_than_obeyed() -> None:
    """A header asking for an hour must not park a corpus indexing run for an hour.

    60s is the same ceiling the openai SDK applies, so this keeps the behaviour the SDK layer had
    rather than inventing a more trusting one. Past the cap the policy falls back to its own
    backoff, which is the pre-existing behaviour and bounded.
    """
    delays = _delays(_rate_limited_with({"Retry-After": "3600"}))

    assert len(delays) == 1
    assert delays[0] <= 2.0


def test_a_malformed_retry_after_falls_back_to_the_jittered_draw() -> None:
    """A provider that sends nonsense must not take the retry path down with it."""
    delays = _delays(_rate_limited_with({"Retry-After": "soon"}))

    assert len(delays) == 1
    assert delays[0] <= 2.0


@pytest.mark.timeout(300)  # the `voyageai_sdk` import, if this is the first to ask
def test_a_voyage_error_is_paced_too_though_it_carries_no_response(voyageai_sdk: Any) -> None:
    """The second cloud embedder must not be quietly left out of its own fix.

    ``voyageai`` hangs the headers straight off the exception (``VoyageError.headers``) and has
    no ``response`` attribute at all, so a reader that only walks ``exc.response.headers``
    returns None for every Voyage error and the pacing silently covers one call site of the two
    it claims. Voyage does send the header; its own requestor reads it.

    The headers are built as the CASE-INSENSITIVE mapping the SDK actually hands over
    (`requests.Response.headers`), spelled the RFC's canonical way. A plain dict with a lowercase
    key would pass while asserting a shape voyageai never produces, and would leave the reader
    free to become case-sensitive without anything going red — which is the live risk, since the
    reader looks up lowercase keys only.
    """
    from requests.structures import CaseInsensitiveDict

    exc = voyageai_sdk.error.RateLimitError(
        "rate limit exceeded", headers=CaseInsensitiveDict({"Retry-After": "20"})
    )

    delays = _delays(exc)

    assert len(delays) == 1
    assert delays[0] >= 20.0


@pytest.mark.timeout(300)  # the `voyageai_sdk` import, if this is the first to ask
def test_a_header_value_that_is_not_a_string_falls_back_instead_of_raising(voyageai_sdk: Any) -> None:
    """The reader duck-types, so it must survive a mapping that is not ``httpx.Headers``.

    The reader duck-types whatever mapping it is handed, and an SDK's ``headers`` can hold a
    value of any type whichever mapping class carries it. ``parsedate_to_datetime``
    raises ``AttributeError`` rather than ``ValueError`` on a non-string (it calls ``.split()``
    before validating), and that escapes a ``(TypeError, ValueError)`` guard. Inside
    ``retry_with_backoff``'s ``except Exception`` block it would replace the provider's error, so
    an indexing run dies on a stdlib string-method error instead of retrying.
    """
    exc = voyageai_sdk.error.RateLimitError("rate limit exceeded", headers={"retry-after": ["20"]})

    delays = _delays(exc)

    assert len(delays) == 1
    assert delays[0] <= 2.0


def test_no_header_leaves_the_full_jitter_backoff_exactly_as_it_was() -> None:
    """The unpaced path is unchanged: two delays, each a draw under its own doubling cap.

    Asserting the ceilings rather than the values keeps this a test of the schedule rather than
    of `random`, and it is what would catch the header work leaking a constant into the case
    where no header was sent.
    """
    delays = _delays(_rate_limited_with({}), attempts=3)

    assert len(delays) == 2
    assert 0.0 <= delays[0] <= 0.5
    assert 0.0 <= delays[1] <= 1.0


def test_the_cap_is_honoured_at_60_and_refused_at_61() -> None:
    """Pins `_MAX_RETRY_AFTER_S` from BOTH sides.

    The suite previously bounded the cap only from above, with 3600 as the sole absurd value, so
    raising the constant to 600.0 left every test green. That is the mutant that matters: the
    cap's whole justification is that a header must not park a corpus index inside what the
    caller believes is a bounded retry, and ten minutes is exactly that parking.
    """
    at_cap = _delays(_rate_limited_with({"Retry-After": "60"}))
    assert at_cap[0] >= 60.0

    over_cap = _delays(_rate_limited_with({"Retry-After": "61"}))
    assert over_cap[0] <= 2.0  # back on the jittered draw, whose first cap is 0.5


def test_the_jitter_is_added_on_top_of_what_the_provider_asked_for() -> None:
    """Pins the decision `retry_with_backoff`'s docstring argues for explicitly.

    Sleeping exactly `asked` satisfies every one-sided `>= 20.0` assertion in this file, so the
    "ON TOP rather than drawn within" choice was unpinned. It is not cosmetic: a fleet handed the
    same `Retry-After` by the same provider wakes in lockstep without it, which is the thundering
    herd the jitter exists to break up.
    """
    exc = _rate_limited_with({"Retry-After": "20"})
    seen = {_delays(exc)[0] for _ in range(12)}

    assert all(d >= 20.0 for d in seen)  # never less than the provider asked for
    assert len(seen) > 1, f"every draw slept identically: {seen}"


def test_an_http_date_without_a_zone_is_read_as_utc() -> None:
    """Pins the naive-datetime branch, which no test reached.

    RFC 9110 says a date with no zone is UTC. Without the `tzinfo` fix, subtracting an aware
    `now` raises `TypeError` inside `_read_retry_after` — which the blanket `except` then turns
    into a SILENT loss of pacing rather than a crash, so nothing would have gone red.
    """
    naive = (datetime.now(timezone.utc) + timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S")
    delays = _delays(_rate_limited_with({"Retry-After": naive}))

    assert 20.0 <= delays[0] < 40.0


def test_a_junk_millisecond_header_does_not_discard_a_good_one_beside_it() -> None:
    """`retry-after-ms` is read first, and used to commit before it was known to parse.

    An unparseable value returned None from inside the blanket `except`, throwing away the
    `Retry-After` sitting right next to it and dropping the caller back onto the 1.5s unpaced
    budget — the precise failure this module exists to prevent. openai's own reader falls
    through; this one did not.
    """
    delays = _delays(_rate_limited_with({"retry-after-ms": "soon", "Retry-After": "20"}))

    assert delays[0] >= 20.0


def test_the_cap_applies_to_every_spelling_not_just_plain_seconds() -> None:
    """`_capped` is called from three branches; only one of them was pinned.

    Deleting `_capped` from the `retry-after-ms` branch, or from the HTTP-date branch, left the
    whole retry suite green — so a proxy answering `retry-after-ms: 3600000`, or a date an hour
    ahead, could park a corpus index for an hour inside what the caller believes is a bounded
    retry. That is the precise failure the constant's own comment says it exists to prevent, and
    it was reachable through two of the three doors.
    """
    an_hour_in_ms = _delays(_rate_limited_with({"retry-after-ms": "3600000"}))
    assert an_hour_in_ms[0] <= 2.0

    an_hour_ahead = format_datetime(datetime.now(timezone.utc) + timedelta(hours=1), usegmt=True)
    assert _delays(_rate_limited_with({"Retry-After": an_hour_ahead}))[0] <= 2.0


def test_a_date_in_the_past_is_not_treated_as_a_wait() -> None:
    """The lower limb of `_capped` on the date branch, which nothing reached.

    A provider whose clock is behind ours, or a retry of a response held in a proxy, yields a
    negative interval. `0.0 < seconds` refuses it; without that the delay would be negative and
    `time.sleep` would raise `ValueError` inside the retry loop.
    """
    past = format_datetime(datetime.now(timezone.utc) - timedelta(minutes=5), usegmt=True)

    assert _delays(_rate_limited_with({"Retry-After": past}))[0] <= 2.0


def test_the_millisecond_spelling_wins_when_both_are_sent() -> None:
    """OpenAI sends both headers together, so which one wins is a real decision, not a tie-break.

    Its own reader prefers the millisecond form for being more precise than integer seconds, and
    this one is written to match. Nothing pinned it: the only other test carrying both headers
    has an unparseable ms value, so it cannot tell the two orders apart — the same one-sided
    asymmetry `test_status_code_is_read_before_status` exists to close for `_is_transient`.
    """
    delays = _delays(_rate_limited_with({"retry-after-ms": "9000", "Retry-After": "30"}))

    assert 9.0 <= delays[0] < 10.5, f"the seconds spelling appears to have won: {delays}"
