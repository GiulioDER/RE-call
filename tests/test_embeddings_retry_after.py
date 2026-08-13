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

import pytest

from recall.embeddings import retry_with_backoff
from tests.provider_stub_helpers import EMBEDDINGS_OK, provider_stub

openai = pytest.importorskip("openai")

#: Module scope for the same reason as in `test_embeddings_retry_policy`: `voyageai` drags
#: `transformers` in behind it and takes ~75s cold, which is most of the 120s per-test timeout,
#: and collection is not clocked. `importorskip` would skip the whole module for its absence.
try:
    import voyageai.error
except ImportError:  # pragma: no cover - exercised only without the extra
    voyageai = None  # type: ignore[assignment]


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
    """Every delay `retry_with_backoff` takes while exhausting `attempts` on `exc`."""
    slept: list[float] = []

    def _always_fails() -> None:
        raise exc

    with pytest.raises(type(exc)):
        retry_with_backoff(_always_fails, attempts=attempts, sleep=slept.append)
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


def test_a_voyage_error_is_paced_too_though_it_carries_no_response() -> None:
    """The second cloud embedder must not be quietly left out of its own fix.

    ``voyageai`` hangs the headers straight off the exception (``VoyageError.headers``) and has
    no ``response`` attribute at all, so a reader that only walks ``exc.response.headers``
    returns None for every Voyage error and the pacing silently covers one call site of the two
    it claims. Voyage does send the header; its own requestor reads it.
    """
    if voyageai is None:
        pytest.skip("needs the voyage extra (pip install recall[voyage])")
    exc = voyageai.error.RateLimitError("rate limit exceeded", headers={"retry-after": "20"})

    delays = _delays(exc)

    assert len(delays) == 1
    assert delays[0] >= 20.0


def test_a_header_value_that_is_not_a_string_falls_back_instead_of_raising() -> None:
    """The reader duck-types, so it must survive a mapping that is not ``httpx.Headers``.

    Voyage's ``headers`` is a plain dict, which can hold anything. ``parsedate_to_datetime``
    raises ``AttributeError`` rather than ``ValueError`` on a non-string (it calls ``.split()``
    before validating), and that escapes a ``(TypeError, ValueError)`` guard. Inside
    ``retry_with_backoff``'s ``except Exception`` block it would replace the provider's error, so
    an indexing run dies on a stdlib string-method error instead of retrying.
    """
    if voyageai is None:
        pytest.skip("needs the voyage extra (pip install recall[voyage])")
    exc = voyageai.error.RateLimitError("rate limit exceeded", headers={"retry-after": ["20"]})

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
