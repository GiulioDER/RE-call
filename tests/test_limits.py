"""Per-tenant rate limits and the indexing spend quota.

The limiter is deliberately pure arithmetic over an injected clock, so every one of these runs
without a database, a transport, or a sleep. A rate limiter tested with `time.sleep` is a slow
test that also cannot express "and now it is exactly one hour later".
"""
from __future__ import annotations

import threading

import pytest

from recall_mcp.limits import (
    DEFAULT_CALLS_PER_MIN,
    OFF,
    Rate,
    RateLimited,
    RateLimiter,
    _Bucket,
    _rate_from_env,
    limiter_from_env,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter(capacity: float, per_second: float, clock: FakeClock) -> RateLimiter:
    return RateLimiter({"read": Rate(capacity, per_second)}, clock=clock)


# --------------------------------------------------------------------------------------------
# The bucket
# --------------------------------------------------------------------------------------------


def test_a_tenant_may_spend_its_full_burst_immediately():
    """A fresh tenant starts full — nobody should be throttled on their first call."""
    clock = FakeClock()
    lim = _limiter(5, 1.0, clock)
    for _ in range(5):
        lim.check("acme", "read")


def test_the_call_after_the_burst_is_refused_with_a_usable_retry():
    clock = FakeClock()
    lim = _limiter(5, 1.0, clock)  # 1 token/sec
    for _ in range(5):
        lim.check("acme", "read")

    with pytest.raises(RateLimited) as exc:
        lim.check("acme", "read")
    assert exc.value.retry_after_seconds == pytest.approx(1.0)


def test_waiting_the_advertised_time_actually_works():
    """The retry_after must be honest: obeying it has to succeed, or clients hot-loop on it."""
    clock = FakeClock()
    lim = _limiter(5, 1.0, clock)
    for _ in range(5):
        lim.check("acme", "read")
    with pytest.raises(RateLimited) as exc:
        lim.check("acme", "read")

    clock.advance(exc.value.retry_after_seconds)
    lim.check("acme", "read")  # must not raise


def test_refill_is_capped_at_the_burst_size():
    """An idle tenant banks tokens up to capacity, not forever — otherwise a quiet week funds a
    flood that the limit was written to prevent."""
    clock = FakeClock()
    lim = _limiter(5, 1.0, clock)
    clock.advance(10_000)

    for _ in range(5):
        lim.check("acme", "read")
    with pytest.raises(RateLimited):
        lim.check("acme", "read")


def test_one_tenants_spending_does_not_throttle_another():
    """The whole point of per-TENANT budgets."""
    clock = FakeClock()
    lim = _limiter(2, 1.0, clock)
    lim.check("acme", "read")
    lim.check("acme", "read")
    with pytest.raises(RateLimited):
        lim.check("acme", "read")

    lim.check("globex", "read")  # unaffected
    lim.check("globex", "read")


def test_budgets_are_independent_per_key():
    """Exhausting reads must not lock a tenant out of forgetting its own data."""
    clock = FakeClock()
    lim = RateLimiter({"read": Rate(1, 1.0), "forget": Rate(1, 1.0)}, clock=clock)
    lim.check("acme", "read")
    with pytest.raises(RateLimited):
        lim.check("acme", "read")

    lim.check("acme", "forget")


def test_a_cost_larger_than_the_bucket_is_refused_as_impossible_not_throttled():
    """A 1 GB index against a 200 MB quota can never succeed by waiting.

    Returning a retry_after here would be a lie the client would obey forever, so it is reported
    as permanent and the message says to raise the limit.
    """
    clock = FakeClock()
    lim = RateLimiter({"index_bytes": Rate(100.0, 1.0)}, clock=clock)

    with pytest.raises(RateLimited) as exc:
        lim.check("acme", "index_bytes", 500.0)
    assert exc.value.retry_after_seconds == 0.0
    assert "never succeed" in str(exc.value)


def test_an_unconfigured_key_is_unlimited():
    """This is how `off` is represented — the disabled path must cost nothing and never raise."""
    lim = RateLimiter({}, clock=FakeClock())
    for _ in range(1000):
        lim.check("acme", "read")


def test_a_zero_cost_call_does_not_consume_budget():
    """Indexing an empty directory embeds nothing, so it should not be billed."""
    clock = FakeClock()
    lim = RateLimiter({"index_bytes": Rate(10.0, 1.0)}, clock=clock)
    for _ in range(50):
        lim.check("acme", "index_bytes", 0)
    lim.check("acme", "index_bytes", 10.0)  # full budget still there


def test_a_clock_that_goes_backwards_does_not_grant_free_refills():
    clock = FakeClock()
    lim = _limiter(2, 1.0, clock)
    lim.check("acme", "read")
    lim.check("acme", "read")

    clock.now -= 500  # should be impossible with a monotonic clock; must not pay out anyway
    with pytest.raises(RateLimited):
        lim.check("acme", "read")

    # ...and it must still be refused on the NEXT call. Clamping only the elapsed delta refuses
    # the payout on the backwards reading itself and then moves the reference point back with it,
    # so the following call measures from the rewound timestamp and mints the whole rewound
    # interval — the free refill deferred by one call rather than denied. Stopping at the
    # assertion above is what let that through.
    clock.now += 3  # three real seconds: worth 3 tokens, not 503
    with pytest.raises(RateLimited):
        lim.check("acme", "read")


def test_an_out_of_order_clock_reading_cannot_rewind_the_bucket():
    """The same rewind, reachable WITHOUT a clock jump.

    `check` reads the clock and then takes the lock, so two threads can acquire in the opposite
    order to their readings and the later-acquiring thread presents the older `now`. Driven here
    directly against the bucket, which is what that interleaving produces: a stale reading must
    not move the reference point backwards, or the next honest reading pays out the difference.
    """
    b = _Bucket(Rate(capacity=10.0, per_second=1.0), now=0.0)
    assert b.take(10.0, 20.0) == 0.0     # drained at t=20; reference point is now 20
    assert b.take(1.0, 10.0) > 0.0       # stale reading arriving late: credits nothing, refused

    # Three real seconds after the drain is worth three tokens. Had the stale reading moved the
    # reference point back to t=10, the same call would measure thirteen seconds and pay out the
    # bucket's full capacity instead.
    assert b.take(5.0, 23.0) > 0.0, (
        "a stale reading rewound the bucket: three real seconds paid out five or more tokens"
    )


def test_concurrent_callers_cannot_overspend_the_budget():
    """The lock is load-bearing: tool bodies run in worker threads, so this race is real.

    Without it, two threads both read "enough tokens" before either writes, and the tenant spends
    more than its budget — the failure a limiter exists to prevent, under exactly the concurrency
    the server runs at.
    """
    clock = FakeClock()
    lim = _limiter(50, 0.000001, clock)  # effectively no refill during the test
    allowed = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        for _ in range(10):
            try:
                lim.check("acme", "read")
                allowed.append(1)
            except RateLimited:
                pass

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(allowed) == 50, f"budget was 50 but {len(allowed)} calls were admitted"


def test_a_rate_must_be_positive():
    with pytest.raises(ValueError):
        Rate(0, 1.0)
    with pytest.raises(ValueError):
        Rate(1.0, -1.0)


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


def test_defaults_apply_when_nothing_is_configured(monkeypatch):
    for scope in DEFAULT_CALLS_PER_MIN:
        monkeypatch.delenv(f"RECALL_RATE_{scope.upper()}_PER_MIN", raising=False)
    monkeypatch.delenv("RECALL_INDEX_BYTES_PER_HOUR", raising=False)

    limits = limiter_from_env().limits()
    assert set(limits) == {"read", "write", "forget", "index_bytes"}
    assert limits["read"].capacity == DEFAULT_CALLS_PER_MIN["read"]


def test_off_disables_exactly_one_limit(monkeypatch):
    monkeypatch.setenv("RECALL_RATE_READ_PER_MIN", OFF)
    limits = limiter_from_env().limits()
    assert "read" not in limits
    assert "write" in limits and "index_bytes" in limits


@pytest.mark.parametrize(
    "raw",
    [
        "", "lots", "0", "-5", "nan", "inf", "Infinity", "+inf", "-inf", "1e400", "2" + "0" * 400,
        # The underflow end of the same range: the smallest positive double. It is finite and
        # positive, so every check above passes it — but `value / window_seconds` is exactly
        # 0.0, which `Rate` rejects. Unhandled, that ValueError comes out of
        # `limiter_from_env()` and the server never starts: a fall-back-to-default contract
        # that instead kills the process. (Merely *tiny* values like 1e-320 are not this bug —
        # their quotient is still a nonzero subnormal, so they build a valid, useless Rate.
        # Refusing those would be a new policy about minimum sane rates, not a fix.)
        "5e-324",
    ],
)
def test_a_malformed_or_non_positive_limit_falls_back_to_the_default(monkeypatch, raw):
    """Never read as "unlimited" — and never fatal either.

    `0` is the dangerous one: it reads as "no limit" to one person and "nothing allowed" to
    another. Guessing wrong in a spend control silently removes the cap, so it is refused and the
    documented way to disable a limit is the word `off`.

    The infinities are the same mistake wearing a different hat, and they are the ones that
    actually reach production: `float()` parses `inf` happily and, worse, OVERFLOWS a long
    numeric literal to it — so an operator typing a generous budget with too many zeros gets an
    unlimited bucket. Unlike `off`, which announces itself in the log, that removal is silent.

    The subnormals are that overflow reflected: a value small enough that the derived per-second
    rate underflows to zero. Validating only the parsed number catches one end of the range and
    not the other, and the miss is the worse of the two — it raises out of startup rather than
    being logged and defaulted.
    """
    monkeypatch.setenv("RECALL_RATE_WRITE_PER_MIN", raw)
    limits = limiter_from_env().limits()
    assert limits["write"].capacity == DEFAULT_CALLS_PER_MIN["write"]


def test_a_valid_override_is_applied(monkeypatch):
    monkeypatch.setenv("RECALL_RATE_WRITE_PER_MIN", "7")
    limits = limiter_from_env().limits()
    assert limits["write"].capacity == 7.0
    assert limits["write"].per_second == pytest.approx(7.0 / 60.0)


def test_the_byte_quota_is_expressed_per_hour(monkeypatch):
    monkeypatch.setenv("RECALL_INDEX_BYTES_PER_HOUR", "3600")
    rate = limiter_from_env().limits()["index_bytes"]
    assert rate.capacity == 3600.0
    assert rate.per_second == pytest.approx(1.0)


def test_rate_from_env_reports_disabled_as_none(monkeypatch):
    monkeypatch.setenv("X", OFF)
    assert _rate_from_env("X", 10.0, 60.0) is None
    monkeypatch.setenv("X", "OFF")  # case-insensitive
    assert _rate_from_env("X", 10.0, 60.0) is None
