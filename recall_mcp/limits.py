"""Per-tenant rate limits and an indexing spend quota.

The pre-flight caps in `service.index_memory` bound ONE request: at most
`RECALL_INDEX_MAX_FILES` files and `RECALL_INDEX_MAX_BYTES` bytes. They say nothing about how
many requests a client may make, so a caller that stays politely under the per-call cap can
still issue it in a loop and direct unbounded cloud-embedding spend. This module bounds the
aggregate.

WHY THE TENANT AND NOT THE PRINCIPAL. Two tokens issued to the same tenant are the same
blast radius and the same bill; letting a tenant multiply its budget by minting another token
would make the quota advisory. Rate limiting the principal instead would isolate two clients
from each other but not cap the tenant, which is the thing you actually pay for.

ONE PRIMITIVE, TWO USES. A token bucket meters calls and bytes identically — the only
difference is what a token represents and how fast it refills. Calls debit 1; an index request
debits the byte count it is about to embed. Bytes are the load-bearing one: request COUNT is a
poor proxy for spend when one call can carry 20 MB and the next 200 bytes.

BUCKETS ARE PER PROCESS, AND THAT IS A REAL LIMIT. Nothing is shared across workers, so N
server processes admit roughly N times these rates. This is honest for the deployment the auth
work targets — one process behind TLS — and it is the first thing to revisit before running a
fleet. A shared limiter needs Redis or the database, and a network round trip on every call.

FAILS OPEN BY CONFIGURATION, NEVER BY ACCIDENT. A limit can be switched off, but only by
writing `off`; anything malformed falls back to the default rather than being read as
"unlimited". A typo must not silently remove the cap.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from recall.observability import get_logger

_log = get_logger("limits")

#: Scope-keyed call budgets. Reads are cheap and interactive, so they get a wide allowance;
#: writes embed and cost money; forget is irreversible and no agent needs to do it in bulk.
#: `capacity` is the burst — a client may spend it at once, then is paced by `per_second`.
DEFAULT_CALLS_PER_MIN: dict[str, float] = {
    "read": 120.0,
    "write": 20.0,
    "forget": 10.0,
}
#: Aggregate embedding spend, in bytes of source text per hour per tenant. 200 MB is ~10x the
#: 20 MB single-request cap, so an ordinary re-index of a large corpus fits comfortably while a
#: loop calling `recall_index` at the cap is stopped after ten iterations rather than never.
DEFAULT_INDEX_BYTES_PER_HOUR = 200 * 1024 * 1024

_SECONDS_PER_MIN = 60.0
_SECONDS_PER_HOUR = 3600.0
#: The literal that disables a limit. A WORD, not a number: `0` reads as both "no limit" and
#: "nothing allowed" depending on who is looking, and that ambiguity in a spend control is how
#: a cap gets removed by someone who meant to tighten it.
OFF = "off"


class RateLimited(RuntimeError):
    """A tenant exceeded its budget. Carries the wait so a caller can be told when to retry."""

    def __init__(self, message: str, *, retry_after_seconds: float) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class Rate:
    """`capacity` tokens, refilled at `per_second`. Both must be > 0."""

    capacity: float
    per_second: float

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.per_second <= 0:
            raise ValueError("capacity and per_second must both be > 0")


class _Bucket:
    """A single token bucket. Not thread-safe on its own — `RateLimiter` holds the lock."""

    __slots__ = ("_rate", "_tokens", "_updated")

    def __init__(self, rate: Rate, now: float) -> None:
        self._rate = rate
        self._tokens = rate.capacity  # start full: a fresh tenant is not born throttled
        self._updated = now

    def take(self, cost: float, now: float) -> float:
        """Debit `cost`. Returns 0.0 on success, else the seconds until it would succeed."""
        # Refill for elapsed time first. `max(0, ...)` because a monotonic clock should never go
        # backwards, but a bucket that credited itself on a negative delta would be a free
        # refill, so the arithmetic refuses rather than trusts.
        elapsed = max(0.0, now - self._updated)
        # `max` on the WRITE too, not just on the delta. Clamping only `elapsed` refuses the free
        # refill on the backwards reading itself but then moves the reference point back, so the
        # NEXT call measures from the rewound timestamp and mints the whole rewound interval —
        # deferring the free refill by one call rather than denying it. This is reachable without
        # any clock jump: `check` reads the clock outside the lock, so two threads can enter in
        # the opposite order to their readings.
        self._updated = max(self._updated, now)
        self._tokens = min(self._rate.capacity, self._tokens + elapsed * self._rate.per_second)

        if cost > self._rate.capacity:
            # Larger than the bucket can EVER hold: waiting cannot help, so this is not a
            # throttle but a permanent refusal, and it must say so rather than hand back a
            # retry_after that will fail identically forever.
            raise RateLimited(
                f"request costs {cost:,.0f} but the budget holds at most "
                f"{self._rate.capacity:,.0f} — it can never succeed; raise the limit",
                retry_after_seconds=0.0,
            )
        if self._tokens >= cost:
            self._tokens -= cost
            return 0.0
        return (cost - self._tokens) / self._rate.per_second


class RateLimiter:
    """Per-(tenant, key) token buckets behind one lock.

    The lock is not decorative: tool bodies run in worker threads (`_to_thread`), so the byte
    debit for an index call genuinely races other requests. Read-modify-write on a bucket
    without it would let two concurrent calls each see enough tokens and both spend them.
    """

    def __init__(
        self,
        rates: dict[str, Rate],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rates = dict(rates)
        self._clock = clock
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def limits(self) -> dict[str, Rate]:
        return dict(self._rates)

    def check(self, tenant: str, key: str, cost: float = 1.0) -> None:
        """Debit `cost` from `tenant`'s `key` budget, or raise `RateLimited`.

        A key with no configured rate is unlimited — that is how `off` is represented, so the
        disabled case costs nothing and cannot itself fail.
        """
        rate = self._rates.get(key)
        if rate is None:
            return
        if cost <= 0:
            return  # nothing to meter; an empty index request should not consume a token

        # Clock read INSIDE the lock: read outside it, two threads can acquire in the opposite
        # order to their readings, so the later-acquiring thread presents an older `now` and
        # rewinds the bucket's reference point. A monotonic read is nanoseconds and the lock
        # already spans a dict lookup, so making the read and the read-modify-write one critical
        # section costs nothing measurable.
        with self._lock:
            now = self._clock()
            bucket = self._buckets.get((tenant, key))
            if bucket is None:
                # Bounded by the token file's tenant set, which is fixed at startup, so this
                # dict cannot grow without limit from client input.
                bucket = _Bucket(rate, now)
                self._buckets[(tenant, key)] = bucket
            wait = bucket.take(cost, now)

        if wait > 0:
            _log.warning("tenant %r rate-limited on %s (retry in %.1fs)", tenant, key, wait)
            raise RateLimited(
                f"rate limit exceeded for {key!r}: budget is {rate.capacity:,.0f} with "
                f"{rate.per_second * _SECONDS_PER_MIN:,.1f} restored per minute. "
                f"Retry in {wait:.1f}s.",
                retry_after_seconds=wait,
            )


def _rate_from_env(name: str, default: float, window_seconds: float) -> Rate | None:
    """Read one limit. Returns None when explicitly disabled; the default when malformed."""
    raw = os.environ.get(name)
    if raw is None:
        value = default
    elif raw.strip().lower() == OFF:
        _log.warning("%s=off — this limit is disabled", name)
        return None
    else:
        try:
            value = float(raw)
        except ValueError:
            _log.warning("ignoring malformed %s=%r; using default %s", name, raw, default)
            value = default
        else:
            # Rejected rather than clamped, and NOT read as "unlimited": a 0 or negative here is
            # someone reaching for the off switch with a number, and guessing which way they
            # meant it is the one mistake a spend control must not make.
            #
            # `isfinite` covers NaN and, more importantly, the INFINITIES. `float()` parses
            # "inf" happily, and it also OVERFLOWS any sufficiently long numeric literal to it
            # WITHOUT raising — so an operator typing a generous budget with one zero too many
            # gets an unlimited bucket. That is precisely the failure this module exists to
            # prevent, arriving by the most ordinary route available, and it is silent: `off`
            # announces itself in the log, an accidental infinity would not.
            #
            # The DERIVED rate is validated too, not just the parsed capacity. A value can be
            # finite and positive and still divide to exactly 0.0 — any subnormal smaller than
            # `window_seconds * sys.float_info.min` underflows — and `Rate.__post_init__` then
            # raises ValueError straight out of `limiter_from_env()`, killing the server at
            # startup. That breaks this module's contract in the one direction it promises never
            # to break it: a bad value falls back to the default, it does not take the process
            # down. Overflow was closed above; this is the same hole at the other end.
            if not math.isfinite(value) or value <= 0 or value / window_seconds <= 0:
                _log.warning(
                    "ignoring %s=%r — not a finite positive number, or too small to yield a "
                    "non-zero rate (use %r to disable); using default %s",
                    name, raw, OFF, default,
                )
                value = default
    return Rate(capacity=value, per_second=value / window_seconds)


def limiter_from_env() -> RateLimiter:
    """Build the limiter the server uses, from `RECALL_RATE_*` / `RECALL_INDEX_BYTES_PER_HOUR`."""
    rates: dict[str, Rate] = {}
    for scope, default in DEFAULT_CALLS_PER_MIN.items():
        rate = _rate_from_env(f"RECALL_RATE_{scope.upper()}_PER_MIN", default, _SECONDS_PER_MIN)
        if rate is not None:
            rates[scope] = rate
    byte_rate = _rate_from_env(
        "RECALL_INDEX_BYTES_PER_HOUR", float(DEFAULT_INDEX_BYTES_PER_HOUR), _SECONDS_PER_HOUR
    )
    if byte_rate is not None:
        rates["index_bytes"] = byte_rate
    return RateLimiter(rates)
