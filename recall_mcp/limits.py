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

BUCKETS ARE LOCAL OR FLEET-WIDE BY CONFIGURATION. The local limiter is per process and is suitable
for stdio or one-process development. Authenticated fleet deployments must select Redis or Valkey,
which shares tenant buckets across tasks using an atomic Lua reservation and server time. Redis
outages fail closed for mutations and use only the configured bounded read fallback.

FAILS OPEN BY CONFIGURATION, NEVER BY ACCIDENT. A limit can be switched off, but only by
writing `off`; anything malformed falls back to the default rather than being read as
"unlimited". A typo must not silently remove the cap.
"""
from __future__ import annotations

import math
import os
import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from mcp.server.mcpserver.exceptions import ToolError
from recall.constants import MAX_IDEMPOTENCY_RESULT_BYTES
from recall.observability import get_logger
from recall.errors import IdempotencyConflict, RecallError

_log = get_logger("limits")

#: Scope-keyed call budgets. Reads are cheap and interactive, so they get a wide allowance;
#: writes embed and cost money; forget is irreversible and no agent needs to do it in bulk.
#: `capacity` is the burst — a client may spend it at once, then is paced by `per_second`.
DEFAULT_CALLS_PER_MIN: dict[str, float] = {
    "read": 120.0,
    "write": 20.0,
    "forget": 10.0,
    # Admin actions (calibration publish) change what a whole tenant serves; they are rare by
    # nature and share nobody else's budget, so a runaway publish loop cannot starve indexing.
    "admin": 10.0,
}
#: Aggregate embedding spend, in bytes of source text per hour per tenant. 200 MB is ~10x the
#: 20 MB single-request cap, so an ordinary re-index of a large corpus fits comfortably while a
#: loop calling `recall_index` at the cap is stopped after ten iterations rather than never.
DEFAULT_INDEX_BYTES_PER_HOUR = 200 * 1024 * 1024
#: The budget key for the byte quota, named once. A bare literal at a debit site that mistyped it
#: would silently meter nothing: `RateLimiter.check` treats an unknown key as unlimited.
INDEX_BYTES_BUDGET = "index_bytes"

_SECONDS_PER_MIN = 60.0
_SECONDS_PER_HOUR = 3600.0
#: The literal that disables a limit. A WORD, not a number: `0` reads as both "no limit" and
#: "nothing allowed" depending on who is looking, and that ambiguity in a spend control is how
#: a cap gets removed by someone who meant to tighten it.
OFF = "off"


class RateLimited(RuntimeError, ToolError, RecallError):
    """A tenant exceeded its budget and the MCP client should receive the retry guidance.

    `ToolError` is an MCP transport classification, not a change to the domain contract. Keeping
    `RuntimeError` first preserves existing callers that catch the historical builtin, while the
    `ToolError` base prevents MCP 2.1 from redacting this anticipated, actionable refusal as an
    unexpected tool crash.
    """

    def __init__(self, message: str, *, retry_after_seconds: float) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class RateLimiterUnavailable(RuntimeError, ToolError, RecallError):
    """The shared limiter could not be reached.

    This is intentionally distinct from :class:`RateLimited`.  An operator needs to know that a
    tenant was refused because the budget was exhausted, rather than because the enforcement
    backend is unavailable.  The server maps the distinction to fail closed writes and bounded
    read fallback.
    """

    def __init__(self, message: str = "centralized rate limiter is unavailable", *, retry_after_seconds: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class IdempotencyReplay(RuntimeError, ToolError, RecallError):
    """A completed mutation has a replayable result for this idempotency key."""

    def __init__(self, result: str) -> None:
        super().__init__("replaying the completed idempotent mutation")
        self.result = result

class IdempotencyResultMissing(RuntimeError, ToolError, RecallError):
    """Redis reserved a mutation key, but its response is not in Redis."""

    def __init__(self, idempotency_key: str) -> None:
        super().__init__("idempotency key was already used, but its Redis result is unavailable")
        self.idempotency_key = idempotency_key


class AsyncRateLimiter(Protocol):
    """Async authorization choke point implemented by local and Redis limiters."""

    async def check(
        self,
        tenant: str,
        key: str,
        cost: float = 1.0,
        *,
        idempotency_key: str | None = None,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
        read_only: bool = False,
    ) -> None: ...

    async def close(self) -> None: ...


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

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)
        self._updated = max(self._updated, now)
        self._tokens = min(self._rate.capacity, self._tokens + elapsed * self._rate.per_second)

    def level(self, now: float) -> float:
        """Refill for elapsed time and report the token level, without debiting."""
        self._refill(now)
        return self._tokens

    def drain(self, cost: float, now: float) -> None:
        """Debit `cost`, saturating at 0. Never raises.

        Unlike `take`, a cost larger than the bucket is NOT a user error here — for a
        failure counter it just means the bucket stays empty. `take` reserves the raise for
        the spend limiter, where an impossible cost is a misconfiguration worth shouting about.
        """
        self._refill(now)
        self._tokens = max(0.0, self._tokens - cost)


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

    async def check_async(
        self,
        tenant: str,
        key: str,
        cost: float = 1.0,
        *,
        idempotency_key: str | None = None,
        read_only: bool = False,
    ) -> None:
        """Async adapter used by the MCP server without changing local semantics."""
        del idempotency_key, read_only
        self.check(tenant, key, cost)

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


def _token_bucket_lua() -> str:
    return """
local now = redis.call('TIME')
local now_ms = (tonumber(now[1]) * 1000) + math.floor(tonumber(now[2]) / 1000)
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens'))
local updated = tonumber(redis.call('HGET', KEYS[1], 'updated'))
if tokens == nil then
  tokens = capacity
  updated = now_ms
end
tokens = math.min(capacity, tokens + math.max(0, now_ms - updated) * refill / 1000)
local duplicate = 0
if KEYS[2] ~= '' and redis.call('EXISTS', KEYS[2]) == 1 then
  duplicate = 1
  if tonumber(ARGV[5]) == 1 then
    return {1, 0, duplicate}
  end
  return {0, 0, 2}
end
if cost > capacity then
  return {0, 0, duplicate}
end
if tokens < cost then
  local wait_ms = math.ceil((cost - tokens) / refill * 1000)
  return {0, wait_ms, duplicate}
end
tokens = tokens - cost
redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated', now_ms)
redis.call('PEXPIRE', KEYS[1], ttl_ms)
if KEYS[2] ~= '' then
  redis.call('SET', KEYS[2], ARGV[6], 'PX', ttl_ms)
end
return {1, 0, duplicate}
"""


class RedisRateLimiter:
    """Fleet wide token buckets backed by Redis server time and one atomic Lua reservation."""

    def __init__(
        self,
        redis_url: str,
        rates: dict[str, Rate],
        *,
        deployment: str = "default",
        key_prefix: str = "recall:rate",
        timeout_seconds: float = 0.25,
        fallback_read_budget: float = 3.0,
        max_connections: int = 32,
        redis_client: Any | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if not math.isfinite(fallback_read_budget) or fallback_read_budget < 0:
            raise ValueError("fallback_read_budget must be >= 0")
        if max_connections <= 0:
            raise ValueError("max_connections must be > 0")
        self._redis_url = redis_url
        self._rates = dict(rates)
        self._deployment = self._safe_part(deployment)
        self._prefix = self._safe_part(key_prefix)
        self._timeout = timeout_seconds
        self._fallback_read_budget = fallback_read_budget
        self._max_connections = max_connections
        self._fallback = RateLimiter({"read": Rate(fallback_read_budget, 1.0)} if fallback_read_budget else {})
        self._redis = redis_client
        self._script_sha: str | None = None
        self.requires_idempotency = True
        self._metrics = {
            "limiter_requests": "recall_rate_limiter_requests_total",
            "limiter_refused": "recall_rate_limiter_refused_total",
            "limiter_errors": "recall_rate_limiter_redis_errors_total",
            "limiter_fallback": "recall_rate_limiter_fallback_total",
        }

    @staticmethod
    def _safe_part(value: str) -> str:
        if not value or len(value) > 128:
            raise ValueError("rate limiter key components must be nonempty and at most 128 characters")
        return value

    @property
    def limits(self) -> dict[str, Rate]:
        return dict(self._rates)

    @staticmethod
    def _request_hash(request_id: str) -> str:
        material = json.dumps(request_id, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _keys(
        self, tenant: str, budget: str, request_id: str | None
    ) -> tuple[str, str]:
        tenant_hash = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
        base = f"{self._prefix}:{self._deployment}:{tenant_hash}:{self._safe_part(budget)}"
        idem = ""
        if request_id:
            request_hash = self._request_hash(request_id)
            idem = f"{self._prefix}:{self._deployment}:{tenant_hash}:idempotency:{request_hash}"
        return base, idem

    def _result_key(self, tenant: str, request_id: str, operation: str | None = None) -> str:
        tenant_hash = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
        request_hash = self._request_hash(request_id)
        return f"{self._prefix}:{self._deployment}:{tenant_hash}:idempotency-result:{request_hash}"

    async def _client(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            from redis.asyncio import Redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover, depends on optional production extra
            raise RateLimiterUnavailable("redis package is not installed") from exc
        self._redis = Redis.from_url(
            self._redis_url,
            socket_connect_timeout=self._timeout,
            socket_timeout=self._timeout,
            max_connections=self._max_connections,
            decode_responses=False,
        )
        return self._redis

    async def check(
        self,
        tenant: str,
        key: str,
        cost: float = 1.0,
        *,
        idempotency_key: str | None = None,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
        read_only: bool = False,
    ) -> None:
        rate = self._rates.get(key)
        if rate is None or cost <= 0:
            return
        started = time.perf_counter()
        try:
            client = await self._client()
            bucket, idem = self._keys(tenant, key, idempotency_key, idempotency_operation)
            ttl_ms = max(1000, int((rate.capacity / rate.per_second) * 2000))
            reservation = json.dumps(
                {
                    "operation": idempotency_operation,
                    "request_fingerprint": idempotency_fingerprint,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            args = [
                rate.capacity,
                rate.per_second,
                cost,
                ttl_ms,
                1 if read_only else 0,
                reservation,
            ]
            try:
                if self._script_sha is None:
                    self._script_sha = await client.script_load(_token_bucket_lua())
                result = await client.evalsha(self._script_sha, 2, bucket, idem, *args)
            except Exception as exc:  # BROAD-CATCH: error-translation
                if type(exc).__name__ != "NoScriptError":
                    raise
                self._script_sha = await client.script_load(_token_bucket_lua())
                result = await client.evalsha(self._script_sha, 2, bucket, idem, *args)
            allowed, wait_ms, duplicate = (int(value) for value in result)
            if duplicate == 2:
                if idempotency_key:
                    stored = await client.get(idem)
                    if stored is not None:
                        try:
                            metadata = json.loads(
                                stored.decode("utf-8")
                                if isinstance(stored, bytes)
                                else str(stored)
                            )
                        except (TypeError, ValueError):
                            metadata = None
                        if isinstance(metadata, dict) and (
                            metadata.get("operation") != idempotency_operation
                            or metadata.get("request_fingerprint") != idempotency_fingerprint
                        ):
                            raise IdempotencyConflict()
                    replay = await self.get_idempotency_result(
                        tenant,
                        idempotency_key,
                        operation=idempotency_operation,
                        request_fingerprint=idempotency_fingerprint,
                    )
                    if replay is not None:
                        raise IdempotencyReplay(replay)
                    raise IdempotencyResultMissing(idempotency_key)
                raise RateLimited(
                    f"idempotency key for {key!r} was already used; replay the original result "
                    "instead of executing the mutation again",
                    retry_after_seconds=0.0,
                )
            if allowed != 1:
                retry = max(0.001, wait_ms / 1000.0)
                self._metric("limiter_refused", budget=key)
                raise RateLimited(
                    f"rate limit exceeded for {key!r}; retry after {retry:.1f}s",
                    retry_after_seconds=retry,
                )
            if duplicate:
                self._metric("limiter_requests", budget=key, result="idempotent_replay")
            else:
                self._metric("limiter_requests", budget=key, result="reserved")
        except (RateLimited, IdempotencyResultMissing, IdempotencyReplay, IdempotencyConflict):
            raise
        except Exception as exc:  # BROAD-CATCH: error-translation
            self._metric("limiter_errors", budget=key)
            if read_only:
                if self._fallback_read_budget <= 0:
                    self._metric("limiter_refused", budget=key, result="fallback_disabled")
                    raise RateLimiterUnavailable(
                        "centralized rate limiter is unavailable and read fallback is disabled",
                        retry_after_seconds=1.0,
                    ) from exc
                try:
                    self._fallback.check(tenant, "read")
                except RateLimited:
                    self._metric("limiter_refused", budget=key, result="fallback_exhausted")
                    raise RateLimiterUnavailable(
                        "centralized rate limiter is unavailable and the bounded read fallback is exhausted",
                        retry_after_seconds=1.0,
                    ) from exc
                self._metric("limiter_fallback", budget=key)
                return
            raise RateLimiterUnavailable(
                f"centralized rate limiter is unavailable ({type(exc).__name__})",
                retry_after_seconds=1.0,
            ) from exc
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._metric("limiter_latency_ms", budget=key, value=elapsed_ms)

    async def get_idempotency_result(
        self,
        tenant: str,
        idempotency_key: str,
        *,
        operation: str | None = None,
        request_fingerprint: str | None = None,
    ) -> str | None:
        """Return a completed mutation result, if one was durably recorded in Redis."""
        if not idempotency_key:
            return None
        client = await self._client()
        raw = await client.get(self._result_key(tenant, idempotency_key, operation))
        if raw is None:
            return None
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if request_fingerprint is None:
            return text
        try:
            envelope = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(envelope, dict) or envelope.get("_recall_idempotency_result") != 1:
            return None
        if envelope.get("request_fingerprint") != request_fingerprint:
            raise IdempotencyConflict()
        result = envelope.get("result")
        return result if isinstance(result, str) else None

    async def store_idempotency_result(
        self,
        tenant: str,
        idempotency_key: str,
        result: str,
        *,
        operation: str | None = None,
        request_fingerprint: str | None = None,
    ) -> None:
        """Store a bounded mutation response for safe retries with the same request key."""
        if not idempotency_key:
            return
        if len(result.encode("utf-8")) > MAX_IDEMPOTENCY_RESULT_BYTES:
            raise ValueError("idempotent mutation result exceeds the 512 KiB replay limit")
        rate = self._rates.get("write") or self._rates.get("admin") or self._rates.get("forget")
        ttl_ms = max(60_000, int(((rate.capacity / rate.per_second) if rate else 3600) * 2000))
        client = await self._client()
        value = result
        if request_fingerprint is not None:
            value = json.dumps(
                {
                    "_recall_idempotency_result": 1,
                    "request_fingerprint": request_fingerprint,
                    "result": result,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        await client.set(self._result_key(tenant, idempotency_key, operation), value, px=ttl_ms)

    async def release_idempotency_reservation(
        self, tenant: str, idempotency_key: str, *, operation: str | None = None
    ) -> None:
        """Release a reservation after a verified pre-side-effect failure."""
        if not idempotency_key:
            return
        client = await self._client()
        _bucket, idem = self._keys(tenant, "write", idempotency_key)
        await client.delete(idem)

    def _metric(self, name: str, **labels: object) -> None:
        from recall.observability import METRICS

        value = labels.pop("value", None)
        string_labels = {key: str(label) for key, label in labels.items()}
        if value is None:
            METRICS.increment(name, 1, **string_labels)
        else:
            METRICS.observe(name, float(str(value)), **string_labels)

    async def close(self) -> None:
        if self._redis is not None:
            close = getattr(self._redis, "aclose", None)
            if callable(close):
                await close()

    def check_sync(
        self,
        tenant: str,
        key: str,
        cost: float = 1.0,
        *,
        idempotency_key: str | None = None,
        read_only: bool = False,
    ) -> None:
        """Bridge the synchronous indexing callback from the worker thread to the async limiter."""
        import asyncio

        asyncio.run(
            self.check(
                tenant,
                key,
                cost,
                idempotency_key=idempotency_key,
                read_only=read_only,
            )
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


class FailedAuthThrottle:
    """Caps the work an authentication-failure storm can drive.

    The SDK's TokenVerifier protocol hands a verifier only the token string — no request
    object, no remote address — so this throttle is process-global, and it says so rather
    than pretending per-client fairness. It is consulted only where a failed authentication
    is EXPENSIVE: the OIDC path, whose JWKS fetch and RSA verify are the work worth capping.
    The static token path does not gate on it (a digest lookup is cheap), so a valid static
    token is never refused because of someone else's failures.

    ⚠️ On the OIDC path the gate is not free of collateral: while the bucket is drained a
    VALID OIDC token is refused too, because `allow()` cannot tell it from garbage without
    doing the very validation the gate defers. That is a deliberate availability-for-integrity
    trade on the one path where the work is expensive; the per-IP ASGI middleware in front of
    the SDK's bearer middleware is the way to narrow it, and it is not built yet. Revisit
    before serving a fleet.

    `allow()` reports the level without debiting; `record_failure()` debits one, saturating
    at zero (never raising, even under a sub-1 per-minute configuration).
    """

    def __init__(self, rate: Rate | None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._rate = rate
        self._clock = clock
        self._lock = threading.Lock()
        self._bucket: _Bucket | None = None

    def allow(self) -> bool:
        if self._rate is None:
            return True
        with self._lock:
            now = self._clock()
            if self._bucket is None:
                self._bucket = _Bucket(self._rate, now)
            return self._bucket.level(now) >= 1.0

    def record_failure(self) -> None:
        if self._rate is None:
            return
        with self._lock:
            now = self._clock()
            if self._bucket is None:
                self._bucket = _Bucket(self._rate, now)
            self._bucket.drain(1.0, now)


def failed_auth_throttle_from_env() -> FailedAuthThrottle:
    """Build the pre-auth failure throttle from `RECALL_RATE_AUTH_FAILURES_PER_MIN`."""
    rate = _rate_from_env("RECALL_RATE_AUTH_FAILURES_PER_MIN", 60.0, _SECONDS_PER_MIN)
    return FailedAuthThrottle(rate)


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
        rates[INDEX_BYTES_BUDGET] = byte_rate
    return RateLimiter(rates)


def _limiter_rates_from_env() -> dict[str, Rate]:
    rates: dict[str, Rate] = {}
    for scope, default in DEFAULT_CALLS_PER_MIN.items():
        rate = _rate_from_env(f"RECALL_RATE_{scope.upper()}_PER_MIN", default, _SECONDS_PER_MIN)
        if rate is not None:
            rates[scope] = rate
    byte_rate = _rate_from_env(
        "RECALL_INDEX_BYTES_PER_HOUR", float(DEFAULT_INDEX_BYTES_PER_HOUR), _SECONDS_PER_HOUR
    )
    if byte_rate is not None:
        rates[INDEX_BYTES_BUDGET] = byte_rate
    return rates


def async_limiter_from_env() -> AsyncRateLimiter | None:
    """Resolve the configured local or Redis limiter without opening a network connection."""
    backend = os.environ.get("RECALL_RATE_LIMIT_BACKEND", "local").strip().lower()
    if os.environ.get("RECALL_ENV", "development").strip().lower() == "production" and backend != "redis":
        raise ValueError("production deployments require RECALL_RATE_LIMIT_BACKEND=redis")
    if backend in {"off", "none"}:
        return None
    if backend in {"local", "memory", "in-memory"}:
        return _AsyncLocalLimiter(limiter_from_env())
    if backend != "redis":
        raise ValueError("RECALL_RATE_LIMIT_BACKEND must be local, redis, or off")
    redis_url = os.environ.get("RECALL_REDIS_URL", "").strip()
    if not redis_url:
        raise ValueError("RECALL_REDIS_URL is required when RECALL_RATE_LIMIT_BACKEND=redis")
    timeout = float(os.environ.get("RECALL_REDIS_TIMEOUT_SECONDS", "0.25"))
    fallback = float(os.environ.get("RECALL_RATE_READ_FALLBACK_BUDGET", "3"))
    max_connections = int(os.environ.get("RECALL_REDIS_MAX_CONNECTIONS", "32"))
    return RedisRateLimiter(
        redis_url,
        _limiter_rates_from_env(),
        deployment=os.environ.get("RECALL_DEPLOYMENT", "default"),
        key_prefix=os.environ.get("RECALL_RATE_LIMIT_KEY_PREFIX", "recall:rate"),
        timeout_seconds=timeout,
        fallback_read_budget=fallback,
        max_connections=max_connections,
    )


class _AsyncLocalLimiter:
    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter

    @property
    def limits(self) -> dict[str, Rate]:
        return self._limiter.limits()

    async def check(
        self,
        tenant: str,
        key: str,
        cost: float = 1.0,
        *,
        idempotency_key: str | None = None,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
        read_only: bool = False,
    ) -> None:
        del idempotency_key, idempotency_operation, idempotency_fingerprint, read_only
        self._limiter.check(tenant, key, cost)

    async def close(self) -> None:
        return None
