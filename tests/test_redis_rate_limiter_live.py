from __future__ import annotations

import asyncio
import os

import pytest

from recall_mcp.limits import Rate, RateLimited, RedisRateLimiter


def test_two_process_equivalents_share_one_global_budget() -> None:
    url = os.environ.get("RECALL_REDIS_URL")
    if not url or os.environ.get("RECALL_RATE_LIMIT_BACKEND") != "redis":
        pytest.skip("Redis contract service is not configured")
    pytest.importorskip("redis")
    # Since SEC-003 (2026-10-30) a replayed key spends one token too, so the shared budget is
    # three: the first call, the reservation and its replay. The reservation key is deleted as
    # well, so a rerun against the same Redis starts clean.
    async def run() -> None:
        first = RedisRateLimiter(url, {"read": Rate(3, 0.001)}, key_prefix="recall:test")
        second = RedisRateLimiter(url, {"read": Rate(3, 0.001)}, key_prefix="recall:test")
        try:
            client = await first._client()
            await client.delete(
                first._keys("contract-tenant", "read", None)[0],
                first._keys("contract-tenant", "read", "same-request")[1],
            )
            await first.check("contract-tenant", "read")
            await second.check("contract-tenant", "read", idempotency_key="same-request", read_only=True)
            await second.check("contract-tenant", "read", idempotency_key="same-request", read_only=True)
            with pytest.raises(RateLimited):
                await first.check("contract-tenant", "read")
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_idempotency_reservation_lives_for_the_receipt_window() -> None:
    """BUG-001, against a real Redis: the reservation TTL is the durable receipt window.

    Red proof: failed on 3ada1afd0 against redis:7.2 with AssertionError, the reservation PTTL
    being at most 120000 ms (twice the per-minute refill window).
    """
    url = os.environ.get("RECALL_REDIS_URL")
    if not url or os.environ.get("RECALL_RATE_LIMIT_BACKEND") != "redis":
        pytest.skip("Redis contract service is not configured")
    pytest.importorskip("redis")
    from recall.constants import IDEMPOTENCY_RECEIPT_RETENTION_SECONDS

    async def run() -> None:
        limiter = RedisRateLimiter(url, {"write": Rate(20, 20 / 60)}, key_prefix="recall:test")
        try:
            client = await limiter._client()
            bucket, idem = limiter._keys("contract-tenant", "write", "reservation-ttl")
            await client.delete(bucket, idem)
            await limiter.check(
                "contract-tenant",
                "write",
                idempotency_key="reservation-ttl",
                idempotency_operation="recall_index",
            )
            ttl_ms = await client.pttl(idem)
            assert ttl_ms > (IDEMPOTENCY_RECEIPT_RETENTION_SECONDS - 60) * 1000
            await client.delete(bucket, idem)
        finally:
            await limiter.close()

    asyncio.run(run())


def test_check_sync_from_a_worker_leaves_the_shared_client_usable() -> None:
    """PERF-001: the indexing callback debits bytes from a worker thread with `check_sync`.

    The main event loop and that worker share one cached redis.asyncio client. A connection opened
    or reused on the worker's throwaway loop was bound to a loop that then closed, so the next call
    failed. Non-read budgets surface that as RateLimiterUnavailable instead of hiding it in the read
    fallback. Red proof: failed on e60e79883 against redis:7.2 with RateLimiterUnavailable from
    "got Future attached to a different loop" (recall_mcp.limits.RedisRateLimiter.check_sync ran
    asyncio.run on the worker).
    """
    url = os.environ.get("RECALL_REDIS_URL")
    if not url or os.environ.get("RECALL_RATE_LIMIT_BACKEND") != "redis":
        pytest.skip("Redis contract service is not configured")
    pytest.importorskip("redis")
    anyio = pytest.importorskip("anyio")

    async def run() -> None:
        limiter = RedisRateLimiter(
            url,
            {"write": Rate(1000, 1000), "index_bytes": Rate(1e9, 1e6)},
            key_prefix="recall:test-cross-loop",
        )
        try:
            for _ in range(3):
                await limiter.check("contract-tenant", "write")
                await anyio.to_thread.run_sync(
                    lambda: limiter.check_sync("contract-tenant", "index_bytes", 1.0)
                )
            await limiter.check("contract-tenant", "write")
        finally:
            await limiter.close()

    asyncio.run(run())


def test_a_replayed_key_still_spends_budget() -> None:
    """SEC-003 (owner decision 2026-10-30): a replay costs one token, so one key cannot be
    repeated without limit.

    Red proof: failed on e60e79883 against redis:7.2: the third call raised
    IdempotencyResultMissing again instead of RateLimited, because the Lua script returned on a
    duplicate key before any debit.
    """
    url = os.environ.get("RECALL_REDIS_URL")
    if not url or os.environ.get("RECALL_RATE_LIMIT_BACKEND") != "redis":
        pytest.skip("Redis contract service is not configured")
    pytest.importorskip("redis")
    from recall_mcp.limits import IdempotencyResultMissing

    async def run() -> None:
        limiter = RedisRateLimiter(url, {"write": Rate(2, 0.001)}, key_prefix="recall:test-replay")
        try:
            client = await limiter._client()
            bucket, idem = limiter._keys("contract-tenant", "write", "replayed-key")
            await client.delete(bucket, idem)
            await limiter.check(
                "contract-tenant", "write", idempotency_key="replayed-key", idempotency_operation="op"
            )
            with pytest.raises(IdempotencyResultMissing):
                await limiter.check(
                    "contract-tenant", "write", idempotency_key="replayed-key", idempotency_operation="op"
                )
            with pytest.raises(RateLimited):
                await limiter.check(
                    "contract-tenant", "write", idempotency_key="replayed-key", idempotency_operation="op"
                )
            await client.delete(bucket, idem)
        finally:
            await limiter.close()

    asyncio.run(run())
