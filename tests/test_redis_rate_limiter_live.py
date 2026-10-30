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
    async def run() -> None:
        first = RedisRateLimiter(url, {"read": Rate(2, 0.001)}, key_prefix="recall:test")
        second = RedisRateLimiter(url, {"read": Rate(2, 0.001)}, key_prefix="recall:test")
        try:
            client = await first._client()
            await client.delete(first._keys("contract-tenant", "read", None)[0])
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
