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
