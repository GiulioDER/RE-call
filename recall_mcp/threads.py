"""Offloading blocking work from the MCP event loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import anyio.to_thread

_T = TypeVar("_T")


async def _to_thread(fn: Callable[[], _T]) -> _T:
    """Run a blocking tool body off the event loop.

    MCPServer executes async tool bodies on the event loop, so each tool explicitly offloads its
    blocking embedder, database, reranker and indexing work before it can monopolize the loop.

    `anyio.to_thread` rather than `asyncio.to_thread` because the MCP SDK runs on AnyIO: this inherits
    its worker-thread limiter and cancellation scope instead of starting a second, unmanaged pool
    beside it.
    """
    return await anyio.to_thread.run_sync(fn)
