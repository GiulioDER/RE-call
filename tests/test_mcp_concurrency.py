"""The MCP server must not do its work on the event loop.

FastMCP awaits an async tool and CALLS A SYNC ONE INLINE — see
`mcp/server/fastmcp/utilities/func_metadata.py`:

    if fn_is_async:
        return await fn(**arguments_parsed_dict)
    else:
        return fn(**arguments_parsed_dict)

There is no thread offload. So a sync tool that embeds a query, makes two database round trips
and optionally runs a cross-encoder blocks the entire loop for its whole duration: effective
concurrency is one, and the server cannot even answer a ping meanwhile. `recall_index` blocks it
for the length of a corpus index.

These tests pin the fix — the tools are coroutines that hand their blocking work to a worker
thread — rather than the symptom, which only shows up under load.
"""
from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from recall_mcp.server import build_server


def _tools(server):
    return {t.name: t for t in server._tool_manager.list_tools()}


def test_every_tool_is_async():
    """A sync tool is awaited inline by FastMCP — the whole point of the change."""
    tools = _tools(build_server())
    assert set(tools) == {"recall_search", "recall_index", "recall_stats"}
    for name, tool in tools.items():
        assert tool.is_async, f"{name} is sync and would block the event loop"
        assert inspect.iscoroutinefunction(tool.fn), name


def test_the_event_loop_stays_responsive_while_a_tool_blocks():
    """The behavioural check: a blocking tool body must not stall other loop work.

    A sync tool holding the loop for 300 ms would delay every concurrent task by ~300 ms. Handed
    to a thread, the loop keeps ticking and the heartbeat below records many ticks.
    """
    from recall_mcp import server as srv

    async def scenario() -> tuple[int, float]:
        ticks = 0
        stop = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        hb = asyncio.create_task(heartbeat())
        t0 = time.perf_counter()
        await srv._to_thread(lambda: time.sleep(0.3))
        elapsed = time.perf_counter() - t0
        stop.set()
        await hb
        return ticks, elapsed

    ticks, elapsed = asyncio.run(scenario())
    assert elapsed >= 0.3  # the work really did take its time
    assert ticks > 10, f"loop was starved: only {ticks} ticks while a tool ran for {elapsed:.2f}s"


def test_concurrent_tool_bodies_overlap():
    """Four 200 ms bodies offloaded to threads finish in well under the 800 ms serial cost."""
    from recall_mcp import server as srv

    async def scenario() -> float:
        t0 = time.perf_counter()
        await asyncio.gather(*(srv._to_thread(lambda: time.sleep(0.2)) for _ in range(4)))
        return time.perf_counter() - t0

    elapsed = asyncio.run(scenario())
    assert elapsed < 0.6, f"tool bodies serialised: {elapsed:.2f}s for 4x200ms"


@pytest.mark.parametrize("name", ["recall_search", "recall_index", "recall_stats"])
def test_tools_still_declare_their_schema(name):
    """Async-ifying must not change the wire contract clients depend on."""
    tool = _tools(build_server())[name]
    assert tool.description
    assert tool.parameters["type"] == "object"
