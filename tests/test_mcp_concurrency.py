"""The MCP server must not do its work on the event loop.

MCPServer executes async tool bodies on the event loop, so a sync tool would run its blocking body
there too:

    if fn_is_async:
        return await fn(**arguments_parsed_dict)
    else:
        return fn(**arguments_parsed_dict)

There is no implicit thread offload. So a sync tool that embeds a query, makes two database round trips
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


#: The four tools that predate the evidence boundary. Kept as their own name so the additive claim
#: — "a tool was ADDED, none was changed or removed" — is a control this test asserts rather than
#: something a reader has to infer from a rewritten set literal.
ORIGINAL_TOOLS = {"recall_search", "recall_index", "recall_forget", "recall_stats"}
ALL_TOOLS = ORIGINAL_TOOLS | {
    "recall_evidence",
    "recall_reasoning_audit",
    "recall_reasoning_projection",
    "recall_reasoning_proposals",
    "recall_reasoning_query",
    # Read only, and deliberately unaccompanied: there is no `recall_rewrite_apply`, because
    # the MCP client is the model and a reviewer id it can type is a field, not a person.
    "recall_rewrite_plan",
}


def test_every_tool_is_async():
    """A sync tool would block the event loop — the whole point of the change.

    The set is asserted EXACTLY, not as a superset, so a tool added later still trips this and has
    to be shown async on purpose. `recall_evidence` did trip it, which is the test working:
    adding it was deliberate, so the expectation moves and the control below records that the four
    it joined are all still there.
    """
    tools = _tools(build_server())
    assert ORIGINAL_TOOLS <= set(tools), "an existing tool disappeared from the server"
    assert set(tools) == ALL_TOOLS
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


@pytest.mark.parametrize("name", sorted(ALL_TOOLS))
def test_tools_still_declare_their_schema(name):
    """Async-ifying must not change the wire contract clients depend on.

    Driven off `ALL_TOOLS` rather than a second hand-written list: a new tool that was added to one
    list and forgotten in the other would be held to the async rule and exempted from the schema
    rule, which is the quieter half of the same omission.
    """
    tool = _tools(build_server())[name]
    assert tool.description
    assert tool.parameters["type"] == "object"
