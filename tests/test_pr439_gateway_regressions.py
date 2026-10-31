"""PERF-001 transport lifetime, using the real gateway with a harmless SDK transport double.

Baseline 8ef55caf0 reinitializes once per call. The same test must fail its
startup-count assertion before the repair and pass on the remote test host after the repair.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from recall.desktop.models import RuntimeMode, RuntimeProfile
from recall.desktop.runtime import LocalDatabaseRuntime, RuntimeErrorBase, SdkMcpGateway


def test_local_gateway_reuses_session_and_closes_each_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    starts: list[str] = []
    initializes: list[str] = []
    closes: list[str] = []
    tasks: dict[str, Any] = {}

    @asynccontextmanager
    async def transport(parameters: Any):
        tenant = parameters.env["RECALL_TENANT"]
        starts.append(tenant)
        tasks[tenant] = asyncio.current_task()
        try:
            yield tenant, None
        finally:
            assert asyncio.current_task() is tasks[tenant], "transport exited in a different task"
            closes.append(tenant)

    class Session:
        def __init__(self, tenant: str, writer: Any) -> None:
            self.tenant = tenant

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            initializes.append(self.tenant)

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            assert asyncio.current_task() is tasks[self.tenant]
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"tenant": self.tenant}))])

    monkeypatch.setattr("mcp.client.stdio.stdio_client", transport)
    monkeypatch.setattr("mcp.client.session.ClientSession", Session)
    runtime = LocalDatabaseRuntime(RuntimeProfile(mode=RuntimeMode.LOCAL_DATABASE, dsn="postgresql://example.invalid/recall"))
    alpha = runtime._gateway_for("alpha")
    beta = runtime._gateway_for("beta")
    try:
        assert alpha.call("recall_stats", {}) == {"tenant": "alpha"}
        assert alpha.call("recall_stats", {}) == {"tenant": "alpha"}
        assert beta.call("recall_stats", {}) == {"tenant": "beta"}
        assert sorted(starts) == ["alpha", "beta"], "cached gateways restarted their MCP servers"
        assert sorted(initializes) == ["alpha", "beta"]
        assert closes == [], "stdio session closed before runtime.stop"
    finally:
        runtime.stop()
    assert sorted(closes) == ["alpha", "beta"]
    with pytest.raises(RuntimeErrorBase, match="closed"):
        alpha.call("recall_stats", {})


def test_real_stdio_server_keeps_state_between_calls(tmp_path: Path) -> None:
    """PERF-001: a harmless real SDK server must initialize once across public calls.

    Baseline restarts this process and returns counter=1 twice. The unchanged
    second-counter assertion must fail red before passing with session reuse.
    """
    script = tmp_path / "counter_server.py"
    script.write_text(
        "from mcp.server import MCPServer\n"
        "server = MCPServer('gateway-lifetime-test')\n"
        "counter = 0\n"
        "@server.tool()\n"
        "def count() -> dict[str, int]:\n"
        "    global counter\n"
        "    counter += 1\n"
        "    return {'counter': counter}\n"
        "server.run()\n",
        encoding="utf-8",
    )
    gateway = SdkMcpGateway(
        RuntimeProfile(mode=RuntimeMode.LOCAL_DATABASE, dsn="postgresql://example.invalid/recall"),
        command=[sys.executable, str(script)],
    )
    try:
        assert gateway.call("count", {}) == {"counter": 1}
        assert gateway.call("count", {}) == {"counter": 2}, "real stdio server restarted between calls"
    finally:
        gateway.close()


@pytest.mark.parametrize("phase", ["dispatch", "idle"])
def test_transport_cancellation_resolves_requests_and_exits(
    monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    """PERF-001 rework: no request or executor thread may be stranded by cancellation.

    The first persistent-worker implementation fails the intended reply/worker
    assertions. A bounded observation is used so the test can clean up that old
    worker instead of hanging pytest. Its source and hash are in the audit receipt.
    """
    initialized = threading.Event()
    idle_read = threading.Event()
    transport_closed = threading.Event()
    completed = threading.Event()
    owner: dict[str, Any] = {}
    stolen: list[Any] = []
    errors: list[BaseException] = []

    @asynccontextmanager
    async def transport(parameters: Any):
        try:
            yield None, None
        finally:
            transport_closed.set()

    class Session:
        def __init__(self, *args: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            owner.update(loop=asyncio.get_running_loop(), task=asyncio.current_task())
            initialized.set()
            if phase == "dispatch":
                owner["loop"].call_soon(owner["task"].cancel)

        async def call_tool(self, *args: Any) -> Any:
            await asyncio.sleep(0)
            return SimpleNamespace(content=[SimpleNamespace(text='{"ok": true}')])

    async def executor_handoff(function: Any, *args: Any) -> Any:
        # Deterministically put the old worker between executor dequeue and reply delivery.
        value = function(*args)
        stolen.append(value)
        await asyncio.sleep(0)
        return value

    monkeypatch.setattr("mcp.client.stdio.stdio_client", transport)
    monkeypatch.setattr("mcp.client.session.ClientSession", Session)
    if phase == "dispatch":
        monkeypatch.setattr(asyncio, "to_thread", executor_handoff)
    gateway = SdkMcpGateway(
        RuntimeProfile(mode=RuntimeMode.LOCAL_DATABASE, dsn="postgresql://example.invalid/recall"),
        command=["harmless-transport-double"],
    )
    get = gateway._requests.get

    def observed_get(*args: Any, **kwargs: Any) -> Any:
        if gateway._requests.empty():
            idle_read.set()
        return get(*args, **kwargs)

    monkeypatch.setattr(gateway._requests, "get", observed_get)

    def call() -> None:
        try:
            gateway.call("read", {})
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    try:
        assert initialized.wait(10), "transport fixture never initialized"
        if phase == "dispatch":
            assert completed.wait(2), "dequeued request never received its cancellation outcome"
            assert len(errors) == 1
        else:
            assert completed.wait(2)
            assert not errors
            assert idle_read.wait(2), "worker never reached its idle read"
            owner["loop"].call_soon_threadsafe(owner["task"].cancel)
            assert transport_closed.wait(2)
            assert gateway._thread is not None
            gateway._thread.join(timeout=2)
            assert not gateway._thread.is_alive(), "idle queue read stranded the worker's executor"
    finally:
        # Release the first implementation's deliberately stranded resources after a red assertion.
        for request in stolen:
            if request is not None and not request[2].done():
                request[2].set_exception(RuntimeErrorBase("test cleanup"))
        gateway._requests.put(None)
        gateway.close()
        caller.join(timeout=2)
