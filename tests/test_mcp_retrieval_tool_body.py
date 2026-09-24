"""`recall_search` and `recall_evidence` differ in exactly what they are meant to, and no more.

Both tools run one shared body (plan, timer, federated execution, run, trust-refusal mapping,
render). What each must keep of its own is pinned here, at the registered MCP tool itself: the
latency metric's `tool` tag, and the federation mode. Evidence forces federation `off`, because
evidence cards are persisted through the primary store's provenance controller and a rescued hit
from another tenant must not be revalidated as the primary tenant's; search serves the configured
mode.

Red proofs (2026-09-24, base ``6ad0764f``), node
``tests/test_mcp_retrieval_tool_body.py::test_each_retrieval_tool_keeps_its_metric_tag_and_federation_mode``:

* mutating ``recall_evidence`` to pass ``_federation_config_for(state)`` unchanged fails with
  ``assert ('evidence', 'active') == ('evidence', 'off')``;
* mutating ``recall_evidence``'s timer tag to ``"search"`` fails with
  ``assert ('search', 'off') == ('evidence', 'off')``.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import recall_mcp.server as server
from recall.federation import FederationConfig
from recall_mcp.server import build_server
from recall_mcp.settings import Settings


class _Stop(Exception):
    """Raised from the stubbed federated execution once it has recorded its inputs."""


class _Plan:
    def as_dict(self) -> dict[str, object]:
        return {}


def _call(name: str, monkeypatch) -> tuple[str, str]:
    seen: dict[str, str] = {}

    @contextmanager
    def timer(metric: str, **labels: str):
        if metric == "recall_tool_latency_ms":
            seen["tag"] = labels["tool"]
        yield

    def prepare(plan, *, config, **kwargs):
        seen["mode"] = config.mode
        raise _Stop

    monkeypatch.setattr(server.METRICS, "timer", timer)
    monkeypatch.setattr(server, "prepare_federated_execution", prepare)
    monkeypatch.setattr(server, "_retrieval_plan_for", lambda *args, **kwargs: _Plan())

    tools = {tool.name: tool for tool in build_server()._tool_manager.list_tools()}
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context={
                "store": SimpleNamespace(tenant="default"),
                "embedder": object(),
                "calibration": None,
                "federation_config": FederationConfig(mode="active"),
                "settings": Settings.from_env({}),
            }
        )
    )
    with pytest.raises(_Stop):
        asyncio.run(tools[name].fn(ctx=context, query="how many requests per second?"))
    return seen["tag"], seen["mode"]


def test_each_retrieval_tool_keeps_its_metric_tag_and_federation_mode(monkeypatch) -> None:
    assert _call("recall_search", monkeypatch) == ("search", "active")
    assert _call("recall_evidence", monkeypatch) == ("evidence", "off")
