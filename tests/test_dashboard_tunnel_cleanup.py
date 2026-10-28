"""BUG-001: the dashboard command owns only the SSH tunnel it starts.

Startup failure must reap an owned tunnel, and successful serving must still close
the server and reap that tunnel. Reusing an already-listening forward must start
and stop no SSH process. All sockets, processes and serving are replaced at their
external boundaries; the command is driven through its registered CLI handler.

Proof plan: run both startup-failure nodes against baseline 970dbeb7b before the
repair. Prove the normal-shutdown control with a mutation deleting the production
terminate call; prove the reused-forward control by deleting the production reuse
branch in _open_tunnel. Each must fail its ownership/cleanup assertion, then the
same nodes must pass after restoring the production bytes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from recall.cli_commands import dashboard_cmd
from recall.dashboard import server as dashboard_server


class FakeTunnel:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")

    def wait(self, timeout: float | None = None) -> int:
        self.events.append("wait")
        self.returncode = 0
        return 0


class FakeServer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def serve_forever(self) -> None:
        self.events.append("serve_forever")

    def server_close(self) -> None:
        self.events.append("server_close")


def _arguments(monkeypatch: pytest.MonkeyPatch, root: Path) -> argparse.Namespace:
    for name in (
        "RECALL_DSN", "RECALL_SERVING_DSN", "RECALL_DASHBOARD_DSN_FILE",
        "RECALL_DASHBOARD_TUNNEL", "RECALL_DASHBOARD_CONFIG_HOST",
        "RECALL_DASHBOARD_CONFIG_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    parser = argparse.ArgumentParser()
    dashboard_cmd.register(parser.add_subparsers())
    return parser.parse_args([
        "dashboard", "--root", str(root), "--no-browser",
        "--tunnel", "test-host", "--tunnel-ports", "55433:5432",
    ])


@pytest.mark.parametrize("failure", ["app", "bind"])
def test_startup_failure_reaps_owned_tunnel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str,
) -> None:
    """BUG-001: no successfully opened owned tunnel survives failed startup."""
    args = _arguments(monkeypatch, tmp_path)
    events: list[str] = []
    tunnel = FakeTunnel(events)
    monkeypatch.setattr(dashboard_cmd, "_open_tunnel", lambda host, ports: tunnel)

    def app_fails(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated app startup failure")

    def bind_fails(app: object) -> None:
        raise OSError("simulated bind failure")

    if failure == "app":
        monkeypatch.setattr(dashboard_server, "DashboardApp", app_fails)
        with pytest.raises(RuntimeError, match="simulated app startup failure"):
            args.func(args)
    else:
        monkeypatch.setattr(dashboard_server, "serve", bind_fails)
        with pytest.raises(SystemExit) as stopped:
            args.func(args)
        assert stopped.value.code == 2

    assert events == ["terminate", "wait"], "BUG-001: failed startup left its owned SSH tunnel unreaped"


def test_normal_shutdown_closes_server_and_reaps_owned_tunnel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Compatibility control: successful serving retains the owned cleanup."""
    args = _arguments(monkeypatch, tmp_path)
    events: list[str] = []
    monkeypatch.setattr(dashboard_cmd, "_open_tunnel", lambda host, ports: FakeTunnel(events))
    monkeypatch.setattr(dashboard_server, "serve", lambda app: FakeServer(events))

    args.func(args)

    assert events == ["serve_forever", "server_close", "terminate", "wait"], "normal shutdown lost server or owned-tunnel cleanup"


def test_failed_startup_does_not_start_or_stop_a_reused_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Compatibility control: an existing forward is not this command's process."""
    args = _arguments(monkeypatch, tmp_path)
    started: list[object] = []
    events: list[str] = []

    def start_ssh(*args: Any, **kwargs: Any) -> FakeTunnel:
        started.append(args)
        return FakeTunnel(events)

    def bind_fails(app: object) -> None:
        raise OSError("simulated bind failure")

    monkeypatch.setattr(dashboard_cmd, "_port_open", lambda port: True)
    monkeypatch.setattr(dashboard_cmd.subprocess, "Popen", start_ssh)
    monkeypatch.setattr(dashboard_server, "serve", bind_fails)

    with pytest.raises(SystemExit) as stopped:
        args.func(args)

    assert stopped.value.code == 2
    assert started == [], "a reused forward caused this command to start another SSH process"
    assert events == [], "this command attempted to stop a forward it did not own"
