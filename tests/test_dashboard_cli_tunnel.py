"""`recall dashboard --tunnel`: what the command will and will not hand to ssh.

Invariants and the failure each one catches:
- C1 a `--tunnel` value that starts with `-` is refused before ssh is started, so it cannot be read
  as an ssh option (`-oProxyCommand=...` would run a local command).
- C2 `--tunnel-ports` must be LOCAL:REMOTE with both in range.
- C3 a forward already listening on the local port is reused, not doubled by a second ssh.

Red proof, 2026-10-06, each mutation alone against `recall/cli_commands/dashboard_cmd.py`, failing in
the named assertion (JUnit XML), then restored and green:
- T1 (C1) the `host.startswith("-")` refusal removed: ssh was started with the option.
- T2 (C2) the range check in `_parse_ports` removed: `1:70000` was accepted.
- T3 (C3) the reuse branch in `_open_tunnel` removed: a second ssh was started.
"""

from __future__ import annotations

from typing import Any

import pytest

from recall.cli_commands import dashboard_cmd


class FakeSsh:
    def __init__(self, running: bool) -> None:
        self.returncode = None if running else 255

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        pass


class NoSsh:
    """Stands in for `subprocess.Popen`: records every command; the 'ssh' exits at once unless `running`."""

    def __init__(self) -> None:
        self.started: list[list[str]] = []
        self.running = False

    def __call__(self, command: list[str], **kwargs: Any) -> FakeSsh:
        self.started.append(command)
        return FakeSsh(self.running)


@pytest.fixture
def ssh(monkeypatch: pytest.MonkeyPatch) -> NoSsh:
    recorder = NoSsh()
    monkeypatch.setattr(dashboard_cmd.subprocess, "Popen", recorder)
    return recorder


def test_a_tunnel_host_that_looks_like_an_option_is_refused(ssh: NoSsh, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_cmd, "_port_open", lambda port: False)
    with pytest.raises(SystemExit) as stopped:
        dashboard_cmd._open_tunnel("-oProxyCommand=calc", "55433:55432")
    assert stopped.value.code == 2
    assert ssh.started == [], "ssh was started with an option smuggled in as the host"


@pytest.mark.parametrize("spec", ["55433", "a:b", "0:55432", "1:70000", ":55432"])
def test_tunnel_ports_must_be_two_ports_in_range(spec: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        dashboard_cmd._parse_ports(spec)
    assert stopped.value.code == 2


def test_tunnel_ports_parse() -> None:
    assert dashboard_cmd._parse_ports("55433:55432") == (55433, 55432)


def test_a_forward_already_listening_is_reused(ssh: NoSsh, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_cmd, "_port_open", lambda port: True)
    ssh.running = True
    reused = dashboard_cmd._open_tunnel("vps", "55433:55432")
    assert ssh.started == [], "a second ssh was started beside a working forward"
    assert reused is None
