"""`recall dashboard --reports-tenant` binds the review queue to tenants, and refuses an empty one.

Invariant: names are stripped, and a tenant name the queue would ask for with no name in it is refused at the command line,
with exit 2, before anything is opened; any non-empty name is accepted, because it reaches SQL only
as a bound parameter (a pattern here would silently drop names the store accepts, such as one that
starts with an underscore, which the cookie pattern `TENANT_NAME` refuses).

Red proof, 2026-10-06, on a Linux host, each mutation alone against
`recall/cli_commands/dashboard_cmd.py`, JUnit XML, then restored and green:
- C1 the empty-name check removed: `test_an_empty_reports_tenant_is_refused`, the command went on
  to serve (the test's stand-in raised `_Served`) instead of exiting 2.
- C2 the names checked against the cookie pattern `TENANT_NAME` instead:
  `test_any_named_tenant_reaches_the_app`, exit 2 for `_bench`.
- C3 the names no longer stripped: both tests, "an empty tenant name was accepted" for a blank name,
  and (' memory ', '_bench') != ('memory', '_bench').
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from recall.cli_commands import dashboard_cmd


class _Served(Exception):
    """Raised in place of opening a socket: reaching it means the command got past its checks."""


def _run(monkeypatch, tmp_path: Path, *extra: str) -> list[object]:
    seen: list[object] = []

    def serve(app: object) -> None:
        seen.append(app)
        raise _Served

    monkeypatch.setattr("recall.dashboard.server.serve", serve)
    for name in ("RECALL_DASHBOARD_DSN_FILE", "RECALL_DASHBOARD_TUNNEL"):
        monkeypatch.delenv(name, raising=False)  # register() reads them as defaults
    parser = argparse.ArgumentParser()
    dashboard_cmd.register(parser.add_subparsers())
    args = parser.parse_args(["dashboard", "--root", str(tmp_path), "--no-browser", *extra])
    with pytest.raises((_Served, SystemExit)) as raised:
        args.func(args)
    seen.append(raised.value)
    return seen


def test_an_empty_reports_tenant_is_refused(monkeypatch, tmp_path: Path, capsys) -> None:
    outcome = _run(monkeypatch, tmp_path, "--reports-tenant", "  ")[-1]
    assert isinstance(outcome, SystemExit) and outcome.code == 2, "an empty tenant name was accepted"
    assert "--reports-tenant needs a tenant name" in capsys.readouterr().err


def test_any_named_tenant_reaches_the_app(monkeypatch, tmp_path: Path) -> None:
    seen = _run(monkeypatch, tmp_path, "--reports-tenant", " memory ", "--reports-tenant", "_bench")
    assert isinstance(seen[-1], _Served), f"a named tenant was refused: {seen[-1]!r}"
    assert seen[0].reports_tenants == ("memory", "_bench")  # type: ignore[attr-defined]
    default, _ = _run(monkeypatch, tmp_path)
    assert default.reports_tenants == ("memory",)  # type: ignore[attr-defined]
