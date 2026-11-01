"""Review regressions for STAKES-001 and DAT-002 at real consumer boundaries.

Red proof replays the preserved pre-review production bytes as plausible mutations:
unguarded execute calls in desktop.main/setup_wizard, and unhandled lock entry
in wiring.register_local_scope. Source hashes are preserved with the proof receipt.
The same test nodes run unchanged against the repaired boundaries.
"""
from __future__ import annotations

import errno
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.cli import main as cli_main
from recall.desktop.main import uninstall_main
from recall.wizard import uninstall
from recall.wizard.stack import COMPOSE_NAME
from recall.wizard.wiring import LocalScopeRegistration, ServerBlock, register_local_scope
from tests.test_wizard_uninstall import _docker, _install


@pytest.mark.parametrize("boundary", ["desktop", "cli"])
@pytest.mark.parametrize("failure", ["changed-compose", "snapshot-creation", "snapshot-read"])
def test_uninstall_execution_failures_are_visible_before_destructive_commands(
    tmp_path, monkeypatch, boundary, failure
):
    """STAKES-001: pre-review mutation escapes as raw error instead of a visible refusal."""
    root = _install(tmp_path)
    compose = root / COMPOSE_NAME
    docker = _docker()
    monkeypatch.setattr(uninstall, "_run", docker)
    original_mkstemp = uninstall.tempfile.mkstemp
    original_read = Path.read_bytes
    armed = False

    def fail_create(*args, **kwargs):
        if armed and kwargs.get("prefix") == ".recall-uninstall-":
            raise OSError("cannot create reviewed snapshot")
        return original_mkstemp(*args, **kwargs)

    def fail_read(path, *args, **kwargs):
        if armed and path == compose:
            raise OSError("cannot verify reviewed snapshot")
        return original_read(path, *args, **kwargs)

    def confirm(*args):
        nonlocal armed
        if failure == "changed-compose":
            document = json.loads(compose.read_text())
            document["volumes"]["unreviewed"] = {"name": "unreviewed-test-volume"}
            compose.write_text(json.dumps(document), encoding="utf-8")
        elif failure == "snapshot-creation":
            monkeypatch.setattr(uninstall.tempfile, "mkstemp", fail_create)
        else:
            monkeypatch.setattr(Path, "read_bytes", fail_read)
        armed = True
        return True

    if boundary == "desktop":
        notifications = []
        try:
            result = uninstall_main(
                ["--data-root", str(root), "--purge-data"],
                confirm=confirm, notify=lambda *args: notifications.append(args), runner=docker,
            )
        except (uninstall.UninstallRefusal, OSError) as exc:
            result = exc
        assert result == 1, "execution refusal must return a visible failure instead of escaping"
        assert notifications and notifications[-1][0] == "Cannot uninstall"
        assert "Uninstall finished" not in [item[0] for item in notifications]
    else:
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("builtins.input", lambda *args: "y" if confirm() else "n")
        visible = None
        try:
            cli_main(["uninstall", "--data-root", str(root), "--purge-data"])
        except SystemExit as exc:
            visible = str(exc)
        except (uninstall.UninstallRefusal, OSError):
            pass
        assert visible, "the CLI must translate execute failures into its normal SystemExit report"
    assert not any("down" in call or "rm" in call for call in docker.calls)
    assert compose.exists(), "a refused uninstall must remain retryable"


def test_registration_lock_failure_preserves_skip_report_and_body_errors(tmp_path, monkeypatch):
    """DAT-002 mutation: direct unhandled lock entry raises before the historical skip result.

    SEC-001 writer adaptation: inject at backup publication rather than the old
    Path.write_text transport. Red proof: the first private-writer fix caught
    OSError and failed the unchanged expected-exception assertion; propagation
    restores the original protected-body contract.
    """
    missing = tmp_path / "missing-client.json"
    blocks = (ServerBlock("recall-docs", "docs", {}, "fixture"),)
    original_open = os.open

    def denied(path, *args, **kwargs):
        if str(path).endswith(".recall-lock"):
            raise PermissionError(errno.EACCES, "lock denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", denied)
    try:
        report = register_local_scope(blocks, project_root=tmp_path, config_path=missing)
    except OSError as exc:
        report = exc
    assert isinstance(report, LocalScopeRegistration), "lock entry must preserve the skip-report contract"
    assert report.skipped_reason and not report.registered
    assert not missing.exists()

    # The acquisition catch must not swallow an existing protected-body/write failure.
    monkeypatch.setattr(os, "open", original_open)
    config = tmp_path / "client.json"
    config.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    original_replace = os.replace

    def broken_backup(source, destination, *args, **kwargs):
        if Path(destination).name.endswith(".recall-backup"):
            raise OSError("existing body write failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", broken_backup)
    with pytest.raises(OSError, match="existing body write failure"):
        register_local_scope(blocks, project_root=tmp_path, config_path=config)
