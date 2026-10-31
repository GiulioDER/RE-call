"""PR440/446 safety regressions. Red baseline: 160a23a3f, unchanged test nodes.

Each test asserts behavior at existing consumer boundaries. The baseline fails the
stated assertion, while controls in the same node preserve intended behavior.
"""
from __future__ import annotations

import io
import json
import os
import stat
from pathlib import Path

import pytest

from recall.wizard.stack import COMPOSE_NAME, add_tenant_services
from recall.wizard.uninstall import UninstallRefusal, execute, plan_uninstall
from tests.test_wizard_uninstall import _docker, _install


def _registered_install(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    root = _install(tmp_path, project_root=project)
    config = tmp_path / "client.json"
    config.write_text(json.dumps({"projects": {str(project): {"mcpServers": {
        "recall": {"command": "python", "cwd": str(project)},
        "other": {"command": "other", "env": {"TOKEN": "fake-credential-marker"}},
    }}}}), encoding="utf-8")
    config.chmod(0o600)
    return project, root, config


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode-bit exposure, exercised on the remote test runner")
def test_unregister_keeps_credentials_private_at_first_write(tmp_path, monkeypatch):
    """SEC-001: _unregister at baseline writes the replacement at 0644 before chmod.

    Red assertion: every credential-bearing write must already be private. Final
    chmod cannot repair a prior exposure. Other registration and removal controls
    are checked against the actual committed configuration.
    """
    project, root, config = _registered_install(tmp_path)
    plan = plan_uninstall(data_root=root, claude_config_path=config,
                          profile_path=tmp_path / "missing-profile", runner=_docker())
    observed = []
    original_open = io.open

    class Writer:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, key):
            return getattr(self.handle, key)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, value):
            token = b"fake-credential-marker" if isinstance(value, bytes) else "fake-credential-marker"
            if token in value:
                observed.append(stat.S_IMODE(os.fstat(self.handle.fileno()).st_mode))
            return self.handle.write(value)

    def watch_open(*args, **kwargs):
        handle = original_open(*args, **kwargs)
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        return Writer(handle) if any(c in mode for c in "wax+") else handle

    monkeypatch.setattr(io, "open", watch_open)
    old_umask = os.umask(0o022)
    try:
        report = execute(plan, claude_config_path=config, runner=_docker())
    finally:
        os.umask(old_umask)
    assert observed, "the actual backup and replacement must be observed"
    assert all(mode & 0o077 == 0 for mode in observed), observed
    assert not report.failed
    servers = json.loads(config.read_text())["projects"][str(project)]["mcpServers"]
    assert "recall" not in servers
    assert servers["other"]["env"]["TOKEN"] == "fake-credential-marker"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture, exercised on the remote test runner")
def test_unregister_does_not_follow_predictable_temporary_symlink(tmp_path):
    """SEC-001: baseline writes credentials through client.json.tmp into a foreign file."""
    _project, root, config = _registered_install(tmp_path)
    sentinel = tmp_path / "foreign.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    config.with_name(config.name + ".tmp").symlink_to(sentinel)
    plan = plan_uninstall(data_root=root, claude_config_path=config,
                          profile_path=tmp_path / "missing-profile", runner=_docker())
    report = execute(plan, claude_config_path=config, runner=_docker())
    assert sentinel.read_text() == "untouched"
    assert not config.is_symlink()
    assert not report.failed


def test_uninstall_refuses_a_compose_change_after_confirmation(tmp_path):
    """STAKES-001: baseline sends an unreviewed volume declaration to compose down -v."""
    root = _install(tmp_path)
    plan = plan_uninstall(data_root=root, purge_data=True, runner=_docker(),
                          profile_path=tmp_path / "missing-profile")
    compose = root / COMPOSE_NAME
    document = json.loads(compose.read_text())
    document["volumes"]["unreviewed"] = {"name": "unreviewed-test-volume"}
    compose.write_text(json.dumps(document), encoding="utf-8")
    docker = _docker()
    try:
        execute(plan, purge_data=True, runner=docker)
    except UninstallRefusal:
        pass
    assert not docker.calls, "changed configuration must be refused before destructive commands"
    assert compose.exists()

    control = _install(tmp_path / "control")
    control_plan = plan_uninstall(data_root=control, purge_data=True, runner=_docker(),
                                  profile_path=tmp_path / "missing-profile")
    accepted = _docker()
    assert not execute(control_plan, purge_data=True, runner=accepted).failed
    assert any("down" in call and "-v" in call for call in accepted.calls)


def test_uninstall_executes_only_reviewed_compose_bytes(tmp_path):
    """STAKES-001: baseline reads mutable config at execution, even after a late edit.

    The injected Docker boundary performs the edit after execute's freshness check.
    The command must still name a snapshot of the reviewed declaration.
    """
    root = _install(tmp_path)
    plan = plan_uninstall(data_root=root, purge_data=True, runner=_docker(),
                          profile_path=tmp_path / "missing-profile")
    original = (root / COMPOSE_NAME).read_bytes()
    seen = []

    def docker(command):
        if "down" in command:
            changed = json.loads(original)
            changed["volumes"]["late"] = {"name": "unreviewed-test-volume"}
            (root / COMPOSE_NAME).write_text(json.dumps(changed), encoding="utf-8")
            executed = Path(command[command.index("-f") + 1])
            seen.append(executed.read_bytes())
        return 0, ""

    execute(plan, purge_data=True, runner=docker)
    assert seen == [original], "Docker must consume exactly the reviewed configuration"
    assert not list(root.glob(".recall-uninstall-*.json")), "private snapshot must be cleaned"


def test_explicit_relative_bind_is_refused_without_rewriting_stack(tmp_path):
    """ENV-001: baseline treats source pgdata as a volume despite explicit type bind."""
    path = tmp_path / COMPOSE_NAME
    document = {"name": "recall-safety", "services": {"db": {"volumes": [
        {"type": "bind", "source": "pgdata", "target": "/var/lib/postgresql"}
    ]}}, "volumes": {"pgdata": None}}
    original = json.dumps(document).encode()
    path.write_bytes(original)
    with pytest.raises(ValueError, match="WAL"):
        add_tenant_services(path, {"new-docs": {"RECALL_TENANT": "new-docs"}})
    assert path.read_bytes() == original
    document["services"]["db"]["volumes"][0]["type"] = "volume"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert add_tenant_services(path, {"new-docs": {"RECALL_TENANT": "new-docs"}}) == ("new-docs",)
