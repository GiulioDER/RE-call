"""DAT-002: shared config edits must preserve another writer's committed removal.

Red baseline160a23a3f: the first writer reads old JSON, waits while the second
removes a registration, then overwrites that removal. The final content fails.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from recall.wizard.uninstall import execute, plan_uninstall
from recall.wizard.wiring import ServerBlock, register_local_scope
from tests.test_wizard_uninstall import _docker, _install


@pytest.mark.parametrize("first_operation", ["uninstall", "register"])
def test_cooperating_configuration_writers_do_not_restore_removed_registration(
    tmp_path, monkeypatch, first_operation
):
    """Targets the real writer RMW paths; no assertion depends on scheduler timing."""
    projects = [tmp_path / "project-a", tmp_path / "project-b"]
    for project in projects:
        project.mkdir()
    roots = [_install(tmp_path / label, project_root=project)
             for label, project in zip(("a", "b"), projects, strict=True)]
    config = tmp_path / "client.json"
    config.write_text(json.dumps({"unrelated": "kept", "projects": {
        str(project): {"mcpServers": {name: {"command": "python", "cwd": str(project)}}}
        for project, name in zip(projects, ("recall-a", "recall-b"), strict=True)
    }}), encoding="utf-8")
    plans = [plan_uninstall(data_root=root, claude_config_path=config, runner=_docker(),
                            profile_path=tmp_path / "absent-profile") for root in roots]
    original_read = Path.read_text
    first_read = threading.Event()
    release_first = threading.Event()
    second_read = threading.Event()
    second_started = threading.Event()

    def gated_read(path, *args, **kwargs):
        value = original_read(path, *args, **kwargs)
        if path == config:
            if threading.current_thread().name.endswith("_0") and not first_read.is_set():
                first_read.set()
                assert release_first.wait(10), "coordinator must release first writer"
            elif threading.current_thread().name.endswith("_1"):
                second_read.set()
        return value

    monkeypatch.setattr(Path, "read_text", gated_read)

    def first():
        if first_operation == "register":
            return register_local_scope(
                (ServerBlock("recall-new", "new", {}, "fixture"),),
                project_root=projects[0], config_path=config, interpreter="python",
            )
        return execute(plans[0], claude_config_path=config, runner=_docker())

    def second():
        second_started.set()
        return execute(plans[1], claude_config_path=config, runner=_docker())

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="config-writer") as executor:
        first_future = executor.submit(first)
        try:
            assert first_read.wait(5), "must enter the real first configuration read"
            second_future = executor.submit(second)
            assert second_started.wait(5)
            if second_read.wait(1):
                second_future.result(timeout=5)
        finally:
            release_first.set()
        first_report = first_future.result(timeout=5)
        second_report = second_future.result(timeout=5)
    assert not second_report.failed
    if first_operation == "uninstall":
        assert not first_report.failed
    else:
        assert "recall-new" in first_report.registered
    result = json.loads(config.read_text())
    assert result["projects"][str(projects[1])]["mcpServers"] == {}, (
        "the later writer restored a removed registration"
    )
    servers_a = result["projects"][str(projects[0])]["mcpServers"]
    assert ("recall-a" in servers_a) == (first_operation == "register")
    assert ("recall-new" in servers_a) == (first_operation == "register")
    assert result["unrelated"] == "kept"
