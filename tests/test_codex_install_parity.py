"""The Codex installer writes the hook config the shared `recall_hooks` readers need.

Findings ENV-001 (with its duplicates DAT-001 and DEPLOY-002) and DEPLOY-004 of the 2026-11-04
CCA hunt of `recall_hooks`, fixed at the owner's request ("make it the same as we do here").

Red proof: each test was run on the test host against the pre-fix `recall/codex.py`, failed in its
own assertion, and passed on the fix; the node ids and verdicts are in the pull request. No test
imports a symbol the fix introduced (`hook_config_document` is reached only through the
installer), so each one reaches the pre-fix code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recall.codex import _hook_entries, install_codex_integration


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project: Path, **extra: Any) -> dict[str, Any]:
    codex_home = tmp_path / "codex"
    marketplace = tmp_path / "agents" / "plugins" / "marketplace.json"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.chdir(project)
    install_codex_integration(
        dsn="postgresql://example/recall",
        tenant="memory",
        embedder="fastembed",
        hooks_path=codex_home / "hooks.json",
        marketplace_path=marketplace,
        plugin_destination=marketplace.parent / ".codex" / "plugins" / "recall",
        python_executable="python-test",
        print_fn=lambda *args, **kwargs: None,
        **extra,
    )
    return json.loads((codex_home / "re-call" / "recall-hook.json").read_text(encoding="utf-8"))


def test_env001_the_codex_config_admits_events_in_its_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENV-001. Invariant: the config the Codex installer writes passes the project boundary the
    shared write-time hook and session-end index apply, for an event in the install's project.

    Defect caught: the Codex writer omitted `project_root`, which both readers require, so
    write-time retrieval and the memory index never ran on any Codex install, even after a
    reinstall. Production line: `_write_hook_config` in `recall/codex.py`.
    """
    from recall_hooks import write_time

    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    config = _install(tmp_path, monkeypatch, project)
    admitted = write_time._event_is_in_configured_project(config, {"cwd": str(project / "src")})
    assert admitted, f"the Codex config refuses its own project: {sorted(config)}"
    assert write_time.settings(config)["connection_mode"] == "relay"


def test_env001_a_codex_reinstall_keeps_the_operators_general_knowledge_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENV-001, the rest of "the same as Claude Code". Invariant: reinstalling Codex keeps an
    operator's `general_knowledge` block, as the Claude Code installer does.

    Defect caught: the Codex writer rewrote the file without the block, and an absent block means
    enabled, so an explicit `enabled: false` was switched back on by the next install.
    Production line: `_write_hook_config` in `recall/codex.py`.
    """
    project = tmp_path / "project"
    project.mkdir()
    config_path = tmp_path / "codex" / "re-call" / "recall-hook.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"general_knowledge": {"enabled": False}}), encoding="utf-8")
    config = _install(tmp_path, monkeypatch, project)
    assert config.get("general_knowledge") == {"enabled": False}, config


def test_deploy004_the_codex_hook_command_isolates_the_session_directory() -> None:
    """DEPLOY-004. Invariant: every Codex hook runs `python -P -m recall_hooks.codex`, so the
    session's directory is not first on sys.path.

    Defect caught: without `-P`, a session opened in a checkout holding another `recall_hooks`
    or `recall` ran that copy instead of the installed one. The Claude Code installer already
    passes `-P`. Production line: the `args` of `command_group` in `recall.codex._hook_entries`.
    """
    for event, groups in _hook_entries("python-test").items():
        args = groups[0]["hooks"][0]["args"]
        assert args[0] == "-P", (event, args)
