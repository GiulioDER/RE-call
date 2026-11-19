"""Lite setup and the wizard give Claude Code and Codex the hooks, skills and servers.

`recall setup` (PostgreSQL) has long installed hooks and skills for Claude Code and the full Codex
integration. `recall setup --lite` and `recall wizard` registered Claude Code's MCP servers and
nothing else, so a Codex user got no RE-call at all from them. Both now call
`recall.wizard.wiring.install_client_extras`. Every client config here lives under the fake home
the autouse `_confine_claude_client_config` fixture provides, so nothing real is touched.

Invariants and the failure each one catches:
- X1 Lite setup registers in `<project>/.codex/config.toml` the server Claude Code got, with the
  same command, arguments, working directory and environment. Catches a Codex server that would
  start a different interpreter or serve another store.
- X2 Lite setup installs Codex's seven hook events and a hook config bound to the project, with the
  Lite memory folder as its memo store and write-time search off (it reads PostgreSQL only).
- X3 Lite setup installs Claude Code's hooks and the same hook config shape, and every bundled
  skill for both clients, each into its own client's directory.
- X4 A hook config already bound to another project is left alone and reported, for both clients.
- X5 `.codex/config.toml`: the user's own content and a server name the user defined are kept and
  reported, and a second run leaves exactly one RE-call block.
- X6 A `.codex/config.toml` that git tracks is refused, so machine settings are never committed.
- X7 Prompt-time search uses the configured memo folder inside its project only, and the project's
  own store elsewhere.
- X8 SessionEnd does not run the PostgreSQL indexer against a Lite (`sqlite:///`) store.
- X9 The wizard binds the hooks to its writable memory tenant with write-time search on, and
  registers every one of its servers for Codex.

Red proof: see the pull request that added this file; each test failed in its own assertion
against a named mutation of the production line, then passed with the line restored.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from recall.lite.setup import SERVER_NAME, run_lite_setup
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos

SKILLS = {"check-memory-before-acting", "keep-memory-current", "re-call", "session-close", "session-open"}
CODEX_EVENTS = {
    "SessionStart", "PreCompact", "UserPromptSubmit", "PreToolUse", "PostToolUse", "SubagentStart",
    "SessionEnd",
}


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Both clients present, each with its config home under the fake home."""
    home = Path.home()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr("recall.claude_code.claude_code_detected", lambda: True)
    monkeypatch.setattr("recall.codex.codex_code_detected", lambda: True)
    return home


def _lite(tmp_path: Path, project: Path | None = None):  # noqa: ANN202
    project = project or tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    if not (project / "notes").exists():
        _write_memos(project / "notes", 48)
    claude = tmp_path / "claude.json"
    if not claude.exists():
        claude.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    report = run_lite_setup(
        project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(),
        embedder_name="content-words", config_path=claude,
    )
    return project, claude, report


def _claude_server(claude: Path) -> dict[str, object]:
    projects = json.loads(claude.read_text(encoding="utf-8"))["projects"]
    return next(entry["mcpServers"][SERVER_NAME] for entry in projects.values())


def _codex_servers(project: Path) -> dict[str, dict[str, object]]:
    return tomllib.loads((project / ".codex" / "config.toml").read_text(encoding="utf-8"))["mcp_servers"]


def test_lite_registers_the_same_server_for_codex(tmp_path: Path, clients: Path) -> None:
    """X1."""
    project, claude, report = _lite(tmp_path)
    expected = _claude_server(claude)
    server = _codex_servers(project)[SERVER_NAME]
    for key in ("command", "args", "cwd", "env"):
        assert server[key] == expected[key], f"the Codex server's {key} differs from Claude Code's"
    assert server["startup_timeout_sec"] == 60
    assert any(client == "codex" and "registered" in text for client, text in report.clients), report.render()


def test_lite_installs_codex_hooks_bound_to_the_project(tmp_path: Path, clients: Path) -> None:
    """X2."""
    project, _claude, _report = _lite(tmp_path)
    hooks = json.loads((clients / ".codex" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    assert set(hooks) == CODEX_EVENTS, "Codex did not get every RE-call hook event"
    config = json.loads((clients / ".codex" / "re-call" / "recall-hook.json").read_text(encoding="utf-8"))
    assert config["dsn"].startswith("sqlite:///") and config["project_root"] == str(project.resolve())
    assert config["memory_root"] == str((project / "notes").resolve()), "prompt-time would not read the notes"
    assert config["write_time"]["enabled"] is False, "write-time search was left on for a SQLite store"


def test_lite_installs_claude_hooks_and_both_clients_skills(tmp_path: Path, clients: Path) -> None:
    """X3."""
    project, _claude, _report = _lite(tmp_path)
    settings = json.loads((clients / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert "UserPromptSubmit" in settings["hooks"], "Claude Code got no hooks"
    config = json.loads((clients / ".claude" / "recall-hook.json").read_text(encoding="utf-8"))
    assert config["memory_root"] == str((project / "notes").resolve())
    assert config["write_time"]["enabled"] is False
    claude_skills = {p.parent.name for p in (clients / ".claude" / "skills").glob("*/SKILL.md")}
    codex_skills = {p.parent.name for p in (clients / ".agents" / "skills").glob("*/SKILL.md")}
    assert claude_skills == SKILLS, "Claude Code did not get every skill"
    assert codex_skills == SKILLS, "Codex did not get every skill in its own skills folder"


def test_hooks_bound_to_another_project_are_left_alone(tmp_path: Path, clients: Path) -> None:
    """X4."""
    other = {"dsn": "postgresql://x@127.0.0.1:1/x", "project_root": str(tmp_path / "other"), "chunks": 7}
    for path in (clients / ".claude" / "recall-hook.json", clients / ".codex" / "re-call" / "recall-hook.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(other), encoding="utf-8")
    _project, _claude, report = _lite(tmp_path)
    for path in (clients / ".claude" / "recall-hook.json", clients / ".codex" / "re-call" / "recall-hook.json"):
        assert json.loads(path.read_text(encoding="utf-8")) == other, f"{path} was rebound"
    bound = [client for client, text in report.clients if text.startswith("hooks left bound to")]
    assert sorted(bound) == ["claude code", "codex"], report.render()


def test_codex_config_keeps_the_users_content_and_names(tmp_path: Path, clients: Path) -> None:
    """X5."""
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    user = 'model = "mine"\n\n[mcp_servers.other]\ncommand = "other"\n'
    (project / ".codex" / "config.toml").write_text(user, encoding="utf-8")
    _lite(tmp_path, project)
    _lite(tmp_path, project)
    text = (project / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert text.startswith(user) and text.count("# BEGIN RE-call") == 1, text
    assert set(_codex_servers(project)) == {"other", SERVER_NAME}

    taken = tmp_path / "taken"
    (taken / ".codex").mkdir(parents=True)
    mine = f'[mcp_servers.{SERVER_NAME}]\ncommand = "hand-written"\n'
    (taken / ".codex" / "config.toml").write_text(mine, encoding="utf-8")
    _taken, _claude, report = _lite(tmp_path, taken)
    assert (taken / ".codex" / "config.toml").read_text(encoding="utf-8") == mine, "a user server was replaced"
    assert any(client == "codex" and "NOT registered" in text for client, text in report.clients), report.render()


def test_a_tracked_codex_config_is_refused(tmp_path: Path, clients: Path) -> None:
    """X6."""
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    tracked = 'model = "shared"\n'
    (project / ".codex" / "config.toml").write_text(tracked, encoding="utf-8")
    for args in (["init", "-q"], ["add", ".codex/config.toml"]):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    _project, _claude, report = _lite(tmp_path, project)
    assert (project / ".codex" / "config.toml").read_text(encoding="utf-8") == tracked
    assert any("tracked by git" in text for _client, text in report.clients), report.render()


def test_prompt_time_reads_the_configured_folder_only_in_its_project(
    tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X7."""
    from recall_hooks import prompt_time

    project = tmp_path / "project"
    notes = project / "notes"
    notes.mkdir(parents=True)
    config = clients / ".claude" / "recall-hook.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({"dsn": "sqlite:///x", "project_root": str(project), "memory_root": str(notes)}),
        encoding="utf-8",
    )
    seen: list[str] = []
    monkeypatch.setattr(prompt_time, "find_store", lambda cwd, override: seen.append(override))
    prompt = "which decision did we take about the release checklist and the changelog"
    prompt_time.user_prompt_submit({"prompt": prompt, "cwd": str(project)})
    prompt_time.user_prompt_submit({"prompt": prompt, "cwd": str(tmp_path / "elsewhere")})
    assert seen == [str(notes), ""], "the configured folder leaked into another project"


def test_session_end_does_not_index_a_lite_store(
    tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X8."""
    import recall.memory_index
    import recall_hooks

    project = tmp_path / "project"
    (project / "memory").mkdir(parents=True)
    config = clients / ".claude" / "recall-hook.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({"dsn": "sqlite:///C:/store.db", "project_root": str(project)}), encoding="utf-8"
    )
    calls: list[object] = []
    monkeypatch.setattr(recall.memory_index, "index_memory_directory", lambda **kw: calls.append(kw))
    assert recall_hooks._index_and_refresh({"cwd": str(project)}) == 0
    assert calls == [], "the PostgreSQL indexer ran against a SQLite store"


def test_the_wizard_wires_hooks_to_its_memory_tenant(tmp_path: Path, clients: Path) -> None:
    """X9."""
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "memory").mkdir()
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    report = run_headless(
        _config(tmp_path, project_root=str(root)), services=_CountingSpy(), claude_config_path=client
    )
    writable = [block for block in report.servers if block.env.get("RECALL_INDEX_MODE") == "legacy"]
    assert len(writable) == 1 and report.servers
    for path in (clients / ".claude" / "recall-hook.json", clients / ".codex" / "re-call" / "recall-hook.json"):
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["tenant"] == writable[0].tenant, f"{path} serves the wrong tenant"
        assert config["write_time"]["enabled"] is True
        assert config["memory_root"] == str((tmp_path / "memory").resolve())
    assert set(_codex_servers(root)) == {block.name for block in report.servers}
    assert "codex: registered" in report.render()
