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

The tests after X9 each name the CCA finding they pin (run 20261119T182333Z).

Red proof: see the pull requests that added these tests; each failed in its own assertion
against a named mutation of the production line, then passed with the line restored.
"""

from __future__ import annotations

import json
import os
import stat
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


# ----------------------------------------------------------------------------------------------
# CCA run 20261119T182333Z: one test per confirmed P2 (the finding id is in each docstring).
# ----------------------------------------------------------------------------------------------


def _hook_config(home: Path, client: str) -> dict[str, object]:
    path = home / ".claude" / "recall-hook.json" if client == "claude" else home / ".codex" / "re-call" / "recall-hook.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _git_init(project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True, capture_output=True)


def test_the_codex_plugin_server_gets_the_block_settings(tmp_path: Path, clients: Path) -> None:
    """DAT-001: the Codex hook config is what the plugin's server launches from, so it carries the
    block's own settings as `server_env`, and none of the five keys the launcher sets itself."""
    project, _claude, _report = _lite(tmp_path)
    server_env = _hook_config(clients, "codex").get("server_env", {})
    assert isinstance(server_env, dict)
    assert server_env.get("RECALL_INDEX_ROOT") == str((project / "notes").resolve()), "the plugin server lost the block settings"
    assert server_env.get("RECALL_TRUST_MODE") == "strict"
    assert not set(server_env) & {"RECALL_SERVING_DSN", "RECALL_TENANT", "RECALL_EMBEDDER", "RECALL_TABLE", "RECALL_TRANSPORT"}


def test_git_that_cannot_answer_in_a_work_tree_refuses_the_write(
    tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAKES-001: inside a git work tree, no answer from git (missing, timed out, refused) is not
    "not tracked": the credentials are not written."""
    project = tmp_path / "project"
    _git_init(project)
    monkeypatch.setattr("recall.codex._git", lambda *_a: None)
    _project, _claude, report = _lite(tmp_path, project)
    assert not (project / ".codex" / "config.toml").exists(), "credentials written though git could not answer"
    assert any("git could not say" in text for _client, text in report.clients), report.render()


def test_an_untracked_codex_config_is_added_to_gitignore(tmp_path: Path, clients: Path) -> None:
    """STAKES-004: in a git work tree, `.codex/config.toml` is ignored before it is written."""
    project = tmp_path / "project"
    _git_init(project)
    _lite(tmp_path, project)
    assert (project / ".codex" / "config.toml").exists()
    ignored = subprocess.run(
        ["git", "-C", str(project), "check-ignore", "-q", "--", ".codex/config.toml"], capture_output=True
    )
    assert ignored.returncode == 0, "the credential file is not ignored by git"


def test_a_linked_codex_folder_is_refused(tmp_path: Path, clients: Path) -> None:
    """SEC-002: a `.codex` that is a link (a junction on Windows) is refused, so the block is not
    written outside the project."""
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(outside), str(project / ".codex"))
    else:
        os.symlink(outside, project / ".codex")
    _project, _claude, report = _lite(tmp_path, project)
    assert not (outside / "config.toml").exists(), "the block was written through the link"
    assert any("through a link" in text for _client, text in report.clients), report.render()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_an_existing_codex_config_becomes_private(tmp_path: Path, clients: Path) -> None:
    """STAKES-003: a world-readable `.codex/config.toml` is rewritten 0600 once it holds the env."""
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    existing = project / ".codex" / "config.toml"
    existing.write_text('model = "mine"\n', encoding="utf-8")
    existing.chmod(0o644)
    _lite(tmp_path, project)
    assert stat.S_IMODE(existing.stat().st_mode) == 0o600, "credentials left world-readable"


def test_a_skill_the_user_wrote_is_left_alone(tmp_path: Path, clients: Path) -> None:
    """STAKES-002: a skill of the same name that RE-call did not install is neither overwritten nor
    hidden; RE-call's own copies are marked so a later run can refresh them."""
    mine = clients / ".agents" / "skills" / "session-open" / "SKILL.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("---\nname: session-open\ndescription: mine\n---\nmy own\n", encoding="utf-8")
    _project, _claude, report = _lite(tmp_path)
    assert mine.read_text(encoding="utf-8").endswith("my own\n"), "the user's skill was overwritten"
    assert any("left session-open alone" in text for client, text in report.clients if client == "codex"), report.render()
    assert (clients / ".agents" / "skills" / "re-call" / ".installed-by-recall").exists()


def test_the_claude_plugin_counts_only_when_it_runs_here(tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DEPLOY-003: an install for another project, or a disabled plugin, does not stand in for the
    hooks; a user-scope, enabled install whose command resolves does."""
    from recall.claude_code import claude_plugin_installed

    project = tmp_path / "project"
    project.mkdir()
    records = clients / ".claude" / "plugins" / "installed_plugins.json"
    records.parent.mkdir(parents=True)
    settings = clients / ".claude" / "settings.json"
    monkeypatch.setattr("recall.claude_code.shutil.which", lambda name: f"/bin/{name}")

    def plugin(entry: dict[str, object], enabled: object = True) -> bool:
        records.write_text(json.dumps({"version": 2, "plugins": {"recall@re-call": [entry]}}), encoding="utf-8")
        settings.write_text(json.dumps({"enabledPlugins": {"recall@re-call": enabled}}), encoding="utf-8")
        return claude_plugin_installed(project)

    assert plugin({"scope": "project", "projectPath": str(tmp_path / "other")}) is False, "another project's install counted"
    assert plugin({"scope": "user"}, enabled=False) is False, "a disabled plugin counted"
    assert plugin({"scope": "user"}) is True
    assert plugin({"scope": "local", "projectPath": str(project)}) is True


def test_session_end_indexes_the_configured_memo_folder(
    tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENV-002: the wizard's memo folder may sit outside the project; SessionEnd indexes that folder,
    the one prompt-time search reads, not `<project>/memory`."""
    import recall.memory_index
    import recall_hooks

    project = tmp_path / "project"
    (project / "memory").mkdir(parents=True)
    memos = tmp_path / "data" / "memory"
    memos.mkdir(parents=True)
    config = clients / ".claude" / "recall-hook.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({"dsn": "postgresql://x@127.0.0.1:1/x", "project_root": str(project), "memory_root": str(memos)}),
        encoding="utf-8",
    )
    calls: list[Path] = []
    monkeypatch.setattr(recall.memory_index, "index_memory_directory", lambda **kw: calls.append(kw["memory_dir"]))
    recall_hooks._index_and_refresh({"cwd": str(project)})
    assert calls == [memos.resolve()], "SessionEnd indexed a different folder than prompt-time reads"


def test_a_lite_hook_config_carries_the_chunk_count(tmp_path: Path, clients: Path) -> None:
    """DAT-006: the SessionStart digest needs a count, and a Lite store's is read from its file."""
    _project, _claude, report = _lite(tmp_path)
    assert report.chunks > 0
    assert _hook_config(clients, "claude")["chunks"] == report.chunks, "the Lite count stayed 0"


def test_a_codex_registration_completes_the_wizard(tmp_path: Path, clients: Path) -> None:
    """DOC-001: with no Claude Code config but Codex registered, the install is complete and the
    report does not say no client was registered."""
    from recall.wizard.headless import run_headless
    from tests.test_wizard_headless import _Spy
    from tests.test_wizard_state import _config

    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "memory").mkdir()
    report = run_headless(
        _config(tmp_path, project_root=str(root)),
        services=_Spy(),
        claude_config_path=tmp_path / "absent" / ".claude.json",
    )
    assert report.codex_registered
    assert report.ok, report.render()
    assert "no client was registered" not in report.render()


def test_operator_hook_switches_survive_a_rerun(tmp_path: Path, clients: Path) -> None:
    """DAT-002: prompt-time turned off, its tuning, and a `cold` connection mode stay as the operator
    set them in this project's hook config."""
    project = tmp_path / "project"
    project.mkdir()
    config = clients / ".claude" / "recall-hook.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({
            "dsn": "sqlite:///x", "project_root": str(project.resolve()),
            "prompt_time": {"enabled": False, "k": 7}, "write_time": {"enabled": False, "connection_mode": "cold"},
        }),
        encoding="utf-8",
    )
    _lite(tmp_path, project)
    after = _hook_config(clients, "claude")
    assert after["prompt_time"] == {"enabled": False, "k": 7}, "the operator's prompt-time switch was reset"
    assert after["write_time"]["connection_mode"] == "cold"


def test_the_hook_config_homes_are_confined() -> None:
    """STAKES-005: `recall_hooks.claude_config_home` reads these before CLAUDE_CONFIG_DIR, so the
    autouse fixture must clear them or a test writes into the developer's real directory.
    Red proof sets RECALL_HOOK_CONFIG_HOME in the pytest process's environment."""
    assert "RECALL_HOOK_CONFIG_HOME" not in os.environ, "a test can reach the developer's hook config"
    assert "RECALL_PROJECT_CONFIG_HOME" not in os.environ


def test_an_unusable_memo_folder_skips_the_index_not_the_count(
    tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENV-002 follow-up (L5.5): a missing or relative `memory_root` indexes nothing, and the cached
    count is still refreshed, as with a missing `<project>/memory`."""
    import recall.memory_index
    import recall_hooks

    project = tmp_path / "project"
    project.mkdir()
    config = clients / ".claude" / "recall-hook.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    indexed: list[object] = []
    refreshed: list[object] = []
    monkeypatch.setattr(recall.memory_index, "index_memory_directory", lambda **kw: indexed.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda cfg=None: refreshed.append(cfg) or 0)
    for folder in (str(tmp_path / "gone"), "relative/notes"):
        config.write_text(
            json.dumps({"dsn": "postgresql://x@127.0.0.1:1/x", "project_root": str(project), "memory_root": folder}),
            encoding="utf-8",
        )
        recall_hooks._index_and_refresh({"cwd": str(project)})
    assert indexed == [], "an unusable memo folder was indexed"
    assert len(refreshed) == 2, "the count refresh was skipped"


# ----------------------------------------------------------------------------------------------
# CCA deferred run 20261119T203938Z: one test per confirmed P3 with a behaviour change.
# ----------------------------------------------------------------------------------------------


def _server(project: Path, **env: str) -> dict[str, dict[str, object]]:
    return {"recall-memory": {"command": "python", "args": ["-m", "recall_mcp.server"], "cwd": str(project), "env": env}}


def test_a_path_outside_the_bmp_registers(tmp_path: Path, clients: Path) -> None:
    """DAT-004: a value with a character outside the BMP is written as itself, not as a UTF-16
    surrogate pair TOML refuses, and DEL is escaped."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    project.mkdir()
    value = "C:/notes/" + chr(0x1F600) + chr(0x7F)
    try:
        reg = register_project_servers(_server(project, RECALL_INDEX_ROOT=value), project_root=project)
    except ValueError as exc:  # the write guard: the rendered TOML did not parse
        pytest.fail(f"the rendered config was not valid TOML: {exc}")
    assert reg.registered == ("recall-memory",), reg.skipped_reason
    assert _codex_servers(project)["recall-memory"]["env"]["RECALL_INDEX_ROOT"] == value


def test_a_skill_that_cannot_be_copied_is_named(
    tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-002: a skill whose copy fails is named in the report, not silently missing."""
    import recall.claude_code

    real = recall.claude_code.install_user_skill

    def failing(source: Path, *, name: str | None = None, **kwargs: object) -> None:
        if name == "re-call":
            raise PermissionError("denied")
        real(source, name=name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(recall.claude_code, "install_user_skill", failing)
    _project, _claude, report = _lite(tmp_path)
    codex = [text for client, text in report.clients if client == "codex" and text.startswith("skills ")]
    assert codex and "NOT installed: re-call" in codex[0], report.render()


def test_session_end_indexes_with_the_corpus_settings(
    tmp_path: Path, clients: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DAT-007: the session-end index embeds with the settings the corpus was built under (here an
    embedding profile), laid over the hook's own environment."""
    import recall.memory_index
    import recall_hooks

    project = tmp_path / "project"
    (project / "memory").mkdir(parents=True)
    config = clients / ".claude" / "recall-hook.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({
            "dsn": "postgresql://x@127.0.0.1:1/x", "project_root": str(project),
            "server_env": {"RECALL_EMBED_PROFILE": "a-profile"},
        }),
        encoding="utf-8",
    )
    seen: list[object] = []
    monkeypatch.setattr(recall.memory_index, "index_memory_directory", lambda **kw: seen.append(kw.get("env")))
    recall_hooks._index_and_refresh({"cwd": str(project)})
    assert seen and isinstance(seen[0], dict), "the indexer ran without the corpus settings"
    assert seen[0]["RECALL_EMBED_PROFILE"] == "a-profile"
    assert seen[0].get("PATH") == os.environ.get("PATH"), "the hook's own environment was dropped"


def test_the_managed_block_is_matched_by_whole_lines_and_kept_in_place(tmp_path: Path, clients: Path) -> None:
    """CODE-002 / DOC-007: a marker quoted inside the user's TOML is not RE-call's block, and the
    user's text after the block stays after it on a rerun."""
    from recall.codex import MANAGED_BEGIN, register_project_servers

    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    target = project / ".codex" / "config.toml"
    target.write_text(f'note = "{MANAGED_BEGIN}"\n', encoding="utf-8")
    first = register_project_servers(_server(project, A="1"), project_root=project)
    assert first.registered, f"a quoted marker was taken for RE-call's block: {first.skipped_reason}"
    target.write_text(target.read_text(encoding="utf-8") + '\n[profiles.mine]\nmodel = "x"\n', encoding="utf-8")
    register_project_servers(_server(project, A="2"), project_root=project)
    text = target.read_text(encoding="utf-8")
    assert text.index("# END RE-call") < text.index("[profiles.mine]"), "the user's later table moved above the block"
    assert tomllib.loads(text)["profiles"]["mine"]["model"] == "x"


def test_a_windows_edited_config_keeps_its_shape(tmp_path: Path, clients: Path) -> None:
    """DAT-005: a BOM is accepted, CRLF line endings stay CRLF, and the original is backed up once."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    target = project / ".codex" / "config.toml"
    original = b"\xef\xbb\xbfmodel = \"mine\"\r\n"
    target.write_bytes(original)
    reg = register_project_servers(_server(project, A="1"), project_root=project)
    assert reg.registered, f"a BOM made the file unreadable: {reg.skipped_reason}"
    raw = target.read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b""), "LF lines written into a CRLF file"
    backup = project / ".codex" / "config.toml.recall-backup"
    assert backup.exists(), "no backup of the user's file"
    assert backup.read_bytes() == b'model = "mine"\r\n', "the backup is not the user's file"
    register_project_servers(_server(project, A="2"), project_root=project)
    assert backup.read_bytes() == b'model = "mine"\r\n', "the backup was overwritten"


def test_a_hook_config_left_alone_says_why(tmp_path: Path, clients: Path) -> None:
    """CODE-005, and BUG-005 by the owner's decision (2026-11-19): an unreadable config and a
    hosted one that names no project are described as such; a local config that names no project
    is bound here. CODE-001 (run 20261119T220947Z): hosted and local are decided as the hooks
    decide them (`recall_hooks.hosted_mode`), so a dsn beside an endpoint is local."""
    from recall.wizard.wiring import _hooks_left_alone

    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "recall-hook.json"
    path.write_text("{not json", encoding="utf-8")
    assert _hooks_left_alone(path, project) == f"hooks left alone: {path} cannot be read"
    path.write_text(json.dumps({"endpoint": "https://example.invalid"}), encoding="utf-8")
    assert _hooks_left_alone(path, project) == f"hooks left alone: {path} is a hosted config that names no project"
    path.write_text(json.dumps({"dsn": "x"}), encoding="utf-8")
    assert _hooks_left_alone(path, project) is None, "a local config that serves no project was not taken over"
    path.write_text(json.dumps({"dsn": "x", "endpoint": "https://example.invalid"}), encoding="utf-8")
    assert _hooks_left_alone(path, project) is None, "a config the hooks run as local was left alone as hosted"


def test_the_backup_is_ignored_when_the_config_already_was(tmp_path: Path, clients: Path) -> None:
    """DAT-005 follow-up (L5.5): the backup can hold an earlier block's credentials, so it is ignored
    by git even when an exact rule already ignored `.codex/config.toml` itself."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    _git_init(project)
    (project / ".gitignore").write_text(".codex/config.toml\n", encoding="utf-8")
    (project / ".codex").mkdir()
    (project / ".codex" / "config.toml").write_text('model = "mine"\n', encoding="utf-8")
    register_project_servers(_server(project, A="1"), project_root=project)
    assert (project / ".codex" / "config.toml.recall-backup").exists()
    ignored = subprocess.run(
        ["git", "-C", str(project), "check-ignore", "-q", "--", ".codex/config.toml.recall-backup"], capture_output=True
    )
    assert ignored.returncode == 0, "the backup of a credential file is not ignored by git"


# ----------------------------------------------------------------------------------------------
# Owner decisions 2026-11-19 on the uncertain findings of the deferred run.
# ----------------------------------------------------------------------------------------------


def test_a_codex_plugin_counts_only_once_codex_enabled_it(clients: Path) -> None:
    """DEPLOY-004: a marketplace entry only offers the plugin; its skills load once Codex records
    `[plugins."recall@re-call"] enabled = true`, so until then RE-call copies its own."""
    from recall.codex import codex_plugin_installed

    market = clients / ".agents" / "plugins" / "marketplace.json"
    market.parent.mkdir(parents=True)
    market.write_text(json.dumps({"name": "re-call", "plugins": [{"name": "recall"}]}), encoding="utf-8")
    config = clients / ".codex" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('model = "x"\n', encoding="utf-8")
    assert codex_plugin_installed() is False, "an offered but not enabled plugin counted"
    config.write_text('[plugins."recall@re-call"]\nenabled = true\n', encoding="utf-8")
    assert codex_plugin_installed() is True


def test_a_later_run_keeps_the_servers_an_earlier_one_registered(tmp_path: Path, clients: Path) -> None:
    """STAKES-007: as Claude Code's local scope does, Codex keeps an earlier run's servers."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    project.mkdir()
    first = {"a": {"command": "python", "args": ["-m", "x"], "cwd": str(project), "env": {"K": "1"}}}
    second = {"b": {"command": "python", "args": ["-m", "y"], "cwd": str(project), "env": {"K": "2"}}}
    register_project_servers(first, project_root=project)
    reg = register_project_servers(second, project_root=project)
    assert set(_codex_servers(project)) == {"a", "b"}, "the earlier run's server was dropped"
    assert reg.kept == ("a",) and reg.registered == ("b",)
    assert _codex_servers(project)["a"]["env"] == {"K": "1"}


def test_the_report_names_the_backup_and_what_trust_enables(tmp_path: Path, clients: Path) -> None:
    """SEC-004 and the architect's backup point: the report says where the original was copied and
    which servers already in the file trusting the project also enables."""
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    (project / ".codex" / "config.toml").write_text('[mcp_servers.other]\ncommand = "other"\n', encoding="utf-8")
    _project, _claude, report = _lite(tmp_path, project)
    line = next(text for client, text in report.clients if client == "codex" and text.startswith("registered"))
    assert "backed up the file as it was to" in line, line
    assert "which also enables other" in line, "trusting the project would enable a server the report hides"


def test_a_local_hook_config_naming_no_project_is_taken_over(tmp_path: Path, clients: Path) -> None:
    """BUG-005 (owner decision): Lite binds a local hook config that names no project to this one."""
    stale = clients / ".claude" / "recall-hook.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"dsn": "postgresql://x@127.0.0.1:1/x"}), encoding="utf-8")
    project, _claude, report = _lite(tmp_path)
    assert _hook_config(clients, "claude").get("project_root") == str(project.resolve()), "the config was not bound here"
    # DOC-001 (run 20261119T220947Z): it served every project until now, so the takeover is reported.
    line = next(text for client, text in report.clients if client == "claude code" and "hooks" in text)
    assert "named no project" in line, line


def test_a_codex_takeover_names_the_plugin_server_that_reads_it(tmp_path: Path, clients: Path) -> None:
    """DOC-001, architect's REVISE (run 20261119T220947Z): the Codex plugin's server reads Codex's hook
    config in every project, bound or not, so taking one over is reported with that consequence."""
    stale = clients / ".codex" / "re-call" / "recall-hook.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"dsn": "postgresql://x@127.0.0.1:1/x"}), encoding="utf-8")
    _project, _claude, report = _lite(tmp_path)
    line = next(text for client, text in report.clients if client == "codex" and text.startswith("hooks"))
    assert "named no project" in line and "reads its store in every Codex project" in line, line


# ----------------------------------------------------------------------------------------------
# CCA run 20261119T220947Z, on the owner decisions above.
# ----------------------------------------------------------------------------------------------


def _with_block(project: Path, entries: str) -> None:
    """Write `entries` between RE-call's markers in the project's `.codex/config.toml`."""
    from recall.codex import MANAGED_BEGIN, MANAGED_END

    (project / ".codex").mkdir(parents=True, exist_ok=True)
    (project / ".codex" / "config.toml").write_text(f"{MANAGED_BEGIN}\n{entries}{MANAGED_END}\n", encoding="utf-8")


def test_a_server_planted_in_the_block_is_removed(tmp_path: Path, clients: Path) -> None:
    """SEC-001: a project can arrive with a server between RE-call's markers. No run on this machine
    wrote it, so it is removed and reported, never kept as RE-call's and hidden from the trust advice."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    _with_block(project, '[mcp_servers.evil]\ncommand = "evil"\nargs = []\ncwd = "."\n\n[mcp_servers.evil.env]\n')
    reg = register_project_servers(_server(project), project_root=project)
    assert "evil" not in _codex_servers(project), "a server RE-call never wrote was kept in its block"
    assert reg.kept == () and reg.dropped == ("evil",)


def test_a_server_disabled_by_hand_in_the_block_is_not_revived(tmp_path: Path, clients: Path) -> None:
    """STAKES-001: a server RE-call wrote and the user then disabled no longer matches what RE-call
    wrote, so a later run removes it rather than writing it back enabled with its DSN."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    project.mkdir()
    first = {"a": {"command": "python", "args": ["-m", "x"], "cwd": str(project), "env": {"K": "1"}}}
    register_project_servers(first, project_root=project)
    target = project / ".codex" / "config.toml"
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace("[mcp_servers.\"a\"]\n", "[mcp_servers.\"a\"]\nenabled = false\n"), encoding="utf-8")
    assert _codex_servers(project)["a"]["enabled"] is False
    reg = register_project_servers(_server(project), project_root=project)
    after = _codex_servers(project)
    # An absent `enabled` is Codex's default, on, so only a missing entry or an explicit false will do.
    assert "a" not in after or after["a"].get("enabled") is False, "a disabled server came back enabled"
    assert reg.dropped == ("a",)


def test_a_block_entry_with_values_rendering_cannot_carry_is_removed(tmp_path: Path, clients: Path) -> None:
    """DAT-001: an entry whose env holds a table, or whose args hold a date, cannot be rendered
    again; it is removed, and the run registers its own servers instead of failing every time."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    _with_block(
        project,
        '[mcp_servers.n]\ncommand = "c"\nargs = [1979-05-27]\ncwd = "."\n\n[mcp_servers.n.env]\nZ = {z = "q"}\n',
    )
    try:
        reg = register_project_servers(_server(project), project_root=project)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"an entry rendering cannot carry broke the run: {exc}")
    assert reg.registered == ("recall-memory",), reg
    assert "n" not in _codex_servers(project)


def test_the_plugin_key_uses_the_marketplace_name(clients: Path) -> None:
    """ENV-001: the installer keeps the name an existing personal marketplace had, and Codex keys an
    enabled plugin as `<plugin>@<that name>`."""
    from recall.codex import codex_plugin_installed

    market = clients / ".agents" / "plugins" / "marketplace.json"
    market.parent.mkdir(parents=True)
    market.write_text(json.dumps({"name": "personal", "plugins": [{"name": "recall"}]}), encoding="utf-8")
    config = clients / ".codex" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[plugins."recall@personal"]\nenabled = true\n', encoding="utf-8")
    assert codex_plugin_installed() is True, "an enabled plugin in a marketplace of another name was missed"


def test_the_trust_advice_names_a_server_under_a_name_recall_also_uses(tmp_path: Path, clients: Path) -> None:
    """CODE-003: a server the user defined under RE-call's name is left alone, and trusting the
    project enables it too, so the trust advice names it."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    (project / ".codex" / "config.toml").write_text(
        '[mcp_servers.recall-memory]\ncommand = "mine"\n\n[mcp_servers.other]\ncommand = "other"\n', encoding="utf-8"
    )
    servers = {**_server(project), "second": {"command": "python", "args": [], "cwd": str(project), "env": {}}}
    reg = register_project_servers(servers, project_root=project)
    assert set(reg.others) == {SERVER_NAME, "other"}, "trusting enables a server the advice leaves out"


def test_a_recorded_server_the_user_then_defined_is_left_to_the_user(tmp_path: Path, clients: Path) -> None:
    """L5.5 of run 20261119T220947Z: a server RE-call recorded writing, whose name the user then also
    defines outside the block, is not rendered again, which would declare it twice and fail every run."""
    from recall.codex import register_project_servers

    project = tmp_path / "project"
    project.mkdir()
    register_project_servers({"lite": _server(project)["recall-memory"]}, project_root=project)
    target = project / ".codex" / "config.toml"
    target.write_text('[mcp_servers.lite]\ncommand = "mine"\n\n' + target.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        reg = register_project_servers({"wiz": _server(project)["recall-memory"]}, project_root=project)
    except ValueError as exc:
        pytest.fail(f"a name defined twice broke the run: {exc}")
    assert _codex_servers(project)["lite"]["command"] == "mine"
    assert reg.registered == ("wiz",) and reg.kept == () and reg.dropped == ()
