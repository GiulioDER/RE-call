"""PR430 through PR439 regressions, proved against the pre-fix source on the remote test host.

Baseline: 8ef55caf0. Each defect assertion must fail before its repair. The
audit receipt records the unchanged node IDs, baseline failures and green run.
All credential strings here are harmless placeholders.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import recall.claude_code as integration
import recall.setup as setup
import recall_hooks
from recall.desktop.models import RuntimeMode, RuntimeProfile
from recall.desktop.runtime import LocalDatabaseRuntime
from recall.seed import SeedPlan, seed_corpus


@pytest.mark.parametrize("event", ["SessionStart", "PreCompact", "SessionEnd"])
def test_lifecycle_hooks_ignore_project_packages(tmp_path: Path, event: str) -> None:
    """SEC-001: real subprocess must not import the cwd's harmless marker package."""
    project = tmp_path / "untrusted"
    package = project / "recall_hooks"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("print('PROJECT_PACKAGE_EXECUTED')\n", encoding="utf-8")
    (package / "__main__.py").write_text("", encoding="utf-8")
    handler = integration.hook_entries(sys.executable)[event][0]["hooks"][0]
    env = os.environ.copy()
    env.pop("PYTHONSAFEPATH", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["CLAUDE_CONFIG_DIR"] = str(tmp_path / "empty-config")
    if env.get("MUTANT_UNDER_TEST") == "stats":
        env.pop("MUTANT_UNDER_TEST")
    result = subprocess.run(
        [handler["command"], *handler["args"]], cwd=project, env=env,
        input="{}", text=True, capture_output=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PROJECT_PACKAGE_EXECUTED" not in result.stdout


@pytest.mark.parametrize("event", ["pre_compact", "session_end"])
@pytest.mark.parametrize("boundary", ["foreign", "missing-root", "relative-root", "memory-symlink"])
def test_local_lifecycle_rejects_foreign_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: str, boundary: str,
) -> None:
    """SEC-002: event cwd and resolved memory source must belong to the configured project."""
    project = tmp_path / "owned"
    foreign = tmp_path / "foreign"
    project.mkdir()
    (foreign / "memory").mkdir(parents=True)
    config: dict[str, Any] = {"dsn": "postgresql://example.invalid/recall", "project_root": str(project)}
    cwd = foreign
    if boundary == "missing-root":
        config.pop("project_root")
    elif boundary == "relative-root":
        config["project_root"] = "owned"
        monkeypatch.chdir(tmp_path)
        (project / "memory").mkdir()
        cwd = project
    elif boundary == "memory-symlink":
        try:
            (project / "memory").symlink_to(foreign / "memory", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")
        cwd = project
    indexed: list[Path] = []
    stopped: list[str] = []
    monkeypatch.setattr(recall_hooks, "load_config", lambda: config)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)
    monkeypatch.setattr("recall.memory_index.index_memory_directory", lambda **kw: indexed.append(kw["memory_dir"]))
    monkeypatch.setattr("recall_hooks.relay.stop", lambda session_id: stopped.append(session_id))
    monkeypatch.setattr("recall_hooks.mcp_cleanup.close_own_mcp_transports", lambda *a, **kw: None)
    assert getattr(recall_hooks, event)({"cwd": str(cwd), "session_id": "test-session"}) == 0
    assert indexed == [], "foreign or unbound memory reached the configured corpus"
    if event == "session_end":
        assert stopped == ["test-session"], "index confinement must not skip session cleanup"


def test_registration_redacts_keyword_password(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-003: successful public registration must not print a keyword DSN password."""
    config = tmp_path / "client.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(integration, "client_config_path", lambda: config)
    marker = "PLACEHOLDER_REGISTRATION_PASSWORD"
    output = io.StringIO()
    result = integration.register_mcp_server(
        dsn=f"host=localhost dbname=recall user=tester password={marker}",
        project_root=tmp_path, embedder="hashing",
        print_fn=lambda *a, **kw: print(*a, **kw, file=output),
    )
    assert result == "registered"
    assert "Registered MCP server" in output.getvalue()
    assert marker not in output.getvalue(), "successful registration printed a password"


def test_real_driver_percent_token_is_redacted(tmp_path: Path) -> None:
    """SEC-004: the real driver's token-only parse error must be scrubbed by seeding."""
    marker = "PLACEHOLDER%4GSEED"
    note = tmp_path / "README.md"
    note.write_text("A harmless test fact.\n", encoding="utf-8")
    lines: list[str] = []
    result = seed_corpus(
        dsn=f"postgresql://tester:{marker}@localhost:1/recall", embedder_name="hashing",
        plan=SeedPlan(root=tmp_path, files=(note,), total_bytes=note.stat().st_size),
        print_fn=lambda *a, **kw: lines.append(" ".join(map(str, a))),
    )
    rendered = "\n".join(lines)
    assert result == 0
    assert "percent-encoded" in rendered
    assert marker not in rendered, "driver password token reached the seeding error"


def test_seed_keeps_whole_dsn_redaction_for_short_password(tmp_path: Path) -> None:
    """SEC-004 control: token scrubbing must retain the former whole-DSN protection.

    The first repair, using scrub_dsn_secrets alone, fails this unchanged
    assertion on the remote test host because the shared scrubber deliberately skips short tokens.
    """
    dsn = "not-a-dsn://tester:x@localhost/recall"
    note = tmp_path / "README.md"
    note.write_text("A harmless fact.\n", encoding="utf-8")
    lines: list[str] = []
    result = seed_corpus(
        dsn=dsn, embedder_name="hashing",
        plan=SeedPlan(tmp_path, (note,), note.stat().st_size),
        print_fn=lambda *a, **kw: lines.append(" ".join(map(str, a))),
    )
    rendered = "\n".join(lines)
    assert result == 0
    assert "Could not seed" in rendered
    assert dsn not in rendered, "whole-DSN echo lost its existing short-password protection"
    assert ":x@" not in rendered


def _setup_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registration: str = "registered") -> tuple[dict[str, Any], str]:
    """Run the actual setup consumer with local installation files and no model or DB I/O."""
    monkeypatch.chdir(tmp_path)
    note = tmp_path / "README.md"
    note.write_text("A harmless project fact.\n", encoding="utf-8")
    probe = setup.HardwareProbe(2, None, False, 10**10, False, False, False)
    monkeypatch.setattr(setup, "probe_hardware", lambda: probe)
    monkeypatch.setattr(setup, "_module_available", lambda name: False)
    monkeypatch.setattr(setup, "_prepare_schema_for_embedder", lambda **kw: None)
    monkeypatch.setattr(setup, "_reasoning_interview", lambda *a, **kw: {})
    monkeypatch.setattr(setup, "claude_code_detected", lambda: True)
    monkeypatch.setattr(setup, "codex_code_detected", lambda: False)
    monkeypatch.setattr(setup, "plugin_skill_sources", lambda: {})
    monkeypatch.setattr(setup, "plan_seed", lambda root: SeedPlan(root, (note,), note.stat().st_size))
    monkeypatch.setattr(setup, "_ask_yes_no", lambda _i, _p, question, **kw: question.startswith(("Is data security", "Seed the corpus", "Register the recall")))
    monkeypatch.setattr(setup, "_choose", lambda _i, _p, question, *a, **kw: setup.Choice("test", "hashing" if "embedder" in question else "", "test"))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(setup, "seed_corpus", lambda **kw: seen.setdefault("seed", kw))
    monkeypatch.setattr(setup, "register_mcp_server", lambda **kw: registration)
    monkeypatch.setattr(integration, "hook_config_path", lambda: tmp_path / "hook.json")
    monkeypatch.setattr(integration, "refresh_stats", lambda config: 0)
    install = integration.install_hooks

    def install_locally(**kwargs: Any) -> None:
        seen["hooks"] = kwargs
        install(**kwargs, path=tmp_path / "settings.json")

    monkeypatch.setattr(setup, "install_hooks", install_locally)
    output = io.StringIO()
    setup.run_setup_wizard(
        dsn="postgresql://example.invalid/recall", table="selected_chunks",
        env_path=tmp_path / ".env", input_fn=lambda prompt="": "",
        print_fn=lambda *a, **kw: print(*a, **kw, file=output),
    )
    return seen, output.getvalue()


@pytest.mark.parametrize("destination", ["seed", "hooks"])
def test_setup_preserves_selected_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str) -> None:
    """BUG-001: PR431's table selector must reach both new integration paths."""
    seen, _ = _setup_install(tmp_path, monkeypatch)
    if destination == "seed":
        table = seen["seed"].get("table", "chunks")
    else:
        table = json.loads((tmp_path / "hook.json").read_text(encoding="utf-8"))["table"]
    assert table == "selected_chunks", f"{destination} silently selected the default table"


@pytest.mark.parametrize("status", ["skipped", "already-registered"])
def test_setup_honors_registration_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    """BUG-003: a refused registration must not install hooks or announce complete wiring."""
    seen, output = _setup_install(tmp_path, monkeypatch, status)
    assert "hooks" not in seen, "registration refusal still rewrote global hooks"
    assert "Claude Code is wired up" not in output


def test_local_profile_overrides_inherited_serving_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-002: use effective server Settings, not only the legacy DSN overlay."""
    from recall_mcp.settings import Settings

    monkeypatch.setenv("RECALL_SERVING_DSN", "postgresql://foreign.invalid/other")
    monkeypatch.setenv("RECALL_EMBEDDER", "hashing")
    selected = "postgresql://selected.invalid/recall"
    runtime = LocalDatabaseRuntime(RuntimeProfile(mode=RuntimeMode.LOCAL_DATABASE, dsn=selected))
    gateway = runtime._gateway_for("selected")
    env = dict(os.environ)
    env.update(gateway.env or {})  # type: ignore[attr-defined]
    effective = Settings.from_env(env)
    assert effective.serving_dsn == selected, "server used an inherited database after probing the selected one"
    assert effective.tenant == "selected"


class _ProbeConnection:
    def __init__(self, schema_create: bool = True) -> None:
        self.schema_create = schema_create

    def __enter__(self) -> "_ProbeConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def execute(self, query: str) -> SimpleNamespace:
        value: Any = None
        if "SELECT version()" in query:
            value = "PostgreSQL test"
        elif "to_regtype" in query:
            value = "vector"
        elif "has_database_privilege" in query:
            value = False
        elif "has_schema_privilege" in query:
            value = self.schema_create
        elif "SELECT current_user" in query:
            value = "serving_role"
        return SimpleNamespace(fetchone=lambda: (value,))


def test_install_preflight_checks_schema_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    """DAT-002: schema CREATE permits table installation without database CREATE."""
    from recall.wizard.database import probe_database

    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: _ProbeConnection())
    report = probe_database("postgresql://example.invalid/recall")
    assert report.usable, "schema creation permission was mistaken for database CREATE"


def test_serving_runtime_does_not_require_ddl(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-002: the serving check must reach health without requiring DDL rights."""
    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: _ProbeConnection(False))
    runtime = LocalDatabaseRuntime(RuntimeProfile(mode=RuntimeMode.LOCAL_DATABASE, dsn="postgresql://example.invalid/recall"))
    health: list[bool] = []
    monkeypatch.setattr(runtime, "health", lambda: health.append(True) or {"status": "ready"})
    error = None
    try:
        runtime.start()
    except Exception as exc:
        error = exc
    assert error is None, f"serving account was refused before health: {error}"
    assert health == [True]


def test_desktop_extra_includes_local_server_pool() -> None:
    """DEPLOY-001: the desktop-only dependency declaration must cover local server pooling."""
    import tomllib
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    names = {Requirement(value).name for value in project["project"]["optional-dependencies"]["desktop"]}
    assert "psycopg-pool" in names, "desktop installation omits its local server's pool dependency"
