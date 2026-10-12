"""Setup consumer regressions from CCA PR900.

Original head449aa0a1: all behavioral nodes first failed their intended
assertions before source repairs. Settings use persisted registration; strict
refusal uses trusted_search. CLI controls retain PostgreSQL credential refusal.
Configured followups are executed through actual CLI consumers with only the
doctor I/O and HTTP serving replaced. No real client configuration is touched.
"""

from __future__ import annotations

import importlib
import json
import shlex
from types import SimpleNamespace

import pytest

from recall.backends import lite_route_refusal
from recall.lite import LiteStore
from recall.lite.setup import lite_dsn, run_lite_setup
from recall.runtime_route import resolve_runtime_route
from recall.trust import trusted_search
from recall.trust_policy import TrustFailureCode, TrustRefusal
from recall_mcp.settings import Settings
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos


@pytest.fixture(autouse=True)
def isolated_lite_environment(monkeypatch):
    """Record absent keys too, so CLI dotenv loading cannot leak into later tests."""
    for key in (
        "OPENROUTER_API_KEY",
        "VOYAGE_API_KEY",
        "RECALL_EMBED_DIMENSIONS",
        "RECALL_MULTIMODAL_ENABLED",
        "RECALL_DSN",
        "RECALL_SERVING_DSN",
        "RECALL_MIGRATION_DSN",
        "RECALL_EMBED_PROFILE",
        "RECALL_MODEL_CACHE",
        "RECALL_MODEL_SHA256",
        "RECALL_ENV",
        "RECALL_INDEX_MODE",
        "RECALL_TRANSPORT",
        "RECALL_TRUST_MODE",
        "RECALL_ALLOW_INSECURE_DSN",
        "RECALL_EMBEDDER",
        "RECALL_INDEX_ROOT",
    ):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setenv("RECALL_RUN_PAID_TESTS", "0")


def registered_case(root, monkeypatch, *, memos=48, custom=False, ignored=False):
    root.mkdir(parents=True, exist_ok=True)
    notes = root / "notes"
    _write_memos(notes, memos)
    config = root / "client.json"
    config.write_text(json.dumps({"projects": {}}))
    if ignored:
        (root / ".gitignore").write_text(".recall/\n")
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(root / "cache.sqlite"))
    embedder = ContentWordEmbedder()
    report = run_lite_setup(
        project_root=root,
        memory=notes,
        embedder=embedder,
        embedder_name=embedder.name,
        config_path=config,
        store_path=root / "memories.db" if custom else None,
    )
    document = json.loads(config.read_text())
    entries = [
        env
        for project in document["projects"].values()
        for env in project.get("mcpServers", {}).values()
    ]
    assert len(entries) == 1, "invalid registration positive control"
    assert report.chunks == memos, "invalid indexed-corpus positive control"
    return report, entries[0]["env"], embedder


def test_registered_store_survives_parent_override(tmp_path, monkeypatch):
    report, env, _ = registered_case(tmp_path / "project", monkeypatch)
    expected = lite_dsn(report.store)
    assert Settings.from_env(env).serving_dsn == expected, "invalid clean-parent control"
    parent = {"RECALL_SERVING_DSN": "sqlite:///other-project.db"}
    actual = Settings.from_env({**parent, **env}).serving_dsn
    assert actual == expected, "inherited serving DSN displaced the configured store"


def test_local_launch_survives_parent_configuration(tmp_path, monkeypatch):
    report, env, _ = registered_case(tmp_path / "project", monkeypatch)
    control = Settings.from_env(env)
    assert not resolve_runtime_route(env).uses_generation, "invalid clean route control"
    assert control.transport == "stdio" and control.trust_policy.strict, (
        "invalid clean launch control"
    )
    parent = {
        "RECALL_ENV": "production",
        "RECALL_INDEX_MODE": "generation",
        "RECALL_TRANSPORT": "streamable-http",
        "RECALL_TRUST_MODE": "development",
    }
    merged = {**parent, **env}
    route = resolve_runtime_route(merged)
    settings = Settings.from_env(merged)
    failures = {
        "generation": route.uses_generation,
        "route_refusal": lite_route_refusal(lite_dsn(report.store), route) is not None,
        "transport": settings.transport != "stdio",
        "trust": not settings.trust_policy.strict,
    }
    assert not any(failures.values()), f"parent configuration broke local launch: {failures}"


def test_custom_memory_file_receives_privacy_advice(tmp_path, monkeypatch):
    control, _, _ = registered_case(tmp_path / "default", monkeypatch, ignored=True)
    assert not any(".gitignore" in note for note in control.notes), "invalid default-ignore control"
    custom, _, _ = registered_case(tmp_path / "custom", monkeypatch, custom=True, ignored=True)
    assert custom.store.name == "memories.db" and custom.store.exists(), (
        "invalid custom-store control"
    )
    assert any("memories.db" in note and ".gitignore" in note for note in custom.notes), (
        "custom memory database received no applicable privacy advice"
    )


def test_registered_policy_refuses_missing_calibration(tmp_path, monkeypatch):
    report, env, embedder = registered_case(tmp_path / "project", monkeypatch, memos=6)
    assert report.calibration == "missing", "invalid missing-calibration positive control"
    control = Settings.from_env(env)
    with LiteStore(report.store, dim=embedder.dim) as store:
        try:
            trusted_search(
                store,
                embedder,
                "who services the valve",
                k=5,
                policy=control.trust_policy,
                env=control.env,
            )
        except TrustRefusal as refusal:
            assert refusal.code is TrustFailureCode.CALIBRATION_MISSING, "invalid strict control"
        else:
            raise AssertionError("invalid strict positive control did not refuse")
        configured = Settings.from_env({"RECALL_TRUST_MODE": "development", **env})
        try:
            trusted_search(
                store,
                embedder,
                "who services the valve",
                k=5,
                policy=configured.trust_policy,
                env=configured.env,
            )
        except TrustRefusal as refusal:
            assert refusal.code is TrustFailureCode.CALIBRATION_MISSING
            return
        raise AssertionError("registered strict policy allowed a missing-calibration answer")


def test_lite_dispatch_ignores_unused_postgres_credentials(tmp_path, monkeypatch):
    import recall.cli as cli
    import recall.cli_commands.setup_wizard as setup

    for key in (
        "OPENROUTER_API_KEY",
        "VOYAGE_API_KEY",
        "RECALL_EMBED_DIMENSIONS",
        "RECALL_MULTIMODAL_ENABLED",
        "RECALL_DSN",
        "RECALL_SERVING_DSN",
        "RECALL_MIGRATION_DSN",
        "RECALL_EMBED_PROFILE",
        "RECALL_ENV",
        "RECALL_INDEX_MODE",
        "RECALL_ALLOW_INSECURE_DSN",
    ):
        monkeypatch.delenv(key, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    _write_memos(project / "notes", 6)
    monkeypatch.chdir(project)
    cli = importlib.reload(cli)
    monkeypatch.setattr(setup, "_make_embedder", lambda name: ContentWordEmbedder())
    tail = ["setup", "--lite", "--memory", str(project / "notes"), "--no-register"]
    safe = "postgresql://recall:recall@localhost:5432/recall"
    unsafe = "postgresql://recall:recall@db.example.invalid/recall"
    cli.main(["--dsn", safe, *tail])
    assert (project / ".recall/memory.db").exists(), "invalid healthy Lite dispatch control"
    with pytest.raises(SystemExit):
        cli.main(["--dsn", unsafe, "setup"])
    errors = []
    for prefix in (["--dsn", unsafe], ["--dsn", safe, "--migration-dsn", unsafe]):
        try:
            cli.main([*prefix, *tail])
        except (SystemExit, PermissionError) as exc:
            errors.append(type(exc).__name__)
    assert not errors, f"SQLite setup rejected unused PostgreSQL credentials: {errors}"


def test_printed_followups_select_the_created_store(tmp_path, monkeypatch, capsys):
    """DOC001/ENV002/DEPLOY001: original setup printed no executable configured guide.

    Raw baseline repro separately demonstrated that the advertised bare commands
    target PostgreSQL/no database. The corrected contract prints configured
    commands; exercise those actual commands rather than requiring silent dotenv
    edits. Explicit doctor and dashboard controls preserve existing behavior.
    """
    import recall.cli as cli
    import recall.cli_commands.setup_wizard as setup
    import recall.doctor as doctor
    import recall.dashboard.server as dashboard

    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    monkeypatch.chdir(root)
    cli = importlib.reload(cli)
    monkeypatch.setattr(setup, "_make_embedder", lambda name: ContentWordEmbedder())
    cli.main(["setup", "--lite", "--memory", str(root / "notes"), "--no-register"])
    printed = capsys.readouterr().out
    expected = lite_dsn(root / ".recall/memory.db")
    assert (root / ".recall/memory.db").exists(), "invalid setup control"
    seen = []

    def checks(**kwargs):
        seen.append(kwargs["dsn"])
        return SimpleNamespace(render=lambda: "probe doctor", exit_code=lambda: 0)

    monkeypatch.setattr(doctor, "run_checks", checks)
    cli.main(["--dsn", expected, "doctor"])
    assert seen[-1] == expected, "invalid explicit doctor control"
    dbs = []

    class Server:
        def serve_forever(self):
            pass

        def server_close(self):
            pass

    def serve(app):
        dbs.append(app.db)
        return Server()

    monkeypatch.setattr(dashboard, "serve", serve)
    selected = root / "dsn.txt"
    selected.write_text(expected)
    cli.main(["dashboard", "--root", str(root), "--db-dsn-file", str(selected), "--no-browser"])
    assert dbs[-1] is not None and dbs[-1].dsn == expected, "invalid explicit dashboard control"
    commands = [
        shlex.split(line.strip())
        for line in printed.splitlines()
        if line.strip().startswith("env ")
    ]
    assert len(commands) == 2, "setup provided no executable configured followups"
    executed = []
    for command in commands:
        assert command.pop(0) == "env"
        while command and command[0] != "recall":
            key, separator, value = command.pop(0).partition("=")
            assert separator and key.startswith("RECALL_"), "invalid printed environment"
            monkeypatch.setenv(key, value)
        assert command and command.pop(0) == "recall"
        if "dashboard" in command:
            command.append("--no-browser")
        cli = importlib.reload(cli)
        cli.main(command)
        executed.append(command[0])
    assert set(executed) == {"doctor", "dashboard"}
    assert seen[-1] == expected and dbs[-1] is not None and dbs[-1].dsn == expected
    assert not (root / ".env").exists(), "setup unexpectedly rewrote project configuration"


@pytest.mark.parametrize("filename, valid_pattern", [
    ("memories[1].db", r"memories\[1\].db*"),
    ("#memories.db", r"\#memories.db*"),
    ("!memories.db", r"\!memories.db*"),
])
def test_custom_store_ignore_advice_covers_literal_git_paths(tmp_path, monkeypatch, filename, valid_pattern):
    """SEC001 treeb15d768f: raw patterns wrongly silenced custom-store advice.

    Git check-ignore is the real consumer. Validate a known literal pattern,
    reject the raw rule, then require the advised rule to cover DB and sidecars.
    """
    import subprocess

    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True)
    _write_memos(root / "notes", 6)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    ignore = root / ".gitignore"
    paths = [filename, filename + "-wal", filename + "-shm"]

    def ignored(relative):
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--no-index", "-q", "--", relative],
            check=False, capture_output=True, text=True,
        )
        assert result.returncode in (0, 1), "invalid Git consumer control"
        return result.returncode == 0

    ignore.write_text(valid_pattern + "\n")
    assert all(ignored(path) for path in paths), "invalid literal-pattern positive control"
    ignore.write_text(filename + "*\n")
    assert not ignored(filename), "raw metacharacter pattern unexpectedly covers the database"
    model = ContentWordEmbedder()
    report = run_lite_setup(
        project_root=root, memory=root / "notes", embedder=model, embedder_name=model.name,
        store_path=root / filename, register=False,
    )
    assert report.store.exists() and report.chunks == 6, "invalid actual-store control"
    advice = [note for note in report.notes if ".gitignore" in note]
    assert advice, "wrong Git pattern silenced privacy advice for the actual database"
    target = advice[0].split("`")[1]
    ignore.write_text(target + "\n")
    assert all(ignored(path) for path in paths), "advised pattern misses actual database or sidecars"
    repeated = run_lite_setup(
        project_root=root, memory=root / "notes", embedder=model, embedder_name=model.name,
        store_path=root / filename, register=False,
    )
    assert not any(".gitignore" in note for note in repeated.notes), "valid advice was not recognized"