from __future__ import annotations

import json
import io

from recall.codex import install_codex_integration


def test_codex_plugin_bundle_has_manifest_hooks_and_shared_skills() -> None:
    from recall.codex import codex_plugin_source

    root = codex_plugin_source()
    assert root is not None
    manifest = json.loads((root / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "recall"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert json.loads((root / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["recall"]
    assert (root / "hooks" / "hooks.json").is_file()
    assert (root / "skills" / "check-memory-before-acting" / "SKILL.md").is_file()


def test_codex_install_is_idempotent_and_preserves_user_configuration(tmp_path, monkeypatch) -> None:
    codex_home = tmp_path / "codex"
    marketplace = tmp_path / "agents" / "plugins" / "marketplace.json"
    plugin_destination = marketplace.parent / ".codex" / "plugins" / "recall"
    hooks = codex_home / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text(
        json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "user-hook"}]}]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    for _ in range(2):
        install_codex_integration(
            dsn="postgresql://example/recall",
            tenant="memory",
            embedder="fastembed:BAAI/bge-large-en-v1.5",
            hooks_path=hooks,
            marketplace_path=marketplace,
            plugin_destination=plugin_destination,
            python_executable="python-test",
            print_fn=lambda *args, **kwargs: None,
        )

    document = json.loads(marketplace.read_text(encoding="utf-8"))
    assert [item["name"] for item in document["plugins"]].count("recall") == 1
    entry = next(item for item in document["plugins"] if item["name"] == "recall")
    assert entry["policy"]["installation"] == "INSTALLED_BY_DEFAULT"
    assert entry["source"]["path"] == "./.codex/plugins/recall"
    assert (plugin_destination / ".codex-plugin" / "plugin.json").is_file()

    hook_document = json.loads(hooks.read_text(encoding="utf-8"))
    assert any(
        group["hooks"][0]["command"] == "user-hook"
        for group in hook_document["hooks"]["UserPromptSubmit"]
    )
    for event in ("SessionStart", "PreCompact", "UserPromptSubmit", "PreToolUse", "SessionEnd"):
        groups = hook_document["hooks"][event]
        assert sum("recall_hooks.codex" in json.dumps(group) for group in groups) == 1

    config = json.loads((codex_home / "re-call" / "recall-hook.json").read_text(encoding="utf-8"))
    assert config == {
        "dsn": "postgresql://example/recall",
        "tenant": "memory",
        "embedder": "fastembed:BAAI/bge-large-en-v1.5",
        "table": "chunks",
        "write_time": {"enabled": True},
        "prompt_time": {"enabled": True},
    }


def test_codex_hook_adapter_normalises_prompt_and_uses_codex_config(tmp_path, monkeypatch) -> None:
    import recall_hooks
    from recall_hooks import codex

    captured = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        captured["payload"] = json.load(codex.sys.stdin)
        return 0

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(recall_hooks, "main", fake_main)
    monkeypatch.setattr(codex.sys, "stdin", codex.io.StringIO(json.dumps({"text": "remember this"})))

    assert codex.main(["user-prompt-submit"]) == 0
    assert captured["argv"] is None
    assert captured["payload"]["prompt"] == "remember this"
    assert captured["payload"]["cwd"]
    assert "RECALL_HOOK_CONFIG_HOME" not in codex.os.environ


def test_codex_mcp_launcher_refuses_missing_configuration(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("RECALL_CODEX_CONFIG", str(tmp_path / "missing.json"))
    from recall_mcp.codex_server import main

    assert main() == 2
    assert "cannot read" in capsys.readouterr().err


def test_codex_mcp_launcher_uses_installed_config_over_inherited_environment(
    tmp_path, monkeypatch
) -> None:
    from recall_mcp import codex_server

    config = tmp_path / "recall-hook.json"
    config.write_text(
        json.dumps(
            {
                "dsn": "postgresql://installed/recall",
                "tenant": "installed-tenant",
                "embedder": "installed-embedder",
                "table": "installed_chunks",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RECALL_CODEX_CONFIG", str(config))
    monkeypatch.setenv("RECALL_SERVING_DSN", "postgresql://stale/recall")
    monkeypatch.setenv("RECALL_TENANT", "stale-tenant")
    captured = {}
    monkeypatch.setattr(
        "recall_mcp.server.main",
        lambda: captured.update(
            {key: codex_server.os.environ[key] for key in (
                "RECALL_SERVING_DSN", "RECALL_TENANT", "RECALL_EMBEDDER", "RECALL_TABLE"
            )}
        ),
    )

    assert codex_server.main() == 0
    assert captured == {
        "RECALL_SERVING_DSN": "postgresql://installed/recall",
        "RECALL_TENANT": "installed-tenant",
        "RECALL_EMBEDDER": "installed-embedder",
        "RECALL_TABLE": "installed_chunks",
    }


def test_setup_installs_codex_integration_by_default_when_detected(tmp_path, monkeypatch) -> None:
    from recall.setup import HardwareProbe, run_setup_wizard
    from recall.seed import SeedPlan

    captured = {}
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=100 * 1024**3,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=False,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr("recall.setup._prepare_schema_for_embedder", lambda **kwargs: None)
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: False)
    monkeypatch.setattr("recall.setup.codex_code_detected", lambda: True)
    monkeypatch.setattr(
        "recall.setup.plan_seed",
        lambda root, **kwargs: SeedPlan(root=root, files=(), total_bytes=0),
    )
    monkeypatch.setattr(
        "recall.setup.install_codex_integration",
        lambda **kwargs: captured.update(kwargs),
    )
    answers = iter(["y", "2", "1", "1", "n", "n", "y", "n"])
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *args, **kwargs: print(*args, **kwargs, file=output),
    )

    assert captured["dsn"] == "postgresql://example/recall"
    assert captured["embedder"] == "fastembed"
    assert "Codex is wired up" in output.getvalue()


def test_claude_bundle_keeps_claude_hook_entrypoint() -> None:
    from recall.codex import codex_plugin_source

    codex_root = codex_plugin_source()
    assert codex_root is not None
    claude_hooks = json.loads(
        (codex_root.parent / "plugin" / "hooks" / "hooks.json").read_text(encoding="utf-8")
    )
    assert all(
        hook["command"] == "recall-hooks"
        for groups in claude_hooks["hooks"].values()
        for group in groups
        for hook in group["hooks"]
    )
