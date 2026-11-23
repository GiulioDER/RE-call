"""Plain `recall setup` (PostgreSQL) points prompt-time search at the `memory/` folder it made (#1426).

Invariant: with the scaffold accepted, the hook config's `memory_root` is the `memory/` folder setup
made, for Claude Code and for Codex, read back through the reader the prompt-time hook uses
(`recall_hooks.prompt_time.settings` and `find_store`). Failure mode it catches: setup scaffolded
the folder but wrote no `memory_root`, so prompt-time search read the client's own
`projects/<slug>/memory` folder, not the one the scaffolded `CLAUDE.md` tells the agent to write to.

Red proof receipt ``setup-memory-root-01``, 2026-11-19, against master `fae294875` (after #1422,
which added `memory_root` to the hook config, and #1442), `recall/setup.py` and `recall/codex.py`
at that commit: `test_prompt_time_reads_the_scaffolded_memory_folder` failed on its `store == ...`
assertion for Claude Code (`''`, no `memory_root` written). The negative control
`test_a_declined_scaffold_writes_no_memory_root` passed there, as it must. Mutation of the fix,
applied alone and restored: `memory_root=None` in the Codex call (`_run_codex_wiring`) turns the
first test red on its Codex assertion.

No database, network or model: schema preparation, seeding and memory indexing are stubbed, the
hashing embedder is chosen, and the hooks and the Codex integration are written for real under
`tmp_path`.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall import setup
from recall.seed import SeedPlan


def _run_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, scaffold: bool
) -> dict[str, Any]:
    """Run plain setup with both clients detected and every other question at its default."""
    from recall.codex import install_codex_integration
    from recall.wizard import wiring

    project = tmp_path / "project"
    project.mkdir()
    claude_home = tmp_path / "claude"
    codex_home = tmp_path / "codex"
    monkeypatch.chdir(project)
    monkeypatch.delenv("RECALL_HOOK_CONFIG_HOME", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    probe = setup.HardwareProbe(2, None, False, 10**10, False, False, False)
    monkeypatch.setattr(setup, "probe_hardware", lambda: probe)
    monkeypatch.setattr(setup, "_module_available", lambda name: name == "mcp")
    monkeypatch.setattr(setup, "_prepare_schema_for_embedder", lambda **kw: None)
    monkeypatch.setattr(setup, "_reasoning_interview", lambda *a, **kw: {})
    monkeypatch.setattr(setup, "index_memory_directory", lambda **kw: None)
    monkeypatch.setattr(setup, "plan_seed", lambda root, **kw: SeedPlan(Path(root), (), 0))
    monkeypatch.setattr(setup, "plugin_skill_sources", lambda: {})
    monkeypatch.setattr(setup, "claude_code_detected", lambda: True)
    monkeypatch.setattr(setup, "codex_code_detected", lambda: True)
    monkeypatch.setattr("recall.claude_code.refresh_stats", lambda config: 0)

    def _yes_no(_input: Any, _print: Any, question: str, *, default: bool | None = None) -> bool:
        if question.startswith("Scaffold"):
            return scaffold
        return bool(default)

    monkeypatch.setattr(setup, "_ask_yes_no", _yes_no)
    monkeypatch.setattr(
        setup,
        "_choose",
        lambda _i, _p, title, *a, **kw: setup.Choice("t", "hashing" if "mbedder" in title else "", "t"),
    )
    monkeypatch.setattr(
        wiring,
        "register_local_scope",
        lambda blocks, **kw: SimpleNamespace(
            skipped_reason="", registered=("recall",), conflicts=(),
            project_keys=(str(project),), config_path=tmp_path / "client.json",
        ),
    )
    marketplace = codex_home / "agents" / "plugins" / "marketplace.json"
    monkeypatch.setattr(
        setup,
        "install_codex_integration",
        functools.partial(
            install_codex_integration,
            hooks_path=codex_home / "hooks.json",
            marketplace_path=marketplace,
            plugin_destination=marketplace.parent / ".codex" / "plugins" / "recall",
        ),
    )
    setup.run_setup_wizard(
        dsn="postgresql://example.invalid/recall",
        env_path=project / ".env",
        input_fn=lambda _prompt="": "",
        print_fn=lambda *a, **kw: None,
    )
    return {
        "project": project,
        "claude_config": json.loads((claude_home / "recall-hook.json").read_text(encoding="utf-8")),
        "codex_config": json.loads(
            (codex_home / "re-call" / "recall-hook.json").read_text(encoding="utf-8")
        ),
    }


def test_prompt_time_reads_the_scaffolded_memory_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall_hooks import prompt_time

    run = _run_setup(tmp_path, monkeypatch, scaffold=True)
    memory = (run["project"] / "memory").resolve()
    assert (memory / "MEMORY.md").is_file(), "the scaffold did not run"

    for client in ("claude_config", "codex_config"):
        store = prompt_time.settings(run[client])["store"]
        assert store == str(memory), f"{client}: prompt-time search does not read {memory}"
        assert prompt_time.find_store(str(run["project"]), store) == memory, client


def test_a_declined_scaffold_writes_no_memory_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: with no folder made, the client's own project memory stays the store."""
    from recall_hooks import prompt_time

    run = _run_setup(tmp_path, monkeypatch, scaffold=False)

    assert not (run["project"] / "memory").exists()
    for client in ("claude_config", "codex_config"):
        assert "memory_root" not in run[client], client
        assert prompt_time.settings(run[client])["store"] == "", client
