"""`recall setup` hands the server it registers the choices the server needs (issues 1393, 1394).

Invariant: what the interview chose for reranking, for the reasoning arm and for a cloud
embedder's API key reaches the environment of the MCP server setup registers, read back through
the readers that server uses. Failure mode it catches: setup wrote those answers only to `./.env`,
which only the CLI loads (`recall/cli.py`), while `recall.claude_code.server_env` gave the server
DSN, tenant, trust mode, table and embedder alone, so reranking and the reasoning arm never ran in
the server and a cloud embedder started without its key.

Red proof receipt ``setup-server-env-01``, 2026-11-19, against master `63b7c3ea2`: the reranker
and reasoning test failed on its first assertion (`resolve_reranker` returned None for the
registered environment) and the API key test failed on `KeyError: 'VOYAGE_API_KEY'`; the bge
test's red run is recorded in its own docstring. The negative control passed there, as it must.
Production symbols targeted: `recall.setup.run_setup_wizard` (`_run_claude_wiring`),
`recall.claude_code.server_env` and `recall.setup.reranker_choices`.

The interview is answered by the prompt just printed, so a question added later does not shift
every answer. Registration is captured before it writes any client configuration, and no network
call is made: the reasoning probe is stubbed and the `openai` SDK is replaced by a stand-in.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.setup import HardwareProbe, run_setup_wizard


def _registered_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    security: str,
    embedder: str,
    reranker: str,
    reasoning: bool,
    keys: dict[str, str] | None = None,
    total_ram_bytes: int | None = None,
) -> tuple[dict[str, str], str]:
    from recall.wizard import wiring

    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=100 * 2**30,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
        total_ram_bytes=total_ram_bytes,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr("recall.setup._prepare_schema_for_embedder", lambda **kw: None)
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr("recall.setup.codex_code_detected", lambda: False)
    monkeypatch.setattr("recall.setup.install_hooks", lambda **kw: None)
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    blocks: list = []

    def _capture(new_blocks, **kwargs):
        blocks.extend(new_blocks)
        return SimpleNamespace(
            skipped_reason="", registered=("recall",), conflicts=(),
            project_keys=(str(tmp_path),), config_path=tmp_path / "client.json",
        )

    monkeypatch.setattr(wiring, "register_local_scope", _capture)
    printed: list[str] = []
    menu: dict[str, str] = {}

    def _print(*args, **kwargs) -> None:
        text = " ".join(str(a) for a in args)
        printed.append(text)
        match = re.match(r"\s+(\d+)\. ([^:]+):", text)
        if match:
            menu[match.group(2).strip()] = match.group(1)

    def _answer(_prompt: str = "") -> str:
        last = printed[-1] if printed else ""
        if last.endswith("Select a number: ") or last.startswith("Select a number"):
            wanted = reranker if "ms marco reranker" in menu else embedder
            number = menu[wanted]
            menu.clear()
            return number
        if "Is data security necessary" in last:
            return security
        if "Enable the optional reasoning arm" in last:
            return "y" if reasoning else "n"
        if "Register the recall MCP server with Claude Code" in last:
            return "y"
        if last.rstrip().endswith("]"):
            return "n"
        name = last.strip().rstrip(":")
        if name in (keys or {}):
            return (keys or {})[name]
        if last.startswith("Model id"):
            return "qwen-test"
        return ""

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=_answer,
        print_fn=_print,
    )
    assert len(blocks) == 1, "setup registered no server"
    rendered = wiring.mcp_config(tuple(blocks), project_root=tmp_path)
    return dict(rendered["mcpServers"]["recall"]["env"]), "\n".join(printed)


def test_the_reranker_and_reasoning_arm_reach_the_registered_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.reasoning_expansion import resolve_expansion_provider
    from recall.rerank import DEFAULT_RERANKER_MODEL
    from recall_mcp.factories import resolve_reranker

    env, _ = _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker="ms marco reranker", reasoning=True,
    )

    spec = resolve_reranker(env)
    assert spec is not None and spec[0] == DEFAULT_RERANKER_MODEL, env
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: object()))
    assert resolve_expansion_provider(env) is not None, env


def test_declined_choices_reach_the_registered_server_as_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: a declined reranker and reasoning arm must not switch either on."""
    from recall.reasoning_expansion import resolve_expansion_provider
    from recall_mcp.factories import resolve_reranker

    env, _ = _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker="none", reasoning=False,
    )

    assert resolve_reranker(env) is None
    assert resolve_expansion_provider(env) is None


def test_a_cloud_embedder_key_reaches_the_registered_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, output = _registered_env(
        tmp_path, monkeypatch,
        security="n", embedder="voyage cloud", reranker="none", reasoning=False,
        keys={"VOYAGE_API_KEY": "test-voyage-key-not-real"},
    )

    assert env["RECALL_EMBEDDER"].startswith("voyage"), env["RECALL_EMBEDDER"]
    assert env["VOYAGE_API_KEY"] == "test-voyage-key-not-real"
    # Keys nobody entered are not written as empty values.
    assert "OPENAI_API_KEY" not in env and "OPENROUTER_API_KEY" not in env
    assert "test-voyage-key-not-real" not in output, "setup must not print the key"


def test_the_bge_reranker_reaches_the_registered_server_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server refuses a reranker model without a pinned revision
    (`recall_mcp.factories.resolve_reranker`), so setup must write the pin with the model: the
    revision the LoCoMo arms were measured with (`scripts/run_locomo_arms.sh`).

    Red proof, 2026-11-19, on the test host: against `63b7c3ea2` it failed on the `spec ==`
    assertion (`None`, nothing reached the server); with the fix but the revision dropped from
    the bge choice in `recall.setup.reranker_choices` it failed on `pytest.fail`, the server
    refusing the unpinned model. Restored, it passes."""
    from recall_mcp.factories import resolve_reranker

    env, _ = _registered_env(
        tmp_path, monkeypatch,
        security="n", embedder="fastembed", reranker="bge reranker", reasoning=False,
        total_ram_bytes=16 * 2**30,
    )

    try:
        spec = resolve_reranker(env)
    except ValueError as exc:
        pytest.fail(f"the registered server refuses the reranker setup chose: {exc}")
    assert spec == ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"), env
