"""`recall setup`'s OpenAI-compatible embedder with only an OpenAI key (#1397).

Setup offers "openai compatible cloud" when only `OPENAI_API_KEY` was entered, and since #1411 it
copies that key into the server it registers. The `openai` spellings in
`recall.embeddings.resolve_embedder` read only `OPENROUTER_API_KEY`, so neither setup's own
indexing nor that server could build the embedder.

No test here calls a provider. The `openai` SDK is replaced in `sys.modules` by a module whose
client records its constructor arguments and returns one fixed vector, so the real
`OpenAICompatEmbedder` runs up to the transport and no further.

Red proof, 2026-11-19, against master dac217681's `recall/embeddings.py` (after #1411):

- `test_an_openai_spelling_with_only_an_openai_key_reaches_api_openai_com` and
  `test_a_server_setup_registers_with_only_an_openai_key_can_embed` each failed at their
  `pytest.fail` with "OpenAICompatEmbedder needs OPENROUTER_API_KEY", because `resolve_embedder`
  passed `api_key=source.get("OPENROUTER_API_KEY")`. Deleting the `OPENAI_DIRECT_BASE_URL`
  branch from the fixed `resolve_embedder` fails both the same way.
- `test_an_openrouter_key_still_wins_when_both_are_present` passes before the fix: it guards the
  precedence the fix keeps. Dropping `and not source.get("OPENROUTER_API_KEY")` from that branch
  failed it at `assert 'https://api.openai.com/v1' == 'https://openrouter.ai/api/v1'`.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from recall.embeddings import resolve_embedder
from recall.setup import HardwareProbe, embedder_choices, run_setup_wizard

OPENAI = "https://api.openai.com/v1"
OPENROUTER = "https://openrouter.ai/api/v1"


@pytest.fixture
def fake_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a stand-in `openai` module; return what its client was constructed with."""
    seen: dict[str, Any] = {}

    class _Embeddings:
        def create(self, **request: Any) -> Any:
            seen["model"] = request["model"]
            item = types.SimpleNamespace(index=0, embedding=[0.0] * 1536)
            return types.SimpleNamespace(data=[item] * len(request["input"]))

    class _OpenAI:
        def __init__(self, *, api_key: str, base_url: str, max_retries: int) -> None:
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            self.embeddings = _Embeddings()

    module = types.ModuleType("openai")
    module.OpenAI = _OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    # A key in the shell running the suite must not decide the route under test.
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "VOYAGE_API_KEY", "RECALL_EMBED_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    return seen


def test_an_openai_spelling_with_only_an_openai_key_reaches_api_openai_com(fake_openai_sdk):
    """Invariant: the key setup offered the embedder for is the key the embedder uses.

    Failure mode: only `OPENAI_API_KEY` is set, and construction raises for a missing
    `OPENROUTER_API_KEY`, so setup's own indexing and the registered server both fail.
    """
    try:
        resolve_embedder("openai:text-embedding-3-small", env={"OPENAI_API_KEY": "sk-test"})
    except RuntimeError as exc:
        pytest.fail(f"an OpenAI key alone did not build the openai embedder: {exc}")

    assert fake_openai_sdk["base_url"] == OPENAI
    assert fake_openai_sdk["api_key"] == "sk-test"
    assert fake_openai_sdk["model"] == "text-embedding-3-small"


def test_an_openrouter_key_still_wins_when_both_are_present(fake_openai_sdk):
    """Invariant: a deployment that already embeds `openai` through OpenRouter keeps doing so.

    Failure mode: adding a second key silently moves an existing corpus's embedder to another
    endpoint, which for the bare `openai` spelling is another model id.
    """
    resolve_embedder(
        "openai:text-embedding-3-small",
        env={"OPENAI_API_KEY": "sk-test", "OPENROUTER_API_KEY": "router-key"},
    )

    assert fake_openai_sdk["base_url"] == OPENROUTER
    assert fake_openai_sdk["api_key"] == "router-key"


def test_setup_indexing_and_the_server_pick_one_endpoint_when_the_shell_has_an_openrouter_key(
    fake_openai_sdk, monkeypatch
):
    """Invariant: one RECALL_EMBEDDER resolves to one endpoint whether or not `env` is explicit.

    Failure mode (CCA BUG-001, run 20261119T182325Z): setup indexes with `env=` holding only the
    keys typed into it, while the registered server inherits OPENROUTER_API_KEY from the shell.
    The direct-OpenAI branch looked for that key in `env` only, so setup embedded the corpus at
    api.openai.com and the server embedded queries at OpenRouter.

    Red proof, 2026-11-19, against `recall/embeddings.py` at 2e546bec1 (the condition read
    `not source.get("OPENROUTER_API_KEY")`): failed at the first assertion,
    `'https://api.openai.com/v1' == 'https://openrouter.ai/api/v1'`.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    resolve_embedder("openai:text-embedding-3-small", env={"OPENAI_API_KEY": "sk-test"})
    setup_route = fake_openai_sdk["base_url"]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resolve_embedder("openai:text-embedding-3-small")
    server_route = fake_openai_sdk["base_url"]

    assert setup_route == server_route
    assert server_route == OPENROUTER


def _probe() -> HardwareProbe:
    return HardwareProbe(
        cpu_count=8,
        gpu=None,
        free_bytes=100 * 1024**3,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=False,
    )


def _register_through_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    keys: tuple[str, str, str],
    embedder_label: str,
) -> tuple[dict[str, Any], str]:
    """Run the real interview and the real registration; return the server block and the output.

    Only what would touch the machine is replaced: the hardware probe, the schema check against a
    database that does not exist, the hooks installer, and the seed plan. `register_mcp_server`
    and `register_local_scope` run for real against a `.claude.json` under `tmp_path`.
    """
    config_dir = tmp_path / "cfg"
    project = tmp_path / "proj"
    config_dir.mkdir()
    project.mkdir()
    (config_dir / ".claude.json").write_text(json.dumps({"projects": {}}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.chdir(project)
    monkeypatch.setattr("recall.setup.probe_hardware", _probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr("recall.setup._schema_prepare_state", lambda *a, **k: ("compatible", None))
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr("recall.setup.codex_code_detected", lambda: False)
    monkeypatch.setattr("recall.setup.plugin_skill_sources", lambda: {})
    monkeypatch.setattr("recall.setup.install_hooks", lambda **kw: None)
    from recall.seed import SeedPlan

    monkeypatch.setattr(
        "recall.setup.plan_seed", lambda root, **kw: SeedPlan(root=Path(root), files=(), total_bytes=0)
    )

    entered = dict(zip(("VOYAGE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"), keys))
    menu = embedder_choices(
        _probe(), security_required=False, cloud_keys={k: v for k, v in entered.items() if v}
    )
    position = next(i for i, c in enumerate(menu, start=1) if c.label == embedder_label)
    answers = iter(
        [
            "n",  # data security not required, so the key prompts appear
            *keys,
            str(position),
            "1",  # reranker: none
            "n",  # reasoning arm declined
            "n",  # scaffold declined
            "y",  # register with Claude Code
            "n",  # write-time hook
            "n",  # prompt-time hook
        ]
    )
    printed: list[str] = []
    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        claude_md_path=tmp_path / "CLAUDE.md",
        memory_dir=tmp_path / "memory",
        input_fn=lambda _p="": next(answers),
        print_fn=lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
    )
    # Every scripted answer consumed: one left over would be answering a prompt that is gone, and
    # would answer the next prompt added at the end (CCA CODE-012). Red proof, 2026-11-19: with
    # the removed calibrate answer put back, this failed at `assert 'n' is None`.
    assert next(answers, None) is None, "the setup script has answers for prompts setup no longer asks"
    document = json.loads((config_dir / ".claude.json").read_text(encoding="utf-8"))
    servers = [
        server
        for entry in document["projects"].values()
        for server in entry.get("mcpServers", {}).values()
    ]
    assert len(servers) == 1, f"expected one registered server, found {len(servers)}"
    return servers[0], "\n".join(printed)


def test_a_server_setup_registers_with_only_an_openai_key_can_embed(
    tmp_path, monkeypatch, fake_openai_sdk
):
    """Invariant: the server setup registers can build the embedder setup offered, from its env.

    Failure mode: setup offers the openai embedder for an OpenAI key and #1411 copies that key into
    the registration, but the server's embedder asks for OPENROUTER_API_KEY and raises inside a
    process whose log the client does not show.
    """
    server, output = _register_through_setup(
        tmp_path, monkeypatch, keys=("", "sk-openai-only", ""), embedder_label="openai compatible cloud"
    )
    server_env = server["env"]

    assert server_env["RECALL_EMBEDDER"] == "openai:text-embedding-3-small"
    assert "OPENAI_API_KEY" in server_env, "the registered server cannot reach the embedder's key"
    assert server_env["OPENAI_API_KEY"] == "sk-openai-only"
    # The key rides in the environment block only: never the command line, never the report.
    assert "sk-openai-only" not in json.dumps([server["command"], server["args"]])
    assert "sk-openai-only" not in output
    # The server's own resolution, from exactly that environment, reaches the provider with it.
    try:
        resolve_embedder(server_env["RECALL_EMBEDDER"], env=server_env)
    except RuntimeError as exc:
        pytest.fail(f"the registered server cannot build its embedder: {exc}")
    assert fake_openai_sdk["api_key"] == "sk-openai-only"
    assert fake_openai_sdk["base_url"] == OPENAI

