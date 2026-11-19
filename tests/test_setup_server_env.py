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

from recall.rerank import PINNED_RERANKER_SHA256
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
    codex_home: Path | None = None,
) -> tuple[dict[str, str], str]:
    """The registered Claude Code server's environment, and everything setup printed.

    With `codex_home`, setup also wires Codex, through the real installer redirected into that
    directory, so the caller can launch the Codex server from the file it wrote."""
    from recall.wizard import wiring

    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
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
    monkeypatch.setattr("recall.setup.codex_code_detected", lambda: codex_home is not None)
    if codex_home is not None:
        import functools

        from recall.codex import install_codex_integration

        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        marketplace = codex_home / "agents" / "plugins" / "marketplace.json"
        monkeypatch.setattr(
            "recall.setup.install_codex_integration",
            functools.partial(
                install_codex_integration,
                hooks_path=codex_home / "hooks.json",
                marketplace_path=marketplace,
                plugin_destination=marketplace.parent / ".codex" / "plugins" / "recall",
            ),
        )
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
        if "Register the recall MCP server with Codex" in last:
            return "y" if codex_home is not None else "n"
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


@pytest.mark.parametrize("reranker", ["ms marco reranker", "bge reranker"])
def test_a_setup_reranker_runs_one_search_at_a_time_in_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reranker: str
) -> None:
    """Invariant: a server setup registers with a local reranker admits one search at a time.

    Failure mode (CCA PERF-002, owner decision 2026-11-19): RECALL_RERANK=1 resolves the legacy
    profile, which admits four concurrent searches, each a cross-encoder predict on every core,
    in every stdio server a client opens, with no latency budget.

    Red proof, 2026-11-19, against `recall.setup._server_settings` at 9d7b61420 (no concurrency
    written): both cases failed at `assert 4 == 1`.
    """
    from recall.profiles import resolve_retrieval_profile
    from recall_mcp.factories import resolve_reranker

    env, _ = _registered_env(
        tmp_path, monkeypatch,
        security="n", embedder="fastembed", reranker=reranker, reasoning=False,
        total_ram_bytes=16 * 2**30,  # bge is offered only where it fits
    )
    assert resolve_reranker(env) is not None, env
    assert resolve_retrieval_profile(env).max_concurrency == 1


def test_setup_warns_when_an_exported_profile_would_stop_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: when the shell setup runs in exports a RECALL_RETRIEVAL_PROFILE that conflicts
    with the reranker it registers, setup says so before it registers.

    Failure mode (CCA DEPLOY-003, owner decision 2026-11-19: warn at setup): the client lays the
    registration over its own environment, and RECALL_RETRIEVAL_PROFILE=fast with RECALL_RERANK=1
    makes the server refuse to start, which a client shows only as a server with no tools.

    Red proof, 2026-11-19, against `recall.setup.run_setup_wizard` at 4b070f4c0 (no check): failed
    at the first assertion, nothing printed about the profile.
    """
    from recall.profiles import resolve_retrieval_profile

    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    env, output = _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker="ms marco reranker", reasoning=False,
    )
    assert "RECALL_RETRIEVAL_PROFILE=fast" in output and "refuse to start" in output, output
    # The warning is true: the server the client would start does refuse.
    with pytest.raises(ValueError, match="conflicts"):
        resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "fast", **env})


_PINNED_QUALITY = {
    "RECALL_RERANK_PATH": "/models/ms-marco",
    "RECALL_RERANK_SHA256": PINNED_RERANKER_SHA256,
}


@pytest.mark.parametrize(
    ("exported", "extra", "reranker", "expected"),
    [
        ("quality", {}, "ms marco reranker", "refuse to start"),
        ("quality", _PINNED_QUALITY, "ms marco reranker", "one search at a time"),
        ("code", {}, "ms marco reranker", "refuse to start"),
        ("code", {"RECALL_ACCEPT_REMOTE_MODEL_CODE": "1"}, "ms marco reranker", None),
        ("fast", {}, "none", None),
        ("", {}, "ms marco reranker", None),
    ],
    ids=["quality-unpinned", "quality-pinned", "code-no-consent", "code-consent", "no-reranker",
         "no-profile"],
)
def test_the_profile_warning_names_only_what_will_happen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exported: str,
    extra: dict[str, str],
    reranker: str,
    expected: str | None,
) -> None:
    """What setup says matches what the server's own startup check does with that profile.

    A refusal names it (quality without its pinned artifact, code without the coreb-code
    consent); a pinned quality server that starts gets the one-search-at-a-time note; a profile
    that starts at one search already, no reranker, and no profile get nothing.

    Red proof, 2026-11-19: against 4b070f4c0 (no check) `quality-unpinned`, `quality-pinned` and
    `code-no-consent` failed at `assert expected in output`. Against the first version of the
    check, which used `recall.profiles.resolve_retrieval_profile` instead of the server's startup
    check, `quality-unpinned` (it printed the note) and `code-no-consent` (it printed nothing)
    failed there. The three no-message cases passed in both, as they must.
    """
    for name in ("RECALL_RERANK_PATH", "RECALL_RERANK_SHA256", "RECALL_ACCEPT_REMOTE_MODEL_CODE",
                 "RECALL_ROUTING_MODE", "RECALL_SEARCH_CONCURRENCY", "RECALL_SEARCH_QUEUE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", exported)
    for name, value in extra.items():
        monkeypatch.setenv(name, value)
    _, output = _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker=reranker, reasoning=False,
    )
    if expected is None:
        assert "RECALL_RETRIEVAL_PROFILE" not in output, output
    else:
        assert expected in output, output


@pytest.mark.parametrize(
    ("name", "value"), [("RECALL_SEARCH_CONCURRENCY", "lots"), ("RECALL_ROUTING_MODE", "bogus")]
)
def test_the_profile_check_never_stops_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """A malformed value in the shell must not make setup itself fail: setup finishes and
    registers the server whatever the warning says. Red proof, 2026-11-19: against the first
    version of the check (no guard on the second resolve) the RECALL_SEARCH_CONCURRENCY case
    failed with `ValueError: RECALL_SEARCH_CONCURRENCY must be an integer` out of setup; with the
    first `except ValueError` mutated to `except KeyError`, the RECALL_ROUTING_MODE case failed
    with `ValueError: routing mode must be shadow or active`."""
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "quality")
    for key, pinned in _PINNED_QUALITY.items():
        monkeypatch.setenv(key, pinned)
    monkeypatch.setenv(name, value)
    env, _ = _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker="ms marco reranker", reasoning=False,
    )
    assert env["RECALL_SEARCH_CONCURRENCY"] == "1"


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
    # No reranker, no cap: the default concurrency is left alone.
    assert "RECALL_SEARCH_CONCURRENCY" not in env
    # Off is left unset rather than written as "0": an explicit RECALL_RERANK=0 makes the server
    # refuse to start beside a RECALL_RETRIEVAL_PROFILE=quality the client exports.
    from recall.profiles import resolve_retrieval_profile

    assert resolve_retrieval_profile({**env, "RECALL_RETRIEVAL_PROFILE": "quality"}).name == "quality"


def test_a_declined_reasoning_arm_overrides_one_the_client_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: a reasoning arm the user declined stays off in the server, whatever the client
    inherited.

    Failure mode (CCA ENV-001, run 20261119T182325Z): `_server_settings` dropped every "0", so a
    declined arm wrote nothing into the registration. The client lays the registration over its
    own environment, so an inherited RECALL_REASONING_EXPANSION=1 (with its model and key) made
    paid expansion calls the user had declined.

    Red proof, 2026-11-19, against `recall.setup._server_settings` at 2e546bec1: failed at
    `assert <OpenAIExpansionProvider ...> is None`.
    """
    from recall.reasoning_expansion import resolve_expansion_provider

    env, _ = _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker="none", reasoning=False,
    )
    inherited = {
        "RECALL_REASONING_EXPANSION": "1",
        "RECALL_REASONING_EXPANSION_MODEL": "inherited/model",
        "RECALL_REASONING_EXPANSION_API_KEY": "inherited-key-not-real",
    }
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: object()))

    # What a merging client starts the server with: its own environment, then the block.
    assert resolve_expansion_provider({**inherited, **env}) is None


@pytest.mark.parametrize(
    "case",
    [
        {"security": "y", "embedder": "fastembed", "reranker": "ms marco reranker", "reasoning": True},
        {
            "security": "n", "embedder": "voyage cloud", "reranker": "none", "reasoning": False,
            "keys": {
                "VOYAGE_API_KEY": "test-voyage-key-not-real",
                "OPENAI_API_KEY": "test-openai-key-not-real",
                "OPENROUTER_API_KEY": "test-openrouter-key-not-real",
            },
        },
    ],
    ids=["local-options-on", "every-provider-key"],
)
def test_every_answer_setup_writes_also_reaches_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: dict
) -> None:
    """Invariant: each non-empty key setup writes to `.env`, apart from the CLI's own DSN and
    embedder (handed to the server under its own names), is in the server's registration.

    Failure mode (CCA CODE-001, run 20261119T182325Z): `SERVER_SETTING_KEYS` is a hand list kept
    apart from the prompts that write `.env`, so a key added to the interview reaches `.env`, which
    the server never reads, and not the server: issues 1393 and 1394 again.

    Red proof, 2026-11-19, by mutation of `recall.setup.SERVER_SETTING_KEYS` with
    "OPENROUTER_API_KEY" removed: `every-provider-key` failed at the assertion below, naming
    {'OPENROUTER_API_KEY'}.
    Against `_server_settings` at 2e546bec1 the same case also failed, naming
    {'RECALL_REASONING_EXPANSION'}: the declined arm's "0" was dropped (CCA ENV-001).
    """
    env, _ = _registered_env(tmp_path, monkeypatch, **case)
    written = {}
    for raw in (tmp_path / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = raw.strip().partition("=")
        if sep and not key.startswith("#") and value.strip():
            written[key.strip()] = value.strip()
    cli_only = {"RECALL_DSN", "RECALL_EMBEDDER"}
    # RECALL_RERANK=0 is left out on purpose (see the negative control above).
    expected = {k for k, v in written.items() if k not in cli_only and not (k == "RECALL_RERANK" and v == "0")}

    missing = expected - set(env)
    assert not missing, f"written to .env but not handed to the server: {missing}"


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


def _codex_server_env(monkeypatch: pytest.MonkeyPatch, config: Path) -> tuple[int, dict[str, str]]:
    """Launch the Codex MCP entry point on `config` and return what the server would start with."""
    import os

    from recall_mcp import codex_server

    monkeypatch.setenv("RECALL_CODEX_CONFIG", str(config))
    captured: dict[str, str] = {}
    monkeypatch.setattr("recall_mcp.server.main", lambda: captured.update(os.environ))
    return codex_server.main(), captured


def test_the_reranker_and_reasoning_arm_reach_the_codex_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same choices through the Codex registration, which `recall setup` writes to
    `recall-hook.json` and `recall_mcp.codex_server` launches from.

    Red proof, 2026-11-19, on the test host against `dac217681`: failed on the `spec` assertion,
    `resolve_reranker` returning None, because the Codex launcher set only DSN, tenant, embedder,
    table and transport. The one-search-at-a-time cap (CCA PERF-002) is asserted here too: red
    against 9d7b61420 at `assert 4 == 1`."""
    import os

    from recall.profiles import resolve_retrieval_profile
    from recall.rerank import DEFAULT_RERANKER_MODEL
    from recall.reasoning_expansion import resolve_expansion_provider
    from recall_mcp.factories import resolve_reranker

    for key in ("RECALL_RERANK", "RECALL_REASONING_EXPANSION", "RECALL_SEARCH_CONCURRENCY"):
        monkeypatch.delenv(key, raising=False)
    codex_home = tmp_path / "codex"
    _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker="ms marco reranker", reasoning=True,
        codex_home=codex_home,
    )

    status, env = _codex_server_env(monkeypatch, codex_home / "re-call" / "recall-hook.json")

    assert status == 0
    spec = resolve_reranker(env)
    assert spec is not None and spec[0] == DEFAULT_RERANKER_MODEL
    assert resolve_retrieval_profile(env).max_concurrency == 1
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: object()))
    assert resolve_expansion_provider(env) is not None
    assert "RECALL_RERANK" not in os.environ, "the launcher must restore the environment it changed"


def test_a_cloud_embedder_key_reaches_the_codex_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red proof, 2026-11-19, on the test host against `dac217681`: failed on
    `KeyError: 'VOYAGE_API_KEY'`."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    codex_home = tmp_path / "codex"
    _registered_env(
        tmp_path, monkeypatch,
        security="n", embedder="voyage cloud", reranker="none", reasoning=False,
        keys={"VOYAGE_API_KEY": "test-voyage-key-not-real"}, codex_home=codex_home,
    )

    status, env = _codex_server_env(monkeypatch, codex_home / "re-call" / "recall-hook.json")

    assert status == 0
    assert env["VOYAGE_API_KEY"] == "test-voyage-key-not-real"


def test_a_codex_server_env_cannot_redirect_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The five keys the launcher owns win over the same keys in `server_env`.

    Red proof, 2026-11-19: with `server_env` applied after the five keys in
    `recall_mcp.codex_server.main`, it failed on the DSN assertion."""
    import json

    config = tmp_path / "recall-hook.json"
    config.write_text(json.dumps({
        "dsn": "postgresql://installed/recall",
        "tenant": "installed",
        "server_env": {"RECALL_SERVING_DSN": "postgresql://elsewhere/recall", "RECALL_TENANT": "other"},
    }), encoding="utf-8")

    status, env = _codex_server_env(monkeypatch, config)

    assert status == 0
    assert env["RECALL_SERVING_DSN"] == "postgresql://installed/recall"
    assert env["RECALL_TENANT"] == "installed"


@pytest.mark.parametrize("server_env", [["RECALL_RERANK=1"], {"RECALL_RERANK": 1}])
def test_a_malformed_codex_server_env_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_env: object
) -> None:
    """Refused rather than half applied. Red proof, 2026-11-19, on the test host against
    `dac217681`: both cases failed on `assert 0 == 2`, the launcher starting the server."""
    import json

    config = tmp_path / "recall-hook.json"
    config.write_text(
        json.dumps({"dsn": "postgresql://installed/recall", "server_env": server_env}),
        encoding="utf-8",
    )

    status, env = _codex_server_env(monkeypatch, config)

    assert status == 2
    assert env == {}, "the server must not start"


def test_the_codex_server_gets_the_trust_mode_the_claude_server_gets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 1427. Setup's fresh corpus is uncalibrated, which a strict server refuses with
    `INDEX_NOT_READY`, so setup registers the Claude Code server under `development`. The Codex
    server it wires got no trust mode and so ran strict, refusing the corpus Claude Code serves.

    Red proof, 2026-11-19, on the test host against `2e546bec1`: failed on the `strict` assertion,
    the Codex launcher's environment resolving to the strict default."""
    from recall.trust_policy import TrustPolicy

    monkeypatch.delenv("RECALL_TRUST_MODE", raising=False)
    codex_home = tmp_path / "codex"
    claude_env, _ = _registered_env(
        tmp_path, monkeypatch,
        security="y", embedder="fastembed", reranker="none", reasoning=False,
        codex_home=codex_home,
    )

    status, codex_env = _codex_server_env(monkeypatch, codex_home / "re-call" / "recall-hook.json")

    assert status == 0
    assert TrustPolicy.from_env(claude_env).strict is False
    assert TrustPolicy.from_env(codex_env).strict is False, codex_env.get("RECALL_TRUST_MODE")
