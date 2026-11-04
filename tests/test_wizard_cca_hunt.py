"""Regression tests for the DEEP CCA hunt of the first-run code: `recall setup`, the wizard,
`recall index`, `recall scopes`, `recall lite refresh` and `recall uninstall`.

Each test names the finding it pins. Red proof for every test was taken on the remote test host
against the pre-fix snapshot of the production files (base 57e1d5990), by
`python -m cca_checks tautology verify`; the verdicts are recorded in the pull request. A test
that needs a symbol the fix added imports it after its first assertion, so the pre-fix run fails
on behaviour rather than on the import.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


# ----------------------------------------------------------------------------------------------
# SEC-001 (P1): a `git.exe` in the current directory must not run in place of git on Windows
# ----------------------------------------------------------------------------------------------


def test_the_cli_turns_off_the_current_directory_executable_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEC-001. Invariant: every `recall` invocation sets NoDefaultCurrentDirectoryInExePath=1
    before it can start a child, so Windows CreateProcess stops searching the current directory
    for bare names such as `git` and `docker`. Failure caught: `recall index .` inside a cloned
    repository that carries a `git.exe` at its root ran that file. Target: `recall.cli.main`."""
    from recall import cli

    # Set (not deleted) so monkeypatch restores it afterwards and the 1 does not leak.
    monkeypatch.setenv("NoDefaultCurrentDirectoryInExePath", "0")
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert os.environ.get("NoDefaultCurrentDirectoryInExePath") == "1"


# ----------------------------------------------------------------------------------------------
# STAKES-001 (P1): the wizard must not delete what `recall setup` wrote into the shared block
# ----------------------------------------------------------------------------------------------


def _managed_block(path: Path) -> dict[str, str]:
    from recall._env import _unquote

    values: dict[str, str] = {}
    inside = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "# recall setup begin":
            inside = True
        elif line == "# recall setup end":
            break
        elif inside and "=" in line:
            key, _sep, value = line.partition("=")
            values[key] = _unquote(value)
    return values


def test_the_wizard_keeps_the_keys_recall_setup_wrote_into_the_shared_env_block(
    tmp_path: Path,
) -> None:
    """STAKES-001. Invariant: `write_project_files` updates RECALL_DSN and RECALL_EMBEDDER in the
    managed `.env` block and keeps every other key there. Failure caught: the block was rebuilt
    from the wizard's two keys alone, deleting the provider API key and RECALL_CALIBRATION that
    `recall setup` had written, with no backup. Target: `recall.wizard.wiring.write_project_files`."""
    from recall.setup import _update_env_block
    from recall.wizard.wiring import write_project_files

    root = tmp_path / "project"
    root.mkdir()
    env = root / ".env"
    env.write_text("UNMANAGED=kept\n", encoding="utf-8")
    _update_env_block(
        env,
        {
            "RECALL_DSN": "postgresql://old:pw@127.0.0.1:5432/recall",
            "VOYAGE_API_KEY": "fake-provider-key",
            "RECALL_CALIBRATION": "/somewhere/calibration.json",
            "RECALL_RERANK": "1",
        },
    )

    write_project_files(
        project_root=root,
        dsn="postgresql://new:pw@127.0.0.1:5433/recall",
        embedder="hashing",
        memory_dir=root / "memory",
    )

    block = _managed_block(env)
    assert block.get("VOYAGE_API_KEY") == "fake-provider-key", block
    assert block.get("RECALL_CALIBRATION") == "/somewhere/calibration.json", block
    assert block.get("RECALL_RERANK") == "1", block
    assert block["RECALL_DSN"] == "postgresql://new:pw@127.0.0.1:5433/recall"
    assert block["RECALL_EMBEDDER"] == "hashing"
    assert "UNMANAGED=kept" in env.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# CODE-005, ENV-003: the server blocks pin what an inherited environment could otherwise change
# ----------------------------------------------------------------------------------------------


def test_a_calibrated_server_survives_an_inherited_legacy_index_mode() -> None:
    """CODE-005. Invariant: the generation block resolves to the generation route even when the
    client's own environment carries RECALL_INDEX_MODE=legacy, because Claude Code lays the block
    over that environment. Failure caught: production plus an inherited legacy mode raised
    RouteConfigurationError and the server had no tools. Target: `wiring._generation_block`."""
    from recall.runtime_route import resolve_runtime_route
    from recall.wizard.wiring import _generation_block

    block = _generation_block(
        SimpleNamespace(tenant="acme-docs"),  # type: ignore[arg-type]
        dsn="postgresql://u:p@127.0.0.1:5432/recall",
        embedder="hashing",
        trust="strict",
        fact_write_dsn=None,
        rationale="test",
    )
    inherited = {"RECALL_INDEX_MODE": "legacy", **block.env}
    assert resolve_runtime_route(inherited).uses_generation


def test_the_memory_server_may_index_the_memory_folder_it_serves(tmp_path: Path) -> None:
    """ENV-003. Invariant: the writable memory block sets RECALL_INDEX_ROOT to its own root, and the
    Docker service built from it does not carry that host path. Failure caught: with no index root
    the MCP index path defaulted to the server's cwd and refused the configured memory folder.
    Target: `wiring._legacy_block` and `stack.tenant_service`."""
    from recall.wizard.stack import tenant_service
    from recall.wizard.wiring import _legacy_block

    root = tmp_path / "memory"
    block = _legacy_block(
        SimpleNamespace(tenant="acme-memory", root=root),  # type: ignore[arg-type]
        dsn="postgresql://u:p@127.0.0.1:5432/recall",
        embedder="hashing",
        fact_write_dsn=None,
    )
    assert block.env.get("RECALL_INDEX_ROOT") == str(root)
    service = tenant_service(block.env, image="recall-wizard:test")
    assert "RECALL_INDEX_ROOT" not in service["environment"]  # type: ignore[operator]


# ----------------------------------------------------------------------------------------------
# DAT-003: a failed controller login must not leave a role holding the fact grants
# ----------------------------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, row: Any) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row


class _FakeConnection:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, statement: Any, params: Any = None) -> _FakeCursor:
        self._log.append(statement if isinstance(statement, str) else repr(statement))
        return _FakeCursor(None)

    def commit(self) -> None:
        return None

    def transaction(self) -> _FakeConnection:
        return self


def test_a_failed_controller_login_grants_nothing_to_the_new_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DAT-003. Invariant: when the wizard creates the fact-controller role and that role then
    cannot log in, no privilege has been granted to it and the login error is what surfaces.
    Failure caught: the grants committed with CREATE ROLE, so the rollback's DROP ROLE was refused
    (a role holding privileges cannot be dropped), hid the login error and left a LOGIN role with
    the fact-ledger grants. Target: `headless._RealServices.configure_fact_boundary`."""
    import psycopg

    from tests.test_wizard_state import _config

    from recall.wizard.headless import _RealServices

    owner = "postgresql://owner:pw@127.0.0.1:1/recall"
    controller = "postgresql://fact_writer:factpw@127.0.0.1:1/recall"
    executed: list[str] = []

    def connect(dsn: str, **_kwargs: Any) -> _FakeConnection:
        if dsn == controller:
            raise psycopg.OperationalError("password authentication failed for user fact_writer")
        return _FakeConnection(executed)

    monkeypatch.setattr(psycopg, "connect", connect)
    services = _RealServices(_config(tmp_path))

    with pytest.raises(psycopg.OperationalError, match="password authentication failed"):
        services.configure_fact_boundary(
            owner, serving_role="reader", controller_dsn=controller
        )
    assert not [s for s in executed if "GRANT" in s.upper()], executed
    assert any("DROP ROLE" in s for s in executed), executed


# ----------------------------------------------------------------------------------------------
# DEPLOY-001: an install whose servers cannot start is not reported complete
# ----------------------------------------------------------------------------------------------


def test_registering_servers_without_the_mcp_extra_is_a_reported_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-001. Invariant: when `mcp` is not importable, the wizard names the missing extra in
    the report and the install is not ok. Failure caught: the registered servers each died on
    `import mcp` while the wizard printed install complete. Target: `headless.run_headless`."""
    import importlib.util

    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _config, _CountingSpy

    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        return None if name == "mcp" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    root = tmp_path / "project"
    root.mkdir()
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")

    report = run_headless(
        _config(tmp_path, project_root=str(root)),
        services=_CountingSpy(),
        claude_config_path=client,
    )

    wiring = [f.error for f in report.failures if f.tenant == "wiring"]
    assert any('recall-rag[mcp]' in error for error in wiring), report.failures
    assert report.ok is False


# ----------------------------------------------------------------------------------------------
# DEPLOY-003: a slow first image download is not killed inside `up --wait`
# ----------------------------------------------------------------------------------------------


def test_the_stack_pulls_its_images_before_starting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-003. Invariant: `bring_up` pulls the images in a step of their own before
    `up --wait`, so the download does not run on the start timeout. Failure caught: the first pull
    ran inside a 300 s `up` and a slow link was reported as a stack that could not start.
    Target: `recall.wizard.stack.bring_up`."""
    from recall.wizard import stack

    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(stack.subprocess, "run", run)
    stack.bring_up(tmp_path / "compose.json", project_name="p", services=("db",))

    assert len(calls) == 2, calls
    assert "pull" in calls[0] and "db" in calls[0], calls
    assert "up" in calls[1], calls


# ----------------------------------------------------------------------------------------------
# SEC-002, STAKES-005: uninstall removes only what this install owns
# ----------------------------------------------------------------------------------------------


def test_uninstall_leaves_another_installers_server_with_the_same_cwd(tmp_path: Path) -> None:
    """SEC-002. Invariant: a registration is removed only when its name is one of this install's
    tenants as well as carrying the project's cwd. Failure caught: the `recall setup --lite`
    server (and any hand-written one) under the same cwd was deleted by the Docker install's
    uninstall. Target: `uninstall._registrations` through `plan_uninstall` and `execute`."""
    from recall.wizard.uninstall import execute, plan_uninstall
    from tests.test_wizard_uninstall import _docker, _install

    project = tmp_path / "work"
    project.mkdir()
    data_root = _install(tmp_path, project_root=project)
    config = tmp_path / "claude.json"
    config.write_text(
        json.dumps(
            {
                "projects": {
                    str(project): {
                        "mcpServers": {
                            "acme-docs": {"command": "python", "cwd": str(project)},
                            "recall-memory": {"command": "python", "cwd": str(project)},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    plan = plan_uninstall(data_root=data_root, claude_config_path=config, runner=_docker())
    assert {i.name for i in plan.removing("registration")} == {"acme-docs"}
    execute(plan, claude_config_path=config, runner=_docker())

    servers = json.loads(config.read_text(encoding="utf-8"))["projects"][str(project)][
        "mcpServers"
    ]
    assert set(servers) == {"recall-memory"}


def test_a_users_own_dockerfile_in_the_data_folder_is_neither_replaced_nor_removed(
    tmp_path: Path,
) -> None:
    """STAKES-005. Invariant: a `Dockerfile` the wizard did not write is never overwritten by
    `write_compose` (which refuses before writing anything) and never planned for removal by
    uninstall. Failure caught: the user's file was replaced on install and deleted on uninstall.
    Target: `stack.write_compose` and `uninstall.plan_uninstall`."""
    from recall.wizard.stack import COMPOSE_NAME, DOCKERFILE_NAME, write_compose
    from recall.wizard.uninstall import plan_uninstall
    from tests.test_wizard_uninstall import _docker, _install

    folder = tmp_path / "project-with-docker"
    folder.mkdir()
    theirs = folder / DOCKERFILE_NAME
    theirs.write_text("FROM node:20\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not written by the recall wizard"):
        write_compose(folder / COMPOSE_NAME, {"name": "p", "services": {}})
    assert theirs.read_text(encoding="utf-8") == "FROM node:20\n"
    assert not (folder / COMPOSE_NAME).exists()

    data_root = _install(tmp_path / "install")
    (data_root / DOCKERFILE_NAME).write_text("FROM node:20\n", encoding="utf-8")
    plan = plan_uninstall(data_root=data_root, runner=_docker())
    assert str(data_root / DOCKERFILE_NAME) not in {i.name for i in plan.removing("file")}


# ----------------------------------------------------------------------------------------------
# STAKES-002: the terminal wizard saves its config where uninstall looks for it
# ----------------------------------------------------------------------------------------------


def test_the_terminal_wizard_suggests_saving_beside_the_chosen_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAKES-002. Invariant: with a data folder chosen, the suggested config path is
    `<data_root>/wizard.json`, the one place `recall uninstall --data-root` reads. Failure caught:
    it was `~/.recall/wizard.json` whatever folder was chosen, so uninstall could not find the
    install's MCP registrations and a second install overwrote the first one's config.
    Target: `setup_wizard._cmd_wizard`."""
    from recall.cli_commands.setup_wizard import _cmd_wizard
    from recall.wizard import interactive

    data_root = tmp_path / "D" / "recall-data"
    suggested: list[str] = []

    class _Prompter:
        def ask(self, prompt: str, default: str | None = None) -> str:
            suggested.append(str(default))
            return str(default)

    def write_config(document: dict[str, Any], target: Path) -> Path:
        suggested.append(f"target={target}")
        raise SystemExit("stopped by the test")

    monkeypatch.setattr(interactive, "stdin_prompter", lambda: _Prompter())
    monkeypatch.setattr(
        interactive, "ask_config", lambda prompter, default_root: {"data_root": str(data_root)}
    )
    monkeypatch.setattr(interactive, "write_config", write_config)

    with pytest.raises(SystemExit, match="stopped by the test"):
        _cmd_wizard(argparse.Namespace(gui=False, config=None, headless=False))
    assert suggested[-1] == f"target={data_root / 'wizard.json'}", suggested


# ----------------------------------------------------------------------------------------------
# BUG-003: a rerun trusts the database, not the state file, about what is promoted
# ----------------------------------------------------------------------------------------------


def test_a_recorded_generation_the_database_no_longer_serves_is_rebuilt(tmp_path: Path) -> None:
    """BUG-003. Invariant: a resumed run reuses a recorded corpus only while the database still
    serves that generation. Failure caught: after a purged volume or a recreated database at the
    same address, the state file's record was reused, strict servers were wired over tenants with
    no generation and the install was reported complete. Target: `headless.run_headless`."""
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _config, _CountingSpy

    config = _config(tmp_path)
    state = tmp_path / "state.json"
    first = _CountingSpy()
    run_headless(config, services=first, state_path=state)
    assert "default-docs" in first.built

    class _DatabaseWasRecreated(_CountingSpy):
        def active_generation(self, tenant: str) -> str | None:
            return None

    second = _DatabaseWasRecreated()
    report = run_headless(config, services=second, state_path=state)
    assert "default-docs" in second.built, "a generation the database lost must be rebuilt"
    assert "default-docs" not in report.reused


# ----------------------------------------------------------------------------------------------
# BUG-001, BUG-002: user-facing refusals are messages, not tracebacks
# ----------------------------------------------------------------------------------------------


def test_recall_index_prints_an_indexing_refusal_instead_of_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-001. Invariant: a ValueError the indexer raises for the person running the command (a
    missing path, a file outside the scan) ends `recall index` with that message. Failure caught:
    it escaped `recall.cli.main` as a traceback. Target: `index_search._cmd_index`."""
    from recall import cli
    from recall.index import Indexer

    def refuse(self: Any, path: Any, glob: Any = None) -> Any:
        raise ValueError("./memroy has unsupported suffix '<none>'; pass --glob")

    monkeypatch.setattr(Indexer, "index_path", refuse)
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "store.db"
    with pytest.raises(SystemExit) as caught:
        cli.main(
            ["--dsn", f"sqlite:///{db.as_posix()}", "--embedder", "hashing", "index", "./memroy"]
        )
    assert "recall index:" in str(caught.value.code)
    assert "--glob" in str(caught.value.code)


def test_lite_refresh_reports_an_unusable_embedder_instead_of_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """BUG-002. Invariant: a typed embedder `recall lite refresh` cannot resolve ends the command
    with a message. Failure caught: the ValueError was re-raised bare and reached the console as a
    traceback. Target: `lite_cmd._cmd_refresh`."""
    from tests.test_lite_refresh import _setup_with_model_name

    from recall import cli

    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    project = tmp_path / "project"
    (project / "memory").mkdir(parents=True)
    for i in range(3):
        (project / "memory" / f"m{i}.md").write_text(f"# memo {i}\n\nbody {i}\n", encoding="utf-8")
    monkeypatch.chdir(project)
    db, _asked = _setup_with_model_name(project, monkeypatch)

    with pytest.raises(SystemExit) as caught:
        cli.main(
            ["--embedder", "no-such-embedder", "lite", "refresh", "--store", str(db),
             "--memory", str(project / "memory")]
        )
    assert "no-such-embedder" in str(caught.value.code)


# ----------------------------------------------------------------------------------------------
# CODE-004: `recall scopes` lists the store `recall search` filters
# ----------------------------------------------------------------------------------------------


def test_scopes_follows_the_runtime_route_not_the_environment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CODE-004. Invariant: with RECALL_INDEX_MODE=generation under development, `recall scopes`
    lists the generation store, as `recall search` does. Failure caught: it chose by RECALL_ENV
    alone and listed the legacy table, whose folders search then never found.
    Target: `index_search._cmd_scopes`."""
    import recall.generation_store as generation_store
    from recall.cli_commands import index_search

    monkeypatch.setenv("RECALL_ENV", "development")
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    used: list[str] = []

    class _Store:
        def __init__(self, label: str) -> None:
            self._label = label

        def __enter__(self) -> _Store:
            used.append(self._label)
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def scope_inventory(self, dimension: str) -> list[Any]:
            return []

        def scope_undeclared_count(self, dimension: str) -> int:
            return 0

    monkeypatch.setattr(generation_store, "GenerationStore", lambda *a, **k: _Store("generation"))
    monkeypatch.setattr(index_search, "PgVectorStore", lambda *a, **k: _Store("legacy"))
    index_search._cmd_scopes(
        argparse.Namespace(
            dsn="postgresql://u:p@127.0.0.1:1/recall", tenant="t", table="chunks",
            dimension="folder", _runtime_route=None,
        )
    )
    assert used == ["generation"]


# ----------------------------------------------------------------------------------------------
# PERF-002: the most common commands do not load the MCP lifecycle module at startup
# ----------------------------------------------------------------------------------------------


def test_loading_the_index_and_search_commands_does_not_load_the_mcp_lifecycle_module() -> None:
    """PERF-002. Invariant: importing `recall.cli_commands.index_search` (what `recall index` and
    `recall search` load) does not import `recall_mcp.lifecycle`, which only forget and withdraw
    use and which drags in pydantic. Failure caught: a module-level import charged every index and
    search about half a second. Target: `index_search` module imports."""
    probe = (
        "import sys, recall.cli_commands.index_search; "
        "print('recall_mcp.lifecycle' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    ).stdout.strip()
    assert out == "False"
