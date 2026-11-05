"""Regression tests for the second DEEP CCA hunt of the first-run code: the stack, the uninstaller,
the machine probe and the MCP wiring (`recall.wizard.stack`, `recall.wizard.uninstall`,
`recall.wizard.probe` and `recall.wizard.wiring`).

Each test names the finding it pins. Red proof for every test is taken on the remote test host
against the pre-fix snapshot of the production files, and the verdicts are recorded in the pull
request. A test that needs a symbol the fix added imports it after its first assertion, so the
pre-fix run fails on behaviour rather than on the import.

Nothing here reaches Docker, a database, the network or the real home directory: subprocess calls
are stubbed, and every client config and project folder lives under `tmp_path`.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

_SWITCH = "NoDefaultCurrentDirectoryInExePath"


# ----------------------------------------------------------------------------------------------
# SEC-006: the bare-name docker, wsl, systeminfo and nvidia-smi launches must not search the
# current directory, whichever entry point reached them
# ----------------------------------------------------------------------------------------------


def _recording_run(seen: list[str | None], result: Any = None) -> Any:
    def run(command: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(os.environ.get(_SWITCH))
        if result is not None:
            return result
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run


def test_bring_up_turns_off_the_current_directory_search_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-006. Invariant: `stack.bring_up` sets NoDefaultCurrentDirectoryInExePath=1 in its own
    environment before it starts `docker`. Failure caught: the desktop console scripts
    (recall-install) reach `bring_up` without `recall.cli.main`, so a `docker.exe` in the folder the
    installer was launched from ran in place of Docker. Target: `recall.wizard.stack.bring_up`."""
    from recall.wizard import stack

    # Set (not deleted) so monkeypatch restores it afterwards and the 1 does not leak.
    monkeypatch.setenv(_SWITCH, "0")
    seen: list[str | None] = []
    monkeypatch.setattr(stack.subprocess, "run", _recording_run(seen))

    stack.bring_up(tmp_path / "docker-compose.recall.yml", project_name="p")

    assert seen and all(value == "1" for value in seen), seen


def test_the_uninstall_runner_turns_off_the_current_directory_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEC-006. Invariant: `uninstall._run` sets NoDefaultCurrentDirectoryInExePath=1 before it
    starts a command. Failure caught: recall-uninstall reaches `docker ps`, `docker compose down`
    and `docker volume rm` by bare name without `recall.cli.main`. Target:
    `recall.wizard.uninstall._run`."""
    from recall.wizard import uninstall

    monkeypatch.setenv(_SWITCH, "0")
    seen: list[str | None] = []
    monkeypatch.setattr(uninstall.subprocess, "run", _recording_run(seen))

    uninstall._run(["docker", "ps"])

    assert seen == ["1"]


def test_the_probe_turns_off_the_current_directory_search_for_run_and_which(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEC-006. Invariant: `probe._run` and the `shutil.which` lookups in the probes run with
    NoDefaultCurrentDirectoryInExePath=1. Failure caught: recall-desktop's probe page reaches
    `docker info`, `wsl`, `systeminfo` and `nvidia-smi` without `recall.cli.main`, and
    `shutil.which` on Windows also checks the current directory first. Target:
    `recall.wizard.probe._run` and `recall.wizard.probe.probe_docker`."""
    from recall.wizard import probe

    monkeypatch.setenv(_SWITCH, "0")
    seen_which: list[str | None] = []

    def which(name: str, *args: Any, **kwargs: Any) -> None:
        seen_which.append(os.environ.get(_SWITCH))
        return None

    monkeypatch.setattr(probe.shutil, "which", which)
    assert probe.probe_docker() == (False, False)
    assert seen_which == ["1"], "shutil.which looked up docker with the current-directory search on"

    monkeypatch.setenv(_SWITCH, "0")
    seen_run: list[str | None] = []
    monkeypatch.setattr(probe.subprocess, "run", _recording_run(seen_run))
    probe._run(["systeminfo"], 1.0)
    assert seen_run == ["1"]


# ----------------------------------------------------------------------------------------------
# NUM-005 and NUM-006: a failed compose call must report the cause, which compose prints LAST
# ----------------------------------------------------------------------------------------------


def test_a_failed_bring_up_reports_the_tail_of_compose_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NUM-005. Invariant: the RuntimeError from `stack.bring_up` keeps the END of compose's
    output. Failure caught: the first 400 characters were kept, which are per-resource progress
    lines, and the cause (`dependency failed to start`, a port bind error) was cut off. Target:
    `recall.wizard.stack.bring_up`."""
    from recall.wizard import stack

    def run(command: Any, *args: Any, **kwargs: Any) -> Any:
        if "pull" in command:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise subprocess.CalledProcessError(
            1, command, output="", stderr=" Container x Creating\n" * 60 + "CAUSE: unhealthy"
        )

    monkeypatch.setattr(stack.subprocess, "run", run)

    with pytest.raises(RuntimeError) as raised:
        stack.bring_up(tmp_path / "docker-compose.recall.yml", project_name="p")
    assert "CAUSE: unhealthy" in str(raised.value)


def test_a_failed_compose_down_reports_the_tail_of_its_output(tmp_path: Path) -> None:
    """NUM-006. Invariant: a failed `docker compose down` is reported with the END of its combined
    output. Failure caught: `output[:200]` kept the `Container x Stopping` progress lines and
    dropped the actual failure from the uninstall report. Target: `recall.wizard.uninstall.execute`."""
    from recall.wizard.uninstall import execute, plan_uninstall
    from tests.test_wizard_uninstall import _install

    data_root = _install(tmp_path)

    def runner(command: Any) -> tuple[int, str]:
        if "ps" in command:
            return 0, "recall-acme-1234abcd-db-1"
        if "down" in command:
            return 1, " Container x Stopping\n" * 40 + "CAUSE: daemon error"
        return 0, ""

    kwargs: dict[str, Any] = {
        "claude_config_path": tmp_path / "claude.json", "runner": runner,
    }
    plan = plan_uninstall(data_root=data_root, profile_path=tmp_path / "none.json", **kwargs)
    report = execute(plan, **kwargs)

    reasons = [reason for _item, reason in report.failed]
    assert reasons and all("CAUSE: daemon error" in reason for reason in reasons), reasons


# ----------------------------------------------------------------------------------------------
# DEPLOY-009: uninstall removes the recipe the services build from, and the image they built
# ----------------------------------------------------------------------------------------------


def _versioned_install(tmp_path: Path) -> tuple[Path, Path]:
    from recall.wizard.stack import COMPOSE_NAME, GENERATED_DOCKERFILE_HEADER
    from tests.test_wizard_uninstall import _install

    data_root = _install(tmp_path)
    compose = data_root / COMPOSE_NAME
    document = json.loads(compose.read_text(encoding="utf-8"))
    for name in ("recall-acme-docs", "recall-acme-code"):
        document["services"][name] = {
            "image": "recall-wizard:0.15.0",
            "build": {"context": ".", "dockerfile": "Dockerfile.recall-0123456789ab"},
        }
    compose.write_text(json.dumps(document), encoding="utf-8")
    recipe = data_root / "Dockerfile.recall-0123456789ab"
    recipe.write_text(GENERATED_DOCKERFILE_HEADER + "\nFROM python:3.13-slim\n", encoding="utf-8")
    return data_root, recipe


def test_uninstall_removes_the_versioned_recipe_and_names_the_built_image(tmp_path: Path) -> None:
    """DEPLOY-009. Invariant: `plan_uninstall` plans the generated recipe each service names in
    `build.dockerfile`, `execute` removes it, and the `recall-wizard:*` image is named in the plan
    but never removed (its tag is shared by every install of that version). Failure caught: only
    the fixed `Dockerfile` was planned, so after any completed install the live
    `Dockerfile.recall-<digest>` stayed in data_root and nothing told the user about the
    LibreOffice-sized image. Target: `recall.wizard.uninstall.plan_uninstall` and `execute`."""
    from recall.wizard.uninstall import execute, plan_uninstall
    from tests.test_wizard_uninstall import _docker

    data_root, recipe = _versioned_install(tmp_path)
    runner = _docker()
    kwargs: dict[str, Any] = {"claude_config_path": tmp_path / "claude.json", "runner": runner}

    plan = plan_uninstall(data_root=data_root, profile_path=tmp_path / "none.json", **kwargs)

    assert str(recipe) in {item.name for item in plan.removing("file")}, plan.render()
    assert not plan.removing("image")
    assert [item.name for item in plan.keeping() if item.kind == "image"] == [
        "recall-wizard:0.15.0"
    ]

    report = execute(plan, **kwargs)
    assert not recipe.exists()
    assert not [call for call in runner.calls if call[:3] == ["docker", "image", "rm"]]
    assert not report.failed, report.render()


def test_uninstall_leaves_a_named_recipe_it_did_not_write(tmp_path: Path) -> None:
    """DEPLOY-009, the guard half. Invariant: a recipe a service names is removed only when it
    carries the generated header. Failure caught: a version of the fix that trusted the name alone
    would delete a user's own Dockerfile from a data folder that is also their project. Target:
    `recall.wizard.uninstall.plan_uninstall`."""
    from recall.wizard.uninstall import plan_uninstall
    from tests.test_wizard_uninstall import _docker

    data_root, recipe = _versioned_install(tmp_path)
    recipe.write_text("FROM theirs\n", encoding="utf-8")

    plan = plan_uninstall(
        data_root=data_root, profile_path=tmp_path / "none.json",
        claude_config_path=tmp_path / "claude.json", runner=_docker(),
    )

    assert str(recipe) not in {item.name for item in plan.removing("file")}
    assert [item for item in plan.keeping() if item.kind == "image"], (
        "the image is named (red pre-fix: no image was ever planned)"
    )


# ----------------------------------------------------------------------------------------------
# STAKES-006: the install-time backup of ~/.claude.json is written once and never replaced
# ----------------------------------------------------------------------------------------------


def _block(name: str = "acme-docs") -> Any:
    from recall.wizard.wiring import ServerBlock

    return ServerBlock(name=name, tenant=name, env={"RECALL_TENANT": name}, rationale="test")


def test_a_second_registration_keeps_the_pre_recall_backup(tmp_path: Path) -> None:
    """STAKES-006. Invariant: `<config>.recall-backup` holds the client config from BEFORE recall
    was first registered, and a later registration writes its own uniquely named backup. Failure
    caught: every registration overwrote the fixed backup with the current config, so after a
    re-run it already contained recall's entries, contradicting what uninstall documents it to be.
    Target: `recall.wizard.wiring._register_local_scope_locked`."""
    from recall.wizard.wiring import register_local_scope

    client = tmp_path / ".claude.json"
    original = json.dumps({"projects": {}, "mine": True})
    client.write_text(original, encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()

    first = register_local_scope(
        (_block(),), project_root=project, config_path=client, interpreter="python"
    )
    assert first.recorded
    after_first = client.read_text(encoding="utf-8")

    second = register_local_scope(
        (_block(), _block("acme-code")), project_root=project, config_path=client,
        interpreter="python",
    )
    assert second.recorded

    fixed = client.with_name(client.name + ".recall-backup")
    assert fixed.read_text(encoding="utf-8") == original, "the pre-recall copy was overwritten"
    later = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(client.name + ".recall-backup-")
    ]
    assert len(later) == 1
    assert later[0].read_text(encoding="utf-8") == after_first


# ----------------------------------------------------------------------------------------------
# CODE-006: unregistration rewrites ~/.claude.json through the registration's private writer
# ----------------------------------------------------------------------------------------------


def test_unregistration_writes_through_the_private_configuration_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CODE-006. Invariant: the uninstall rewrite of the client config goes through
    `wiring._write_private_configuration`, the same writer registration uses. Failure caught: the
    two halves used different writers (an mkstemp 0600 writer against `atomic_write_bytes`, which
    copies the original's mode), so a hardening of one never reached the other. Target:
    `recall.wizard.uninstall._unregister_locked`."""
    from recall.wizard import uninstall, wiring

    project = tmp_path / "project"
    project.mkdir()
    config = tmp_path / "claude.json"
    config.write_text(
        json.dumps({"projects": {str(project): {"mcpServers": {
            "acme-docs": {"command": "python", "cwd": str(project)},
        }}}}),
        encoding="utf-8",
    )
    written: list[Path] = []
    real = wiring._write_private_configuration

    def recording(path: Path, content: str) -> None:
        written.append(Path(path))
        real(path, content)

    monkeypatch.setattr(wiring, "_write_private_configuration", recording)

    assert uninstall._unregister(("acme-docs",), project, config) is None

    assert config.resolve() in written
    document = json.loads(config.read_text(encoding="utf-8"))
    assert document["projects"][str(project)]["mcpServers"] == {}


# ----------------------------------------------------------------------------------------------
# SEC-003: the `.env` the wizard writes is kept out of git
# ----------------------------------------------------------------------------------------------


def test_the_project_env_is_added_to_gitignore_inside_a_work_tree(tmp_path: Path) -> None:
    """SEC-003. Invariant: when `project_root` is inside a git work tree and its `.gitignore` does
    not ignore `.env`, `write_project_files` adds `.env` to it, keeping existing lines. Failure
    caught: the wizard wrote the DSN, password included, and the provider keys into
    `<project>/.env` with nothing keeping it out of a `git add .`. Target:
    `recall.wizard.wiring.write_project_files`."""
    from recall.wizard.wiring import write_project_files

    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".gitignore").write_text("node_modules/", encoding="utf-8")

    write_project_files(
        project_root=project, dsn="postgresql://u:secret@db.example.invalid/recall",
        embedder="hashing", memory_dir=tmp_path / "memory",
    )

    lines = (project / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines == ["node_modules/", ".env"], lines

    # A second run adds nothing: the entry is already there.
    write_project_files(
        project_root=project, dsn="postgresql://u:secret@db.example.invalid/recall",
        embedder="hashing", memory_dir=tmp_path / "memory",
    )
    assert (project / ".gitignore").read_text(encoding="utf-8").splitlines() == lines


def test_no_gitignore_is_written_outside_a_work_tree(tmp_path: Path) -> None:
    """SEC-003, the guard half. Invariant: a project that is not in a git work tree gets no
    `.gitignore` from the wizard. Failure caught: a fix that wrote one unconditionally would drop an
    unexplained file into every non-git folder. Target: `recall.wizard.wiring.write_project_files`.
    Not red-provable against the pre-fix code, which never wrote one: a deliberate guard."""
    from recall.wizard.wiring import write_project_files

    project = tmp_path / "project"
    project.mkdir()
    if any((folder / ".git").exists() for folder in (project, *project.resolve().parents)):
        pytest.skip("the temporary directory is itself inside a git work tree")

    write_project_files(
        project_root=project, dsn="postgresql://u:p@127.0.0.1:5487/recall",
        embedder="hashing", memory_dir=tmp_path / "memory",
    )
    assert not (project / ".gitignore").exists()


# ----------------------------------------------------------------------------------------------
# ENV-011: the host server blocks carry the installing shell's embedding cache settings
# ----------------------------------------------------------------------------------------------


def test_host_blocks_carry_the_embedding_cache_settings_and_containers_drop_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENV-011. Invariant: the server blocks copy RECALL_EMBED_CACHE and RECALL_EMBED_CACHE_MAX_MB
    from the installing environment, and `stack.tenant_service` drops both inside a container.
    Failure caught: a user who moved or switched off the cache for the install had the memory
    server's `recall_index` recreate it at the platform default, up to 512 MB. Target:
    `recall.wizard.wiring._base_env` and `recall.wizard.stack.tenant_service`."""
    from recall.wizard.corpora import default_plan
    from recall.wizard.stack import tenant_service
    from recall.wizard.wiring import server_blocks

    cache = str(tmp_path / "cache.sqlite")
    monkeypatch.setenv("RECALL_EMBED_CACHE", cache)
    monkeypatch.setenv("RECALL_EMBED_CACHE_MAX_MB", "64")
    plan = default_plan(
        embedder="hashing", docs_root=tmp_path / "docs", code_root=tmp_path / "code",
        memory_root=tmp_path / "memory",
    )

    blocks, _unservable = server_blocks(
        plan, dsn="postgresql://recall:recall@127.0.0.1:5487/recall",
        promoted=frozenset(spec.tenant for spec in plan.corpora), serving=frozenset(),
    )

    assert blocks
    for block in blocks:
        assert block.env.get("RECALL_EMBED_CACHE") == cache, block
        assert block.env.get("RECALL_EMBED_CACHE_MAX_MB") == "64", block
    environment = tenant_service(blocks[0].env, image="recall-wizard:test")["environment"]
    assert isinstance(environment, dict)
    assert "RECALL_EMBED_CACHE" not in environment, "a host cache path means nothing in a container"
    assert "RECALL_EMBED_CACHE_MAX_MB" not in environment


# ----------------------------------------------------------------------------------------------
# NUM-002: the RAM floor allows for reporting slack and never prints the floor it says is missed
# ----------------------------------------------------------------------------------------------


def _system(total_ram_bytes: int) -> Any:
    from recall.wizard.probe import SystemProbe

    return SystemProbe(
        hardware=None,  # type: ignore[arg-type]
        total_ram_bytes=total_ram_bytes,
        available_ram_bytes=None,
        docker_installed=True,
        docker_running=True,
        wsl2_available=None,
        virtualization_enabled=None,
        cuda_vram_bytes=None,
        python_executable="python",
        python_is_store_build=False,
    )


def test_a_machine_sold_with_4_gb_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """NUM-002. Invariant: `blockers` applies the module's RAM reporting slack to the 4 GiB floor,
    as `heavy_local_models_are_feasible` does for 8 GiB. Failure caught: a nominal 4 GB machine,
    which reports 3.8 to 3.99 GiB, was refused, with the message "4.0 GB of memory, below the 4 GB
    needed". Target: `recall.wizard.probe.blockers`."""
    from recall.wizard.probe import blockers

    found = blockers(_system(int(3.96 * 1024**3)))
    assert not [line for line in found if "memory" in line], found


def test_the_ram_message_never_prints_the_floor_and_says_gib() -> None:
    """NUM-002. Invariant: the refusal floors the reported figure to one decimal and labels it GiB,
    so the printed value is always strictly below the number it is said to miss. Failure caught:
    `{gib:.1f} GB` rounded 2.96 GiB up to "3.0 GB" and called GiB GB. Target:
    `recall.wizard.probe.blockers`."""
    from recall.wizard.probe import blockers

    found = [line for line in blockers(_system(int(2.96 * 1024**3))) if "memory" in line]
    assert len(found) == 1, found
    assert "reports 2.9 GiB of memory, below the 4 GiB" in found[0], found[0]
