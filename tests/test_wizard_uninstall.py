"""The uninstaller, tested for what it must NOT remove as hard as for what it must.

An uninstaller is judged by its false positives. Removing too little leaves a person one list to
finish by hand; removing too much loses work that was never the installer's to delete. Every test
below that begins "never" or "keeps" is the more important half of this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recall.wizard.stack import COMPOSE_NAME, DOCKERFILE_NAME
from recall.wizard.uninstall import UninstallRefusal, execute, plan_uninstall

PROJECT = "recall-acme-1234abcd"


def _install(root: Path, *, project_root: Path | None = None) -> Path:
    """A data folder shaped like one the installer produced, with the corpus roots INSIDE it.

    Inside deliberately: that is what the installer suggests, so it is what most installs look like,
    and it is the layout in which a careless uninstall destroys the user's documents.
    """
    data_root = root / "recall"
    data_root.mkdir(parents=True)
    (data_root / COMPOSE_NAME).write_text(
        json.dumps(
            {
                "name": PROJECT,
                "services": {"db": {"image": "pgvector/pgvector:pg18"}, "recall-docs": {}},
                "volumes": {"pgdata": None},
            }
        ),
        encoding="utf-8",
    )
    (data_root / DOCKERFILE_NAME).write_text("FROM python:3.13-slim\n", encoding="utf-8")
    config: dict[str, Any] = {
        "project": "acme",
        "embedder": "fastembed",
        "corpus_version": "2026-01-01",
        "data_root": str(data_root),
        "docs_root": str(data_root / "docs"),
        "code_root": str(data_root / "code"),
        "memory_root": str(data_root / "memory"),
    }
    if project_root is not None:
        config["project_root"] = str(project_root)
    (data_root / "wizard.json").write_text(json.dumps(config), encoding="utf-8")
    (data_root / "runtime.json").write_text("{}", encoding="utf-8")
    for leaf in ("docs", "code", "memory"):
        folder = data_root / leaf
        folder.mkdir()
        (folder / "mine.md").write_text(f"# {leaf}\n\nwork that predates recall\n", encoding="utf-8")
    return data_root


def _docker(containers: tuple[str, ...] = ("recall-acme-1234abcd-db-1",)) -> Any:
    """A fake docker that answers the container query and records every command."""
    calls: list[list[str]] = []

    def _run(command: Any) -> tuple[int, str]:
        calls.append(list(command))
        if "ps" in command:
            return 0, "\n".join(containers)
        return 0, ""

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def test_the_plan_never_lists_a_corpus_root_for_removal(tmp_path: Path) -> None:
    """⛔ **The defect this module exists to prevent, and the one a person cannot undo.**

    The installer SUGGESTS `<data_root>/docs`, `<data_root>/code` and `<data_root>/memory`, so on a
    default install the user's notes, source and agent memory sit inside the very directory an
    uninstaller is most tempted to `rmtree`. Deleting the index is recoverable by re-indexing.
    Deleting what was indexed is not.
    """
    data_root = _install(tmp_path)

    plan = plan_uninstall(data_root=data_root, runner=_docker())

    removing = {item.name for item in plan.removing()}
    for leaf in ("docs", "code", "memory"):
        assert str(data_root / leaf) not in removing, f"{leaf} is the user's own content"
    kept = {item.name for item in plan.keeping()}
    assert {str(data_root / leaf) for leaf in ("docs", "code", "memory")} <= kept, (
        "and it must be SHOWN as kept: on a default install these sit inside the folder being "
        "uninstalled, so silence here reads as deletion"
    )


def test_execute_leaves_every_corpus_file_on_disk(tmp_path: Path) -> None:
    """The same property, asserted against the filesystem rather than against the plan.

    A plan that says the right thing and an `execute` that removes the directory anyway would pass
    the test above. This one opens the files afterwards.
    """
    data_root = _install(tmp_path)
    plan = plan_uninstall(data_root=data_root, runner=_docker())

    execute(plan, runner=_docker())

    for leaf in ("docs", "code", "memory"):
        survivor = data_root / leaf / "mine.md"
        assert survivor.exists(), f"{survivor} was destroyed by an uninstall"
        assert "predates recall" in survivor.read_text(encoding="utf-8")


def test_only_the_files_the_installer_wrote_are_removed(tmp_path: Path) -> None:
    """A named list, not a glob. A glob grows to cover whatever a user later drops in the folder."""
    data_root = _install(tmp_path)
    stranger = data_root / "notes-i-put-here.txt"
    stranger.write_text("mine", encoding="utf-8")

    plan = plan_uninstall(data_root=data_root, runner=_docker())
    execute(plan, runner=_docker())

    assert not (data_root / COMPOSE_NAME).exists()
    assert not (data_root / DOCKERFILE_NAME).exists()
    assert not (data_root / "wizard.json").exists()
    assert stranger.exists(), "a file the installer did not write must survive"


def test_containers_are_identified_by_compose_label_not_by_name(tmp_path: Path) -> None:
    """⛔ A name pattern would match a SECOND install, and `docker rm -f` does not ask twice.

    Two installs on one machine differ only by the digest in their compose project name, which is
    exactly the part a pattern like `recall-*` ignores.
    """
    data_root = _install(tmp_path)
    docker = _docker()

    plan_uninstall(data_root=data_root, runner=docker)

    query = next(call for call in docker.calls if "ps" in call)
    assert f"label=com.docker.compose.project={PROJECT}" in query, (
        "the containers must be found by the project this install recorded, not by a name shape"
    )
    assert plan_uninstall(data_root=data_root, runner=docker).project_name == PROJECT


def test_a_missing_stack_file_refuses_rather_than_guessing(tmp_path: Path) -> None:
    """Without the stack file there is no safe way to tell this install's resources from another's.

    Refusing is the right answer. The alternative — matching container names by shape — is how an
    uninstaller removes a stack somebody else is using.
    """
    data_root = tmp_path / "recall"
    data_root.mkdir()

    with pytest.raises(UninstallRefusal) as excinfo:
        plan_uninstall(data_root=data_root, runner=_docker())

    assert COMPOSE_NAME in str(excinfo.value)


def test_a_stack_file_without_a_project_name_refuses(tmp_path: Path) -> None:
    """Same reasoning one level down: a nameless stack cannot be identified, only guessed at."""
    data_root = tmp_path / "recall"
    data_root.mkdir()
    (data_root / COMPOSE_NAME).write_text(json.dumps({"services": {"db": {}}}), encoding="utf-8")

    with pytest.raises(UninstallRefusal, match="no project name"):
        plan_uninstall(data_root=data_root, runner=_docker())


def test_the_index_volume_is_kept_unless_purge_is_asked_for(tmp_path: Path) -> None:
    """The indexes are reproducible and expensive. Whoever reinstalls next week wants them.

    Off by default and reported as kept, so the person reclaiming disk learns the flag exists rather
    than discovering the volume months later.
    """
    data_root = _install(tmp_path)
    plan = plan_uninstall(data_root=data_root, runner=_docker())
    docker = _docker()

    report = execute(plan, runner=docker)

    down = next(call for call in docker.calls if "down" in call)
    assert "-v" not in down, "the volume must not be removed without --purge-data"
    assert any("pgdata" in item.name for item in report.kept)
    assert any("--purge-data" in item.detail for item in report.kept), (
        "the report has to name the flag, or the volume is simply lost track of"
    )


def test_purge_data_removes_the_volume(tmp_path: Path) -> None:
    """And the flag has to actually do it, or it is a switch that reports rather than acts.

    ⚠️ `purge_data` goes to BOTH calls. This test originally passed it only to `execute`, and the
    guard added afterwards refused it — correctly. The plan is what a person reads and agrees to,
    so a run that removes more than the plan displayed is the failure the split exists to prevent,
    and a test encoding that mismatch was encoding the defect.
    """
    data_root = _install(tmp_path)
    plan = plan_uninstall(data_root=data_root, purge_data=True, runner=_docker())
    docker = _docker()

    report = execute(plan, purge_data=True, runner=docker)

    down = next(call for call in docker.calls if "down" in call)
    assert "-v" in down
    assert any("pgdata" in item.name for item in report.removed)


def test_only_registrations_this_install_wrote_are_removed(tmp_path: Path) -> None:
    """⛔ The `cwd` marker decides, in the same direction `register_local_scope` reads it.

    That function refuses to OVERWRITE an entry without the marker, on the grounds that it belongs
    to somebody else. Removal has to apply the same test, or the uninstaller becomes a way to lose
    a server the operator configured by hand under a name the wizard also uses.
    """
    project_root = tmp_path / "work"
    project_root.mkdir()
    data_root = _install(tmp_path, project_root=project_root)
    config_path = tmp_path / "claude.json"
    config_path.write_text(
        json.dumps(
            {
                "projects": {
                    str(project_root): {
                        "mcpServers": {
                            "recall": {"command": "python", "cwd": str(project_root)},
                            "recall-handwritten": {"command": "python"},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    plan = plan_uninstall(
        data_root=data_root, claude_config_path=config_path, runner=_docker()
    )
    execute(plan, claude_config_path=config_path, runner=_docker())

    servers = json.loads(config_path.read_text(encoding="utf-8"))["projects"][str(project_root)][
        "mcpServers"
    ]
    assert "recall" not in servers, "the entry this install wrote must go"
    assert "recall-handwritten" in servers, (
        "an entry with no cwd marker was written by hand or by another tool, and an uninstall that "
        "removes it destroys configuration nobody asked it to touch"
    )


def test_the_rest_of_the_client_config_survives(tmp_path: Path) -> None:
    """⚠️ `~/.claude.json` holds every project the user has. A partial write costs all of them.

    The file is rewritten whole and replaced atomically for that reason. Asserted by leaving an
    unrelated project in it and checking it is still there afterwards.
    """
    project_root = tmp_path / "work"
    project_root.mkdir()
    data_root = _install(tmp_path, project_root=project_root)
    config_path = tmp_path / "claude.json"
    config_path.write_text(
        json.dumps(
            {
                "numStartups": 41,
                "projects": {
                    str(project_root): {
                        "mcpServers": {"recall": {"command": "python", "cwd": str(project_root)}}
                    },
                    "C:/some/other/project": {"mcpServers": {"unrelated": {"command": "node"}}},
                },
            }
        ),
        encoding="utf-8",
    )

    plan = plan_uninstall(data_root=data_root, claude_config_path=config_path, runner=_docker())
    execute(plan, claude_config_path=config_path, runner=_docker())

    document = json.loads(config_path.read_text(encoding="utf-8"))
    assert document["numStartups"] == 41, "unrelated keys must survive"
    assert document["projects"]["C:/some/other/project"]["mcpServers"] == {
        "unrelated": {"command": "node"}
    }


def test_a_failure_does_not_abort_the_rest(tmp_path: Path) -> None:
    """⚠️ Stopping at the first error leaves a mixture nobody can reason about.

    Containers gone, files present, registrations live, and no list of which. Removing what can be
    removed and REPORTING the rest leaves one list to finish by hand.
    """
    data_root = _install(tmp_path)
    plan = plan_uninstall(data_root=data_root, runner=_docker())

    def _broken(command: Any) -> tuple[int, str]:
        if "down" in command:
            return 1, "Cannot connect to the Docker daemon"
        return 0, ""

    report = execute(plan, runner=_broken)

    assert report.failed, "the docker failure must be reported"
    assert not (data_root / COMPOSE_NAME).exists(), (
        "the files must still be removed: a failure on one kind of resource is not a reason to "
        "leave every other kind behind"
    )
    assert "Cannot connect to the Docker daemon" in report.render()


def test_the_plan_reads_before_it_writes(tmp_path: Path) -> None:
    """`plan_uninstall` must be safe to call on a whim: it is what a confirmation prompt shows."""
    data_root = _install(tmp_path)

    plan = plan_uninstall(data_root=data_root, runner=_docker())

    assert (data_root / COMPOSE_NAME).exists(), "planning must not delete anything"
    assert (data_root / "wizard.json").exists()
    rendered = plan.render()
    assert "This will remove:" in rendered and "This will KEEP:" in rendered


def test_the_plan_says_the_volume_is_kept_when_purge_is_not_asked_for(tmp_path: Path) -> None:
    """⛔ A confirmation screen must not contradict itself about the one item holding data.

    The first version listed the volume under "This will remove" with a note saying it would only
    be removed with `--purge-data`. Both halves were in the same line. Whoever read it learned
    nothing except that the tool was unsure.
    """
    data_root = _install(tmp_path)

    plan = plan_uninstall(data_root=data_root, runner=_docker())

    rendered = plan.render()
    assert not plan.removing("volume"), "the volume is not being removed, so it must not be listed"
    assert any("pgdata" in item.name for item in plan.keeping())
    remove_section = rendered.split("This will KEEP:")[0]
    assert "pgdata" not in remove_section


def test_executing_a_wider_removal_than_was_shown_is_refused(tmp_path: Path) -> None:
    """⛔ The plan is what the person agreed to.

    Widening the removal after it was displayed defeats the entire plan-then-do split: the screen
    said the indexes were kept and the run would have destroyed them. Refused rather than
    reconciled, because there is no safe way to guess which of the two the caller meant.
    """
    data_root = _install(tmp_path)
    shown = plan_uninstall(data_root=data_root, purge_data=False, runner=_docker())

    with pytest.raises(UninstallRefusal, match="disagree about the index volume"):
        execute(shown, purge_data=True, runner=_docker())


def test_the_plan_lists_only_registrations_this_install_wrote(tmp_path: Path) -> None:
    """⚠️ **This test exists because its mutation survived, and what survived it is interesting.**

    Removing the `cwd` check from the PLAN did not let an unmarked entry be deleted: `_unregister`
    checks the marker again before touching anything, so the entry survived and the safety test
    still passed. That is defence in depth working as intended, and it is worth keeping.

    But it means the plan can be wrong while the removal stays right, and the plan is the thing a
    person reads and agrees to. An uninstaller that announces it will delete a server it then leaves
    alone teaches its user that the confirmation screen is not to be trusted, which costs more than
    the entry itself would have.
    """
    project_root = tmp_path / "work"
    project_root.mkdir()
    data_root = _install(tmp_path, project_root=project_root)
    config_path = tmp_path / "claude.json"
    config_path.write_text(
        json.dumps(
            {
                "projects": {
                    str(project_root): {
                        "mcpServers": {
                            "recall": {"command": "python", "cwd": str(project_root)},
                            "recall-handwritten": {"command": "python"},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    plan = plan_uninstall(data_root=data_root, claude_config_path=config_path, runner=_docker())

    listed = {item.name for item in plan.removing("registration")}
    assert listed == {"recall"}, (
        "the plan must announce exactly what will go; listing an entry the removal will refuse to "
        "touch makes the confirmation screen unreliable"
    )
    kept = {item.name for item in plan.keeping()}
    assert "recall-handwritten" in kept, "and it should say why the other one survives"
