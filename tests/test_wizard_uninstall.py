"""The uninstaller, tested for what it must NOT remove as hard as for what it must.

An uninstaller is judged by its false positives. Removing too little leaves a person one list to
finish by hand; removing too much loses work that was never the installer's to delete. Every test
below that begins "never" or "keeps" is the more important half of this file.
"""

from __future__ import annotations

import json
import os
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
    assert "Cannot connect to the Docker daemon" in report.render()

    # ⚠️ **This expectation CHANGED, deliberately, for finding BUG-004.** It used to assert the
    # compose file was removed too. That is wrong when the teardown failed: the stack file is the
    # only thing naming this install's containers and volume, `plan_uninstall` refuses without it,
    # and its own refusal message tells the user to "restore the file" — which nothing could,
    # because the uninstaller had just deleted it. Leftover containers became unnameable by the tool
    # that left them.
    assert (data_root / COMPOSE_NAME).exists(), (
        "the identity file must survive a failed teardown so the uninstall can be retried"
    )
    assert any("retried" in item.detail for item in report.kept), (
        "and the report has to say why it is still there"
    )

    # The CONTROL, which is the property this test was originally written for and which did not
    # change: a failure on one kind of resource still does not stop the other kinds being removed.
    assert not (data_root / DOCKERFILE_NAME).exists(), (
        "a docker failure is not a reason to leave every other file behind"
    )


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


# ----------------------------------------------------------------------------------------------
# The window, which is the surface the person who needed a graphical installer will use
# ----------------------------------------------------------------------------------------------
#
# These need no Qt. `uninstall_main` takes `confirm` and `notify` as collaborators for the same
# reason `InstallerWindow` takes a runner and a writer: the decision logic is what needs testing,
# and requiring a dialog to exercise it is how that logic ends up untested.


def _recorder() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Somewhere for the dialogs to go, so a test can read what they would have said."""
    asked: list[dict[str, str]] = []
    told: list[dict[str, str]] = []
    return asked, told


def test_the_window_shows_the_whole_plan_including_what_survives(tmp_path: Path) -> None:
    """⛔ A confirmation that does not say what it is confirming is a confirmation of nothing.

    This removes containers and rewrites the MCP client's configuration. The dialog carries
    `UninstallPlan.render()` verbatim, which is the same text the terminal prints, so the two
    surfaces cannot come to disagree about what is going — and so the person sees the line saying
    their documents are kept, which on a default install is the fact that matters most.
    """
    from recall.desktop.main import uninstall_main

    asked, told = _recorder()
    data_root = _install(tmp_path)

    def _confirm(title: str, text: str, detail: str) -> bool:
        asked.append({"title": title, "text": text, "detail": detail})
        return False

    assert uninstall_main(["--data-root", str(data_root)], confirm=_confirm, notify=told.append) == 0

    detail = asked[0]["detail"]
    assert "This will remove:" in detail
    assert "This will KEEP:" in detail
    assert str(data_root / "docs") in detail, "the person must see their own documents survive"


def test_answering_no_removes_nothing(tmp_path: Path) -> None:
    """The dialog defaults to No, and No has to mean it."""
    from recall.desktop.main import uninstall_main

    data_root = _install(tmp_path)

    uninstall_main(
        ["--data-root", str(data_root)],
        confirm=lambda *_args: False,
        notify=lambda *_args: None,
    )

    assert (data_root / COMPOSE_NAME).exists(), "declining must leave the install intact"
    assert (data_root / "wizard.json").exists()


def test_answering_yes_removes_and_reports(tmp_path: Path) -> None:
    """And Yes has to act, and then say what it did."""
    from recall.desktop.main import uninstall_main

    told: list[tuple[str, str]] = []
    data_root = _install(tmp_path)

    uninstall_main(
        ["--data-root", str(data_root)],
        confirm=lambda *_args: True,
        notify=lambda title, text: told.append((title, text)),
        # ⛔ **Injected, because this test used to run REAL docker.** It asserted the stack file was
        # removed, and on a machine with no such stack `docker compose down` fails — which passed
        # only because the failure was being discarded and the file deleted anyway. Once that was
        # fixed the test correctly kept the file and went red. What it is about is the CLI wiring,
        # so the teardown is now a fake that succeeds and the outcome no longer depends on whether
        # this machine has a daemon.
        runner=_docker(),
    )

    assert not (data_root / COMPOSE_NAME).exists()
    assert (data_root / "docs" / "mine.md").exists(), "still never the user's own content"
    assert told and "Removed" in told[-1][1]


def test_a_refusal_is_shown_rather_than_raised(tmp_path: Path) -> None:
    """A frozen binary has nowhere to print a traceback that anybody will read.

    Pointing the uninstaller at the wrong folder is the ordinary mistake, not an exceptional one,
    so the reply has to be a sentence in a window rather than a stack trace on a console the user
    never sees. The exit status still says it failed.
    """
    from recall.desktop.main import uninstall_main

    told: list[tuple[str, str]] = []
    empty = tmp_path / "not-an-install"
    empty.mkdir()

    status = uninstall_main(
        ["--data-root", str(empty)],
        confirm=lambda *_args: True,
        notify=lambda title, text: told.append((title, text)),
    )

    assert status == 1
    assert told, "the refusal must be shown somewhere"
    assert COMPOSE_NAME in told[-1][1]


def test_the_folder_is_asked_for_when_no_argument_is_given(tmp_path: Path) -> None:
    """⛔ **Requiring `--data-root` made the graphical uninstaller unreachable.**

    A Start Menu shortcut, a pinned icon and a double-clicked exe all pass no arguments. argparse
    then wrote its usage error to a stderr that a windowed process does not show and exited 2 — an
    uninstaller that appears to do nothing at all when clicked. The audience that needed a window to
    install is exactly the audience here, and it is the one that never types a flag.
    """
    from recall.desktop.main import uninstall_main

    data_root = _install(tmp_path)
    asked: list[tuple[str, str]] = []

    status = uninstall_main(
        [],
        choose=lambda title, start: asked.append((title, start)) or str(data_root),
        confirm=lambda *_a: True,
        notify=lambda *_a: None,
        # See the sibling test: without this the assertion below depends on whether this machine has
        # a docker daemon, and used to pass only because a failed teardown was being discarded.
        runner=_docker(),
    )

    assert status == 0
    assert asked, "with no argument the person must be asked which install to remove"
    assert not (data_root / COMPOSE_NAME).exists(), "and the chosen install must actually go"


def test_cancelling_the_folder_picker_removes_nothing(tmp_path: Path) -> None:
    """Backing out is a decision, not a failure, and must not be reported as one."""
    from recall.desktop.main import uninstall_main

    data_root = _install(tmp_path)
    told: list[Any] = []

    status = uninstall_main(
        [],
        choose=lambda title, start: "",
        confirm=lambda *_a: pytest.fail("nothing may be confirmed after a cancel"),
        notify=lambda *args: told.append(args),
    )

    assert status == 0, "cancelling is not an error"
    assert not told, "and it needs no dialog telling the person what they just decided"
    assert (data_root / COMPOSE_NAME).exists()


def test_a_json_document_that_is_not_an_object_never_aborts_the_run(tmp_path: Path) -> None:
    """⛔ STAKES-007: `json.loads("null")` returns None, and `.get` on it raises AttributeError.

    That escaped `execute` AFTER `docker compose down` and the volume removal had run and BEFORE
    the files were unlinked: no report returned, containers gone, files present, and nothing
    telling the user which. `execute`'s own docstring promises the opposite. `register_local_scope`
    already guarded this exact shape; the pattern was simply not carried across.
    """
    project_root = tmp_path / "work"
    project_root.mkdir()
    data_root = _install(tmp_path, project_root=project_root)
    config_path = tmp_path / "claude.json"
    config_path.write_text("null", encoding="utf-8")

    plan = plan_uninstall(data_root=data_root, claude_config_path=config_path, runner=_docker())
    report = execute(plan, claude_config_path=config_path, runner=_docker())

    assert report is not None, "a malformed client config must not abort the uninstall"
    assert not (data_root / COMPOSE_NAME).exists(), "the rest of the removal must still happen"


def test_the_volume_removed_is_the_one_the_stack_declared(tmp_path: Path) -> None:
    """⛔ STAKES-005: recomputing the name is what `_compose_project` refuses to do for the project.

    A stack written by a release with a different volume name yields a name that stack never
    declared. `docker volume rm` then says "no such volume", `execute` counts that as REMOVED, and
    the real volume holding the indexes survives while the report says it went, on the one item
    that cannot be undone.
    """
    data_root = tmp_path / "recall"
    data_root.mkdir()
    (data_root / COMPOSE_NAME).write_text(
        json.dumps(
            {
                "name": PROJECT,
                "services": {"db": {}},
                "volumes": {"legacy-pgdata": None, "external-thing": {"external": True}},
            }
        ),
        encoding="utf-8",
    )
    (data_root / "wizard.json").write_text(json.dumps({"project": "acme"}), encoding="utf-8")

    plan = plan_uninstall(data_root=data_root, purge_data=True, runner=_docker())

    names = {item.name for item in plan.removing("volume")}
    assert names == {f"{PROJECT}_legacy-pgdata"}, (
        f"the declared volume must be the one removed, got {names}"
    )


def test_the_client_config_is_backed_up_before_it_is_rewritten(tmp_path: Path) -> None:
    """⛔ STAKES-006: the REMOVAL direction had less protection than the addition direction.

    `register_local_scope` writes a `.recall-backup` before its atomic replace. `_unregister` did
    not, on the one path where a mistake cannot be undone, over a file holding every project the
    MCP client tracks.
    """
    project_root = tmp_path / "work"
    project_root.mkdir()
    data_root = _install(tmp_path, project_root=project_root)
    config_path = tmp_path / "claude.json"
    original = json.dumps(
        {
            "projects": {
                str(project_root): {
                    "mcpServers": {"recall": {"command": "python", "cwd": str(project_root)}}
                }
            }
        }
    )
    config_path.write_text(original, encoding="utf-8")

    plan = plan_uninstall(data_root=data_root, claude_config_path=config_path, runner=_docker())
    execute(plan, claude_config_path=config_path, runner=_docker())

    # ⚠️ **The name is now timestamped, and this expectation changed deliberately.** The fixed
    # `.recall-backup` is what `register_local_scope` writes at INSTALL time, so reusing it here
    # overwrote the only copy of the file from before recall ever touched it. The property this test
    # is about — the previous config survives somewhere before the rewrite — is unchanged.
    backups = [
        path
        for path in config_path.parent.iterdir()
        if path.name.startswith(config_path.name + ".recall-backup")
    ]
    assert len(backups) == 1, "exactly one backup, under a name that is never reused"
    backup = backups[0]
    assert backup.exists(), "the previous config must survive somewhere"
    assert json.loads(backup.read_text(encoding="utf-8")) == json.loads(original), (
        "and it must be the content as it was before the rewrite"
    )


def test_an_unqueryable_docker_is_not_reported_as_a_removal(tmp_path: Path) -> None:
    """⛔ STAKES-010: the fallback item is a GLOB, and a glob under "This will remove" is a lie.

    It reads as a name-pattern teardown, which the module docstring says it refuses to do, and
    `execute` reported the pseudo-name as removed while never naming a real container.
    """
    data_root = _install(tmp_path)

    def _no_docker(command: Any) -> tuple[int, str]:
        return 1, "Cannot connect to the Docker daemon"

    plan = plan_uninstall(data_root=data_root, runner=_no_docker)

    assert not plan.removing("container"), "a container that could not be listed is not a removal"
    assert any("*" in item.name for item in plan.keeping()), (
        "and it must still be SHOWN, so the user knows the list is incomplete"
    )


def test_a_dead_docker_never_reports_a_clean_uninstall(tmp_path: Path) -> None:
    """⛔ **Two correct-looking fixes cancelled each other, and the defect lived BETWEEN them.**

    One stopped reporting an unqueryable docker as a removal, by setting `removing=False` on the
    fallback container item. The other kept the stack file when the teardown failed, by scanning
    `report.failed` — which `execute` only ever populated by iterating that same now-empty tuple. So
    with the daemon unreachable, a failing `docker compose down` was recorded nowhere: no failure,
    no mention in the report, and the stack file naming the still-running containers deleted anyway.
    `plan_uninstall` then refuses on retry and tells the user to restore a file this run removed.

    Reproduced before the fix, verbatim: `failed: []`, `Removed 3 item(s).`, compose file gone.
    Three auditors found it independently, which is what a defect that belongs to no single change
    looks like.
    """
    data_root = _install(tmp_path)

    def dead(command: list[str]) -> tuple[int, str]:
        return (1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock")

    plan = plan_uninstall(data_root=data_root, runner=dead)
    report = execute(plan, runner=dead, claude_config_path=tmp_path / "claude.json")

    assert (data_root / COMPOSE_NAME).exists(), (
        "the stack file is the only thing that names this install's containers, and they are still "
        "running; deleting it makes them unnameable by the tool that left them"
    )
    assert (data_root / "wizard.json").exists(), "and the retry needs the config it resolves from"
    assert any(item.kind == "container" for item, _reason in report.failed), (
        "the teardown failure must be RECORDED even when no container could be listed to hang it on"
    )
    assert "Cannot connect to the Docker daemon" in report.render(), (
        "and the user must be told what went wrong; a failure nobody reports is a failure nobody "
        "can act on"
    )

    # The retry the keep-list exists to enable must actually work.
    retry = plan_uninstall(data_root=data_root, runner=dead)
    assert retry.project_name == PROJECT

    # Control: with a working docker the same files DO go, so this is not a test that passes by
    # never deleting anything.
    healthy = execute(
        plan_uninstall(data_root=data_root, runner=lambda command: (0, "")),
        runner=lambda command: (0, ""),
        claude_config_path=tmp_path / "claude.json",
    )
    assert not (data_root / COMPOSE_NAME).exists()
    assert not healthy.failed


def test_a_volume_the_stack_declared_external_is_never_derived_back(tmp_path: Path) -> None:
    """⛔ The `or (derived,)` fallback undid the external-volume exclusion three lines above it.

    For `{pgdata: {external: true, name: "<project>_pgdata"}}` every declared volume is skipped as
    not ours, the list is empty, the fallback derives `<project>_pgdata` — which is exactly the
    volume just excluded — and `docker volume rm` removes somebody else's data. The fallback is for
    a LEGACY stack that declared no volumes at all, and those are different facts.
    """
    data_root = _install(tmp_path)
    (data_root / COMPOSE_NAME).write_text(
        json.dumps(
            {
                "name": PROJECT,
                "services": {"db": {}},
                "volumes": {"pgdata": {"external": True, "name": f"{PROJECT}_pgdata"}},
            }
        ),
        encoding="utf-8",
    )
    plan = plan_uninstall(data_root=data_root, runner=lambda command: (0, ""), purge_data=True)
    assert [item.name for item in plan.removing("volume")] == [], (
        "a stack whose volumes are all external declares nothing this uninstaller may remove"
    )

    # An explicitly NAMED volume is used verbatim, not derived.
    (data_root / COMPOSE_NAME).write_text(
        json.dumps(
            {
                "name": PROJECT,
                "services": {"db": {}},
                "volumes": {"pgdata": {"name": "chosen-by-the-user"}},
            }
        ),
        encoding="utf-8",
    )
    named = plan_uninstall(data_root=data_root, runner=lambda command: (0, ""), purge_data=True)
    assert [item.name for item in named.removing("volume")] == ["chosen-by-the-user"]

    # And a legacy stack with no `volumes:` key at all still gets the historical derived name,
    # because there it is the only thing available and dropping it would strand the volume.
    (data_root / COMPOSE_NAME).write_text(
        json.dumps({"name": PROJECT, "services": {"db": {}}}), encoding="utf-8"
    )
    legacy = plan_uninstall(data_root=data_root, runner=lambda command: (0, ""), purge_data=True)
    assert [item.name for item in legacy.removing("volume")] == [f"{PROJECT}_pgdata"]


def test_the_backup_never_overwrites_the_one_the_installer_made(tmp_path: Path) -> None:
    """⛔ The install-time `.recall-backup` is the copy from BEFORE recall touched the file.

    Reusing that fixed name here destroyed it, with the already-rewritten content, at the moment
    somebody undoing an install would most want it. `recall/claude_code.py` already had the
    non-clobbering pattern; this now matches it.
    """
    import os
    import stat

    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = _install(tmp_path, project_root=project_root)
    config_path = tmp_path / "claude.json"
    config_path.write_text(
        json.dumps(
            {
                "projects": {
                    str(project_root): {
                        "mcpServers": {"recall": {"command": "python", "cwd": str(project_root)}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    installed_backup = config_path.with_name(config_path.name + ".recall-backup")
    installed_backup.write_text('{"the": "original, from before recall"}', encoding="utf-8")
    if os.name != "nt":
        config_path.chmod(0o600)

    execute(
        plan_uninstall(data_root=data_root, claude_config_path=config_path, runner=_docker()),
        runner=_docker(),
        claude_config_path=config_path,
    )

    assert installed_backup.read_text(encoding="utf-8") == '{"the": "original, from before recall"}', (
        "the install-time backup is the only pre-recall copy of this file and must survive"
    )
    fresh = [
        path
        for path in config_path.parent.iterdir()
        if path.name.startswith(config_path.name + ".recall-backup-")
    ]
    assert len(fresh) == 1, "the uninstall writes its own backup under a name it never reuses"
    if os.name != "nt":
        assert stat.S_IMODE(fresh[0].stat().st_mode) == 0o600, (
            "this file carries bearer tokens; a fresh write at the umask default leaves a "
            "world-readable copy of every credential beside a 0600 original"
        )
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def _profile(path: Path, compose_file: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"compose_file": str(compose_file)}), encoding="utf-8")
    return path


def test_the_desktop_handoff_file_is_removed_at_its_real_location(tmp_path: Path) -> None:
    """⛔ It lives in the user's CONFIG directory, not under `data_root`.

    It was listed as a `data_root`-relative name, so the entry never matched on Windows and every
    uninstall left the desktop app holding a profile pointing at a compose file that had just been
    deleted. The only test touching it fabricated the file under `data_root`, which is precisely
    why the mismatch was invisible: the fixture agreed with the bug.
    """
    data_root = _install(tmp_path)
    profile = _profile(tmp_path / "appdata" / "runtime.json", data_root / COMPOSE_NAME)

    plan = plan_uninstall(data_root=data_root, profile_path=profile, runner=_docker())
    assert str(profile) in [item.name for item in plan.removing("file")]

    execute(
        plan,
        runner=_docker(),
        claude_config_path=tmp_path / "claude.json",
    )
    assert not profile.exists()


def test_another_installs_handoff_file_is_never_touched(tmp_path: Path) -> None:
    """There is ONE profile per machine, so removing it blindly breaks a second install.

    The guard is whose stack it names. A profile pointing somewhere else, a profile that is not a
    JSON object, and a profile naming no stack at all must all be left alone rather than guessed at.
    """
    data_root = _install(tmp_path)
    someone_else = _profile(tmp_path / "appdata" / "runtime.json", tmp_path / "other" / COMPOSE_NAME)

    # Every plan-only case first: `execute` removes the stack file, and `plan_uninstall` needs it.
    for content in (
        json.dumps({"compose_file": str(tmp_path / "other" / COMPOSE_NAME)}),
        "null",
        "[]",
        '{"compose_file": null}',
        "{}",
        "not json at all",
    ):
        someone_else.write_text(content, encoding="utf-8")
        quiet = plan_uninstall(data_root=data_root, profile_path=someone_else, runner=_docker())
        assert str(someone_else) not in [item.name for item in quiet.removing("file")], content

    missing = plan_uninstall(
        data_root=data_root, profile_path=tmp_path / "nope" / "runtime.json", runner=_docker()
    )
    assert not [item for item in missing.removing("file") if "runtime.json" in item.name]

    # Then the destructive half, once, on the case that matters most.
    someone_else.write_text(
        json.dumps({"compose_file": str(tmp_path / "other" / COMPOSE_NAME)}), encoding="utf-8"
    )
    plan = plan_uninstall(data_root=data_root, profile_path=someone_else, runner=_docker())
    execute(plan, runner=_docker(), claude_config_path=tmp_path / "claude.json")
    assert someone_else.exists(), "a second install's profile is not ours to remove"


def test_a_failed_teardown_keeps_the_handoff_file_too(tmp_path: Path) -> None:
    """🔁 **This corrects an earlier judgement of mine, which reasoned to the opposite answer.**

    That argument was: the profile points at an install the user asked to remove, so leaving it
    behind leaves the app pointing at something that should be gone. It is wrong about the state it
    reasons over. If the teardown FAILED, the stack is still running — that is what failure means
    here — and the compose file naming it is kept for exactly that reason. Deleting the app's only
    pointer to a live stack while keeping the file that names it strands a running resource behind a
    tool that can no longer reach it, which is the same defect as deleting the stack file.
    """
    data_root = _install(tmp_path)
    profile = _profile(tmp_path / "appdata" / "runtime.json", data_root / COMPOSE_NAME)

    def dead(command: list[str]) -> tuple[int, str]:
        return (1, "Cannot connect to the Docker daemon")

    plan = plan_uninstall(data_root=data_root, profile_path=profile, runner=dead)
    execute(plan, runner=dead, claude_config_path=tmp_path / "claude.json")

    assert profile.exists(), "the stack is still up; the app must still be able to reach it"
    assert (data_root / COMPOSE_NAME).exists()

    # And the retry removes all three together, which is the state in which removal is correct.
    retry = plan_uninstall(data_root=data_root, profile_path=profile, runner=_docker())
    execute(retry, runner=_docker(), claude_config_path=tmp_path / "claude.json")
    assert not profile.exists()
    assert not (data_root / COMPOSE_NAME).exists()


def test_a_stack_file_that_is_not_an_object_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """⛔ Five `isinstance` guards were added; only the one that DEGRADES had a test.

    The one that changes control flow is `_compose_project`'s, which refuses. A compose file
    containing `null` or a list has no project name, and matching containers on a name pattern
    instead is the single thing this module refuses to do.
    """
    data_root = _install(tmp_path)
    for content in ("null", "[]", '"a string"', "42"):
        (data_root / COMPOSE_NAME).write_text(content, encoding="utf-8")
        with pytest.raises(UninstallRefusal):
            plan_uninstall(data_root=data_root, runner=_docker())

    # Control: a well-formed object still plans, so the refusal is about the SHAPE, not about
    # refusing everything.
    (data_root / COMPOSE_NAME).write_text(
        json.dumps({"name": PROJECT, "services": {"db": {}}}), encoding="utf-8"
    )
    assert plan_uninstall(data_root=data_root, runner=_docker()).project_name == PROJECT


def test_a_missing_stack_file_at_execute_time_does_not_crash(tmp_path: Path) -> None:
    """⛔ **FIX-04.** `compose_failed` was bound only inside `if compose_path.exists():` and read
    unconditionally, so `execute` raised `UnboundLocalError` when the stack file was gone.

    Three auditors reproduced it. It is reachable as a TOCTOU: `uninstall_main` plans, then blocks
    on a confirmation dialog for an unbounded time, then calls `execute`. It also makes a repeat run
    non-idempotent, because the first successful run deletes that very file.

    The docstring one line above says "A partial uninstall must not abort partway", and an
    `UnboundLocalError` before the report is returned is the sharpest possible violation of it.
    """
    data_root = _install(tmp_path)
    plan = plan_uninstall(data_root=data_root, runner=_docker())
    (data_root / COMPOSE_NAME).unlink()

    report = execute(plan, runner=_docker(), claude_config_path=tmp_path / "claude.json")
    assert report is not None, "a vanished stack file is a fact to report, not a crash"


def test_a_failed_unregistration_keeps_the_files_a_retry_needs(tmp_path: Path) -> None:
    """⛔ **FIX-03.** `teardown_failed` counted only failed CONTAINER and VOLUME items.

    So a failed MCP unregistration deleted the stack file, `wizard.json` and the handoff file
    anyway, and the retry became impossible: `plan_uninstall` refuses without the stack file, and
    `project_root` (the only thing that can find the stale `~/.claude.json` entry) lived in
    `wizard.json`. The comment above that line claims the opposite in so many words.
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
                        "mcpServers": {"recall": {"command": "python", "cwd": str(project_root)}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    plan = plan_uninstall(data_root=data_root, claude_config_path=config_path, runner=_docker())
    assert plan.removing("registration"), "the fixture must actually plan an unregistration"

    # Make the rewrite fail the way a locked or unreadable config would.
    config_path.unlink()

    report = execute(plan, runner=_docker(), claude_config_path=config_path)

    assert any(item.kind == "registration" for item, _reason in report.failed)
    assert (data_root / COMPOSE_NAME).exists(), (
        "a failed unregistration must keep the stack file: without it plan_uninstall refuses and "
        "the retry that would finish the job cannot even be built"
    )
    assert (data_root / "wizard.json").exists(), (
        "and wizard.json, which is the only record of project_root and therefore the only way to "
        "find the ~/.claude.json entry that was just left behind"
    )


def test_a_stack_whose_volumes_are_all_external_can_still_be_uninstalled(tmp_path: Path) -> None:
    """⛔ **FIX-05.** The external-volume exclusion defeated the plan/execute agreement guard.

    With every declared volume `external`, the plan holds no volume item, so
    `purge_data != bool(plan.removing("volume"))` was true and `execute` refused the whole
    uninstall before removing anything. `recall/desktop/main.py` calls `execute` outside its
    handler, so in the frozen GUI that is a traceback after the user already confirmed the plan.

    The guard is meant to catch a caller that widens the removal between plan and execute. It must
    compare INTENT, not the emptiness of a list that is legitimately empty.
    """
    data_root = _install(tmp_path)
    (data_root / COMPOSE_NAME).write_text(
        json.dumps(
            {
                "name": PROJECT,
                "services": {"db": {}},
                "volumes": {"pgdata": {"external": True, "name": f"{PROJECT}_pgdata"}},
            }
        ),
        encoding="utf-8",
    )

    plan = plan_uninstall(data_root=data_root, runner=_docker(), purge_data=True)
    assert plan.removing("volume") == (), "an external volume is never ours to remove"

    report = execute(
        plan, runner=_docker(), purge_data=True, claude_config_path=tmp_path / "claude.json"
    )
    assert not (data_root / COMPOSE_NAME).exists(), (
        "the uninstall must proceed: there is simply no volume of ours to purge"
    )
    assert not any(item.kind == "volume" for item in report.removed), (
        "and it must not claim to have removed somebody else's volume"
    )

    # Control: the guard still fires on the case it exists for, a caller widening the removal.
    other = _install(tmp_path / "second")
    widened = plan_uninstall(data_root=other, runner=_docker(), purge_data=False)
    with pytest.raises(UninstallRefusal):
        execute(widened, runner=_docker(), purge_data=True, claude_config_path=tmp_path / "c.json")


def test_the_handoff_file_is_resolved_through_the_writers_own_path(tmp_path: Path, monkeypatch) -> None:
    """⛔ **FIX-10.** Every profile test injected `profile_path=`, so the DEFAULT branch was unpinned.

    An auditor pointed that default back at `data_root / "runtime.json"` — the original bug, where
    the uninstaller looked for the handoff file in the wrong directory and left it behind on every
    Windows uninstall — and 32 tests stayed green. The docstring on the fix says the old unit test
    "fabricated the file under `data_root`, which is precisely why the mismatch was invisible"; the
    replacement inherited the same blind spot from the other side.

    Injecting the writer's own function rather than a path is what pins the delegation without
    touching the developer's real %APPDATA%.
    """
    import recall.desktop.profiles as profiles

    data_root = _install(tmp_path)
    real_profile = tmp_path / "appdata" / "RE-call" / "runtime.json"
    real_profile.parent.mkdir(parents=True)
    real_profile.write_text(
        json.dumps({"compose_file": str(data_root / COMPOSE_NAME)}), encoding="utf-8"
    )
    # A decoy in the place the ORIGINAL bug looked, which must never be chosen.
    (data_root / "runtime.json").write_text(
        json.dumps({"compose_file": str(data_root / COMPOSE_NAME)}), encoding="utf-8"
    )
    monkeypatch.setattr(profiles, "profile_path", lambda: real_profile)

    plan = plan_uninstall(data_root=data_root, runner=_docker())
    planned = [item.name for item in plan.removing("file")]

    assert str(real_profile) in planned, (
        "the handoff file must be resolved through the writer's own path function, which is the "
        "only thing that knows it lives in the user config directory"
    )
    assert str(data_root / "runtime.json") not in planned, (
        "and never guessed at relative to data_root, which is where it was looked for when every "
        "Windows uninstall left it behind"
    )


def test_an_unwritable_backup_refuses_rather_than_writing_it_unsafely(tmp_path: Path, monkeypatch) -> None:
    """⛔ **FIX-19b. The fallback defeated the protection instead of degrading from it.**

    The backup is created with `O_EXCL | O_NOFOLLOW` at 0600 because `~/.claude.json` carries every
    MCP bearer token and its backup name is predictable to the second. But the `except OSError`
    beneath that used to fall back to `write_text`, which FOLLOWS a symlink — and `O_NOFOLLOW`
    reports a planted link by raising `ELOOP`, which is an `OSError`. So the single case the
    exclusive open exists to catch routed straight into the unsafe write. A reviewer found it in
    the error path of the fix itself.

    Refusing is the right degradation: the caller records a failure, the files a retry needs are
    kept, and the config the backup would have protected is left untouched.
    """
    project_root = tmp_path / "work"
    project_root.mkdir()
    data_root = _install(tmp_path, project_root=project_root)
    config_path = tmp_path / "claude.json"
    before = json.dumps(
        {
            "projects": {
                str(project_root): {
                    "mcpServers": {"recall": {"command": "python", "cwd": str(project_root)}}
                }
            }
        }
    )
    config_path.write_text(before, encoding="utf-8")

    real_open = os.open

    def refuse_backup(path, flags, mode=0o777, **kwargs):
        if ".recall-backup-" in str(path):
            raise OSError(40, "Too many levels of symbolic links")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", refuse_backup)

    plan = plan_uninstall(data_root=data_root, claude_config_path=config_path, runner=_docker())
    report = execute(plan, runner=_docker(), claude_config_path=config_path)

    assert config_path.read_text(encoding="utf-8") == before, (
        "the config must be untouched: rewriting it without a backup is the one thing the backup "
        "exists to prevent, and writing the backup through a symlink is worse than not writing it"
    )
    assert any(item.kind == "registration" for item, _reason in report.failed), (
        "and the refusal must be REPORTED, not swallowed"
    )
    reason = next(r for item, r in report.failed if item.kind == "registration")
    assert "backup" in reason.lower() and "nothing was changed" in reason.lower(), reason

    assert not [
        path for path in config_path.parent.iterdir() if ".recall-backup-" in path.name
    ], "and no partial or unsafe backup is left behind"
