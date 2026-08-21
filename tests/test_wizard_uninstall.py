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

    backup = config_path.with_name(config_path.name + ".recall-backup")
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
