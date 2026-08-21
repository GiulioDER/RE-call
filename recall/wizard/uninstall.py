"""Take an install apart, and refuse to take anything else apart with it.

⛔ **The hazard this module exists to avoid is deleting the user's own documents.** The installer
SUGGESTS the corpus roots underneath the data folder — `<data_root>/docs`, `<data_root>/code`,
`<data_root>/memory` — so anyone who accepted the defaults has their notes, their source and their
agent memory inside the directory an uninstaller would be most tempted to remove wholesale. One
`rmtree(data_root)` and the thing recall was indexing is gone along with the index.

So this never removes a directory. It removes **the specific files the installer wrote**, by name,
and reports the corpus roots as kept, with their paths, so the person can see they survived.

Three more rules, each a consequence of something already learned here:

**Nothing is removed that this install did not create.** The containers come from the compose
project name recorded in the stack file, not from a name pattern — a pattern would match a second
install, or somebody else's stack, and `docker rm -f` against the wrong container is not
recoverable. The MCP registrations are removed only where the entry's `cwd` marks it as written by
this project, which is the same test `register_local_scope` uses to decide whether it may overwrite
one. An entry without that marker was written by hand or by another tool, and is left alone.

**Planning and doing are separate calls.** `plan_uninstall` opens nothing and deletes nothing; it
returns what WOULD be removed so a person, a prompt or a window can show it first. An uninstaller
that acts before it can be read is one you can only evaluate afterwards.

**The database volume is opt-in, separately from the files.** It holds the built indexes. Those are
reproducible by re-indexing and expensive to rebuild, which makes them exactly the thing somebody
reinstalling next week wants kept and somebody reclaiming disk wants gone. Guessing either way is
worse than asking.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import suppress
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from recall.wizard.stack import COMPOSE_NAME, DB_VOLUME, DOCKERFILE_NAME

__all__ = [
    "Removable",
    "UninstallPlan",
    "UninstallReport",
    "UninstallRefusal",
    "execute",
    "plan_uninstall",
]

RemovableKind = Literal["container", "volume", "image", "file", "registration"]

#: The files the installer writes into the data folder. Named individually, never globbed: a glob
#: would grow to cover whatever a future version drops there, including things a user put there
#: themselves, and the point of this list is that it is a list.
_INSTALLER_FILES = (
    COMPOSE_NAME,
    DOCKERFILE_NAME,
    "wizard.json",
    "wizard.state.json",
)

# ⛔ **`runtime.json` is NOT in the list above, because it is not under `data_root`.** The
# installer writes it to the user's config directory (`%APPDATA%/RE-call/` on Windows, which is
# the platform this targets), and `recall/wizard/headless.py` says so outright: "written to the
# user's config directory, outside both data_root and project_root". Listing it as a
# `data_root`-relative name meant the entry NEVER matched on Windows, so every uninstall left the
# desktop app holding a profile pointing at a compose file that had just been deleted. The unit
# test fabricated the file under `data_root`, which is precisely why the mismatch was invisible.
#
# It is planned at its real location instead, and only when it names THIS install.
#
# ⚠️ Plain `#`, not `#:`. Sphinx's attribute-doc marker documents the object that FOLLOWS it, and
# this block sits after `_INSTALLER_FILES` closes — so as `#:` it rendered as the documentation for
# `UninstallRefusal`, which has its own docstring.


class UninstallRefusal(RuntimeError):
    """Raised when the install cannot be identified. Carries a sentence a person can act on."""


@dataclass(frozen=True)
class Removable:
    """One thing that would be removed, or one thing deliberately kept."""

    kind: RemovableKind
    name: str
    #: Why it is in the list, or why it is being kept. Shown to the user verbatim.
    detail: str = ""
    #: `False` means "recognised, and deliberately NOT removed". Kept in the plan rather than
    #: filtered out, because the corpus roots surviving is the fact the user most needs to see.
    removing: bool = True


@dataclass(frozen=True)
class UninstallPlan:
    """What an uninstall would do, computed without doing any of it."""

    data_root: Path
    project_name: str
    items: tuple[Removable, ...]
    #: The project whose MCP registrations this install wrote, if the config named one.
    project_root: Path | None = None
    #: Whether this plan was built to purge the index volumes. Recorded rather than inferred from
    #: the item list: a plan that purges but finds every declared volume `external` legitimately
    #: lists none, and inferring intent from that emptiness refused the whole uninstall.
    purge_data: bool = False

    def removing(self, kind: RemovableKind | None = None) -> tuple[Removable, ...]:
        return tuple(
            item
            for item in self.items
            if item.removing and (kind is None or item.kind == kind)
        )

    def keeping(self) -> tuple[Removable, ...]:
        return tuple(item for item in self.items if not item.removing)

    def render(self) -> str:
        """The text a person reads before confirming. Removals first, then what survives."""
        lines = [f"Uninstalling the recall install at {self.data_root}", ""]
        by_kind: dict[str, list[Removable]] = {}
        for item in self.removing():
            by_kind.setdefault(item.kind, []).append(item)
        if by_kind:
            lines.append("This will remove:")
            for kind in ("container", "volume", "image", "file", "registration"):
                for item in by_kind.get(kind, ()):
                    suffix = f"  ({item.detail})" if item.detail else ""
                    lines.append(f"  {kind}: {item.name}{suffix}")
        else:
            lines.append("Nothing to remove: this install appears to be gone already.")
        kept = self.keeping()
        if kept:
            lines.extend(["", "This will KEEP:"])
            for item in kept:
                suffix = f"  ({item.detail})" if item.detail else ""
                lines.append(f"  {item.name}{suffix}")
        return "\n".join(lines)


@dataclass
class UninstallReport:
    """What actually happened, per item. Failures are recorded, never raised."""

    removed: list[Removable] = field(default_factory=list)
    failed: list[tuple[Removable, str]] = field(default_factory=list)
    kept: list[Removable] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Removed {len(self.removed)} item(s)."]
        for item, reason in self.failed:
            lines.append(f"  could not remove {item.kind} {item.name}: {reason}")
        for item in self.kept:
            suffix = f"  ({item.detail})" if item.detail else ""
            lines.append(f"  kept {item.name}{suffix}")
        return "\n".join(lines)


def _run(command: Sequence[str], *, timeout: float = 60.0) -> tuple[int, str]:
    """Run a command, returning its status and combined output rather than raising.

    ⚠️ `encoding="utf-8", errors="replace"` for the reason recorded in `recall/wizard/stack.py` and
    `recall/desktop/runtime.py`: `text=True` alone decodes with the platform codec, and an
    undecodable byte from Docker yields rc=0 with `stdout=None` rather than an exception. On this
    path that would read as a successful removal of something still present.
    """
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return completed.returncode, ((completed.stdout or "") + (completed.stderr or "")).strip()


def _compose_project(
    compose_path: Path,
) -> tuple[str, tuple[str, ...], tuple[str, ...], bool]:
    """The project name, service names and DECLARED VOLUMES recorded in the stack file.

    Read from the FILE rather than recomputed from `data_root` and the project name. Recomputing
    would silently disagree with the running stack the moment the derivation changed, and the
    disagreement would present as an uninstall that removed nothing while reporting success.
    """
    try:
        document = json.loads(compose_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UninstallRefusal(
            f"cannot read the stack file at {compose_path}: {exc}. That file names the containers "
            "and the volume, so without it there is no safe way to tell this install's resources "
            "from another's. Remove them by hand, or restore the file."
        ) from exc
    if not isinstance(document, dict):
        raise UninstallRefusal(
            f"the stack file at {compose_path} is not a JSON object, so it names no containers. "
            "Refusing rather than guessing."
        )
    name = document.get("name")
    services = document.get("services")
    if not isinstance(name, str) or not name:
        raise UninstallRefusal(
            f"the stack file at {compose_path} declares no project name, so its containers cannot "
            "be identified without guessing. Refusing rather than matching on a name pattern, "
            "which would also match another install."
        )
    if not isinstance(services, dict):
        raise UninstallRefusal(f"the stack file at {compose_path} declares no services")
    # ⛔ The volumes are READ, not recomputed, for the same reason the project name is. Deriving
    # `f"{project}_{DB_VOLUME}"` from a constant means a stack written by a release with a different
    # volume name yields a name that stack never declared: `docker volume rm` reports "no such
    # volume", `execute` counts that as removed, and the real volume holding the indexes survives
    # while the report says it went. That is the one irreversible item in the plan.
    #
    # ⚠️ **Three cases, not two.** `declares_volumes` distinguishes "this stack has no `volumes:`
    # key at all", where the historical derived name is the only guess available and is right, from
    # "volumes were declared and every one was external", where deriving a name UNDOES the exclusion
    # three lines below: for `{pgdata: {external: true, name: "<project>_pgdata"}}` the derived name
    # is exactly the external volume just skipped, and `docker volume rm` would remove a volume this
    # code had already decided was not ours.
    declared = document.get("volumes")
    declares_volumes = isinstance(declared, dict) and bool(declared)
    volumes: list[str] = []
    if isinstance(declared, dict):
        for key, spec in declared.items():
            if isinstance(spec, dict) and spec.get("external"):
                # An external volume was not created by this stack and is not ours to remove.
                continue
            explicit = spec.get("name") if isinstance(spec, dict) else None
            volumes.append(str(explicit) if explicit else f"{name}_{key}")
    return name, tuple(str(key) for key in services), tuple(volumes), declares_volumes


def _corpus_roots(data_root: Path) -> tuple[tuple[str, Path], ...]:
    """The folders the install INDEXES, read back from the config it was run from.

    These are the user's own documents, code and agent memory. They are listed so the uninstaller
    can say out loud that it is keeping them, which matters most for the common case: the installer
    suggests them underneath `data_root`, so somebody who took the defaults has their notes inside
    the directory being uninstalled.
    """
    config_path = data_root / "wizard.json"
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(document, dict):
        return ()
    roots: list[tuple[str, Path]] = []
    for key in ("docs_root", "code_root", "memory_root"):
        value = document.get(key)
        if isinstance(value, str) and value:
            roots.append((key, Path(value)))
    return tuple(roots)


def _configured_project_root(data_root: Path) -> Path | None:
    """The project whose MCP client config this install wrote into, if any."""
    try:
        document = json.loads((data_root / "wizard.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    value = document.get("project_root")
    return Path(value) if isinstance(value, str) and value else None


def _registrations(project_root: Path, config_path: Path) -> tuple[Removable, ...]:
    """The MCP servers this project registered, and only those.

    ⛔ The `cwd` marker decides. `register_local_scope` writes one into every block it creates and
    refuses to overwrite an entry lacking it, on the grounds that the entry belongs to somebody
    else. Removal has to apply the same test in the same direction: an entry a person wrote by hand
    under a name this wizard also uses must survive an uninstall, or the uninstaller becomes a way
    to lose configuration nobody asked it to touch.
    """
    from recall.wizard.wiring import _written_by_this_project

    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(document, dict):
        return ()
    projects = document.get("projects")
    if not isinstance(projects, dict):
        return ()

    found: list[Removable] = []
    for key, entry in projects.items():
        if not isinstance(entry, dict):
            continue
        servers = entry.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        for name, definition in servers.items():
            if _written_by_this_project(definition, project_root):
                found.append(
                    Removable("registration", str(name), f"in {key}")
                )
            elif isinstance(definition, dict) and "recall" in str(name):
                found.append(
                    Removable(
                        "registration",
                        str(name),
                        f"in {key}, not written by this install",
                        removing=False,
                    )
                )
    return tuple(found)


#: How the desktop app's handoff file is described in the plan. A constant because `execute`
#: matches on it to decide what to keep when the teardown fails, and the profile lives outside
#: `data_root` so it cannot be recognised by filename alone the way the other two can.
_PROFILE_DETAIL = "the desktop app's handoff file"


def _profile_for(data_root: Path, profile_path: Path | None = None) -> Path | None:
    """The desktop runtime profile, but only when it points at THIS install.

    Lives in the user's config directory rather than under `data_root`, so it has to be resolved
    through the writer's own path function rather than assumed. Guarded by whose stack it names: a
    second install's profile is not ours to remove, and there is exactly one profile file for the
    machine.

    ⛔ **`profile_path` is injectable for the same reason `claude_config_path` is.** Without the
    seam, `plan_uninstall` — documented as safe to call on a whim, opening nothing and deleting
    nothing — read the developer's REAL `%APPDATA%/RE-call/runtime.json` on every test run, and the
    deletion of a machine-global file had no way to be exercised at all. That is how this file's
    predecessor bug survived: the only test touching it fabricated the file in the wrong directory.
    """
    if profile_path is not None:
        resolved = profile_path
    else:
        try:
            from recall.desktop.profiles import profile_path as writers_own_path
        except Exception:  # noqa: BLE001 - no desktop extra means no profile to remove
            return None
        try:
            resolved = writers_own_path()
        except Exception:  # noqa: BLE001 - an unresolvable config dir names no profile
            return None
    try:
        path = resolved
        if not path.exists():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    compose = document.get("compose_file")
    if not isinstance(compose, str) or not compose:
        return None
    try:
        return path if Path(compose).resolve().parent == data_root.resolve() else None
    except (OSError, ValueError):  # pragma: no cover - an unresolvable path names no install
        return None


def plan_uninstall(
    *,
    data_root: Path,
    purge_data: bool = False,
    claude_config_path: Path | None = None,
    profile_path: Path | None = None,
    runner: Callable[[Sequence[str]], tuple[int, str]] | None = None,
) -> UninstallPlan:
    """Work out what removing this install would take away. Opens nothing, deletes nothing.

    ⚠️ **`purge_data` is taken here, not only by `execute`, so the rendered plan tells the truth.**
    Without it the volume appeared under "This will remove" carrying a note saying it would only be
    removed with a flag — a confirmation screen contradicting itself about the one item that holds
    data. Pass the same value to both calls; the plan is what the person agreed to.

    `runner` is injected so the plan can be computed and tested without a Docker daemon; the default
    asks the real one which of the stack's containers exist.
    """
    run = runner or (lambda command: _run(command))
    compose_path = data_root / COMPOSE_NAME
    if not compose_path.exists():
        raise UninstallRefusal(
            f"no recall stack at {data_root}: {COMPOSE_NAME} is not there. Point the uninstaller at "
            "the data folder chosen during installation; it is recorded in that install's "
            "wizard.json."
        )
    project_name, services, declared_volumes, declares_volumes = _compose_project(compose_path)

    items: list[Removable] = []

    # Containers, asked of Docker by compose PROJECT LABEL rather than by name pattern.
    status, output = run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--format",
            "{{.Names}}",
        ]
    )
    if status == 0:
        for line in output.splitlines():
            name = line.strip()
            if name:
                items.append(Removable("container", name, f"compose project {project_name}"))
    else:
        # ⛔ `removing=False`. Reported as UNKNOWN rather than as a removal: a glob under "This
        # will remove" reads as a name-pattern teardown, which is the one thing this module refuses
        # to do, and `execute` would otherwise report the pseudo-name as removed while never naming
        # the real containers.
        items.append(
            Removable(
                "container",
                f"{project_name}-*",
                "docker could not be queried, so the containers could not be listed; "
                "compose will still bring the stack down",
                removing=False,
            )
        )

    # The derived name is the fallback for a LEGACY stack only. See `_compose_project`: a stack
    # that declared volumes and had them all excluded as external must not have one guessed back.
    fallback = () if declares_volumes else (f"{project_name}_{DB_VOLUME}",)
    for volume in declared_volumes or fallback:
        items.append(
            Removable(
                "volume",
                volume,
                "the built indexes"
                if purge_data
                else "the built indexes; kept unless you pass --purge-data",
                removing=purge_data,
            )
        )

    for filename in _INSTALLER_FILES:
        path = data_root / filename
        if path.exists():
            items.append(Removable("file", str(path), "written by the installer"))

    profile = _profile_for(data_root, profile_path)
    if profile is not None:
        items.append(Removable("file", str(profile), _PROFILE_DETAIL))

    project_root = _configured_project_root(data_root)
    if project_root is not None:
        config = claude_config_path or _default_claude_config()
        items.extend(_registrations(project_root, config))

    # ⛔ Listed as KEPT, loudly. These are the user's own files, and on a default install they sit
    # inside `data_root`, which is the directory somebody would otherwise expect to be deleted.
    for key, root in _corpus_roots(data_root):
        items.append(
            Removable("file", str(root), f"{key}: your own content, never removed", removing=False)
        )

    return UninstallPlan(
        data_root=data_root,
        project_name=project_name,
        items=tuple(items),
        project_root=project_root,
        purge_data=purge_data,
    )


def execute(
    plan: UninstallPlan,
    *,
    purge_data: bool = False,
    claude_config_path: Path | None = None,
    runner: Callable[[Sequence[str]], tuple[int, str]] | None = None,
) -> UninstallReport:
    """Carry out `plan`. Every failure is recorded and the rest continues.

    ⚠️ **A partial uninstall must not abort partway.** Stopping at the first error leaves a mixture
    nobody can reason about: containers gone, files present, registrations live. Removing what can
    be removed and REPORTING the rest leaves the person one list to finish by hand.

    `purge_data` also removes the database volume, which holds the built indexes. Off by default:
    they are reproducible by re-indexing and expensive to rebuild, so the person reinstalling next
    week and the person reclaiming disk want opposite things and neither should be guessed.

    ⛔ **Pass the same `purge_data` to `plan_uninstall`.** The plan is what the person read and
    agreed to; executing a wider removal than the one displayed is the failure this whole
    plan-then-do split exists to prevent. Mismatched values are refused rather than reconciled.

    🔁 **The guard compares the plan's own INTENT, not whether it listed any volume.** It used to
    test `purge_data != bool(plan.removing("volume"))`, and the external-volume exclusion made that
    wrong: a stack whose declared volumes are all `external` has nothing this uninstaller may
    remove, so a `--purge-data` plan legitimately lists zero volumes and the guard refused the
    entire uninstall. `recall/desktop/main.py` calls this outside its handler, so in the frozen GUI
    that surfaced as a traceback after the person had already confirmed the plan. Three auditors
    reproduced it. An empty list is not disagreement; a different answer to "were we asked to
    purge?" is.
    """
    if purge_data != plan.purge_data:
        raise UninstallRefusal(
            "the plan shown and the removal asked for disagree about the index volume. The plan is "
            "what the person agreed to, so pass the same purge_data to plan_uninstall and execute "
            "rather than widening the removal after it was displayed."
        )
    run = runner or (lambda command: _run(command))
    report = UninstallReport()

    compose_path = plan.data_root / COMPOSE_NAME
    # ⛔ **Bound before the branch, because it is READ after it.** This was assigned only inside
    # the `exists()` branch and read unconditionally below, so `execute` raised `UnboundLocalError`
    # when the stack file was gone — which is reachable as a TOCTOU (plan, then a confirmation
    # dialog with no time bound, then execute) and makes a repeat run non-idempotent, since the
    # first successful run deletes that very file. Three auditors reproduced it, against a docstring
    # promising that a partial uninstall never aborts partway.
    compose_failed = False
    if compose_path.exists():
        command = [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "-p",
            plan.project_name,
            "down",
            "--remove-orphans",
        ]
        if purge_data:
            # `-v` removes the named volume compose declares. Only here: without `--purge-data`
            # the indexes are kept deliberately.
            command.append("-v")
        status, output = run(command)
        # ⛔ **The teardown status is captured HERE, not inferred from the container items.** It used
        # to be recorded only by looping over `plan.removing("container")`, and the sibling fix that
        # stopped reporting an unqueryable docker as a removal sets `removing=False` on exactly that
        # item — so the two fixes cancelled. Reproduced: with the daemon unreachable at plan time AND
        # at execute time, a failing `compose down` produced `failed: []`, the report printed
        # "Removed 3 item(s).", and the stack file naming the still-running containers was deleted.
        # Three auditors found it independently, which is what a defect that lives BETWEEN two
        # correct-looking fixes looks like.
        compose_failed = status != 0
        for item in plan.removing("container"):
            if status == 0:
                report.removed.append(item)
            else:
                report.failed.append((item, output[:200] or "docker compose down failed"))
        if compose_failed and not plan.removing("container"):
            # Nothing was listed to attach the failure to, so the stack itself carries it. Without
            # this the error never reaches the user at all.
            report.failed.append(
                (
                    Removable(
                        "container",
                        plan.project_name,
                        "the stack's containers",
                        removing=False,
                    ),
                    output[:200] or "docker compose down failed",
                )
            )

    for item in plan.removing("volume"):
        # `compose down -v` above already removed it; this is the belt-and-braces path for a volume
        # left behind by an earlier partial removal. A missing volume is a success, not a failure.
        status, output = run(["docker", "volume", "rm", item.name])
        if status == 0 or "no such volume" in output.lower():
            report.removed.append(item)
        else:
            report.failed.append((item, output[:200]))

    if plan.project_root is not None:
        names = tuple(item.name for item in plan.removing("registration"))
        if names:
            failure = _unregister(
                names, plan.project_root, claude_config_path or _default_claude_config()
            )
            for item in plan.removing("registration"):
                if failure is None:
                    report.removed.append(item)
                else:
                    report.failed.append((item, failure))

    # ⛔ **Two files are kept, and each for its own reason.** The stack file is the only thing that
    # identifies this install's containers and volumes, and `plan_uninstall` refuses without it —
    # so deleting it after a failed teardown leaves resources nothing can name, and the refusal
    # message then tells the user to restore a file this function just removed. `wizard.json` is
    # kept because the retry needs it too: `_configured_project_root` and `_corpus_roots` both read
    # it, so without it the second attempt cannot find the MCP registrations to unwind or say which
    # folders it is keeping.
    #
    # 🔁 **The desktop profile is kept too, and the first version of this reasoned its way to the
    # opposite answer.** That argument was: the profile points at an install the user asked to
    # remove, so leaving it behind leaves the app pointing at something that should be gone. It is
    # wrong about the state it is reasoning about. If the teardown FAILED, the stack is still
    # running — that is what failure means here — and the compose file naming it is being kept for
    # exactly that reason. Deleting the app's only pointer to a stack that is still up, while
    # keeping the file that names it, is the same defect as deleting the stack file itself: it
    # strands a live resource behind a tool that can no longer reach it.
    #
    # The retry removes all three together once the teardown succeeds, which is the state in which
    # removing any of them is correct.
    # 🔁 **Any failure at all, not only a container or volume failure.** This filtered on
    # `kind in {"container", "volume"}`, so a failed MCP unregistration deleted the stack file,
    # `wizard.json` and the handoff file anyway — and then the retry was impossible, because
    # `plan_uninstall` refuses without the stack file and `project_root`, the only thing that can
    # find the `~/.claude.json` entry just left behind, lived in `wizard.json`. The comment below
    # already claimed these files are kept "so the uninstall can be retried"; the predicate did not
    # cover the failure that most needs a retry.
    teardown_failed = compose_failed or bool(report.failed)
    for item in plan.removing("file"):
        path = Path(item.name)
        keep_for_retry = path.name in {COMPOSE_NAME, "wizard.json"} or item.detail == _PROFILE_DETAIL
        if teardown_failed and keep_for_retry:
            report.kept.append(
                Removable(
                    item.kind,
                    item.name,
                    "kept so the uninstall can be retried: something above could not be removed",
                    removing=False,
                )
            )
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            report.failed.append((item, str(exc)))
        else:
            report.removed.append(item)

    report.kept.extend(plan.keeping())
    return report


def _unregister(names: Sequence[str], project_root: Path, config_path: Path) -> str | None:
    """Remove the named servers from every project entry that this install wrote them into.

    Returns a reason on failure and `None` on success, rather than raising: the caller is midway
    through an uninstall and one unwritable config must not discard the report of what already went.

    ⚠️ **Rewritten in full and replaced atomically**, matching how the client's config is written
    everywhere else here. A partial write to `~/.claude.json` costs the user every project entry it
    holds, not just recall's.
    """
    from recall.wizard.wiring import _written_by_this_project

    try:
        original = config_path.read_text(encoding="utf-8")
        document = json.loads(original)
    except (OSError, ValueError) as exc:
        return f"cannot read {config_path}: {exc}"
    if not isinstance(document, dict):
        return f"{config_path} is not a JSON object; refusing to rewrite it"
    projects = document.get("projects")
    if not isinstance(projects, dict):
        return None

    wanted = set(names)
    for entry in projects.values():
        if not isinstance(entry, dict):
            continue
        servers = entry.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        for name in list(servers):
            if name in wanted and _written_by_this_project(servers[name], project_root):
                del servers[name]

    temporary = config_path.with_name(config_path.name + ".tmp")
    try:
        # ⛔ **A backup first, and it must NOT clobber the install-time one.** `register_local_scope`
        # writes `<name>.recall-backup` before its own atomic replace, so copying that fixed name
        # here overwrote the only copy of the file from before recall ever touched it — with the
        # already-rewritten content, at the moment it is most likely to be wanted. `claude_code.py`
        # already had the right pattern and this now matches it: a timestamped name, never reused.
        #
        # 🔁 This used to say "`copy2` and not `write_text`", and the `os.open` rewrite below
        # falsified it: the backup is now written through a descriptor opened at 0600, so the mode
        # is set at CREATION rather than narrowed after the credential bytes are already on disk,
        # which is stronger than what `copy2` gave. Two consequences worth stating rather than
        # discovering: the copy is no longer byte-identical, because it is written with `\n` and a
        # CRLF original is normalised, and `copy2`'s mtime preservation is gone. The content is
        # exactly the text the rewrite below is derived from.
        backup = _backup_path(config_path)
        try:
            # ⛔ **`O_EXCL | O_NOFOLLOW` at 0600, not `copy2` after an `exists()` test.** The name is
            # predictable to the second, and `copy2` opens the destination with plain `open(...,
            # "wb")`: it follows a symlink planted in the window between the check and the copy, and
            # it creates at the umask default before any mode is narrowed. This file holds every
            # bearer token the MCP client tracks, so it is created private rather than made private.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(backup, flags, 0o600)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(original)
            except BaseException:
                with suppress(OSError):
                    os.unlink(backup)
                raise
        except OSError as exc:
            # ⛔ **No fallback that FOLLOWS a symlink.** This used to fall back to
            # `backup.write_text(...)`, and that defeated the protection above rather than degrading
            # from it: `O_NOFOLLOW` reports a planted symlink by raising `ELOOP`, which is an
            # `OSError`, so the one case the exclusive open exists to catch routed straight into a
            # write that follows the link — putting every bearer token in this file wherever the
            # link pointed. A reviewer found this in the error path of my own fix.
            #
            # Refusing is the right degradation here, and it is what the caller already understands:
            # `_unregister` returns a reason, `execute` records it as a failed registration, and
            # (since the teardown-failure fix) the files a retry needs are then kept. The rewrite
            # does NOT proceed, so the config the backup would have protected is untouched.
            return (
                f"cannot create a private backup of {config_path}: {type(exc).__name__}: {exc}. "
                "Refusing to rewrite it, because that file carries access tokens and the backup is "
                "the only way back. Nothing was changed."
            )
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        # `replace` installs a fresh inode at the umask default, discarding a 0600 the user or the
        # client may have set on a file that carries bearer tokens. Copy the mode across first.
        with suppress(OSError, NotImplementedError):
            shutil.copymode(config_path, temporary)
        temporary.replace(config_path)
    except OSError as exc:
        return f"cannot write {config_path}: {exc}"
    return None


def _backup_path(config_path: Path) -> Path:
    """A backup name that is never reused, so an earlier backup is never overwritten.

    ⛔ The fixed `<name>.recall-backup` that `register_local_scope` writes is the copy of the file
    from BEFORE recall was ever registered. Reusing that name here destroyed it, at the exact moment
    somebody undoing an install would want it. The counter, rather than a timestamp alone, is
    because two uninstalls inside the same second are a thing that happens in tests and in scripts.
    """
    import time

    stamp = int(time.time())
    for suffix in range(64):
        tail = f".recall-backup-{stamp}" + (f".{suffix}" if suffix else "")
        candidate = config_path.with_name(config_path.name + tail)
        if not candidate.exists():
            return candidate
    return config_path.with_name(f"{config_path.name}.recall-backup-{stamp}.last")


def _default_claude_config() -> Path:
    from recall.wizard.wiring import claude_config_path

    return claude_config_path()
