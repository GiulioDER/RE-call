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
import subprocess
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
    "runtime.json",
)


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


def _compose_project(compose_path: Path) -> tuple[str, tuple[str, ...]]:
    """The project name and service names recorded in the stack file.

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
    return name, tuple(str(key) for key in services)


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


def plan_uninstall(
    *,
    data_root: Path,
    purge_data: bool = False,
    claude_config_path: Path | None = None,
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
    project_name, services = _compose_project(compose_path)

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
        items.append(
            Removable(
                "container",
                f"{project_name}-*",
                "docker could not be queried; the stack will be brought down by compose anyway",
            )
        )

    items.append(
        Removable(
            "volume",
            f"{project_name}_{DB_VOLUME}",
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
    """
    if purge_data != bool(plan.removing("volume")):
        raise UninstallRefusal(
            "the plan shown and the removal asked for disagree about the index volume. The plan is "
            "what the person agreed to, so pass the same purge_data to plan_uninstall and execute "
            "rather than widening the removal after it was displayed."
        )
    run = runner or (lambda command: _run(command))
    report = UninstallReport()

    compose_path = plan.data_root / COMPOSE_NAME
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
        for item in plan.removing("container"):
            if status == 0:
                report.removed.append(item)
            else:
                report.failed.append((item, output[:200] or "docker compose down failed"))

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

    for item in plan.removing("file"):
        path = Path(item.name)
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
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"cannot read {config_path}: {exc}"
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
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(config_path)
    except OSError as exc:
        return f"cannot write {config_path}: {exc}"
    return None


def _default_claude_config() -> Path:
    from recall.wizard.wiring import claude_config_path

    return claude_config_path()
