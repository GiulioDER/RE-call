"""Add a project to an install that already exists.

The wizard's install path builds a whole stack from nothing. This is the other operation the
desktop UI needs: the user types a name into "+ Add project" and expects a scope they can drop
files into. Until now that produced a combo-box entry and nothing else, so the name pointed at a
tenant with no service, no server block and no corpus, and the failure surfaced later as a message
about a "tenant scope".

**What this deliberately does NOT do is build or calibrate the corpora.** Those take minutes and
need a progress UI, and the user has just asked for a place to put files, not for an index of files
they have not chosen yet. So a new project arrives empty and uncertified, which the UI can already
represent: its ingest page fills it and its calibration page certifies it. An "add" that blocked for
ten minutes before the user could do either would be a worse answer to the question they asked.

The consequence to be honest about: a project added here is **uncalibrated**, so search against it
is not trusted until the user calibrates. That is the same state a freshly installed corpus is in
before the wizard's own calibration stage runs, and it is reported rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from recall.wizard.corpora import CorpusKind, DEFAULT_PROJECT, tenant_for
from recall.wizard.stack import (
    add_tenant_services,
    container_dsn,
    existing_port,
    existing_tenants,
    host_dsn,
)

#: The kinds every project gets, matching what `corpora.default_plan` builds for the install path.
#: A project that arrived with fewer would be a second class of project, and the difference would
#: only show up when something went looking for the missing corpus.
_KINDS: tuple[CorpusKind, ...] = ("docs", "code", "memory")


@dataclass(frozen=True)
class AddedProject:
    """What actually happened, so the caller can report it rather than assume it."""

    project: str
    tenants: tuple[str, ...]
    #: Tenants that already had a service. Non-empty means this was a partial or repeat add, which
    #: is not an error but must not be reported as "created".
    already_present: tuple[str, ...]
    compose_path: Path
    port: int

    @property
    def created_anything(self) -> bool:
        return bool(self.tenants)


class ProjectRefusal(ValueError):
    """The project cannot be added, with a reason a user can act on."""


def compose_path_for(data_root: Path) -> Path:
    return data_root / "docker-compose.yml"


def add_project(
    data_root: Path,
    project: str,
    *,
    embedder: str | None = None,
    image: str | None = None,
) -> AddedProject:
    """Provision `project`'s three tenants in the stack at `data_root`.

    Refuses rather than half-succeeding. Every refusal names what the user should do instead,
    because this is reached from a text box in a GUI and a traceback there is useless.
    """
    if project.strip() == DEFAULT_PROJECT:
        raise ProjectRefusal(
            f"{DEFAULT_PROJECT!r} is the name of the project the installer creates; choose another"
        )
    # Validated by the wizard's own rule rather than a second one written here. The UI used to
    # accept names `tenant_for` refuses — anything with a space or a punctuation mark, anything
    # ending in a corpus kind — and then told the user to run the wizard, which would have refused
    # them. Borrowing the rule means the dialog gives the wizard's reason, at the point of typing.
    try:
        tenants = tuple(tenant_for(project, kind) for kind in _KINDS)
    except ValueError as exc:
        raise ProjectRefusal(str(exc)) from exc

    compose_path = compose_path_for(data_root)
    port = existing_port(compose_path)
    if port is None:
        raise ProjectRefusal(
            f"no recall stack at {data_root}: {compose_path} is missing or does not publish a "
            "database port. Run the installer for this location before adding a project to it."
        )

    if embedder is None:
        embedder = stack_embedder(compose_path)
        if embedder is None:
            raise ProjectRefusal(
                f"cannot read the embedder from {compose_path}, and a project added with a "
                "different one would build a corpus this database cannot hold. Pass it explicitly "
                "if you are certain which the stack uses."
            )

    present = set(existing_tenants(compose_path))
    already = tuple(sorted(t for t in tenants if t in present))

    # The trust posture a NEW project genuinely has. It has no generation, no calibration and no
    # documents, so `production` would be a claim about a corpus that does not exist yet; a strict
    # server would refuse every query and the user would read that as the add having failed.
    # The wizard's wiring stage rewrites this per tenant once something certifies.
    env = {
        tenant: {
            "RECALL_DSN": container_dsn(),
            "RECALL_EMBEDDER": embedder,
            "RECALL_TENANT": tenant,
            "RECALL_TRUST_MODE": "development",
        }
        for tenant in tenants
        if tenant not in present
    }

    if not env:
        added: tuple[str, ...] = ()
    elif image is None:
        added = add_tenant_services(compose_path, env)
    else:
        added = add_tenant_services(compose_path, env, image=image)

    return AddedProject(
        project=project.strip(),
        tenants=added,
        already_present=already,
        compose_path=compose_path,
        port=port,
    )


def stack_embedder(compose_path: Path) -> str | None:
    """The embedder the existing services run, or None if it cannot be read.

    A new corpus MUST use the same one. The embedder is welded to the table's vector dimension, so
    a project added with a different embedder would build a corpus the shared database cannot hold,
    and the failure would arrive at ingest time rather than here. Inheriting it is therefore the
    correct default, not a convenience: the caller should not be able to choose wrong.
    """
    try:
        document = json.loads(compose_path.read_text(encoding="utf-8"))
        services = document["services"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(services, dict):
        return None
    for name, service in services.items():
        if name == "db" or not isinstance(service, dict):
            continue
        environment = service.get("environment")
        if isinstance(environment, dict) and environment.get("RECALL_EMBEDDER"):
            return str(environment["RECALL_EMBEDDER"])
    return None


def host_dsn_for(data_root: Path) -> str | None:
    """The DSN a host process uses to reach this install's database, or None if there is no stack."""
    port = existing_port(compose_path_for(data_root))
    return None if port is None else host_dsn(port)


__all__ = [
    "AddedProject",
    "ProjectRefusal",
    "add_project",
    "compose_path_for",
    "host_dsn_for",
]
