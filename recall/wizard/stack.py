"""The one database the wizard and the desktop UI both use, at a location the user chose.

Until this existed there were two. `docker-compose.desktop.yml` published no ports at all, so its
Postgres was reachable only inside the compose network at the hostname `db`, while the wizard and
every MCP server it registers are HOST processes reaching a host address. Two stores,
one user, and the failure is silent: files added through the UI build generations the agent cannot
see, corpora the wizard builds never appear on the UI's calibration page, and both surfaces report
themselves healthy because each is telling the truth about a different world.

One store, two addresses. Host processes use `127.0.0.1:<published>`; services inside the compose
network use `db:5432`. Those are the same database, and that is the whole point of publishing a
port.

**`data_root` holds the stack; the DATABASE lives on a named volume.** The user's chosen directory
holds the compose file, the generated Dockerfile and `.env` — everything they need to inspect or
stop their own install — but not the index.

🔁 **This paragraph asserted the opposite until 2026-08-19, and the measurement it carried is what
made the wrong half read as authoritative.** It said the database was bind mounted into `data_root`
"because a person installing this on Windows should be able to say where their index goes, find it,
back it up and delete it", and cited: PostgreSQL 18.4 with pgvector 0.8.6 initialises on a
bind-mounted Windows path and serves normally, writing ~47 MB there for an empty database.

That measurement was real and is retained deliberately, because it is why the design survived as
long as it did: a bind mount initialises and serves. What it does not survive is sustained WAL
writing, where Docker Desktop's filesystem passthrough returns EINTR and PostgreSQL treats it as
fatal (`could not write to file "pg_wal/xlogtemp...": Interrupted system call`). The failure is
INTERMITTENT — one full install certified both corpora, the next died mid-run — which makes it a
corruption risk rather than an availability one. See `compose_document` for the detail.

The cost is real and is not hidden: the user's index is no longer under the folder they chose, so
backup and uninstall have to name the volume instead.

⚠️ **The mount point is `/var/lib/postgresql`, NOT `/var/lib/postgresql/data`.** pg18 refuses the
latter outright and says why: it wants a single mount one level up so `pg_upgrade --link` can work
across a version bump without crossing a mount boundary. Measured the same way, by watching the
container exit 1 on the wrong path.

**The document is emitted as JSON.** YAML 1.2 is a JSON superset and `docker compose` parses it
happily, verified including a bind mount whose Windows path contains a space. That buys correct
quoting from `json.dumps` for free and avoids depending on PyYAML, which is importable in this
environment but is NOT a declared dependency of this project and so cannot be relied on.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "StackSpec",
    "compose_document",
    "choose_port",
    "container_dsn",
    "existing_port",
    "host_dsn",
    "wait_for_database",
    "write_compose",
]

#: The image the database runs. Pinned to a major version because the data directory is not
#: portable across them: a bump needs `pg_upgrade`, which is exactly what the mount point above is
#: chosen to keep possible.
DB_IMAGE = "pgvector/pgvector:pg18"

#: Where the database lives. A named volume inside the Docker VM; see `compose_document` for the
#: measured reason it is not a bind mount into the user's chosen directory.
DB_VOLUME = "pgdata"


def _default_image() -> str:
    """The tag the generated stack builds and runs, scoped to the recall version.

    ⚠️ **It used to be the fixed tag `recall-desktop-recall:local`, and that silently ran the wrong
    image.** `image:` alongside `build:` makes Compose REUSE an existing tag rather than build it,
    so on any machine that had ever built the hand-written `docker-compose.desktop.yml` the
    generated stack started that image instead. Measured: a container running an image built two
    days earlier, on Python 3.14 where this Dockerfile pins 3.13, with no `documents` extra and
    with the post-install assertion never executed. Nothing errored.

    Scoping the tag to the version fixes both halves: a machine carrying the legacy tag no longer
    matches, and an upgrade rebuilds instead of silently serving the previous version's image.
    """
    from recall import __version__

    return f"recall-wizard:{__version__}"


_DEFAULT_IMAGE = _default_image()

#: The generated stack's file name, under the user's chosen `data_root`.
#:
#: ⚠️ **Defined here, beside the writer, because a second definition made the whole "+ Add project"
#: feature inert.** `recall/wizard/projects.py` had its own literal `"docker-compose.yml"`, so it
#: looked for a file the installer never writes and refused every add with "no recall stack at
#: {data_root} ... Run the installer for this location" — told to somebody who had just run it.
#: Five auditors found it independently; no test could, because the fixtures built their stacks
#: through the same wrong helper, so the suite agreed with itself.
COMPOSE_NAME = "docker-compose.recall.yml"

#: Written beside the compose file, and referenced by every tenant service's `build:` stanza.
DOCKERFILE_NAME = "Dockerfile"

#: The extras the served image needs. `documents` is what makes LibreOffice worth its layer: the
#: desktop UI classifies .docx/.xlsx/.pptx as DOCUMENTS, so an image without it accepts those files
#: and extracts nothing from them, which is a silent failure rather than a missing feature.
_IMAGE_EXTRAS = "mcp,fastembed,documents"

#: The extras available before `documents` was published. Used when the pinned version predates it.
_LEGACY_IMAGE_EXTRAS = "mcp,fastembed"

#: The first release that publishes the `documents` extra. Pinning it on anything older produces a
#: Dockerfile that CANNOT build, because the post-install import assertion below fails on the
#: packages the extra would have brought.
#:
#: ⛔ This exists because making the Dockerfile follow the INHERITED image tag (so a stack keeps
#: serving the recall it was built with) turned a silent wrong-image bug into an unbuildable one:
#: adding a project to a 0.9.1 stack wrote `recall-rag[...,documents]==0.9.1`, and `documents`
#: first shipped in 0.9.6.
_DOCUMENTS_EXTRA_SINCE = (0, 9, 6)

#: Where pg18 wants its single mount. See the module docstring; the wrong path exits 1.
DB_MOUNT = "/var/lib/postgresql"

#: The port the database listens on INSIDE the compose network. Not the published one.
DB_INTERNAL_PORT = 5432

#: Preferred published port. Deliberately not 5432: a Windows user may already run Postgres, and
#: this repository's own root compose already binds 5432, so defaulting there would collide with
#: the developer's own stack on the machine most likely to be testing it.
DEFAULT_PORT = 5487


@dataclass(frozen=True)
class StackSpec:
    """What to generate. Every field is a decision the user or the wizard already made."""

    #: The user's chosen location. Holds the compose file, the generated Dockerfile and `.env`; the
    #: database itself is on the named volume, not here (see the module docstring). Still required
    #: absolute, and the reason is now Compose's rather than the user's: a relative path is resolved
    #: against whatever directory invoked `docker compose`, not against this file.
    data_root: Path
    #: Published host port. Chosen by `choose_port` at install time rather than fixed, because a
    #: collision here is invisible until the first query and looks like a recall bug.
    port: int
    #: `{project}-{kind}` tenants, one MCP service each. One service per tenant is not a style
    #: choice: on unauthenticated stdio `_require` returns the server's own single store and
    #: ignores the requested tenant, so one server genuinely cannot serve two tenants.
    tenants: tuple[str, ...]
    #: Per-tenant environment, from `wiring.server_blocks`, so the trust decision for a tenant is
    #: made in ONE place and applies to the agent's server and the UI's service alike.
    env: dict[str, dict[str, str]]
    recall_image: str = _DEFAULT_IMAGE
    project_name: str = "recall-desktop"

    def __post_init__(self) -> None:
        if not self.data_root.is_absolute():
            raise ValueError(
                f"data_root must be absolute, not {str(self.data_root)!r}: a relative location "
                "resolves against whatever directory the installer happened to run from, so the "
                "user's index would land somewhere they did not choose and cannot find again"
            )
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be a TCP port, not {self.port}")
        if not self.tenants:
            raise ValueError("a stack with no tenants would start a database nothing can reach")
        missing = [tenant for tenant in self.tenants if tenant not in self.env]
        if missing:
            raise ValueError(
                f"no environment for {', '.join(missing)}. Every tenant's trust posture comes from "
                "`wiring.server_blocks`; inventing one here would put the agent and the UI on "
                "different rules for the same corpus."
            )


def host_dsn(port: int, *, user: str = "recall", password: str = "recall") -> str:
    """The address a HOST process uses: the wizard, and every MCP server it registers."""
    return f"postgresql://{user}:{password}@127.0.0.1:{port}/recall"


def container_dsn(*, user: str = "recall", password: str = "recall") -> str:
    """The address a service INSIDE the compose network uses. Same database, shorter path."""
    return f"postgresql://{user}:{password}@db:{DB_INTERNAL_PORT}/recall"


def choose_port(preferred: int = DEFAULT_PORT, *, attempts: int = 64) -> int:
    """A free TCP port, starting from `preferred`.

    Probed rather than assumed because a busy port does not fail at install time: compose reports
    the bind error, the stack half starts, and the first symptom the user sees is a query that
    cannot connect. Checking costs microseconds.
    """
    for offset in range(attempts):
        candidate = preferred + offset
        if candidate > 65535:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # ⚠️ NO `SO_REUSEADDR` here, and that is the whole correctness of this function.
            # With it set, the bind SUCCEEDS against a port that is already listening — on Windows
            # that flag explicitly permits taking over such an address — so the probe reported
            # every busy port as free and handed one out. Compose would then fail to bind and the
            # user would meet it as a query that cannot connect. Caught by the test that occupies
            # a port and asks for it.
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError(
        f"no free port in {preferred}..{min(preferred + attempts - 1, 65535)}; pass one explicitly"
    )


def existing_tenants(compose_path: Path) -> tuple[str, ...]:
    """The tenants a previous run already provisioned, in sorted order.

    ⚠️ **Read this before regenerating a stack that is meant to GAIN a project.**
    `compose_document` builds the whole document from the tenants it is handed, and `write_compose`
    replaces the file, so a run carrying only the new project's tenants does not add a project: it
    **deletes every existing one**, taking their MCP services with it. The corpora survive in the
    database, orphaned and unreachable, which is the worst shape for a data-loss bug to take,
    because nothing errors and the user's projects simply stop existing in the UI.

    So the caller unions: `existing_tenants(path) + new`. Kept as a separate reader rather than
    folded into `write_compose`, because a silent union would be just as wrong in the other
    direction — a genuine re-install that drops a project must be able to say so.

    Empty for anything unreadable, matching `existing_port`: there is nothing to preserve in a file
    that cannot be parsed, and refusing would block an install over a file the caller is about to
    overwrite anyway.
    """
    try:
        document = json.loads(compose_path.read_text(encoding="utf-8"))
        services = document["services"]
    except (OSError, ValueError, KeyError, TypeError):
        return ()
    if not isinstance(services, dict):
        return ()
    prefix = _service_name("")
    return tuple(sorted(
        name[len(prefix) :] for name in services if name.startswith(prefix) and name != prefix
    ))


def existing_port(compose_path: Path) -> int | None:
    """The published port a previous run already chose, or None if there is no stack yet.

    **The port must be STABLE across runs.** `runtime.json` names a compose file, and the desktop
    UI connects through whatever that file publishes; the server block the agent uses carries the
    host address directly. Re-choosing a free port on every install would silently repoint the
    database out from under both, and the symptom is a UI that shows an empty corpus rather than an
    error. So a re-run reads the port back rather than picking again.

    Returns None for anything unreadable, because an unparseable compose file is not a reason to
    refuse an install: the caller chooses a fresh port and writes a correct one over it.
    """
    try:
        document = json.loads(compose_path.read_text(encoding="utf-8"))
        published = document["services"]["db"]["ports"][0]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    # ⚠️ **Parse from the RIGHT, because the mapping now carries a host IP.**
    # It is `127.0.0.1:5487:5432`, not `5487:5432`, so taking field 0 returns "127.0.0.1", which
    # is not an int: `existing_port` would answer None for a perfectly good stack, the installer
    # would choose a fresh port on every re-run, and `add_project` would refuse every add with
    # "no recall stack at ...". Binding to loopback and this parser are ONE change; shipping the
    # bind alone would trade an exposure for a stack nobody can add to.
    #
    # The container port is always last and the published port immediately precedes it, so
    # `[-2]` reads both the two-field and three-field forms.
    fields = str(published).split(":")
    if len(fields) < 2:
        return None
    try:
        return int(fields[-2])
    except ValueError:
        return None


def compose_document(spec: StackSpec) -> dict[str, object]:
    """The compose document: one database, one MCP service per tenant."""
    services: dict[str, object] = {
        "db": {
            "image": DB_IMAGE,
            "environment": {
                "POSTGRES_USER": "recall",
                "POSTGRES_PASSWORD": "recall",
                "POSTGRES_DB": "recall",
            },
            # PUBLISHED. Without this line the wizard and the agent cannot reach the database the
            # UI is filling, which is the entire defect this module exists to remove.
            # ⚠️ **`127.0.0.1:` is load-bearing, and its absence was a real exposure.** Compose's
            # short syntax with no host IP binds 0.0.0.0, and Docker Desktop's proxy then listens
            # on every interface of the Windows host — with the `recall:recall` credentials this
            # project's own README publishes. Anyone who could reach the machine on this port had
            # read and write on every indexed document.
            #
            # The comment justifying `RECALL_ALLOW_INSECURE_DSN` below asserted "the port that IS
            # published is bound to loopback" as its risk acceptance. That was the claim; this is
            # what makes it true. The stack this replaces published no database port at all, so
            # the generated one had been strictly more exposed than its predecessor.
            "ports": [f"127.0.0.1:{spec.port}:{DB_INTERNAL_PORT}"],
            # ⚠️ **A NAMED VOLUME, not a bind mount into `data_root`, and this was measured the
            # hard way.** Bind-mounting the user's chosen directory is what the wizard promises
            # everywhere else, but PostgreSQL cannot survive it on Windows: Docker Desktop's
            # filesystem passthrough returns EINTR on writes, and postgres treats that as fatal.
            #
            #     FATAL:  could not write to file "pg_wal/xlogtemp.1218": Interrupted system call
            #     LOG:  startup process (PID 1218) exited with exit code 1
            #     LOG:  shutting down due to startup process failure
            #
            # It is INTERMITTENT, which is worse than a clean failure: an earlier full install on
            # the same design built, calibrated and certified both corpora, and the next run died
            # mid-flight. An intermittent WAL write failure is a corruption risk, not just an
            # availability one, so the index cannot live on that mount.
            #
            # The named volume lives inside the Docker VM. `data_root` still holds everything the
            # user needs to see and back up — the compose file, the Dockerfile, `.env` — and the
            # database is reached through the published loopback port. `docker compose down -v`
            # destroys it, which is why the wizard never runs that.
            "volumes": [f"{DB_VOLUME}:{DB_MOUNT}"],
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U recall"],
                "interval": "2s",
                "timeout": "3s",
                "retries": 30,
                # `initdb` runs before the first successful check, and on a first install it can
                # take longer than interval x retries. Without this the FIRST `up --wait` of every
                # new install reports "dependency failed to start: container is unhealthy" and
                # leaves every MCP service at `created`, then a second `up` works — which reads to
                # a user as the installer being broken and then mysteriously fixing itself.
                # Measured on a bind mount; a named volume is far faster, and this is the margin
                # that stops the difference being a failed install.
                "start_period": "180s",
            },
        }
    }

    for tenant in spec.tenants:
        services[_service_name(tenant)] = tenant_service(
            spec.env[tenant], image=spec.recall_image
        )

    # Compose namespaces a named volume under the project, so this becomes
    # `<project_name>_pgdata` and two installs at different data roots cannot collide:
    # `compose_project_for` already hashes the data root into the project name.
    return {"name": spec.project_name, "services": services, "volumes": {DB_VOLUME: None}}


def tenant_service(base_env: dict[str, str], *, image: str) -> dict[str, object]:
    """One tenant's compose service. Shared, so an ADDED project is built exactly like an installed
    one — a second construction site would drift, and the drift would only show up as a service
    that starts differently from its siblings.
    """
    # The tenant's own env, minus the DSN: inside the network the database is `db`, not a host
    # port. Same store, and taking the rest verbatim is what keeps one trust decision per
    # tenant rather than two that can drift.
    environment = {k: v for k, v in base_env.items() if k != "RECALL_DSN"}
    environment["RECALL_DSN"] = container_dsn()
    environment["RECALL_MIGRATION_DSN"] = container_dsn()
    # ⚠️ **Without this the service refuses to start, and the whole generated stack is inert.**
    # `require_secure_dsn` rejects the built-in `recall:recall` credentials against any host it
    # does not consider local, and the compose hostname `db` is not local by that test. So
    # `recall schema apply` and `python -m recall_mcp.server` both exit 1 inside every service
    # here. Demonstrated by running the CLI with this exact DSN: `PermissionError: refusing to
    # start against postgresql://recall:***@db:5432/recall`.
    #
    # It is safe for the same reason the hand-written `docker-compose.desktop.yml` sets it on
    # all four of its services: this DSN never leaves the compose network, and the port that IS
    # published is bound to loopback. The credentials are the thing to change if that stops
    # being true, not this flag.
    environment["RECALL_ALLOW_INSECURE_DSN"] = "1"
    return {
        "image": image,
        # ⚠️ **Without this the generated stack has no image and cannot start.** `image:` alone
        # makes Compose PULL, and this tag is published to no registry: the only other `build:`
        # for it lives in `docker-compose.desktop.yml`, whose context is the source checkout, which
        # a user who installed recall from PyPI does not have. So the wizard writes its own
        # Dockerfile beside the compose file and builds from PyPI instead. `image:` is kept so the
        # build is TAGGED rather than anonymous, and so every tenant service shares one build.
        "build": {"context": ".", "dockerfile": DOCKERFILE_NAME},
        "command": ["sleep", "infinity"],
        "environment": environment,
        "depends_on": {"db": {"condition": "service_healthy"}},
    }


def _publishes_documents(version: str) -> bool:
    """Whether `version` is a release that published the `documents` extra.

    Unparseable versions are treated as CURRENT (True). A pre-release or a local build carries the
    current extras by definition; guessing "legacy" for anything unfamiliar would silently drop
    document extraction from images that should have it, which is the quieter of the two failures
    and therefore the wrong default.
    """
    # ⛔ **Leading digits only.** The first version of this joined EVERY digit in the component,
    # so `"5rc1"` became 51 and `_publishes_documents("0.9.5rc1")` returned True — pinning the
    # `documents` extra on a release that predates it, which is precisely the unbuildable image this
    # function exists to prevent. Measured before the fix: 0.9.5rc1, 0.9.2b3 and 0.9.1rc1 all True.
    #
    # A suffix also means PRE-release, which sorts BELOW the release it precedes: 0.9.6rc1 comes
    # before 0.9.6 and must not claim an extra that 0.9.6 introduced.
    parts: list[int] = []
    prerelease = False
    for piece in version.split(".")[:3]:
        match = re.match(r"(\d+)(.*)", piece.strip())
        if match is None:
            # Genuinely unparseable, which is the documented default: a local or branch build
            # carries the CURRENT extras, because dropping document extraction from an image that
            # should have it is the quieter of the two failures.
            return True
        parts.append(int(match.group(1)))
        prerelease = prerelease or bool(match.group(2))
    exact = tuple(parts) == _DOCUMENTS_EXTRA_SINCE
    return tuple(parts) > _DOCUMENTS_EXTRA_SINCE or (exact and not prerelease)


def dockerfile_text(version: str | None = None) -> str:
    """The Dockerfile the generated stack builds from.

    Installs from PyPI at a PINNED version rather than copying a source tree, because the wizard
    runs from an installed wheel and has no checkout to copy. Pinning to the running version is the
    point: the container that serves a corpus and the process that built it are then the same
    recall, and an upgrade of one without the other becomes a visible rebuild rather than a silent
    version skew across the embedder and the schema.

    ⚠️ It therefore requires the version to EXIST on PyPI. Running the wizard from a checkout whose
    version has not been published yet produces a stack whose build fails, saying so plainly; that
    is better than silently serving a different version than the one the user is running.
    """
    if version is None:
        from recall import __version__

        version = __version__
    # The extras follow the PINNED version, not the running one. See `_DOCUMENTS_EXTRA_SINCE`.
    extras = _IMAGE_EXTRAS if _publishes_documents(version) else _LEGACY_IMAGE_EXTRAS
    # LibreOffice is a large layer and a slow build. It is here for parity with
    # `docker/desktop/Dockerfile`: the UI offers .docx/.xlsx/.pptx, and without it those files are
    # accepted and yield nothing, which reads as recall being bad at documents rather than as a
    # missing dependency.
    return (
        "# Generated by the recall wizard. Rebuilt whenever the compose file is written.\n"
        "FROM python:3.13-slim\n"
        "\n"
        "WORKDIR /opt/recall\n"
        "RUN apt-get update \\\n"
        "    && apt-get install -y --no-install-recommends \\\n"
        "        libreoffice-calc \\\n"
        "        libreoffice-impress \\\n"
        "        libreoffice-writer \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
        f'RUN pip install --no-cache-dir "recall-rag[{extras}]=={version}"\n'
        "\n"
        "# ⚠️ **pip only WARNS about an extra a release does not provide.** So a pin whose version\n"
        "# predates an extra installs cleanly, the image builds, the container runs, and the\n"
        "# missing support is discovered when a user drops a .docx in and gets nothing out of it.\n"
        "# Measured on recall-rag 0.9.5, whose published extras are bench, dev, entail, eval,\n"
        "# extract, fastembed, finetune, langchain, llamaindex, mcp, pool, rerank, s3, sparse and\n"
        "# voyage: `documents` is NOT among them. These imports turn that into a build failure,\n"
        "# where it is cheap and legible, instead of an extraction that silently returns nothing.\n"
        + _import_assertion(extras)
    )


def _import_assertion(extras: str) -> str:
    """The post-install `RUN` that turns a silently-missing extra into a build failure.

    ⛔ **Joined from a list so the line continuations are STRUCTURAL, not hand-placed.** The
    previous version wrote each fragment as its own string literal with a trailing `\\\\`, and
    moving the last fragment into a conditional expression silently made the fragment ABOVE it
    the end of the RUN instruction. Docker then parsed the next line as a top-level instruction
    and failed with `unknown instruction: &&` — so every Dockerfile generated for 0.9.6 or later
    was invalid, which is strictly worse than the unbuildable-pin bug that refactor was fixing.

    The substring tests could not see it: they assert that a fragment is PRESENT, and a
    continuation is a property of the line before it. `test_every_line_of_the_run_block_continues`
    asserts the invariant instead.
    """
    checks = [
        'python -c "import recall_mcp.server"',
        'python -c "import fastembed"',
    ]
    trailer = ""
    if extras == _IMAGE_EXTRAS:
        checks.append('python -c "import pypdf, docx, openpyxl, pptx, bs4"')
    else:
        trailer = (
            "# `documents` is not published for this pinned version, so the extraction\n"
            "# packages are deliberately not asserted; this image reads text formats only.\n"
        )
    body = " \\\n && ".join(checks)
    return f"RUN {body}\n{trailer}"


def write_dockerfile(directory: Path, version: str | None = None) -> Path:
    """Write the Dockerfile the compose file's build stanza points at.

    Always beside the compose file, because a compose document referencing a Dockerfile that is not
    there is broken by construction. `write_compose` calls this for exactly that reason.
    """
    target = directory / DOCKERFILE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(dockerfile_text(version), encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target


def _image_version(image: str) -> str | None:
    """The recall version a `recall-wizard:<version>` tag names, or `None` for any other tag.

    Deliberately narrow. It recognises only tags this module itself writes, because the value is
    used to PIN what a Dockerfile installs, and guessing a version out of somebody's own
    `myco/recall:latest` would install whatever that string happened to look like.
    """
    prefix, separator, version = image.partition(":")
    if prefix != "recall-wizard" or not separator or not version:
        return None
    return version


def _stack_image(services: dict[str, object]) -> str | None:
    """The image the stack's existing tenant services already run, if they agree on one.

    ⚠️ **Defaulting to THIS version's tag adds a service that runs a different recall from its
    siblings.** `_default_image` is scoped to `recall.__version__`, so a stack provisioned by an
    older install carries an older tag, and adding a project after an upgrade would give that one
    project a different image while every existing corpus kept the old one. Compose reuses a tag
    rather than rebuilding it, so both would start and neither would complain: one project answering
    from a different recall than the rest, with nothing on screen to say so.

    Returns None when the services disagree or there are none yet, and the caller then falls back to
    the current version. Disagreement is left alone deliberately: this function exists to keep an
    add consistent with what is there, not to reconcile a stack somebody has already hand-edited.
    """
    tags = {
        service.get("image")
        for name, service in services.items()
        if name != "db" and isinstance(service, dict) and isinstance(service.get("image"), str)
    }
    if len(tags) == 1:
        only = tags.pop()
        return only if isinstance(only, str) else None
    return None


#: A Docker NAMED volume: the shape `docker volume create` accepts. Anything else in the source
#: position of a mount is a host path.
#:
#: ⛔ **Matched positively, so the guard fails CLOSED.** The first version enumerated host paths
#: instead — "contains a separator, or starts with a dot" — after splitting the entry on its FIRST
#: colon. On Windows that splits `C:/Users/me/db:/var/lib/postgresql/data` into the source `"C"`,
#: which has no separator and no leading dot, so the guard passed the exact layout it exists to
#: refuse. Five auditors found it independently and three executed it. The released v0.9.6 wizard
#: wrote precisely that string, `f"{database_dir.as_posix()}:{DB_MOUNT}"` off an absolute
#: `data_root`, so the miss covered every real Windows install on the one platform whose
#: intermittent WAL corruption motivated the guard.
#:
#: Enumerating the unsafe shapes means a shape nobody enumerated is treated as safe. Enumerating the
#: ONE safe shape means a shape nobody anticipated is refused, which is the direction a guard
#: protecting against data loss should fail in.
_NAMED_VOLUME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

#: A Windows absolute path, `C:/…` or `C:\…`. Recognised before the source is parsed, because the
#: drive colon is indistinguishable from the separator between a mount's source and target.
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")


def _is_named_volume(source: str) -> bool:
    return bool(_NAMED_VOLUME.match(source))


def _volume_source(entry: object) -> str | None:
    """The host-or-volume side of one compose volume entry, or None when there is nothing to judge.

    Handles both syntaxes, because `add_tenant_services` is documented to accept a stack "somebody
    else provisioned" and a hand-written or tool-generated file may use either:

    * short — `SOURCE:TARGET[:MODE]`. Parsed from the RIGHT, so a Windows drive letter in the source
      survives. `recall/wizard/stack.py`'s `existing_port` learned the same lesson in the other
      direction when a published port gained a host IP.
    * long — `{"type": "bind", "source": ..., "target": ...}`. The short-syntax-only version skipped
      these silently, so a bind mount declared the long way read as "no bind mount at all".
    """
    if isinstance(entry, dict):
        source = entry.get("source")
        if entry.get("type") == "bind":
            # A bind is a bind whatever its source looks like; return something that cannot be a
            # named volume so the caller refuses.
            return str(source) if isinstance(source, str) else "<bind mount>"
        return str(source) if isinstance(source, str) else None
    if not isinstance(entry, str) or ":" not in entry:
        return None
    # ⛔ **The drive prefix is decided BEFORE any splitting, and that ordering is the fix.**
    # Stripping right-to-left is not enough on its own: `C:/Users/me/db:/var/lib/postgresql/data`
    # strips the target, leaving `C:/Users/me/db`, and then strips AGAIN because `/Users/me/db`
    # also starts with a slash — yielding the source `"C"`, which is a valid named volume. That is
    # the same wrong answer the original first-colon split gave, reached by a longer route, and it
    # survived the first attempt at this fix. A drive-lettered entry is a host path, full stop.
    if _DRIVE_PREFIX.match(entry):
        return entry
    # The container target is an absolute POSIX path, and an optional access mode follows it. Strip
    # from the right until what remains is the source.
    remainder = entry
    for _ in range(2):
        head, _sep, tail = remainder.rpartition(":")
        if not head:
            break
        if tail.startswith("/") or tail in {"ro", "rw", "z", "Z", "cached", "delegated"}:
            remainder = head
            continue
        break
    return remainder


def _refuse_bind_mounted_database(compose_path: Path, db: object) -> None:
    """Refuse to extend a stack whose database still lives on a host bind mount.

    ⛔ **The bind mount is the layout that corrupted a database**, which is why `DB_VOLUME` exists:
    PostgreSQL's WAL writes fail intermittently with EINTR on a Windows bind mount, and an
    intermittent WAL write failure is a corruption risk rather than an availability one. See the
    comment on the `db` service in `compose_document`.

    A stack provisioned before that change still has the old mount. Adding a project to it would put
    a new corpus onto that database — more data on the layout the wizard stopped using precisely
    because it loses data. So this refuses and says what to do, rather than quietly making the
    exposure larger.

    Refusing is safe in a way that migrating is not: moving the volume means moving a live database
    the user may be serving from, which is not something an "add a project" action should attempt
    without being asked.
    """
    if not isinstance(db, dict):
        return
    volumes = db.get("volumes")
    if volumes is None:
        return
    if not isinstance(volumes, list):
        raise ValueError(
            f"the stack at {compose_path} declares its database volumes as "
            f"{type(volumes).__name__}, which this cannot read. Refusing rather than adding a "
            "corpus to a database whose storage layout could not be determined."
        )
    for entry in volumes:
        source = _volume_source(entry)
        if source is None or _is_named_volume(source):
            continue
        raise ValueError(
                f"the stack at {compose_path} keeps its database on the host directory {source!r} "
                "rather than in a Docker volume. That layout loses PostgreSQL WAL writes "
                "intermittently on Windows, which is why newer installs use a named volume, and "
                "adding a project would put another corpus onto it. Back up the data you care "
                "about, then re-provision the stack with a current install."
            )


def add_tenant_services(
    compose_path: Path, env: dict[str, dict[str, str]], *, image: str = _DEFAULT_IMAGE
) -> tuple[str, ...]:
    """Add services for `env`'s tenants to an EXISTING compose file, and return the new tenants.

    ⚠️ **This adds; it does not regenerate.** Rebuilding the document from a union of tenants would
    also rewrite every existing service's environment, and that environment carries
    `RECALL_TRUST_MODE`, which the wiring stage set per tenant from what actually certified. A
    regeneration would quietly reset a certified corpus's posture to whatever the caller happened to
    pass, so the safe operation on a live stack is a strictly additive one: existing services are
    copied through untouched, byte for byte.

    Refuses rather than returning empty when the file cannot be read, which is the opposite of
    `existing_tenants` and `existing_port` and deliberately so: those two are consulted before
    writing a fresh document over the top, where there is nothing to lose. Here the file IS the
    stack, and overwriting an unreadable one would strand every corpus it described.

    A tenant that already has a service is skipped, not overwritten, so a repeated add is a no-op
    rather than a silent reset of that project's trust posture.
    """
    try:
        document = json.loads(compose_path.read_text(encoding="utf-8"))
        services = document["services"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot add to the stack at {compose_path}: {exc}. The file describes the running "
            "containers and every provisioned corpus, so it is not safe to write a new one over it."
        ) from exc
    if not isinstance(services, dict) or "db" not in services:
        raise ValueError(
            f"{compose_path} defines no `db` service, so it is not a recall stack to add to"
        )
    _refuse_bind_mounted_database(compose_path, services["db"])

    inherited = _stack_image(services) or image
    added: list[str] = []
    for tenant, base_env in env.items():
        name = _service_name(tenant)
        if name in services:
            continue
        services[name] = tenant_service(base_env, image=inherited)
        added.append(tenant)

    if added:
        # The Dockerfile follows the tag, not the running wizard. See `write_compose`: regenerating
        # it at the running version left a 0.9.1 stack carrying a 0.9.6 Dockerfile, and Compose
        # would have served the 0.9.1 image under it without a word.
        write_compose(compose_path, document, dockerfile_version=_image_version(inherited))
    return tuple(sorted(added))


def _service_name(tenant: str) -> str:
    """A compose service name for a tenant. Stable, so the UI can map one to the other."""
    return f"recall-{tenant}"


def write_compose(
    path: Path, document: dict[str, object], *, dockerfile_version: str | None = None
) -> None:
    """Write the compose file atomically, as JSON, with LF endings.

    JSON because it is valid YAML and `json.dumps` quotes a Windows path containing spaces
    correctly, which hand-built YAML does not. LF because a file that rewrites every line on every
    platform is a diff nobody can read.

    **Writes the Dockerfile too**, because every tenant service's `build:` stanza names it and a
    compose document pointing at a file that is not there is broken by construction. Coupling them
    here means no caller can produce half a stack: the failure would be a `docker compose up` that
    cannot find the Dockerfile, at the moment the user is trying to start their index.

    ⛔ **`dockerfile_version` exists because writing the RUNNING version here made the stack lie.**
    `add_tenant_services` inherits the existing stack's image tag so a new project runs the same
    recall as its siblings; this then regenerated the Dockerfile at whatever version the wizard
    happened to be. Measured on a 0.9.1 stack under a 0.9.6 wizard: the new service's tag said
    `recall-wizard:0.9.1` and the Dockerfile beside it installed `recall-rag==0.9.6`. Compose reuses
    a tag it already holds rather than building, so the container would start the 0.9.1 image while
    every file on disk claimed 0.9.6 — the same silent-wrong-image failure `_default_image`'s
    docstring records, reintroduced through the inheritance that was meant to prevent it.

    The tag is the thing that decides what actually runs, so the Dockerfile follows the tag.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    write_dockerfile(path.parent, dockerfile_version)


def wait_for_database(dsn: str, *, timeout: float = 120.0, interval: float = 2.0) -> None:
    """Poll `dsn` until it accepts a connection.

    ⚠️ **`docker compose up --wait` is NOT sufficient, and believing it was cost a failed run.**
    The healthcheck is `pg_isready` INSIDE the container, and the postgres entrypoint runs a
    temporary server during `initdb` which answers it. So the healthcheck can pass while the real
    server is still restarting to listen on TCP, and the first host connection dies with
    `server closed the connection unexpectedly`. Measured on a first install against a freshly
    bind-mounted data directory, which is exactly the case every user hits once.

    Polling the address the caller will actually use is the only check that means anything: it
    tests the published port, the bind mount, and the server, rather than one process's opinion of
    its own readiness.
    """
    import time

    import psycopg

    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=5):
                return
        except psycopg.OperationalError as exc:
            last = exc
            time.sleep(interval)
    from recall.store import redacted_dsn

    raise RuntimeError(
        f"the database at {redacted_dsn(dsn)} never accepted a connection within {timeout:.0f}s: "
        f"{last}"
    )


def bring_up(
    compose_path: Path,
    *,
    project_name: str,
    services: tuple[str, ...] = (),
    timeout: float = 300.0,
) -> None:
    """Start the stack and wait for the healthchecks to pass.

    `--wait` rather than a sleep, so this returns when the healthcheck passes rather than when a
    guess expires. A slow first pull on a user's machine is normal and must not look like a failure.

    `services` narrows what is started. The database must come up before the recall image is even
    built, since the build is the slow part of a first install and there is no reason to make the
    user wait for it before their store exists.
    """
    command = [
        "docker",
        "compose",
        "-f",
        str(compose_path),
        "-p",
        project_name,
        "up",
        "-d",
        "--wait",
        *services,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            # ⚠️ See `recall/desktop/runtime.py`: `text=True` alone decodes Docker's output with
            # the platform codec, and an undecodable byte silently yields rc=0 and `stdout=None`
            # rather than an exception. The message below would then name no cause at all.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"could not start the recall stack: {(exc.stderr or exc.stdout or '').strip()[:400]}"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not start the recall stack: {exc}") from exc
