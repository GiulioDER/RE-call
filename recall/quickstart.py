"""One command from `pip install` to a real answer, including the database.

**Why this exists.** Every other entry point in this project asks the reader to provision
PostgreSQL first. `recall demo` does not, and that is worse: it indexes the relative path
``"corpus"``, which exists only in a git clone, so from a `pip install` it indexes an absent
directory. The wizard provisions properly but builds a `recall-wizard:<version>` image on the way,
which is a pip install inside Docker and takes minutes on a first run. Neither is a path a person
evaluating this project will walk, and the measured consequence is a landing page whose headline is
"Get RE-call running in seven steps".

So this module owns exactly one promise: from nothing, produce a store the reader can query, and
show them the two behaviours that distinguish this project from a vector index, which are that a
correction outranks the claim it retracted and that an unanswerable question is refused rather than
answered. Everything else, calibration included, is deliberately out of scope and named as a next
step instead.

Six decisions, each of which is a place this would otherwise go wrong:

**The database service only. No built image.** `recall.wizard.stack` writes a Dockerfile beside its
compose file because every tenant service builds from it. The quickstart has no tenant services:
the CLI runs on the host and talks to the published port, so the slowest step in the wizard's
install is not merely skipped here, it is never expressible.

**Its own compose project, its own volume, its own data root.** A reader trying this must not be
able to damage an install they already have, and the teardown below runs `down -v`, which destroys
a volume. Sharing a project name with the desktop stack would make `recall quickstart --remove`
delete somebody's real index.

**Its own table and tenant.** The 22 demo documents are fiction about a fictional service. Indexed
into `chunks`/`default` they would be indistinguishable from the reader's own memory the moment
they pointed anything real at the same database.

**The corpus comes from the installed package, never from the working directory.** `recall/eval/
corpus/` sits inside the `recall` package and therefore inside the wheel. Resolving it from
`__file__` is what makes this work for the pip-installed reader, who is the only reader this
command is for.

**Re-running reuses the port it already chose.** `choose_port` picks a free one, so a second run
that picked again would leave the first container bound to a port nothing refers to any more, and
this project already has a documented problem with stranded containers.

**Teardown is part of the feature, not an appendix.** A reader who tries this and walks away leaves
a PostgreSQL container running on their machine forever. `--remove` is printed at the end of every
successful run for that reason.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from recall.wizard.stack import (
    DB_IMAGE,
    DB_INTERNAL_PORT,
    DB_MOUNT,
    bring_up,
    choose_port,
    existing_port,
    host_dsn,
    wait_for_database,
)

__all__ = [
    "COMPOSE_NAME",
    "DB_VOLUME",
    "DEMO_QUERIES",
    "PROJECT_NAME",
    "QUICKSTART_PORT",
    "QUICKSTART_TABLE",
    "QUICKSTART_TENANT",
    "compose_document",
    "demo_corpus",
    "docker_unavailable_reason",
    "next_steps",
    "provision",
    "quickstart_root",
    "remove_stack",
]

#: Compose project, volume namespace and container prefix for everything this module starts.
#: Deliberately NOT `recall-desktop`, which `recall.wizard.stack` uses: `remove_stack` runs
#: `down -v`, and a shared project name would make the quickstart's teardown destroy the volume
#: holding a real install's index.
PROJECT_NAME = "recall-quickstart"

#: The generated compose file's name, under `quickstart_root()`.
COMPOSE_NAME = "docker-compose.quickstart.yml"

#: The named volume, namespaced under `PROJECT_NAME` by Compose itself.
DB_VOLUME = "quickstart-pgdata"

#: Preferred published port. Offset from the wizard's 5487 so a reader who has already installed
#: properly can run the quickstart beside it without either stack noticing the other.
QUICKSTART_PORT = 5497

#: Table and tenant for the demo documents. Separate from `DEFAULT_TABLE`/`DEFAULT_TENANT` so the
#: fiction in `recall/eval/corpus/` can never be confused with, or retrieved beside, real memory.
QUICKSTART_TABLE = "quickstart_chunks"
QUICKSTART_TENANT = "quickstart"

#: The three queries, in this order, because the order is the argument.
#:
#: 1. An ordinary answerable question, so the reader sees a confident hit and has a baseline.
#: 2. The superseded pair. `cache_ttl_v2.md` carries ``supersedes: cache_ttl_v1.md`` in its
#:    frontmatter, and v1 is the nearer lexical match for "cache TTL", so a plain vector index
#:    returns the retracted 15 minute answer. This is the entire thesis of the project in one
#:    query, and it is the reason the quickstart shows three results rather than one.
#: 3. A question the corpus cannot answer, refused rather than answered.
#:
#: ⚠️ **The third query was CHOSEN BY MEASUREMENT, and the obvious candidate failed.** The first
#: draft reused `recall demo`'s "how do we handle llamas on mars?", on the reasoning that it was
#: already vetted. Measured against this corpus on 2026-08-22 with `fastembed` and the 0.50
#: development threshold, its top cosine is **0.505**, so it does NOT abstain: it answers, out of
#: `secrets_handling.md`. The caption would have said "refused rather than answered" over an
#: answer. `demo`'s query is vetted for a different corpus (the repository's own six-file
#: `corpus/`), and a vetting does not travel between corpora.
#:
#: The replacement has a real margin rather than a marginal one. Top cosine **0.446**, which is
#: 0.054 clear of the threshold, against 0.021 for the next best candidate considered. A demo that
#: abstains by a hundredth is one that will stop abstaining on somebody's machine.
#:
#: Its distinctive words ("mural", "painted") are absent from the corpus, and absent from
#: `recall/eval/offtopic_subjects.json`, which matters because this module lives under `recall/`
#: and a word from that pool appearing in recall source disqualifies its subject for every
#: synthetic code corpus rooted here.
#:
#: Re-measure all three before changing the corpus or the default embedder:
#:
#:     python -m pytest tests/test_quickstart.py -k supersedes_and_actually_abstains
DEMO_QUERIES: tuple[str, ...] = (
    "why did we pick postgres over a document store",
    "how long do pricing snapshot cache entries live",
    "who painted the mural in the town library",
)


def quickstart_root() -> Path:
    """Where the compose file lives.

    Beside the desktop install's own settings rather than in the working directory, because the
    reader runs this from wherever they happen to be standing and a compose file dropped there is
    litter they did not ask for and will not find again. `RECALL_QUICKSTART_ROOT` overrides it,
    which is what the tests use: pointing this at `tmp_path` is the difference between a test and
    a test that reaches into the developer's real `%APPDATA%`.
    """
    override = os.environ.get("RECALL_QUICKSTART_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    base = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "RE-call" / "quickstart"
    return Path.home() / ".recall" / "quickstart"


def demo_corpus() -> Path:
    """The 22 demo documents, resolved from the installed package.

    ⚠️ **Not the repository's root `corpus/`, and not a relative path.** `recall demo` passes the
    literal string ``"corpus"`` to the indexer, which resolves against the current working
    directory: it works in a clone and indexes nothing anywhere else, which is every reader who
    installed from PyPI. `recall/eval/corpus/` is inside the `recall` package, so it is inside the
    wheel, so `__file__` finds it wherever the package was installed.
    """
    return Path(__file__).resolve().parent / "eval" / "corpus"


def docker_unavailable_reason() -> str | None:
    """Why Docker cannot be used, or None if it can.

    Two checks and not one, because they fail differently and the remedies are different. A missing
    executable means Docker is not installed. A present executable whose daemon does not answer
    means Docker Desktop is not running, which is the overwhelmingly common case on Windows and
    reads to the reader as "recall is broken" if the message does not say otherwise.

    Checked before anything is written, so a reader without Docker gets advice instead of a
    half-provisioned directory.
    """
    if shutil.which("docker") is None:
        # ⚠️ The flag named here is `--existing-dsn`, and a test asserts it. It was `--dsn` in the
        # first draft, which is the parent parser's flag: this advice told a reader without Docker
        # to type an option the subcommand does not accept, which is worse than no advice, because
        # they would conclude the escape hatch is broken rather than that the message is.
        return (
            "docker is not on PATH. Install Docker Desktop from https://docs.docker.com/get-started/"
            " and run this again, or point the quickstart at a PostgreSQL you already have with "
            "`recall quickstart --existing-dsn postgresql://user:password@host:5432/db` (it needs "
            "the pgvector extension available)."
        )
    try:
        completed = subprocess.run(
            # ⚠️ `docker version`, NOT `docker info`, and the reason is the TIMEOUT rather than the
            # average cost. `info` gathers the whole system inventory (containers by state, images,
            # plugins, storage driver, registry config), so what it costs scales with what is on
            # the machine. Measured 2026-08-25 on a workstation holding 37 containers: median
            # 14.11s over five runs, spread 4.01s to 56.36s, a factor of 14 within one hour. The
            # 20 second timeout below is 1.4x the median of the OLD probe and below its observed
            # maximum, which is the failure this replaces: a busy daemon made the quickstart report
            # a perfectly healthy Docker as "did not respond" and sent the reader to fix something
            # that was not broken. `version` is one round trip to the daemon's /version endpoint:
            # median 0.48s, 29.1x faster, and roughly two orders of magnitude clear of the bound.
            #
            # It answers the same question for this caller's purposes. Both return 0 against a live
            # daemon and non-zero against an unreachable one, and both emit the same actionable
            # `error during connect: ...` text that the message below quotes.
            # Record: docs/preregistrations/2026-08-25-docker-probe-latency.md
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            # See `recall/wizard/stack.py`: `text=True` alone decodes with the platform codec, and
            # an undecodable byte yields rc=0 with `stdout=None` rather than raising, so the branch
            # below would report success against no output at all.
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"docker is installed but did not respond: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:300]
        return (
            "docker is installed but the daemon is not answering. Start Docker Desktop (or the "
            f"docker service) and run this again. Docker said: {detail}"
        )
    return None


def compose_document(port: int) -> dict[str, object]:
    """One PostgreSQL service, and nothing else.

    Every field here has a reason recorded at length in `recall.wizard.stack.compose_document`, and
    the three that matter are repeated rather than referenced because getting one wrong is silent:

    `127.0.0.1:` on the published port. Without the host IP, Compose binds 0.0.0.0 and Docker
    Desktop's proxy then serves this database, with the `recall:recall` credentials published in
    this project's README, on every interface of the machine.

    A named volume rather than a bind mount. PostgreSQL cannot survive Docker Desktop's filesystem
    passthrough on Windows: writes return EINTR and the server treats that as fatal, intermittently,
    which is a corruption risk and not merely an availability one.

    `start_period` on the healthcheck. `initdb` on a first run can outlast interval times retries,
    and without the grace period the FIRST `up --wait` of every new install reports the container
    unhealthy and the second one works, which reads as software that is broken and then mysteriously
    is not.
    """
    return {
        "name": PROJECT_NAME,
        "services": {
            "db": {
                "image": DB_IMAGE,
                "environment": {
                    "POSTGRES_USER": "recall",
                    "POSTGRES_PASSWORD": "recall",
                    "POSTGRES_DB": "recall",
                },
                "ports": [f"127.0.0.1:{port}:{DB_INTERNAL_PORT}"],
                "volumes": [f"{DB_VOLUME}:{DB_MOUNT}"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U recall"],
                    "interval": "2s",
                    "timeout": "3s",
                    "retries": 30,
                    "start_period": "180s",
                },
            }
        },
        "volumes": {DB_VOLUME: None},
    }


def write_compose(path: Path, document: dict[str, object]) -> None:
    """Write the compose file atomically, as JSON, with LF endings.

    JSON because it is valid YAML and `json.dumps` quotes a Windows path containing spaces
    correctly. Atomically because a half-written compose file is one `docker compose` reads and
    refuses, and the reader would have to know to delete it.

    Deliberately NOT `recall.wizard.stack.write_compose`, which also writes a Dockerfile: that
    Dockerfile exists to be built by tenant services, this stack has none, and an unbuilt Dockerfile
    sitting beside the quickstart is a file whose purpose nobody can explain later.
    """
    import json

    from recall.atomic_write import atomic_write_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))


def provision(root: Path | None = None) -> tuple[str, int, Path, bool]:
    """Start the database and return `(dsn, port, compose_path, reused)`.

    Idempotent by reading the port back out of an existing compose file rather than choosing again.
    A second run that re-chose would publish a new port, leave the running container bound to the
    old one, and hand the caller a DSN pointing at an empty database while the reader's first run
    is still on the machine holding the data.
    """
    target = root or quickstart_root()
    compose_path = target / COMPOSE_NAME
    recorded = existing_port(compose_path)
    reused = recorded is not None
    port = recorded if recorded is not None else choose_port(QUICKSTART_PORT)
    write_compose(compose_path, compose_document(port))
    # `services=("db",)` is belt and braces: this document declares no other service, so the
    # narrowing cannot change what starts. It is passed anyway to state the intent at the call
    # site, so that adding a service to `compose_document` later cannot silently make the
    # quickstart start it.
    bring_up(compose_path, project_name=PROJECT_NAME, services=("db",))
    dsn = host_dsn(port)
    # `up --wait` returns when the healthcheck passes, and the healthcheck is `pg_isready` INSIDE
    # the container, which the temporary server `initdb` runs also answers. The real server may
    # still be restarting to listen on TCP. Polling the address the caller will actually use is
    # the only check that tests the published port rather than one process's opinion of itself.
    wait_for_database(dsn)
    return dsn, port, compose_path, reused


def next_steps(dsn: str, *, provisioned: bool, compose_path: Path | None) -> tuple[str, ...]:
    """The closing block, as lines, so the one security decision in it is testable directly.

    ⛔ **The literal DSN is emitted only for the stack this module created.** Those credentials are
    `recall:recall`, they are published in this project's README, and the port is bound to loopback,
    so printing them costs nothing and saves the reader retyping. A DSN the reader SUPPLIED is
    theirs and may carry a real password: echoing it would write that password into their
    scrollback, their shell history, and any transcript or recording of the session. The
    placeholder is the whole point of this function existing separately.

    Returned rather than printed because the alternative is testing a security property by parsing
    stdout, and this project already has enough functions that both decide and print.
    """
    shown = dsn if provisioned else "<your dsn>"
    lines = [
        "Next, in the order most people want them:",
        f"  index your own notes    recall --dsn {shown} --table {QUICKSTART_TABLE} \\",
        f"                            --tenant {QUICKSTART_TENANT} index <folder>",
        "  fit a real threshold    recall setup        (the demo threshold above is not one)",
        "  give it to Claude Code  recall setup        (registers the MCP server and hooks)",
    ]
    if provisioned:
        lines.append("  remove all of this      recall quickstart --remove")
    # ⚠️ **Every one of these four, not just the DSN.** The Claude Code plugin asks for a table
    # and a tenant, and this corpus is in NEITHER default: a reader who pasted only the DSN got a
    # server that started cleanly, answered, and returned "0 relevant memory hit(s)" out of an
    # empty `chunks`. Nothing on that path raises, so printing the values in the shape the
    # question asks for them is the only defence there is. Measured 2026-08-25 by driving the
    # stdio server with exactly the plugin's variables against a live quickstart database.
    lines.extend(
        [
            "",
            "Giving this to the Claude Code plugin (/plugin install recall@re-call)?",
            "It asks for four values, and two of them are NOT its defaults:",
            f"  PostgreSQL DSN  {shown}",
            f"  Table           {QUICKSTART_TABLE}",
            f"  Tenant          {QUICKSTART_TENANT}",
            "  Trust mode      development   (uncalibrated corpus; strict correctly refuses it)",
        ]
    )
    if provisioned and compose_path is not None:
        lines.append(f"\nThe compose file is at {compose_path}.")
    return tuple(lines)


def remove_stack(root: Path | None = None) -> bool:
    """Stop the quickstart stack and destroy its volume. True if there was one to remove.

    ⛔ **This runs `down -v`, which is irreversible.** It is safe only because `PROJECT_NAME`,
    `DB_VOLUME` and `quickstart_root()` are exclusive to this module: Compose namespaces a named
    volume under its project, so the volume destroyed here cannot be the one any other install is
    using. Changing `PROJECT_NAME` to something shared would turn this function into a command that
    deletes a reader's real index.

    The demo corpus is 22 fictional documents, so nothing here is worth preserving, which is what
    makes destroying the volume the right default rather than a flag.
    """
    target = root or quickstart_root()
    compose_path = target / COMPOSE_NAME
    if not compose_path.exists():
        return False
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "-p",
            PROJECT_NAME,
            "down",
            "-v",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        # Not `check=True`: a stack that is already gone is the outcome this asks for, and a reader
        # running `--remove` twice should not get a traceback the second time.
        check=False,
    )
    compose_path.unlink(missing_ok=True)
    return True
