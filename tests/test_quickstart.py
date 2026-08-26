"""`recall quickstart`: the one command between a `pip install` and a real answer.

Most of this file is about the ways a first-run command fails INVISIBLY, because that is the only
kind of failure that matters for a command whose entire job is a first impression.

**The corpus must come from the installed package.** `recall demo` indexes the relative path
``"corpus"``, which exists only in a git clone, so from a wheel it indexes an absent directory.
That defect is not visible in this repository: every developer running `recall demo` is standing in
a checkout where the path resolves. `test_demo_corpus_*` pin the property that makes the quickstart
different, which is that its corpus is found from `__file__` and ships inside the wheel.

**The teardown destroys a volume, so the project name is a safety control.** `remove_stack` runs
`docker compose down -v`. Compose namespaces a named volume under its project, so the volume it
destroys is exactly the one this module created, and ONLY because `PROJECT_NAME` is not shared with
`recall.wizard.stack`. Making them equal would turn `recall quickstart --remove` into a command that
deletes a reader's real index, and nothing about the call site would look different.

**Re-running must not strand a container.** `choose_port` returns a free port, so a second run that
chose again would publish a new one, leave the first container bound to a port nothing refers to,
and hand back a DSN pointing at an empty database while the reader's data sat in the old one. This
project already has a documented and measured problem with stranded containers.

No Docker here. `bring_up` and `wait_for_database` are patched on the module under test, following
`tests/test_wizard_stack.py`. The single DB-backed test at the bottom is the one that checks the
command's printed prose against what retrieval actually returns, which no amount of unit testing of
the plumbing can do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall import quickstart as Q
from recall.store import DEFAULT_TABLE, DEFAULT_TENANT

from .conftest import TEST_DSN, requires_db


@pytest.fixture(autouse=True)
def _confine_quickstart_root(tmp_path, monkeypatch) -> None:
    """Point `quickstart_root()` at `tmp_path` for every test in this file.

    Autouse and not opt-in. The real root is `%APPDATA%/RE-call/quickstart`, and a single test that
    forgot to override it would read, write and (through `remove_stack`) delete inside the
    developer's own profile. `recall/wizard/uninstall.py` carries a note recording exactly that
    happening once.
    """
    monkeypatch.setenv("RECALL_QUICKSTART_ROOT", str(tmp_path / "quickstart"))


@pytest.fixture
def no_docker(monkeypatch) -> list[tuple]:
    """Replace the two functions that need a daemon, and record what they were asked to do."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        Q,
        "bring_up",
        lambda path, *, project_name, services=(), timeout=300.0: calls.append(
            ("bring_up", path, project_name, services)
        ),
    )
    monkeypatch.setattr(Q, "wait_for_database", lambda dsn, **kw: calls.append(("wait", dsn)))
    return calls


# --------------------------------------------------------------------------------------------
# The corpus, which is the half of this that `recall demo` gets wrong.
# --------------------------------------------------------------------------------------------


def test_demo_corpus_is_inside_the_installed_package() -> None:
    """Resolved from `__file__`, so it is found wherever the wheel was installed.

    Asserting the path is under `recall/` is the whole test: a corpus resolved any other way is one
    that works in a checkout and silently indexes nothing for the reader who installed from PyPI,
    which is the only reader this command exists for.
    """
    corpus = Q.demo_corpus()
    package_root = Path(Q.__file__).resolve().parent
    assert corpus.is_absolute()
    assert package_root in corpus.parents
    assert corpus.is_dir()


def test_demo_corpus_is_not_the_repository_root_corpus(monkeypatch, tmp_path) -> None:
    """It does not depend on the working directory, which is what `recall demo` gets wrong.

    Changing directory to somewhere with no `corpus/` must not change the answer. `recall demo`
    fails this by construction; that it has never been noticed is because the suite also runs from
    a checkout.
    """
    monkeypatch.chdir(tmp_path)
    assert Q.demo_corpus().is_dir()
    assert not (tmp_path / "corpus").exists()


def test_demo_corpus_holds_the_supersession_pair_the_second_query_needs() -> None:
    """The middle query's whole point is that `cache_ttl_v2` retracts `cache_ttl_v1`.

    Pinned as a corpus fact rather than trusted, because the CLI prints a paragraph asserting this
    relationship in prose. If the frontmatter were edited away, the command would keep printing the
    claim over three ordinary hits and nothing would fail.
    """
    corpus = Q.demo_corpus()
    v1, v2 = corpus / "cache_ttl_v1.md", corpus / "cache_ttl_v2.md"
    assert v1.is_file() and v2.is_file()
    assert "supersedes: cache_ttl_v1.md" in v2.read_text(encoding="utf-8")


def test_the_unanswerable_query_is_about_nothing_in_the_corpus() -> None:
    """The third query has to be unanswerable, and "I chose it carefully" is not a check.

    Necessary but NOT sufficient, and the distinction is the lesson this file paid for: absence
    from the corpus does not imply a cosine below the threshold. `recall demo`'s "llamas on mars"
    satisfies this test and still scores 0.505 against a 0.50 threshold, so it answers. The test at
    the bottom of this file is the one that checks abstention; this one only rules out the
    embarrassing case where the corpus grew a document on the subject.
    """
    unanswerable = Q.DEMO_QUERIES[2]
    corpus_text = " ".join(
        path.read_text(encoding="utf-8").lower() for path in Q.demo_corpus().glob("*.md")
    )
    for word in ("mural", "painted", "town"):
        assert word not in corpus_text, f"{word!r} now appears in the demo corpus"
    assert "mural" in unanswerable


def test_the_unanswerable_querys_distinctive_words_are_not_offtopic_subjects() -> None:
    """⚠️ This module lives under `recall/`, and that has a consequence most files do not have.

    `recall/eval/offtopic_subjects.json` says it plainly: a distinctive word from that pool, written
    anywhere in recall source, disqualifies its subject for every synthetic code corpus rooted at
    this repository. Choosing a quickstart query out of that pool would silently shrink the
    off-topic set the synthetic evaluation draws from, and nothing in that evaluation would report
    the loss.
    """
    import json

    pool = json.loads(
        (Path(Q.__file__).resolve().parent / "eval" / "offtopic_subjects.json").read_text(
            encoding="utf-8"
        )
    )
    words = set(Q.DEMO_QUERIES[2].lower().replace("?", "").split())
    assert not words & set(pool["distinctive"])


def test_demo_queries_are_three_in_a_fixed_order() -> None:
    """The order is the argument: ordinary hit, then retraction, then refusal.

    Pinned because the CLI prints a numbered explanation keyed to it, and a reordering would leave
    every number pointing at the wrong result.
    """
    assert len(Q.DEMO_QUERIES) == 3
    assert "postgres" in Q.DEMO_QUERIES[0]
    assert "cache" in Q.DEMO_QUERIES[1]


# --------------------------------------------------------------------------------------------
# Isolation: nothing this command does may touch an install the reader already has.
# --------------------------------------------------------------------------------------------


def test_table_and_tenant_are_not_the_defaults() -> None:
    """22 fictional documents must not land where real memory lives.

    Indexed into `chunks`/`default` they would be retrieved beside the reader's own notes the first
    time they pointed anything real at the same database, and nothing would report it.
    """
    assert Q.QUICKSTART_TABLE != DEFAULT_TABLE
    assert Q.QUICKSTART_TENANT != DEFAULT_TENANT


def test_project_and_volume_are_not_the_wizard_stacks() -> None:
    """The safety control that makes `down -v` acceptable.

    `remove_stack` destroys a volume. Compose namespaces a named volume under its project, so this
    inequality is the only thing standing between `recall quickstart --remove` and a command that
    deletes the index a reader built with `recall wizard`.
    """
    from recall.wizard import stack as wizard_stack

    assert Q.PROJECT_NAME != wizard_stack.COMPOSE_NAME
    assert Q.PROJECT_NAME != "recall-desktop"
    assert Q.DB_VOLUME != wizard_stack.DB_VOLUME
    assert Q.COMPOSE_NAME != wizard_stack.COMPOSE_NAME


def test_preferred_port_does_not_collide_with_the_wizard_stack() -> None:
    """A reader with a real install must be able to run the quickstart beside it."""
    from recall.wizard import stack as wizard_stack

    assert Q.QUICKSTART_PORT != wizard_stack.DEFAULT_PORT
    assert Q.QUICKSTART_PORT != 5432


def test_quickstart_root_honours_the_environment_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RECALL_QUICKSTART_ROOT", str(tmp_path / "elsewhere"))
    assert Q.quickstart_root() == (tmp_path / "elsewhere").resolve()


# --------------------------------------------------------------------------------------------
# The compose document. Three fields whose absence is silent.
# --------------------------------------------------------------------------------------------


def test_compose_publishes_on_loopback_only() -> None:
    """⚠️ Without the host IP, Compose binds 0.0.0.0.

    Docker Desktop's proxy then serves this database, with the `recall:recall` credentials this
    project's README publishes, on every interface of the machine. Anyone who could reach the host
    on this port would have read and write over everything indexed.
    """
    published = Q.compose_document(6001)["services"]["db"]["ports"][0]
    assert published == "127.0.0.1:6001:5432"
    assert published.startswith("127.0.0.1:")


def test_compose_uses_a_named_volume_not_a_bind_mount() -> None:
    """PostgreSQL cannot survive Docker Desktop's filesystem passthrough on Windows.

    Writes return EINTR and the server treats that as fatal, intermittently, which is a corruption
    risk rather than merely an availability one. A bind mount here would work on the developer's
    machine and destroy a reader's index on theirs.
    """
    volumes = Q.compose_document(6001)["services"]["db"]["volumes"]
    assert volumes == [f"{Q.DB_VOLUME}:/var/lib/postgresql"]
    assert not any(":" in entry and ("/" in entry.split(":")[0]) for entry in volumes)


def test_compose_healthcheck_has_a_start_period() -> None:
    """`initdb` on a first run outlasts interval times retries.

    Without the grace period the FIRST `up --wait` of every new install reports the container
    unhealthy and the second one works, which reads to a reader as software that is broken and then
    mysteriously is not.
    """
    healthcheck = Q.compose_document(6001)["services"]["db"]["healthcheck"]
    assert healthcheck["start_period"] == "180s"


def test_compose_declares_no_service_that_has_to_be_built() -> None:
    """The image build is the slowest step in the wizard's install and is absent here by design.

    A `build:` stanza appearing in this document would silently reintroduce a pip install inside
    Docker, which is minutes, into the command whose only promise is that it is fast.
    """
    services = Q.compose_document(6001)["services"]
    assert set(services) == {"db"}
    assert "build" not in services["db"]


# --------------------------------------------------------------------------------------------
# Provisioning.
# --------------------------------------------------------------------------------------------


def test_provision_writes_the_compose_file_and_waits_on_the_published_dsn(
    tmp_path, no_docker
) -> None:
    """`up --wait` is not sufficient, so the DSN the caller will use is polled directly.

    The healthcheck is `pg_isready` INSIDE the container, and the temporary server `initdb` runs
    answers it, so the check can pass while the real server is still restarting to listen on TCP.
    """
    root = tmp_path / "qs"
    dsn, port, compose_path, reused = Q.provision(root)

    assert compose_path == root / Q.COMPOSE_NAME
    assert compose_path.is_file()
    assert dsn == f"postgresql://recall:recall@127.0.0.1:{port}/recall"
    assert reused is False
    assert ("bring_up", compose_path, Q.PROJECT_NAME, ("db",)) in no_docker
    assert ("wait", dsn) in no_docker


def test_provision_reuses_the_port_it_already_chose(tmp_path, no_docker, monkeypatch) -> None:
    """⛔ Re-choosing on a re-run strands the first container.

    `choose_port` returns a FREE port, so the second run would never return the one already bound
    by the first. The reader would get a DSN pointing at a fresh empty database while their indexed
    data sat in a container nothing referred to any more.
    """
    root = tmp_path / "qs"
    _, first_port, _, first_reused = Q.provision(root)

    # Made loud rather than merely unused: if the second run reaches `choose_port` at all, that is
    # the defect, and a test that only compared the two ports would pass whenever the stub happened
    # to return the same number.
    monkeypatch.setattr(
        Q, "choose_port", lambda *a, **k: pytest.fail("re-run must not choose a new port")
    )
    _, second_port, _, second_reused = Q.provision(root)

    assert first_reused is False and second_reused is True
    assert second_port == first_port


def test_provision_survives_an_unparseable_compose_file(tmp_path, no_docker) -> None:
    """A truncated compose file is not a reason to refuse; it is a reason to write a good one."""
    root = tmp_path / "qs"
    root.mkdir(parents=True)
    (root / Q.COMPOSE_NAME).write_text("{not json", encoding="utf-8")

    dsn, port, compose_path, reused = Q.provision(root)
    assert reused is False
    assert json.loads(compose_path.read_text(encoding="utf-8"))["name"] == Q.PROJECT_NAME
    assert str(port) in dsn


# --------------------------------------------------------------------------------------------
# Teardown.
# --------------------------------------------------------------------------------------------


def test_remove_reports_that_there_was_nothing_to_remove(tmp_path) -> None:
    """A reader who never ran the quickstart must not be told something was deleted."""
    assert Q.remove_stack(tmp_path / "never-created") is False


def test_remove_tears_down_this_project_only_and_deletes_the_compose_file(
    tmp_path, no_docker, monkeypatch
) -> None:
    """The `-p` argument is what confines `down -v` to the volume this module created."""
    root = tmp_path / "qs"
    Q.provision(root)

    seen: list[list[str]] = []
    monkeypatch.setattr(
        Q.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or _completed()
    )
    assert Q.remove_stack(root) is True

    (command,) = seen
    assert command[:2] == ["docker", "compose"]
    assert "-p" in command and command[command.index("-p") + 1] == Q.PROJECT_NAME
    assert command[-2:] == ["down", "-v"]
    assert not (root / Q.COMPOSE_NAME).exists()


def test_remove_does_not_raise_when_the_stack_is_already_gone(tmp_path, no_docker, monkeypatch) -> None:
    """`--remove` run twice is a reader being careful, not a reader making a mistake."""
    root = tmp_path / "qs"
    Q.provision(root)
    monkeypatch.setattr(Q.subprocess, "run", lambda cmd, **kw: _failed())
    assert Q.remove_stack(root) is True
    assert Q.remove_stack(root) is False


def _completed():
    import subprocess

    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _failed():
    import subprocess

    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no such stack")


# --------------------------------------------------------------------------------------------
# Docker preflight.
# --------------------------------------------------------------------------------------------


def test_missing_docker_is_reported_with_both_ways_forward(monkeypatch) -> None:
    """The message has to name the escape, because a reader without Docker still has a path.

    Refusing with "docker not found" and stopping there loses everyone who already runs PostgreSQL,
    which for this project's audience is not a small fraction.
    """
    monkeypatch.setattr(Q.shutil, "which", lambda name: None)
    reason = Q.docker_unavailable_reason()
    assert reason is not None
    assert "--existing-dsn" in reason or "existing" in reason.lower()
    assert "docker.com" in reason or "docker" in reason.lower()


def test_a_stopped_daemon_is_distinguished_from_a_missing_install(monkeypatch) -> None:
    """These are different problems with different remedies and used to render identically.

    "docker is installed but the daemon is not answering" tells a Windows reader to start Docker
    Desktop. "docker is not on PATH" tells them to install it. Collapsing the two sends most of
    them to a download page for software they already have.
    """
    monkeypatch.setattr(Q.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(Q.subprocess, "run", lambda *a, **k: _failed())
    reason = Q.docker_unavailable_reason()
    assert reason is not None
    assert "daemon" in reason


def test_docker_available_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(Q.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(Q.subprocess, "run", lambda *a, **k: _completed())
    assert Q.docker_unavailable_reason() is None


def test_the_daemon_probe_is_the_cheap_one_and_its_timeout_has_real_margin(monkeypatch) -> None:
    """⚠️ `docker info` is not interchangeable with `docker version` here, and the reason is the
    timeout rather than the average cost.

    `info` gathers the whole system inventory, so its cost scales with what is on the machine.
    Measured 2026-08-25 on a workstation holding 37 containers: median **14.11s** over five runs,
    spread **4.01s to 56.36s**, a factor of 14 inside one hour. Against a 60 second bound that is
    1.07x the observed maximum, so a marginally busier daemon makes this function report a
    perfectly healthy Docker as "installed but did not respond" and sends the reader to fix
    something that is not broken. `version` measured median **0.48s**, 29.1x faster.
    Record: `docs/preregistrations/2026-08-25-docker-probe-latency.md`.

    Two assertions, and the second is the one with teeth. Pinning the argv alone would let somebody
    restore `info` under a generous timeout and stay green; pinning the RATIO says what the bound is
    for. 20s against a 0.48s probe is a margin of roughly 40x, and anything under 10x means the
    probe has grown expensive enough to trip its own backstop again.
    """
    seen: dict[str, object] = {}

    def _capture(argv, **kwargs):
        seen["argv"] = argv
        seen["timeout"] = kwargs.get("timeout")
        return _completed()

    monkeypatch.setattr(Q.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(Q.subprocess, "run", _capture)
    assert Q.docker_unavailable_reason() is None

    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[:2] == ["docker", "version"], (
        f"{argv[:2]} is not the cheap probe; `docker info` costs 14s median and 56s worst case on "
        "a machine with a normal number of containers"
    )

    # ⚠️ The denominator is the WORST observed sample, not the median, and it is looked up BY THE
    # PROBE'S OWN ARGV. The first version divided by a frozen `0.48`, which made the assertion
    # algebraically `timeout >= 4.8` — it could not fail on the condition its message named,
    # because the probe's cost never entered it. Keyed this way, restoring `docker info` under a
    # 60s timeout fails here (60/56.36 = 1.06), which is the regression worth catching.
    observed_worst_case = {("docker", "version"): 3.53, ("docker", "info"): 56.36}
    worst = observed_worst_case[tuple(argv[:2])]
    timeout = seen["timeout"]
    assert isinstance(timeout, (int, float))
    assert timeout / worst >= 3, (
        f"timeout={timeout}s leaves only {timeout / worst:.1f}x margin over the slowest observed "
        f"{' '.join(argv[:2])} run ({worst}s), which is how the old probe came to report a healthy "
        "daemon as unresponsive"
    )


# --------------------------------------------------------------------------------------------
# The CLI surface.
# --------------------------------------------------------------------------------------------


def test_cli_remove_does_not_provision(monkeypatch, capsys) -> None:
    """`--remove` must be reachable on a machine where provisioning would fail."""
    from recall import cli

    monkeypatch.setattr(Q, "provision", lambda *a, **k: pytest.fail("must not provision"))
    monkeypatch.setattr(Q, "remove_stack", lambda *a, **k: True)
    cli.main(["quickstart", "--remove"])
    assert "removed" in capsys.readouterr().out


def test_cli_refuses_without_docker_before_writing_anything(monkeypatch, tmp_path) -> None:
    """Preflight comes first, so a reader without Docker gets advice and not a stray directory."""
    from recall import cli

    monkeypatch.setattr(Q, "docker_unavailable_reason", lambda: "docker is not on PATH. Install it")
    monkeypatch.setattr(Q, "provision", lambda *a, **k: pytest.fail("refused runs must not reach here"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["quickstart"])
    assert "docker is not on PATH" in str(excinfo.value)
    assert not (tmp_path / "quickstart").exists()


def test_cli_does_not_declare_that_it_opens_a_database(monkeypatch) -> None:
    """⚠️ `_opens_db` would make `main()` resolve, guard and refuse over a DSN that does not exist yet.

    This command PROVISIONS its database. Declaring the flag would put it behind the secure-DSN
    guard and the broken-.env refusal, both acting on the DEFAULT DSN rather than on the one about
    to be created, so a reader with a bad `.env` would be refused by the very command that exists
    to get them past exactly that.

    Read off the real parser through the real dispatch, because a namespace built here would be a
    second definition free to agree with this test while disagreeing with the CLI.
    """
    from recall import cli
    from recall.cli_commands import setup_wizard

    captured: dict[str, object] = {}
    monkeypatch.setattr(setup_wizard, "_quickstart", lambda args: captured.update(args=args))
    cli.main(["quickstart"])

    args = captured["args"]
    assert getattr(args, "_opens_db", False) is False
    assert args.cmd == "quickstart"


def test_cli_existing_dsn_is_a_distinct_flag_from_the_global_one(monkeypatch) -> None:
    """`--existing-dsn`, never a second `--dsn`.

    The parent parser owns `--dsn` (aliased from `--serving-dsn`, and carrying a default). A
    subparser option of the same name would be a second `--dsn` whose meaning depended on which
    side of the word `quickstart` the reader typed it, and only one of the two would ever be read.
    """
    from recall import cli
    from recall.cli_commands import setup_wizard

    captured: dict[str, object] = {}
    monkeypatch.setattr(setup_wizard, "_quickstart", lambda args: captured.update(args=args))
    cli.main(
        [
            "--dsn",
            "postgresql://global/one",
            "quickstart",
            "--existing-dsn",
            "postgresql://mine/two",
        ]
    )

    args = captured["args"]
    assert args.existing_dsn == "postgresql://mine/two"
    assert args.dsn == "postgresql://global/one"


def _run_quickstart_recording_migrations(monkeypatch, apply) -> list[str]:
    """Drive `recall quickstart --existing-dsn` with `apply_migrations` replaced, reporting calls.

    ⚠️ Patched on `recall.schema`, NOT on `recall.cli_commands.setup_wizard`. `_quickstart`
    imports the name locally (`from recall.schema import ...`), so that import rebinds it inside
    the function on every call and a patch on the command module's attribute is never consulted.
    The first version of this test did exactly that, and its "stub" reached a real
    `psycopg.connect`.
    """
    import recall.schema
    from recall import cli
    from recall.cli_commands import setup_wizard

    migrated: list[str] = []

    def _apply(dsn, *, table, dim):
        migrated.append(table)
        apply(table)

    monkeypatch.setattr(recall.schema, "apply_migrations", _apply)
    monkeypatch.setattr(Q, "provision", lambda *a, **k: pytest.fail("must not provision"))
    monkeypatch.setattr(setup_wizard, "PgVectorStore", lambda *a, **k: _StubStore())
    monkeypatch.setattr(setup_wizard, "Indexer", lambda *a, **k: _StubIndexer())
    monkeypatch.setattr(setup_wizard, "_run_queries", lambda *a, **k: None)
    monkeypatch.setattr(setup_wizard, "_entailment_judge", lambda: None)
    # ⚠️ **`hashing`, NOT the default, and CI is the only place that shows why.** `quickstart`
    # resolves an embedder before it touches the database, and the parent parser's default is
    # `fastembed`, which lives behind an extra CI deliberately does not install. Locally this
    # passed; on CI both of these died with `SystemExit: embedder 'fastembed': ImportError`.
    # Nothing here needs real vectors (every collaborator below is stubbed) and only `.dim` is
    # read, so the built-in embedder is both sufficient and the one that cannot depend on extras.
    # Same shape as the launch-order test that assumed PySide6 was installed.
    monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
    cli.main(
        ["--embedder", "hashing", "quickstart", "--existing-dsn", "postgresql://example/db"]
    )
    return migrated


def test_a_database_with_globals_applied_is_never_touched_beyond_our_table(monkeypatch) -> None:
    """⚠️ The `--existing-dsn` case, and the reason this is a `try` rather than two calls.

    A reader pointing the quickstart at a database they already use has a `chunks` table at THEIR
    embedder's width. Migrating the default target unconditionally asks for it at this command's
    width and raises `SchemaIncompatible: table 'chunks' uses vector(64), requested dimension is
    384`, so the quickstart would refuse to run against exactly the database its own flag invites,
    while breaking the isolation it advertises. Measured against this suite's own container, which
    conftest bootstraps at dim 64.
    """
    migrated = _run_quickstart_recording_migrations(monkeypatch, apply=lambda table: None)
    assert migrated == [Q.QUICKSTART_TABLE], "chunks must not be opened when it is not needed"


def test_a_fresh_database_bootstraps_the_default_target_then_retries(monkeypatch) -> None:
    """⛔ On a FRESH database, `quickstart_chunks` alone raises `SchemaTooOld`.

    Global generation migrations are recorded against the default target and no other table may be
    migrated before them. This is the state every new reader's database is in, and it is invisible
    to the rest of this suite: conftest's autouse `_bootstrap_default_test_schema` has already
    migrated `chunks` before any test runs. That is precisely how the bug survived both the unit
    tests and a manual run against a real container, and it was caught only by pointing the
    command at a genuinely empty one.
    """
    from recall.schema import SchemaTooOld
    from recall.store import DEFAULT_TABLE

    seen: list[str] = []

    def _apply(table: str) -> None:
        # Refuse the quickstart table until the default target has been through, which is what the
        # real guard in `recall/schema.py` does.
        if table != DEFAULT_TABLE and DEFAULT_TABLE not in seen:
            raise SchemaTooOld("global generation migrations must be applied through `chunks`")
        seen.append(table)

    migrated = _run_quickstart_recording_migrations(monkeypatch, apply=_apply)
    assert migrated == [Q.QUICKSTART_TABLE, DEFAULT_TABLE, Q.QUICKSTART_TABLE]
    assert seen == [DEFAULT_TABLE, Q.QUICKSTART_TABLE]


class _StubStore:
    """Enough of `PgVectorStore` for the dispatch to run without a database."""

    table = Q.QUICKSTART_TABLE

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def check_schema(self):
        return None

    def count(self):
        return 22


class _StubIndexer:
    def index_path(self, path):
        return type("Stats", (), {"files": 22, "chunks": 22})()


def test_next_steps_never_echoes_a_dsn_the_reader_supplied() -> None:
    """⛔ A supplied DSN may carry a real password, and the closing block is copyable prose.

    Echoing it writes that password into the reader's scrollback, their shell history, and any
    transcript or recording of the session. This is the reason `next_steps` is a function with a
    return value rather than four `print` calls inline.
    """
    supplied = "postgresql://operator:CHANGEME-placeholder@db.internal:5432/prod"
    lines = "\n".join(Q.next_steps(supplied, provisioned=False, compose_path=None))

    assert "CHANGEME-placeholder" not in lines
    assert "db.internal" not in lines
    assert "<your dsn>" in lines


def test_next_steps_prints_the_quickstarts_own_dsn_in_full() -> None:
    """The stack this module created has README-published credentials bound to loopback.

    Redacting it too would be security theatre that costs the reader a retype: the placeholder
    exists to protect a secret, and this DSN is not one.
    """
    own = "postgresql://recall:recall@127.0.0.1:5497/recall"
    lines = "\n".join(
        Q.next_steps(own, provisioned=True, compose_path=Path("/tmp/x/docker-compose.quickstart.yml"))
    )

    assert own in lines
    assert "quickstart --remove" in lines


def test_next_steps_offers_removal_only_for_a_stack_it_created() -> None:
    """`--remove` against a database the reader supplied would be a lie, and a frightening one."""
    lines = "\n".join(Q.next_steps("postgresql://x/y", provisioned=False, compose_path=None))
    assert "--remove" not in lines


# --------------------------------------------------------------------------------------------
# The one test that checks the prose against retrieval.
# --------------------------------------------------------------------------------------------


@requires_db
def test_the_demo_corpus_actually_supersedes_and_actually_abstains(monkeypatch) -> None:
    """The command prints three claims about its three results. This is the test of those claims.

    Everything above pins plumbing. None of it can tell the difference between a quickstart that
    demonstrates the product and one that prints "the second one is the reason this project exists"
    above three unremarkable hits, which is the failure that would actually cost the reader.

    Uses `fastembed` rather than `HashingEmbedder`: abstention is a cosine threshold decision, and a
    hashing embedder's cosines carry no semantics, so a passing assertion would mean nothing about
    what the reader sees.
    """
    pytest.importorskip("fastembed")
    import uuid

    from recall.embeddings import resolve_embedder
    from recall.index import Indexer
    from recall.schema import apply_migrations
    from recall.store import PgVectorStore

    from .conftest import dev_search

    embedder = resolve_embedder("fastembed")
    table = "qs_" + uuid.uuid4().hex[:8]
    apply_migrations(TEST_DSN, table=table, dim=embedder.dim)
    store = PgVectorStore(TEST_DSN, dim=embedder.dim, table=table, tenant=Q.QUICKSTART_TENANT)
    try:
        store.check_schema()
        stats = Indexer(store, embedder).index_path(Q.demo_corpus())
        assert stats.files >= 20, "the demo corpus lost documents"

        # Claim 1: "an ordinary question, answered from the corpus".
        answered = dev_search(store, embedder, Q.DEMO_QUERIES[0], k=5)
        assert not answered.abstained, "the answerable question was refused"
        assert any("datastore_choice" in (h.provenance.file or "") for h in answered.hits)

        # Claim 2: the retracted answer is retrieved AND marked, and points at its replacement.
        # Retrieval is asserted first and separately: a corpus edit that stopped returning
        # `cache_ttl_v1` at all would leave the verdict assertions vacuously true over an empty
        # list, and the demo would quietly stop demonstrating anything.
        superseded = dev_search(store, embedder, Q.DEMO_QUERIES[1], k=5)
        retracted = [h for h in superseded.hits if "cache_ttl_v1" in (h.provenance.file or "")]
        assert retracted, "the retracted claim is not retrieved at all, so nothing is demonstrated"
        assert all(h.verdict == "superseded" for h in retracted)
        assert all("cache_ttl_v2" in (h.validity.superseded_by or "") for h in retracted)

        # Claim 3: refused, with margin. The boolean alone is not enough: `recall demo`'s query
        # sits 0.005 the WRONG side of this threshold, so a demo can pass "is it absent from the
        # corpus" and still answer. Asserting the margin makes drift toward the edge visible while
        # it is still drift, rather than on the day it flips on somebody else's machine.
        from recall.guards import DEFAULT_GAP_THRESHOLD

        refused = dev_search(store, embedder, Q.DEMO_QUERIES[2], k=5)
        assert refused.abstained, "the unanswerable query was answered"
        top = max((h.cosine for h in refused.hits), default=0.0)
        assert top < DEFAULT_GAP_THRESHOLD - 0.02, (
            f"top cosine {top:.3f} is within 0.02 of the {DEFAULT_GAP_THRESHOLD} threshold; this "
            "query is about to stop abstaining. Re-choose it by measurement, not by intuition."
        )
    finally:
        store.drop_table()
        store.close()
