"""Every `recall` command is classified for a `sqlite:///` DSN, and the unserved ones refuse by name.

A command that cannot serve a lite store used to reach PostgreSQL and die with a psycopg
traceback ("missing '=' after 'sqlite:///...' in connection info string"), which names neither
the command nor the way out.

Invariants and the failure each one catches:
- R1 every top-level command of the full `recall` parser is in exactly one of `LITE_SERVED` and
  `LITE_REFUSED`. This is what keeps the deny list safe: a command added
  later without a decision about lite fails here instead of crashing on a lite DSN.
- R2 each refused command (and the refused forms of the partly served ones) yields a refusal that
  names the command and the lite store; the served forms yield none.
- R3 through the real CLI, a refused command exits by name before anything is opened, so the lite
  file is not created; a served command is not refused.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from recall import cli
from recall.backends import LITE_REFUSED, LITE_SERVED, lite_command_refusal
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture

LITE = "sqlite:///tmp/memory.db"


def test_r1_every_command_is_classified_for_lite() -> None:
    """Red: a command removed from both tables (`demo` dropped from `LITE_REFUSED`) fails
    "not classified for a lite DSN"."""
    # The full parser's own subcommands: the registry's name sets omit commands a module registers
    # beside its main one (`calibrate` is registered by the `calibration` module).
    parser = cli.build_parser()
    registered = {
        name
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for name in action.choices
    }
    unclassified = registered - LITE_SERVED - set(LITE_REFUSED)
    assert not unclassified, f"not classified for a lite DSN: {sorted(unclassified)}"
    assert not LITE_SERVED & set(LITE_REFUSED), "a command is both served and refused on lite"
    assert not (LITE_SERVED | set(LITE_REFUSED)) - registered, "a classified command no longer exists"


def _ns(cmd: str, **extra: object) -> argparse.Namespace:
    return argparse.Namespace(cmd=cmd, dsn=LITE, **extra)


@pytest.mark.parametrize(
    "args",
    [
        _ns("generation", generation_cmd="list"),
        _ns("graph", graph_cmd="rebuild"),
        _ns("reasoning", reasoning_cmd="audit"),
        _ns("idempotency", idempotency_cmd="reconcile"),
        _ns("schema", schema_cmd="status"),
        _ns("calibrate"),
        _ns("demo"),
        _ns("code"),
        _ns("lint", semantic=True),
        _ns("setup", lite=False),
        argparse.Namespace(cmd="quickstart", dsn=None, existing_dsn=LITE),
    ],
    ids=lambda a: a.cmd,
)
def test_r2_refused_forms_name_the_command(args: argparse.Namespace) -> None:
    """Red: `lite_command_refusal` returning None for every command fails each case."""
    refusal = lite_command_refusal(args)
    assert refusal is not None and f"recall {args.cmd}" in refusal and "lite store" in refusal, refusal


@pytest.mark.parametrize(
    "args",
    [
        _ns("schema", schema_cmd="grants"),
        _ns("lint", semantic=False),
        _ns("setup", lite=True),
        argparse.Namespace(cmd="quickstart", dsn=LITE, existing_dsn=None),
        _ns("index"),
        _ns("search"),
        argparse.Namespace(cmd="generation", dsn="postgresql://u:p@db/recall", generation_cmd="list"),
    ],
    ids=lambda a: f"{a.cmd}-{a.dsn.split(':')[0] if a.dsn else 'none'}",
)
def test_r2_served_forms_are_not_refused(args: argparse.Namespace) -> None:
    """Red: `schema grants` no longer exempted (the exemption removed) fails the first case."""
    assert lite_command_refusal(args) is None


@pytest.mark.parametrize("argv", [["generation", "list"], ["schema", "status"], ["setup"], ["demo"]])
def test_r3_the_cli_refuses_before_opening_anything(tmp_path: Path, argv: list[str]) -> None:
    """Red: the `lite_command_refusal` call removed from `recall.cli._main`: the command reaches
    its handler (psycopg, or the setup wizard) instead of exiting by name."""
    path = tmp_path / "memory.db"
    out = run_cli(["--dsn", "sqlite:///" + path.as_posix(), *argv])
    assert "EXIT" in out and f"recall {argv[0]}" in out and "lite store" in out, out
    assert not path.exists(), "a refused command created the lite file"


PG = "postgresql://u:p@db/recall"


def _parsed(argv: list[str]) -> argparse.Namespace:
    """`argv` through the real parser, so a renamed argument cannot hide behind a hand-built Namespace."""
    return cli.build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--dsn", LITE, "lint", ".", "--semantic"],
        ["quickstart", "--existing-dsn", LITE],
    ],
    ids=["lint-semantic", "quickstart-existing-dsn"],
)
def test_r4_refused_forms_through_the_real_parser(argv: list[str]) -> None:
    """CODE-002. Red: `lint`'s `--semantic` stored under another dest (`dest="semantic_lint"` in
    `recall/cli_commands/lint_check.py`): the hand-built R2 case stays green, this one goes red,
    because `lite_command_refusal` reads `semantic` and finds nothing."""
    refusal = lite_command_refusal(_parsed(argv))
    assert refusal is not None and "lite store" in refusal, refusal


def test_r5_schema_is_judged_on_the_dsn_it_opens() -> None:
    """BUG-001. `schema status|plan` open `--migration-dsn` when given, else the serving DSN;
    `schema apply` opens only `--migration-dsn`. Red: judging `args.dsn` for schema (the code before
    this fix) refuses the first case, which opens PostgreSQL, and lets the third through."""
    assert lite_command_refusal(_parsed(["--dsn", LITE, "--migration-dsn", PG, "schema", "status"])) is None
    refused = lite_command_refusal(_parsed(["--dsn", LITE, "schema", "plan"]))
    assert refused is not None and LITE in refused, refused
    applied = lite_command_refusal(_parsed(["--dsn", PG, "--migration-dsn", LITE, "schema", "apply"]))
    assert applied is not None and LITE in applied, f"a lite migration DSN was not refused: {applied}"


def test_r6_a_postgres_run_does_not_import_the_lite_store() -> None:
    """PERF-001. The classification costs a non-lite run nothing: `recall.lite.store` (and through
    it calibration, embeddings and sqlite3) is imported only when a DSN the run judges is
    `sqlite:`. Red: the unconditional `from recall.backends import lite_command_refusal` in
    `recall.cli._main` imports it on `recall schema grants`, which opens nothing."""
    import subprocess
    import sys

    code = (
        "import sys, contextlib, io\n"
        "from recall import cli\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        f"    cli.main(['--dsn', {PG!r}, 'schema', 'grants', '--role', 'recall_reader'])\n"
        "print('recall.lite.store' in sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "False", "a PostgreSQL run imported the lite store"


def test_r7_a_refusal_names_where_the_lite_dsn_came_from() -> None:
    """A user moving to PostgreSQL with a lite DSN still exported is told the variables and the flag
    that chose the lite store, not only "run `recall setup --lite`".

    Red, 2026-10-13, against the pre-fix `lite_command_refusal` (the message ended at the reason):
    fails on "the refusal did not say where the lite DSN came from".
    """
    refusal = lite_command_refusal(argparse.Namespace(cmd="setup", dsn=LITE, lite=False))
    assert refusal is not None
    assert "RECALL_SERVING_DSN" in refusal and "RECALL_DSN" in refusal and "--serving-dsn" in refusal, (
        f"the refusal did not say where the lite DSN came from: {refusal}"
    )
    # `schema apply` opens the migration DSN, so that is the one to name.
    refusal = lite_command_refusal(
        argparse.Namespace(cmd="schema", schema_cmd="apply", migration_dsn=LITE, dsn="postgresql://x/y")
    )
    assert refusal is not None and "--migration-dsn" in refusal and "--serving-dsn" not in refusal, (
        f"the schema refusal named the wrong DSN: {refusal}"
    )
