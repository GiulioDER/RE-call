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
from tests.test_lite_cli import _run

LITE = "sqlite:///tmp/memory.db"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


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
    out = _run(["--dsn", "sqlite:///" + path.as_posix(), *argv])
    assert "EXIT" in out and f"recall {argv[0]}" in out and "lite store" in out, out
    assert not path.exists(), "a refused command created the lite file"
