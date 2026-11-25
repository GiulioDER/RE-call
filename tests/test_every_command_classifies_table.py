"""Every `recall` command is classified for the global `--table` flag.

Invariant: each top-level command of `recall.cli.build_parser()` appears in exactly one of the
three sets below, so a new command cannot arrive without its author deciding what `--table` means
for it.

Failure mode it catches (CCA CODE-003, run 20261119T182325Z): `--table` is a global flag every
command accepts, and #1415 and #1421 were both commands that accepted it and silently acted on
another table. The refusals since are written per command, and the tests that cover them list
their commands by hand, so a new command that neither honours nor refuses the flag would go
unnoticed.

This test does not check that a command in `TABLE_AWARE` really honours or refuses the flag; the
per command tests do (tests/test_generation_route_table_flag.py,
tests/test_generations_commands_table_flag.py). It checks that the decision was made.

Red proof, 2026-11-19, by mutation of `recall.cli.build_parser` with one extra subcommand
registered: failed at the assertion below, naming it as unclassified.
"""

from __future__ import annotations

import argparse

from recall.cli import build_parser

#: Commands that read `--table` or refuse a value they would not honour.
TABLE_AWARE = frozenset(
    {
        "calibrate", "calibration", "demo", "doctor", "forget", "generation", "graph",
        "idempotency", "index", "lint", "lite", "provenance", "reasoning", "records", "schema", "scopes",
        "search", "setup", "withdraw", "wizard",
    }
)

#: Commands that use no chunk table `--table` could name.
NO_CHUNK_TABLE = {
    "backup": "manages Aurora backups and receipts",
    "dashboard": "serves the local dashboard",
    "extract": "edits memo files",
    "rewrite": "edits memo files",
    "manifest": "creates and verifies S3 corpus manifests",
    "secret": "manages provider secrets",
    "uninstall": "removes installed integration files",
    "check": "checks memo files",
    "repair-checks": "reads caller JSON files and opens no store",
}

#: Commands that use a table of their own whatever `--table` says. Listed so the gap is visible.
OWN_TABLE = {
    "code": "indexes into the fixed recall_code table",
    "quickstart": "builds its own quickstart_chunks table",
}


def _commands() -> set[str]:
    parser = build_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(actions) == 1, "expected one top-level subcommand group"
    return set(actions[0].choices)


def test_every_command_is_classified_for_table_exactly_once() -> None:
    classified = [TABLE_AWARE, set(NO_CHUNK_TABLE), set(OWN_TABLE)]
    commands = _commands()

    unclassified = commands - set().union(*classified)
    assert not unclassified, f"decide what --table means for: {sorted(unclassified)}"
    stale = set().union(*classified) - commands
    assert not stale, f"classified but no longer a command: {sorted(stale)}"
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not classified[a] & classified[b], classified[a] & classified[b]
