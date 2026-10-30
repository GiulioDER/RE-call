"""Execute the documented manual schema commands against a fresh disposable database."""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from recall.cli import main
from recall.schema import SchemaError
from recall.store import PgVectorStore
from tests.conftest import TEST_DSN, requires_db


@requires_db
def test_manual_installation_prepares_its_named_corpus():
    """DOC-001: documented provisioning must prepare recall_notes before indexing.

    Red-proof target: docs/INSTALLATION.md at baseline 9807af5f. The unchanged
    node must fail its compatibility assertion with SchemaTooOld, rather than
    failing setup. No embedding or paid API call is involved.
    """
    section = Path("docs/INSTALLATION.md").read_text().split(
        "## Manual schema and corpus path", 1
    )[1].split("## Integration setup", 1)[0]
    commands = [
        shlex.split(line)[1:]
        for line in section.splitlines()
        if line.startswith("recall ") and " schema " in line and line.endswith(" apply")
    ]
    assert commands, "manual installation has no executable schema commands"
    database = "install_doc_" + uuid.uuid4().hex[:12]
    dsn = make_conninfo(TEST_DSN, dbname=database)
    with psycopg.connect(TEST_DSN, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        try:
            for command in commands:
                command[command.index("--migration-dsn") + 1] = dsn
                main(["--dsn", dsn, *command])
            error = None
            try:
                with PgVectorStore(dsn, table="recall_notes", dim=384) as store:
                    store.check_schema()
            except SchemaError as exc:
                error = str(exc)
            assert error is None, f"documented migration left recall_notes unusable: {error}"
        finally:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )
