"""The preflight that decides whether somebody else's PostgreSQL can host this install.

Every case here is a way the install used to fail LATE. The point of the module is that each one
becomes a sentence before anything is built, so the tests are written as "what does the user get
told", not "what does the function return".
"""

from __future__ import annotations

import psycopg
import pytest

from recall.wizard.database import (
    CONNECT_TIMEOUT_SECONDS,
    DatabaseReport,
    Finding,
    is_local_host,
    probe_database,
)
from tests.conftest import TEST_DSN, requires_db, restore_default_chunks_table


# ----------------------------------------------------------------------------------------------
# Local versus remote, which decides whether the credentials guard applies
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dsn, expected",
    [
        ("postgresql://recall:recall@127.0.0.1:5432/recall", True),
        ("postgresql://recall:recall@localhost:5432/recall", True),
        ("postgresql://recall:recall@[::1]:5432/recall", True),
        ("postgresql://recall:pw@db.example.invalid:5432/recall", False),
        ("postgresql://recall:pw@10.0.0.7:5432/recall", False),
    ],
)
def test_local_and_remote_are_told_apart(dsn: str, expected: bool) -> None:
    assert is_local_host(dsn) is expected


def test_a_host_named_after_a_local_one_is_still_remote() -> None:
    """Parsed, not substring-matched.

    A database *named* `localhost` on a remote server is not a contrived case: it is what happens
    when somebody names a database after the machine it replaced. Matching the substring would skip
    the credentials guard for exactly that host.
    """
    assert is_local_host("postgresql://u:p@db.example.invalid:5432/localhost") is False


def test_an_unparseable_dsn_is_treated_as_remote() -> None:
    """The safe direction: we cannot show it is local, and guessing local disables a guard."""
    assert is_local_host("postgresql://[not-a-valid-host/recall") is False


# ----------------------------------------------------------------------------------------------
# The report's own semantics, which are what the caller branches on
# ----------------------------------------------------------------------------------------------


def test_an_undetermined_finding_does_not_block() -> None:
    """⛔ Refusing on something never established would block installs for a broken probe.

    `ok=None` is a third state, not a falsy second one, and this is the assertion that keeps it one.
    """
    report = DatabaseReport(
        dsn="postgresql://u@h/db",
        findings=(Finding(name="privileges", ok=None, detail="could not be determined"),),
    )
    assert report.usable is True
    assert report.blockers == ()


def test_a_blocking_finding_makes_the_database_unusable_and_is_rendered_with_its_advice() -> None:
    report = DatabaseReport(
        dsn="postgresql://u@h/db",
        findings=(
            Finding(name="reachable", ok=True, detail="connected"),
            Finding(
                name="pgvector",
                ok=False,
                detail="not available on this server",
                blocking=True,
                advice="install the pgvector package",
            ),
        ),
    )
    assert report.usable is False
    assert [f.name for f in report.blockers] == ["pgvector"]
    rendered = report.render()
    assert "database NOT usable" in rendered
    assert "install the pgvector package" in rendered, (
        "advice a user cannot see is advice that was not given"
    )


# ----------------------------------------------------------------------------------------------
# Against a real database
# ----------------------------------------------------------------------------------------------


def test_an_unreachable_host_reports_rather_than_raising() -> None:
    """The preflight must never be the thing that fails; its whole job is to report.

    Port 1 is where `conftest` points an unconfigured DSN precisely because nothing listens there
    on any platform, so this needs no database of its own.
    """
    report = probe_database("postgresql://recall:recall@127.0.0.1:1/recall")

    assert report.usable is False
    assert [f.name for f in report.findings] == ["reachable"]
    assert report.findings[0].blocking is True
    assert "ssh -L" in report.findings[0].advice, (
        "the SSH tunnel recipe is the one piece of advice a remote user needs here, and it is "
        "documented rather than built"
    )


def test_the_connect_timeout_is_short_enough_that_a_person_will_wait_for_it() -> None:
    """A wrong hostname must not cost a minute of silence with a person watching."""
    assert 0 < CONNECT_TIMEOUT_SECONDS <= 15


@requires_db
def test_a_healthy_database_reports_every_check_and_blocks_nothing() -> None:
    report = probe_database(TEST_DSN)

    names = [f.name for f in report.findings]
    assert names == ["reachable", "server", "pgvector", "privileges", "schema"], (
        "a check that silently stops running looks identical to one that passed"
    )
    assert report.usable is True
    assert report.server_version, "the server version is what tells a user which database answered"


@requires_db
def test_the_dimension_query_reads_a_real_vector_column() -> None:
    """⚠️ The catalogue fact this rests on, asserted against pgvector rather than assumed.

    `atttypmod` is type-specific. For `varchar(n)` it is n + 4, and assuming that convention here
    would have made every dimension comparison wrong by four — reporting a mismatch on a schema
    that matches, and a match on one that does not. Verified: `vector(384)` gives exactly 384.

    A `vector` column with no declared dimension gives -1, which must not be read as a dimension.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        for declared, expected in ((384, 384), (1536, 1536)):
            connection.execute("DROP TABLE IF EXISTS _dimension_probe")
            connection.execute(f"CREATE TABLE _dimension_probe (embedding vector({declared}))")
            found = connection.execute(
                """
                SELECT a.atttypmod FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = '_dimension_probe' AND a.attname = 'embedding'
                """
            ).fetchone()
            assert found is not None and found[0] == expected, (
                f"vector({declared}) reported atttypmod {found}, so the dimension check is wrong"
            )

        connection.execute("DROP TABLE IF EXISTS _dimension_probe")
        connection.execute("CREATE TABLE _dimension_probe (embedding vector)")
        found = connection.execute(
            """
            SELECT a.atttypmod FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = '_dimension_probe' AND a.attname = 'embedding'
            """
        ).fetchone()
        assert found is not None and found[0] == -1, (
            "an undimensioned vector must be -1, which the probe treats as undetermined"
        )
        connection.execute("DROP TABLE IF EXISTS _dimension_probe")


@requires_db
def test_a_dimension_mismatch_blocks_the_install_before_anything_is_built() -> None:
    """⛔ The failure this module exists for.

    A mismatch does not surface during setup at all: the schema is already there so nothing is
    applied, the build runs, and the FIRST INSERT fails minutes in with a driver error naming
    neither the embedder nor the schema that disagree.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute("DROP TABLE IF EXISTS chunks CASCADE")
        connection.execute("CREATE TABLE chunks (embedding vector(384))")
    try:
        matching = probe_database(TEST_DSN, expected_dimension=384)
        mismatched = probe_database(TEST_DSN, expected_dimension=1536)
    finally:
        # ⚠️ RESTORED, not dropped. `chunks` is shared: the session bootstrap creates it at dim 64
        # and a dozen integration tests assume it. The first version of this line was a bare
        # `DROP TABLE IF EXISTS chunks CASCADE`, which left no table at all for everything that ran
        # afterwards — three `test_wizard_pipeline.py` tests failed in CI with
        # `relation "chunks" does not exist`, while the suite stayed green locally because only some
        # random orders put this test first.
        restore_default_chunks_table()

    assert matching.usable is True
    assert matching.existing_dimension == 384

    assert mismatched.usable is False
    schema = next(f for f in mismatched.findings if f.name == "schema")
    assert schema.blocking is True
    assert "384" in schema.detail and "1536" in schema.detail, (
        "both numbers must appear, or the user cannot tell which side to change"
    )
    assert "Do NOT drop" in schema.advice, (
        "the destructive fix is the obvious one and must not be the recommended one"
    )


def test_a_malformed_dsn_does_not_put_the_password_on_screen() -> None:
    """⛔ The driver echoes the WHOLE connection string when it cannot parse one.

    Measured: `ProgrammingError: missing "=" after "not-a-dsn://user:PASSWORD@x" in connection info
    string`. That detail is rendered onto a label in the desktop settings page, which is on screen,
    in screenshots, and in whatever a user pastes into an issue — and a mistyped DSN is precisely
    when somebody is staring at that label.

    The three WELL-FORMED failure modes were already clean, which is what made this easy to miss:
    unreachable port, bad host and wrong password all report without the secret. Only the parse
    failure leaks, and only the parse failure is the one a person hits while fixing a typo.

    The marker below is a placeholder, not a credential.
    """
    marker = "PLACEHOLDER-NOT-A-REAL-PASSWORD"

    report = probe_database(f"not-a-dsn://recall:{marker}@x")

    assert not report.usable
    assert marker not in report.render(), f"the password reached the report: {report.render()}"
    assert "***" in report.render(), "and it was redacted rather than merely dropped"


def test_the_well_formed_failures_stay_clean_too() -> None:
    """Asserted alongside the leak, so a later change cannot fix one and break these."""
    marker = "PLACEHOLDER-NOT-A-REAL-PASSWORD"

    for dsn in (
        f"postgresql://recall:{marker}@127.0.0.1:1/recall",
        f"postgresql://recall:{marker}@nonexistent.invalid:5432/recall",
    ):
        report = probe_database(dsn)
        assert marker not in report.render(), "a well-formed failure leaked the password"
