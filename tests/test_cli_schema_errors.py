"""A schema fault is operator-fixable by construction, so it must not arrive as a traceback.

Every member of the `SchemaError` family is raised deliberately with a message written for a
person: a database that simply needs `recall schema apply`, a table built for another embedder, a
migrator already holding the lock. All of them used to reach the CLI's caller as a Python
traceback, which reads as a crash in the tool and buries the sentence that says what to do.

These tests need no database. `schema_error_message` is a pure function of the exception, and the
dispatch test raises from a stubbed handler.
"""
from __future__ import annotations

import pytest

from recall.cli import _SCHEMA_REMEDY, main, schema_error_message
from recall.schema import (
    ConcurrentMigrator,
    InterruptedConcurrentIndex,
    MigrationChecksumMismatch,
    SchemaError,
    SchemaIncompatible,
    SchemaTooNew,
    SchemaTooOld,
)

DSN = "postgresql://u:pw@127.0.0.1:1/recall"

#: The two whose own message already ends in a remedy, so the renderer must add nothing.
SELF_EXPLAINING = [
    SchemaTooOld("table 'chunks' needs schema migration(s) ['0001']; run `recall schema apply`"),
    SchemaTooNew("table 'chunks' has unknown schema migration(s) ['0099']; upgrade RE-call"),
]


class TestTheMessageSurvives:
    @pytest.mark.parametrize(
        "exc",
        [
            *SELF_EXPLAINING,
            SchemaIncompatible("table 'chunks' uses vector(384), requested dimension is 64"),
            MigrationChecksumMismatch("migration 0004 checksum drift: committed a, actual b"),
            ConcurrentMigrator("another RE-call schema migrator is already running"),
            InterruptedConcurrentIndex("migration 0009 did not leave valid index 'x'"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_the_exceptions_own_words_are_never_replaced(self, exc: SchemaError) -> None:
        """The diagnosis is the part only the raise site knows. It must be quoted, not summarised."""
        assert str(exc) in schema_error_message(exc)


class TestARemedyIsAddedOnlyWhereOneIsMissing:
    @pytest.mark.parametrize("exc", SELF_EXPLAINING, ids=lambda e: type(e).__name__)
    def test_a_self_explaining_error_gains_nothing(self, exc: SchemaError) -> None:
        """Restating an existing remedy in different words leaves the reader to pick one.

        `SchemaTooOld` already ends "run `recall schema apply`". A second, differently-worded
        instruction underneath it is not extra help; it is two sources of truth.
        """
        assert schema_error_message(exc) == str(exc)

    def test_the_dimension_mismatch_gains_one(self) -> None:
        """The message that motivated this: a precise fact that implies no action.

        `table 'chunks' uses vector(384), requested dimension is 64` is exactly true and tells a
        reader nothing about what to do. It is also the one people hit, because it fires whenever
        a table meets an embedder it was not built for.
        """
        message = schema_error_message(
            SchemaIncompatible("table 'chunks' uses vector(384), requested dimension is 64")
        )
        assert "--embedder" in message
        assert "--table" in message

    def test_a_subclass_inherits_no_remedy(self) -> None:
        """Looked up by exact type on purpose: wrong advice stated confidently is worse than none.

        A future subclass of `SchemaIncompatible` would mean something its parent does not, and
        `isinstance` lookup would hand it the parent's remedy with no sign anything was assumed.
        """
        class NarrowerFault(SchemaIncompatible):
            pass

        exc = NarrowerFault("something more specific went wrong")
        assert schema_error_message(exc) == str(exc)

    def test_every_remedy_is_keyed_to_a_real_schema_error(self) -> None:
        """A remedy keyed to a renamed class is dead code that silently stops applying."""
        for cls in _SCHEMA_REMEDY:
            assert issubclass(cls, SchemaError), cls


class TestTheCliConvertsItRatherThanCrashing:
    def test_a_handler_raising_schema_error_exits_with_the_message(self, monkeypatch) -> None:
        """End to end through `main`, so the dispatch wrapper is covered and not just the renderer."""
        def boom(_args):
            raise SchemaIncompatible("table 'chunks' uses vector(384), requested dimension is 64")

        # Patched before `main` builds the parser: `register` reads this module global when it
        # calls `set_defaults(func=...)`, and that happens inside `main`.
        monkeypatch.setattr("recall.cli_commands.index_search._cmd_search", boom)

        with pytest.raises(SystemExit) as excinfo:
            main(["--dsn", DSN, "search", "anything"])

        code = excinfo.value.code
        assert isinstance(code, str), "a bare status code is the traceback-shaped failure"
        assert "vector(384)" in code
        assert "--embedder" in code

    def test_a_non_schema_exception_still_propagates(self, monkeypatch) -> None:
        """The wrapper must not become a catch-all. A real bug has to keep its traceback."""
        def boom(_args):
            raise RuntimeError("a genuine bug, not an operator-fixable condition")

        monkeypatch.setattr("recall.cli_commands.index_search._cmd_search", boom)

        with pytest.raises(RuntimeError, match="a genuine bug"):
            main(["--dsn", DSN, "search", "anything"])
