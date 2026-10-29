"""Transaction accounting must agree with the SQL actually sent to PostgreSQL.

NUM-001 / FIX-001: the pre-fix _CountingConnection.transaction in 32616a7fe
records two control statements for every block. A failed nested block sends
SAVEPOINT, ROLLBACK TO and RELEASE. The exact regression node is
test_transaction_count_matches_wire_statements[nested-error]; its intended red
assertion compares the trace counter with the independently captured wire count.
The successful cases are controls for behavior that already counted correctly.
"""

from contextlib import nullcontext
import re

import psycopg
import pytest

from recall.observability import PerformanceTrace, performance_trace_scope
from recall.store import _observed_db_call
from tests.conftest import TEST_DSN, requires_db


@requires_db
@pytest.mark.parametrize("outcome", ["commit", "error", "forced", "rollback"])
@pytest.mark.parametrize("nested", [False, True], ids=["outer", "nested"])
def test_transaction_count_matches_wire_statements(tmp_path, nested, outcome):
    trace = PerformanceTrace()
    wire_path = tmp_path / "transaction.trace"

    def operation(conn):
        with conn.transaction(force_rollback=outcome == "forced"):
            conn.execute("SELECT 1 / 0" if outcome == "error" else "SELECT 1")
            if outcome == "rollback":
                raise psycopg.Rollback()

    with psycopg.connect(TEST_DSN, autocommit=True) as connection:
        # The enclosing shared-pool transaction is outside this operation's count.
        with connection.transaction() if nested else nullcontext():
            with wire_path.open("wb") as wire:
                connection.pgconn.trace(wire.fileno())
                try:
                    with performance_trace_scope(trace):
                        if outcome == "error":
                            with pytest.raises(psycopg.errors.DivisionByZero):
                                _observed_db_call(connection, operation)
                        else:
                            _observed_db_call(connection, operation)
                finally:
                    connection.pgconn.untrace()

    sent = re.findall(r"\tF\t\d+\t(?:Query|Parse)\t", wire_path.read_text())
    assert sent, "libpq must expose the statements used as the independent oracle"
    assert trace.snapshot()["counters"]["db_statement_count"] == len(sent)
