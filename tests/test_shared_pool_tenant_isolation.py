"""Tenant context must not survive a transaction, on ANY exit path.

One pool per process means a connection is reused across tenants, so the question that decides
whether that is safe is narrow and mechanical: after the transaction ends, is the tenant GUC
gone? These tests take every exit a request can actually take — success, exception, cancellation,
statement timeout, transaction abort — and ask it.

The last class in the file is the one that keeps the rest honest. Every test above it would still
pass if `_reset` were a no-op, because `SET LOCAL` alone is what clears the GUC. So the reset guard
is exercised directly against a deliberately poisoned connection: a guard that has never been shown
to fire is a hypothesis.
"""

from __future__ import annotations

import threading

import psycopg
import pytest

from recall.pool import TENANT_GUC, SharedPool, TenantContextLeak
from tests.conftest import TEST_DSN, requires_db

pytestmark = requires_db


def _tenant_now(conn: psycopg.Connection) -> str:
    row = conn.execute(f"SELECT current_setting('{TENANT_GUC}', true)").fetchone()
    return "" if row is None or row[0] is None else str(row[0])


@pytest.fixture
def pool():
    p = SharedPool(TEST_DSN, min_size=2, max_size=6)
    yield p
    p.close()


class TestTenantIsSetInsideAndGoneAfter:
    def test_set_inside_the_transaction(self, pool: SharedPool) -> None:
        with pool.tenant_transaction("acme") as conn:
            assert _tenant_now(conn) == "acme"

    def test_cleared_after_a_successful_commit(self, pool: SharedPool) -> None:
        with pool.tenant_transaction("acme"):
            pass
        # A fresh checkout must not inherit it. This is the whole safety property.
        with pool.open().connection() as conn:
            assert _tenant_now(conn) == ""

    def test_empty_tenant_is_refused_rather_than_matching_nothing(self, pool: SharedPool) -> None:
        for bad in ("", "   "):
            with pytest.raises(ValueError):
                with pool.tenant_transaction(bad):
                    pass


class TestEveryExitPathClearsIt:
    """A.8: success, exception, cancellation, timeout, transaction abort."""

    def test_exception_inside_the_transaction(self, pool: SharedPool) -> None:
        with pytest.raises(RuntimeError):
            with pool.tenant_transaction("acme") as conn:
                assert _tenant_now(conn) == "acme"
                raise RuntimeError("caller blew up mid-request")
        with pool.open().connection() as conn:
            assert _tenant_now(conn) == ""

    def test_generator_closed_early_is_a_cancellation(self, pool: SharedPool) -> None:
        """Closing the context manager's generator early models a cancelled request."""
        gen = pool.tenant_transaction("acme")
        conn = gen.__enter__()
        assert _tenant_now(conn) == "acme"
        gen.__exit__(GeneratorExit, GeneratorExit(), None)
        with pool.open().connection() as fresh:
            assert _tenant_now(fresh) == ""

    def test_statement_timeout(self) -> None:
        timed = SharedPool(TEST_DSN, min_size=1, max_size=3, statement_timeout_ms=150)
        try:
            with pytest.raises(psycopg.errors.QueryCanceled):
                with timed.tenant_transaction("acme") as conn:
                    conn.execute("SELECT pg_sleep(3)")
            with timed.open().connection() as conn:
                assert _tenant_now(conn) == ""
        finally:
            timed.close()

    def test_transaction_abort_from_a_sql_error(self, pool: SharedPool) -> None:
        with pytest.raises(psycopg.errors.UndefinedTable):
            with pool.tenant_transaction("acme") as conn:
                conn.execute("SELECT * FROM a_table_that_does_not_exist_9f3a")
        with pool.open().connection() as conn:
            assert _tenant_now(conn) == ""

    def test_nested_rollback_still_clears(self, pool: SharedPool) -> None:
        with pool.tenant_transaction("acme") as conn:
            try:
                with conn.transaction():
                    conn.execute("SELECT 1/0")
            except psycopg.errors.DivisionByZero:
                pass
            # The outer transaction survives, so the tenant is still correctly scoped to it.
            assert _tenant_now(conn) == "acme"
        with pool.open().connection() as conn:
            assert _tenant_now(conn) == ""


class TestConcurrentTenantsNeverSeeEachOther:
    def test_many_threads_each_observe_only_their_own_tenant(self, pool: SharedPool) -> None:
        """The pool is smaller than the thread count, so connections ARE reused across tenants."""
        tenants = [f"tenant_{i:02d}" for i in range(24)]
        seen: dict[str, str] = {}
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(tenants))

        def worker(name: str) -> None:
            try:
                barrier.wait(timeout=30)
                with pool.tenant_transaction(name) as conn:
                    # Read it back several times with work in between: a leak introduced by
                    # another thread mid-transaction would show up here, not just at entry.
                    for _ in range(3):
                        conn.execute("SELECT pg_sleep(0.01)")
                        observed = _tenant_now(conn)
                        if observed != name:
                            raise AssertionError(f"{name} observed {observed!r}")
                    seen[name] = _tenant_now(conn)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in tenants]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"cross-tenant observation: {errors[:3]}"
        assert seen == {t: t for t in tenants}

    def test_a_failing_tenant_does_not_poison_the_next_one(self, pool: SharedPool) -> None:
        for _ in range(8):
            with pytest.raises(RuntimeError):
                with pool.tenant_transaction("noisy"):
                    raise RuntimeError("boom")
        with pool.tenant_transaction("quiet") as conn:
            assert _tenant_now(conn) == "quiet"


class TestTheResetGuardCanActuallyFire:
    """Everything above passes even with `_reset` disabled, because `SET LOCAL` does the work.

    So the guard is tested directly: poison a connection the way a future refactor might (a
    SESSION-scoped set_config, which is exactly the old design) and confirm `_reset` refuses it.
    Without this, `_reset` is a comment that reads like protection.
    """

    def test_session_scoped_tenant_is_detected(self, pool: SharedPool) -> None:
        conn = pool.open().getconn()
        try:
            # is_local => FALSE: survives the transaction. This is the bug class being guarded.
            conn.execute(f"SELECT set_config('{TENANT_GUC}', 'leaked-tenant', false)")
            conn.commit()
            assert _tenant_now(conn) == "leaked-tenant"
            with pytest.raises(TenantContextLeak):
                pool._reset(conn)
            assert pool.reset_discards == 1
        finally:
            conn.close()
            pool.open().putconn(conn)

    def test_a_clean_connection_passes_reset(self, pool: SharedPool) -> None:
        with pool.tenant_transaction("acme"):
            pass
        conn = pool.open().getconn()
        try:
            pool._reset(conn)  # must not raise
        finally:
            pool.open().putconn(conn)

    def test_the_leak_error_does_not_contain_the_tenant_value(self, pool: SharedPool) -> None:
        """An isolation failure must not itself disclose which tenant leaked."""
        conn = pool.open().getconn()
        try:
            conn.execute(f"SELECT set_config('{TENANT_GUC}', 'secret-tenant-name', false)")
            conn.commit()
            with pytest.raises(TenantContextLeak) as excinfo:
                pool._reset(conn)
            assert "secret-tenant-name" not in str(excinfo.value)
        finally:
            conn.close()
            pool.open().putconn(conn)


class TestPoolBounds:
    def test_defaults_are_two_and_twenty(self) -> None:
        from recall.pool import DEFAULT_MAX_POOL_SIZE, DEFAULT_MIN_POOL_SIZE

        assert (DEFAULT_MIN_POOL_SIZE, DEFAULT_MAX_POOL_SIZE) == (2, 20)

    def test_invalid_bounds_are_refused(self) -> None:
        with pytest.raises(ValueError):
            SharedPool(TEST_DSN, min_size=0, max_size=5)
        with pytest.raises(ValueError):
            SharedPool(TEST_DSN, min_size=5, max_size=2)

    def test_stats_expose_saturation_without_identifiers(self, pool: SharedPool) -> None:
        with pool.tenant_transaction("acme"):
            pass
        stats = pool.stats()
        assert stats["max_size"] == 6
        assert stats["open"] is True
        assert "reset_discards" in stats
        assert not any("tenant" in k for k in stats), stats

    def test_guc_name_matches_the_store(self) -> None:
        """Two constants naming one GUC is a drift waiting to happen; pin them together."""
        from recall.store import TENANT_GUC as STORE_GUC

        assert TENANT_GUC == STORE_GUC


class TestConnectionsAreActuallyReused:
    """The pool must POOL. Regression tests for the reset hook leaving connections INTRANS.

    `_reset` probed the tenant GUC with `conn.execute(...)` on a non-autocommit connection and
    never ended the implicit transaction that opened. psycopg_pool checks the transaction status
    after the reset callback and discards anything left non-IDLE, so every connection was closed
    on return and the process-wide pool was really a connect-per-operation loop against the
    database it exists to protect.

    None of the isolation tests above could see it: they assert the tenant is gone, and a
    destroyed connection has no tenant either. The guard tests call `_reset` directly and so never
    exercise the pool's return path at all. These two assert the property the others cannot —
    that the same backend serves successive requests — which is what makes them the regression
    tests for that bug rather than a restatement of the isolation claim.
    """

    def test_successive_transactions_reuse_a_bounded_set_of_backends(
        self, pool: SharedPool
    ) -> None:
        """Ten operations must not mean ten backends.

        The bound is the pool's own size, not one: with `min_size=2` the pool keeps two warm
        connections and hands them out in turn, which is correct pooling. What the bug produced
        was a NEW backend for every single operation, so the discriminating assertion is
        "distinct backends <= pool size", not "== 1".
        """
        pids = []
        for _ in range(10):
            with pool.tenant_transaction("acme") as conn:
                pids.append(conn.execute("SELECT pg_backend_pid()").fetchone()[0])
        assert len(set(pids)) <= 2, (
            f"pool churned: {len(set(pids))} distinct backends for 10 operations "
            f"against a min_size=2 pool"
        )

    def test_the_pool_does_not_report_bad_returns(self, pool: SharedPool) -> None:
        """`returns_bad` is the direct signal: it counts connections destroyed on return.

        The companion bound is against `max_size`, not against the operation count and not
        against `min_size`: psycopg_pool may legitimately grow the pool toward its ceiling. What
        it must never do is open a backend PER OPERATION, which is what `connections_num` climbing
        past `max_size` for a serial workload would mean.
        """
        operations = 10
        for _ in range(operations):
            with pool.tenant_transaction("acme"):
                pass
        stats = pool.stats()
        assert stats["returns_bad"] == 0, f"connections discarded on return: {stats}"
        assert stats["connections_num"] <= stats["max_size"], stats
        assert stats["connections_num"] < operations, stats

    def test_a_poisoned_connection_is_discarded_through_the_real_return_path(
        self, pool: SharedPool
    ) -> None:
        """The guard, exercised through psycopg_pool rather than by calling `_reset` by hand.

        This is the half the original guard test could not reach: it proves the discard actually
        happens on return, and that the next caller gets a different, clean backend.
        """
        with pool.tenant_transaction("acme") as conn:
            poisoned_pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
            # SESSION scope: survives the transaction. Exactly the bug class being guarded.
            conn.execute(f"SELECT set_config('{TENANT_GUC}', 'leaked-tenant', false)")

        with pool.tenant_transaction("globex") as conn:
            assert conn.execute("SELECT pg_backend_pid()").fetchone()[0] != poisoned_pid
            assert _tenant_now(conn) == "globex"
        assert pool.reset_discards == 1
