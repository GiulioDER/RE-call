"""One connection pool per process, with tenant context scoped to a transaction.

Why this replaces one pool per tenant
-------------------------------------
`recall_mcp/stores.py` opened one pool per tenant, per generation, and computed its ceiling as
``len(allowed_tenants) * generations * pool_size``. At the 1,000-tenant target that number is not
a pool, it is a denial of service against your own database.

The per-tenant pool was not arbitrary, though, and it is worth being precise about what it bought,
because this module has to buy the same thing another way. The old design set the tenant GUC
**session-wide** (``set_config(..., is_local => false)``) once, in the pool's ``configure`` hook,
when a connection was opened. That made the tenant a property of the CONNECTION, so the RLS policy
held even if a ``WHERE tenant_id = ...`` predicate was forgotten. The cost is that such a
connection can never be handed to a different tenant, which is exactly what forces a pool each.

The failure it was avoiding is real and worth naming: set the GUC per request on a shared pooled
connection, and any path that returns the connection mid-request — an exception between the
``set_config`` and the query, a cancellation, a timeout — leaves the next caller holding someone
else's tenant.

What makes sharing safe
-----------------------
Three properties together, none of which is sufficient alone:

1. **``SET LOCAL``, not session scope.** ``set_config(..., is_local => true)`` is bound to the
   transaction. On COMMIT *or* ROLLBACK, PostgreSQL discards it. There is no code path — no
   exception, no cancellation, no timeout — where the setting outlives the transaction that set
   it, because the database, not this module, is what unsets it.

2. **An explicit transaction around every operation.** ``SET LOCAL`` outside a transaction block
   is silently a no-op with a warning, which would leave the GUC empty and the RLS policy
   matching nothing. So the transaction is not bookkeeping here, it is what gives the setting
   somewhere to live.

3. **A reset hook that verifies rather than trusts.** ``_reset`` runs before psycopg_pool returns
   a connection, and asserts the GUC really is clear. If it is not, it raises, and psycopg_pool
   discards the connection instead of reusing it. Property 1 says this cannot happen; property 3
   is what makes that a checked claim rather than a comment. A leak becomes a discarded
   connection and a loud error, never a silently mislabelled row.

Explicit ``WHERE tenant_id = %s`` predicates stay exactly as they were. RLS is the backstop for a
forgotten predicate, and the predicate is the backstop for a GUC that was never set. Neither is
load-bearing alone.

What RLS does not protect against
---------------------------------
RLS bounds the damage an **application mistake** can do: a forgotten predicate, a wrong join, a
tenant id threaded through the wrong variable. It is not a defence against a **compromised serving
role**. Anything that can execute arbitrary SQL as the serving role can call ``set_config`` itself
and read any tenant, and a role with ``BYPASSRLS`` or superuser ignores the policy outright. The
security boundary is the credential; RLS is a correctness guard behind it. Treat a leaked serving
DSN as a full corpus disclosure, not as a partial one.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from recall.observability import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import psycopg
    from psycopg_pool import ConnectionPool

_log = get_logger("pool")

#: The GUC the RLS policy reads. Kept in sync with `recall.store.TENANT_GUC` by test.
TENANT_GUC = "recall.tenant_id"

#: Defaults for the per-process pool. Two, not one, so a single stuck connection cannot deadlock
#: a process that needs a second one to make progress; twenty because a search pod's concurrency
#: is bounded by its thread pool long before the database is the constraint, and because 1,000
#: tenants times anything larger is not a number a managed PostgreSQL will grant.
DEFAULT_MIN_POOL_SIZE = 2
DEFAULT_MAX_POOL_SIZE = 20


class TenantContextLeak(RuntimeError):
    """A pooled connection was about to be reused with a tenant still set on it.

    This should be unreachable: `SET LOCAL` is discarded by the database at transaction end. It
    exists because "should be unreachable" is a claim, and an unchecked claim about tenant
    isolation is the kind that is discovered from the wrong side.
    """


def _tenant_on(conn: "psycopg.Connection") -> str:
    """Read the tenant GUC without raising when it was never set.

    `current_setting(name)` errors on an unknown custom GUC; the second argument makes it return
    NULL instead, which is the state we expect and must be able to observe.
    """
    row = conn.execute(f"SELECT current_setting('{TENANT_GUC}', true)").fetchone()
    return "" if row is None or row[0] is None else str(row[0])


class SharedPool:
    """One psycopg pool for the whole process, shared by every tenant.

    Not a `PgVectorStore` replacement: it owns connections and the tenant discipline, and nothing
    else. `store.for_tenant()` becomes a view over one of these rather than a new pool.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = DEFAULT_MIN_POOL_SIZE,
        max_size: int = DEFAULT_MAX_POOL_SIZE,
        connect_kwargs: dict[str, Any] | None = None,
        statement_timeout_ms: int | None = None,
        connect_timeout_s: float = 30.0,
    ) -> None:
        if min_size < 1 or max_size < min_size:
            raise ValueError(f"invalid pool bounds: min_size={min_size}, max_size={max_size}")
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        # `connect_timeout` FIRST so an explicit caller kwarg still wins. Without it a pooled
        # connection to a black-holed host hangs on the TCP handshake with no bound — a
        # protection both other store modes set and this one had silently dropped.
        self._connect_kwargs = {
            "connect_timeout": int(connect_timeout_s),
            **dict(connect_kwargs or {}),
        }
        self._statement_timeout_ms = statement_timeout_ms
        self._connect_timeout_s = connect_timeout_s
        self._lock = threading.Lock()
        #: A SEPARATE lock for the counter, not `_lock`. `close()` holds `_lock` while closing the
        #: pool, and closing can drive connection returns through `_reset` on a worker thread —
        #: which would then wait on a lock the closing thread already holds. One lock for both
        #: jobs is a deadlock between shutdown and the guard.
        self._counter_lock = threading.Lock()
        self._pool: ConnectionPool | None = None
        #: Sticky. Without it `close()` merely nulled `_pool`, so the next `tenant_transaction`
        #: read "not yet opened" and built a WHOLE NEW pool — fresh connections plus a scheduler
        #: and worker threads — after shutdown, with nothing left to close it. The
        #: single-connection mode already guards this class ("silently resurrect a connection
        #: nobody owns"); the shared pool did not.
        self._is_closed = False
        #: Observability for A.8 and the pool-saturation alert. Counting discards separately from
        #: leaks matters: a discard is the guard working, and only a leak is a defect.
        self.reset_discards = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    def _configure(self, conn: "psycopg.Connection") -> None:
        """Per-connection setup. Deliberately does NOT set a tenant.

        Setting a tenant here is the old design, and it is what made a connection un-shareable.
        The vector codec is genuinely per-connection state, so it belongs here; the tenant does
        not, and putting it here again would silently reintroduce the coupling this module exists
        to remove.
        """
        from recall.store import register_vector

        register_vector(conn)
        if self._statement_timeout_ms is not None:
            # SESSION scope on purpose, unlike the tenant: a timeout is a property of this
            # process's connections, carries no tenant information, and must apply to the
            # statements that open a transaction as well as those inside one.
            conn.execute(f"SET statement_timeout = {int(self._statement_timeout_ms)}")
        # psycopg_pool refuses a connection that `configure` left INTRANS, and every `execute`
        # above starts an implicit transaction. Without this the pool discards each connection it
        # opens and then times out with "pool initialization incomplete", which reads like an
        # unreachable database rather than a bug two lines up.
        conn.commit()

    def _reset(self, conn: "psycopg.Connection") -> None:
        """Run by psycopg_pool before a connection returns to the pool.

        Raising here makes psycopg_pool DISCARD the connection rather than reuse it, which is the
        behaviour we want: a connection whose tenant context survived is not safe to hand to the
        next caller under any circumstances, and closing it costs one reconnect.
        """
        if conn.info.transaction_status != conn.info.transaction_status.IDLE:
            conn.rollback()
        leaked = _tenant_on(conn)
        # `_tenant_on` runs a SELECT, and these connections are not autocommit, so the probe
        # ITSELF opens a transaction. psycopg_pool checks the status after this hook and discards
        # anything left non-IDLE — so without this rollback the guard destroyed every connection
        # it inspected and the process-wide pool became a connect-per-operation loop against the
        # database it exists to protect. `_configure` commits for exactly this reason twelve lines
        # above; the same trap, missed twice in one file.
        #
        # Rollback BEFORE the raise, not after: on the leak path psycopg_pool discards the
        # connection anyway, but leaving the ordering to that coincidence would mean the clean
        # path's correctness depended on the error path's side effect.
        conn.rollback()
        if leaked:
            with self._counter_lock:
                # `+=` is a non-atomic read-modify-write and this hook runs on psycopg_pool's
                # worker threads, so concurrent discards could undercount the one number an
                # operator would alert on for tenant leaks.
                self.reset_discards += 1
            _log.error(
                "discarding pooled connection: tenant context survived the transaction",
                extra={"leaked_tenant_len": len(leaked)},  # never the value itself
            )
            raise TenantContextLeak(
                "tenant context survived its transaction; connection discarded"
            )

    def open(self) -> "ConnectionPool":
        if self._is_closed:
            raise RuntimeError("pool is closed")
        if self._pool is not None:
            return self._pool
        with self._lock:
            if self._is_closed:
                raise RuntimeError("pool is closed")
            if self._pool is not None:
                return self._pool
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "SharedPool requires the pool extra: pip install recall[pool]"
                ) from exc
            pool = ConnectionPool(
                self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                kwargs=self._connect_kwargs,
                configure=self._configure,
                reset=self._reset,
                open=False,
            )
            pool.open(wait=True, timeout=self._connect_timeout_s)
            self._pool = pool
            return pool

    @contextmanager
    def tenant_transaction(self, tenant: str) -> Iterator["psycopg.Connection"]:
        """Check out a connection, open a transaction, and scope the tenant GUC to it.

        The ordering is the whole point and is not interchangeable:

            BEGIN  ->  SET LOCAL tenant  ->  caller's work  ->  COMMIT or ROLLBACK

        `SET LOCAL` before `BEGIN` would be a no-op; the tenant would be empty, RLS would match no
        rows, and the caller would see an empty result rather than an error, which is the worst
        of the available failures. Setting it after the caller's work would be worse still.
        """
        if not tenant or not tenant.strip():
            # An empty tenant is not "all tenants" here, but it IS an empty GUC, and an empty GUC
            # against the RLS policy silently matches nothing. Refuse rather than return zero rows
            # that look like an honest answer.
            raise ValueError("tenant must be a non-empty string")
        pool = self.open()
        with pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    f"SELECT set_config('{TENANT_GUC}', %s, true)",  # true => LOCAL
                    (tenant,),
                )
                yield conn

    def close(self) -> None:
        with self._lock:
            self._is_closed = True
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def stats(self) -> dict[str, Any]:
        """Pool telemetry for the saturation alert. No tenant identifiers, no query text."""
        if self._pool is None:
            return {
                "open": False,
                "closed": self._is_closed,
                "max_size": self._max_size,
                "reset_discards": self.reset_discards,
            }
        raw = self._pool.get_stats()
        return {
            "open": True,
            "max_size": self._max_size,
            "pool_size": raw.get("pool_size", 0),
            "pool_available": raw.get("pool_available", 0),
            "requests_waiting": raw.get("requests_waiting", 0),
            "reset_discards": self.reset_discards,
            # Churn. `connections_num` is the count of backends this pool has OPENED; on a healthy
            # pool it stabilises near `pool_size` and stops growing, so its rate against the
            # request rate is the direct detector for "the pool is not pooling".
            #
            # It is here because its absence is what hid a real defect: the reset hook was
            # discarding every connection on return, and `stats()` reported a healthy `pool_size`
            # throughout while `reset_discards` stayed 0 — a counter that only counts tenant
            # leaks cannot distinguish "guard working" from "pool destroying itself".
            "connections_num": raw.get("connections_num", 0),
            "returns_bad": raw.get("returns_bad", 0),
        }


__all__ = [
    "DEFAULT_MAX_POOL_SIZE",
    "DEFAULT_MIN_POOL_SIZE",
    "SharedPool",
    "TENANT_GUC",
    "TenantContextLeak",
]
