"""One `PgVectorStore` per tenant, created on demand and shared across requests.

Why a registry rather than a tenant argument
--------------------------------------------
A store is bound to one tenant for its lifetime. The pool's `configure` hook runs
`set_config('recall.tenant_id', <tenant>, false)` on every connection it opens (`store.py`
`_prepare`), and the RLS policy compares each row against that GUC. The tenant is therefore a
property of the CONNECTION, not of the query — which is exactly what makes the isolation hold
even if a `WHERE tenant_id = ...` predicate is ever forgotten.

The consequence is that you cannot serve two tenants from one store without re-setting the GUC
per request on a pooled connection, and doing that is how cross-tenant leaks happen: a connection
handed back to the pool mid-request, or an exception between `set_config` and the query, and the
next caller inherits someone else's tenant. Keeping one pool per tenant makes that class of bug
unrepresentable rather than merely avoided.

Bounded by configuration, not by traffic
----------------------------------------
`allowed_tenants` comes from the token registry, so a tenant exists only if an operator
provisioned a token for it. There is no request-driven growth: an unauthenticated or unknown
caller never reaches this class, and a known caller can only ever name the one tenant its own
token carries. That is what keeps `len(tenants) * pool_size` a number you can compute at startup
and check against `max_connections`, instead of something that grows until the database refuses
connections.
"""

from __future__ import annotations

import threading

from recall.observability import get_logger
from recall.store import DEFAULT_TABLE, PgVectorStore

_log = get_logger("mcp.stores")


class StoreRegistry:
    """Lazily opens and caches one store per allowed tenant. Thread-safe.

    Tool bodies run in worker threads (`anyio.to_thread` in `server.py`), so `get` is called
    concurrently. The lock is held across store construction — which opens connections and can
    block — deliberately: without it, two threads racing on a cold tenant both build a store and
    one is silently dropped, leaking its entire pool for the life of the process.
    """

    def __init__(
        self,
        *,
        dsn: str,
        dim: int,
        allowed_tenants: frozenset[str],
        pool_size: int,
        statement_timeout_ms: int,
        table: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._dim = dim
        self._allowed = frozenset(allowed_tenants)
        self._pool_size = pool_size
        self._statement_timeout_ms = statement_timeout_ms
        self._stores: dict[str, PgVectorStore] = {}
        self._lock = threading.Lock()
        self._closed = False

    @property
    def allowed_tenants(self) -> frozenset[str]:
        return self._allowed

    @property
    def open_tenants(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._stores)

    def max_connections(self) -> int:
        """Worst-case connection count if every provisioned tenant becomes active at once."""
        return len(self._allowed) * self._pool_size

    def get(self, tenant: str) -> PgVectorStore:
        """Return the store for `tenant`, opening it on first use.

        Raises PermissionError for a tenant outside `allowed_tenants`. That should be impossible
        — the tenant comes from a verified token — so reaching it means an authorisation bug
        upstream, and it fails closed rather than opening a fresh namespace on demand.
        """
        if tenant not in self._allowed:
            raise PermissionError(f"tenant {tenant!r} is not provisioned on this server")
        with self._lock:
            if self._closed:
                raise RuntimeError("StoreRegistry is closed")
            store = self._stores.get(tenant)
            if store is None:
                _log.info("opening store for tenant %r", tenant)
                # `table=` passed explicitly rather than splatted from a conditional dict: a
                # `**kwargs` here is opaque to a type checker, so a wrong name or type in it
                # would only surface as a TypeError at the first tenant open.
                store = PgVectorStore(
                    self._dsn,
                    dim=self._dim,
                    table=self._table or DEFAULT_TABLE,
                    tenant=tenant,
                    pool_size=self._pool_size,
                    statement_timeout_ms=self._statement_timeout_ms,
                )
                try:
                    store.ensure_schema()
                except Exception:
                    # Close the half-built store before propagating: otherwise a tenant whose
                    # schema check fails leaks a full pool on every retry, and a client retrying
                    # a broken tenant exhausts the database's connection slots for everyone else.
                    store.close()
                    raise
                self._stores[tenant] = store
            return store

    def close(self) -> None:
        with self._lock:
            self._closed = True
            stores = list(self._stores.items())
            self._stores.clear()
        for tenant, store in stores:
            try:
                store.close()
            except Exception:  # pragma: no cover - best effort on shutdown
                _log.warning("error closing store for tenant %r", tenant, exc_info=True)
