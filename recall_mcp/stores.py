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
import time

from recall.control_plane import ControlPlane, TenantRoute
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
        generation_mode: bool = False,
        control_plane: ControlPlane | None = None,
        embedding_profile: str | None = None,
        route_poll_seconds: float = 5.0,
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._dim = dim
        self._allowed = frozenset(allowed_tenants)
        self._pool_size = pool_size
        self._statement_timeout_ms = statement_timeout_ms
        self._generation_mode = generation_mode
        self._control_plane = control_plane
        self._embedding_profile = embedding_profile
        self._route_poll_seconds = route_poll_seconds
        self._stores: dict[tuple[str, str], PgVectorStore] = {}
        self._routes: dict[str, tuple[float, TenantRoute | None]] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._listener_stop = threading.Event()
        self._listener: threading.Thread | None = None
        if control_plane is not None:
            self._listener = threading.Thread(
                target=control_plane.watch_routes,
                args=(self.invalidate_route, self._listener_stop),
                daemon=True,
                name="recall-route-listener",
            )
            self._listener.start()

    @property
    def allowed_tenants(self) -> frozenset[str]:
        return self._allowed

    @property
    def control_plane(self) -> ControlPlane | None:
        return self._control_plane

    @property
    def open_tenants(self) -> frozenset[str]:
        with self._lock:
            return frozenset(tenant for tenant, _generation in self._stores)

    def max_connections(self) -> int:
        """Worst-case connection count if every provisioned tenant becomes active at once."""
        generations = 2 if self._control_plane is not None else 1
        return len(self._allowed) * generations * self._pool_size

    def invalidate_route(self, tenant: str) -> None:
        """Drop only routing metadata. Acquired stores remain valid for in-flight requests."""
        with self._lock:
            self._routes.pop(tenant, None)

    def _route(self, tenant: str) -> TenantRoute | None:
        if self._control_plane is None:
            return None
        now = time.monotonic()
        cached = self._routes.get(tenant)
        if cached is not None and now - cached[0] < self._route_poll_seconds:
            return cached[1]
        route = self._control_plane.route(tenant)
        if route is None:
            raise RuntimeError(f"tenant {tenant!r} has no configured index generation route")
        self._routes[tenant] = (now, route)
        return route

    def _get_generation(self, tenant: str, *, shadow: bool = False) -> PgVectorStore | None:
        route = self._route(tenant)
        generation = route.shadow if shadow and route is not None else route.active if route else None
        if shadow and generation is None:
            return None
        if generation is None and self._generation_mode:
            from recall.generation_store import GenerationStore

            key = (tenant, "v1")
            store = self._stores.get(key)
            if store is None:
                _log.info("opening immutable generation store for tenant %r", tenant)
                store = GenerationStore(
                    self._dsn,
                    dim=self._dim,
                    tenant=tenant,
                    migration_target=self._table or DEFAULT_TABLE,
                    pool_size=self._pool_size,
                    statement_timeout_ms=self._statement_timeout_ms,
                )
                try:
                    store.check_schema()
                except Exception:
                    store.close()
                    raise
                self._stores[key] = store
            return store
        table = generation.physical_table if generation is not None else self._table or DEFAULT_TABLE
        generation_id = generation.generation_id if generation is not None else "legacy"
        dimension = generation.dimension if generation is not None else self._dim
        if dimension != self._dim:
            raise RuntimeError(
                f"generation {generation_id!r} dimension {dimension} does not match runtime {self._dim}"
            )
        if (generation is not None and self._embedding_profile is not None
                and not shadow and generation.embedding_profile != self._embedding_profile):
            raise RuntimeError(
                f"generation {generation_id!r} profile {generation.embedding_profile!r} does not "
                f"match runtime {self._embedding_profile!r}"
            )
        key = (tenant, generation_id)
        store = self._stores.get(key)
        if store is None:
            _log.info("opening store for tenant %r generation %r", tenant, generation_id)
            store = PgVectorStore(
                self._dsn,
                dim=dimension,
                table=table,
                tenant=tenant,
                pool_size=self._pool_size,
                statement_timeout_ms=self._statement_timeout_ms,
                generation_id=generation_id,
            )
            try:
                store.ensure_schema()
            except Exception:
                store.close()
                raise
            self._stores[key] = store
        return store

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
            store = self._get_generation(tenant)
            assert store is not None
            return store

    def get_shadow(self, tenant: str) -> PgVectorStore | None:
        if tenant not in self._allowed:
            raise PermissionError(f"tenant {tenant!r} is not provisioned on this server")
        with self._lock:
            if self._closed:
                raise RuntimeError("StoreRegistry is closed")
            return self._get_generation(tenant, shadow=True)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            stores = list(self._stores.items())
            self._stores.clear()
        self._listener_stop.set()
        if self._listener is not None:
            self._listener.join(timeout=6.0)
        for (tenant, generation), store in stores:
            try:
                store.close()
            except Exception:  # pragma: no cover - best effort on shutdown
                _log.warning(
                    "error closing store for tenant %r generation %r",
                    tenant, generation, exc_info=True,
                )
