"""Which store a DSN opens: PostgreSQL, or one SQLite file (`sqlite:///<path>`).

The lite store implements the calls the legacy (non-generation) route makes: indexing, dense and
keyword retrieval, supersession, calibration. It is not a `PgVectorStore` subclass, and it is
returned typed as one on purpose, so the CLI and the server keep one code path; anything outside
that subset raises `recall.lite.LiteUnsupported`, which names the call and says it needs the full
install, and reads as absent to `getattr(store, name, None)` probes.

Generations, manifests, the enterprise control plane and learned sparse retrieval exist only on
Postgres, so a lite DSN on the generation route is refused before anything opens.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, cast

from recall.lite import LiteStore, is_lite_dsn
from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, PgVectorStore

if TYPE_CHECKING:
    from recall.runtime_route import RuntimeRoute

__all__ = ["is_lite_dsn", "lite_route_refusal", "open_legacy_store", "write_session"]


def lite_route_refusal(dsn: str | None, route: RuntimeRoute) -> str | None:
    """Why `route` cannot run on `dsn`, or None when it can."""
    if not is_lite_dsn(dsn) or not route.uses_generation:
        return None
    return (
        f"{route.describe()}: the lite store ({dsn}) has no generations. Use it with "
        "RECALL_ENV=development and RECALL_INDEX_MODE=legacy (the defaults), or point "
        "RECALL_DSN at PostgreSQL for the generation route"
    )


def open_legacy_store(
    dsn: str,
    *,
    dim: int,
    table: str = DEFAULT_TABLE,
    tenant: str = DEFAULT_TENANT,
    pool_size: int | None = None,
    statement_timeout_ms: int | None = None,
) -> PgVectorStore:
    """The legacy-route store for `dsn`: a `LiteStore` for `sqlite:///`, else a `PgVectorStore`.

    The pool size and statement timeout apply to Postgres only; a SQLite file has neither.
    """
    if is_lite_dsn(dsn):
        return cast("PgVectorStore", LiteStore.from_dsn(dsn, dim=dim, tenant=tenant, table=table))
    return PgVectorStore(
        dsn, dim=dim, table=table, tenant=tenant, pool_size=pool_size, statement_timeout_ms=statement_timeout_ms
    )


def write_session(store: object, purpose: str, **kwargs: Any) -> AbstractContextManager[Any]:
    """Hold the lite store's write lane for a write operation; a no-op for any other store.

    Postgres has its own guard on the same operations (`recall.index_lock.single_writer`), so the
    lane exists only where that one cannot reach. kwargs go to `recall.lite.writer_lock.lite_lane`
    (lane, wait_s, coalesce, on_busy).
    """
    if isinstance(store, LiteStore):
        from recall.lite.writer_lock import lite_lane

        return lite_lane(store.path, purpose=purpose, **kwargs)
    return nullcontext(None)
