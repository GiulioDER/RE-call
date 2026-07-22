from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from recall.calibration import load_for
from recall.observability import METRICS, configure_logging, get_logger
from recall.store import DEFAULT_TENANT, redacted_dsn
from recall_mcp.service import index_memory, make_embedder, memory_stats, search_memory

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")
EMBEDDER_NAME = os.environ.get("RECALL_EMBEDDER", "fastembed")
#: Connections the server keeps open. This bounds concurrent in-flight tool calls at the database,
#: which is where the real limit is — more worker threads than connections just queue on the pool.
POOL_SIZE = int(os.environ.get("RECALL_POOL_SIZE", "8"))
#: Tenant this server instance serves. One store is bound to one tenant, so a
#: multi-tenant deployment runs a server (or a store) per tenant rather than switching
#: tenants on a shared connection — see PgVectorStore._prepare.
TENANT = os.environ.get("RECALL_TENANT", DEFAULT_TENANT)
#: Server-side cap on any single statement. A runaway query otherwise holds its connection until
#: the process dies, and a few of those exhaust the pool while the server still looks healthy.
STATEMENT_TIMEOUT_MS = int(os.environ.get("RECALL_STATEMENT_TIMEOUT_MS", "15000"))

_T = TypeVar("_T")

_log = get_logger("mcp")


async def _to_thread(fn: Callable[[], _T]) -> _T:
    """Run a blocking tool body off the event loop.

    FastMCP awaits an async tool and calls a sync one INLINE (`func_metadata.py`:
    ``return await fn(...)`` vs ``return fn(...)``) — there is no thread offload. A sync tool that
    embeds a query, makes two database round trips and maybe runs a cross-encoder therefore blocks
    the whole loop for its duration: one request at a time, with no response to anything else —
    not even a ping — until it finishes. `recall_index` blocks it for an entire corpus index.

    `anyio.to_thread` rather than `asyncio.to_thread` because FastMCP runs on AnyIO: this inherits
    its worker-thread limiter and cancellation scope instead of starting a second, unmanaged pool
    beside it.
    """
    return await anyio.to_thread.run_sync(fn)


@asynccontextmanager
async def _lifespan(_server: FastMCP):
    """Open one PgVectorStore + embedder for the server's lifetime, reused by every tool."""
    from recall.store import PgVectorStore, require_secure_dsn

    # FAIL CLOSED, unlike the CLI's warning: a server is unattended, so a stderr note about
    # published default credentials pointed at a remote database lands in a journal nobody reads
    # while the process comes up looking healthy. RECALL_ALLOW_INSECURE_DSN=1 opts out.
    require_secure_dsn(DEFAULT_DSN)
    try:
        embedder = make_embedder(EMBEDDER_NAME)
        # Pooled + timed out: a server shares this store across concurrent tool calls, and one
        # connection would serialise them however many threads are available to run them.
        store = PgVectorStore(
            DEFAULT_DSN,
            dim=embedder.dim,
            tenant=TENANT,
            pool_size=POOL_SIZE,
            statement_timeout_ms=STATEMENT_TIMEOUT_MS,
        )
    except Exception:
        _log.error(
            "startup failed (dsn=%s, embedder=%r)",
            redacted_dsn(DEFAULT_DSN), EMBEDDER_NAME, exc_info=True,
        )
        raise
    try:
        store.ensure_schema()
    except Exception:
        store.close()
        _log.error("schema check failed", exc_info=True)
        raise
    calibration = load_for(embedder.name)  # None -> uncalibrated fallback, flagged in results
    if calibration is None:
        _log.warning(
            "no calibration for embedder %r — using the default threshold (results will "
            "say calibrated=false). Run `recall calibrate` to fix.", embedder.name,
        )
    if not store.check_rls_effective():
        _log.warning(
            "this database role bypasses row-level security (superuser or BYPASSRLS), so "
            "tenant isolation rests on query predicates alone. Connect as an unprivileged "
            "role for defence in depth."
        )
    try:
        yield {"store": store, "embedder": embedder, "calibration": calibration}
    finally:
        store.close()


def build_server() -> FastMCP:
    """Construct the recall_mcp FastMCP server with its three tools registered."""
    mcp = FastMCP("recall_mcp", lifespan=_lifespan)

    def _state() -> dict:
        ctx = mcp.get_context().request_context.lifespan_context
        if not isinstance(ctx, dict) or "store" not in ctx or "embedder" not in ctx:
            raise RuntimeError(
                "recall_mcp lifespan context is not initialized — tools must be invoked within "
                "the running server (store/embedder are opened in the lifespan)."
            )
        return ctx

    @mcp.tool(
        name="recall_search",
        annotations={"title": "Search agent memory", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def recall_search(query: str, source: str | None = None, k: int = 5) -> str:
        """Search the agent's OWN memory before acting, and get actionable guidance.

        Call this before proposing an idea, forming a hypothesis, or repeating past work:
        if a closed decision or falsified hypothesis surfaces, do not re-litigate it. Every hit
        carries a trust verdict (only `ok` hits should be relied on), a calibrated confidence,
        provenance (indexed_at) and validity (superseded_by / valid_until). When `abstained` is
        true, NO valid hit survived — say you don't know instead of answering from the hits.
        `advice` states what to do.

        Args:
            query: what to recall (natural language).
            source: optional source filter (only search one file/source).
            k: max hits to return (default 5).

        Returns:
            JSON of {query, abstained, reason, calibrated, gap_warning, stale, advice,
            hits:[{source, score, confidence, verdict, superseded_by, valid_until,
            indexed_at, text}]}.
        """
        state = _state()
        with METRICS.timer("recall_tool_latency_ms", tool="search"):
            return await _to_thread(
                lambda: search_memory(
                    state["store"], state["embedder"], query, source=source, k=k,
                    calibration=state.get("calibration"),
                ).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_index",
        annotations={"title": "Add to agent memory", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def recall_index(path: str) -> str:
        """Index a markdown file or folder into the agent's memory so it can be recalled later.

        Re-indexing a file REPLACES its chunks completely (safe to re-run after edits; a shrunk
        file leaves no stale chunks behind).
        `path` is confined to RECALL_INDEX_ROOT (default: the server's working directory).

        Args:
            path: a file or directory path (``**/*.md`` is indexed for directories).

        Returns:
            JSON of {files, chunks, message}.
        """
        state = _state()
        with METRICS.timer("recall_tool_latency_ms", tool="index"):
            return await _to_thread(
                lambda: index_memory(
                    state["store"], state["embedder"], path
                ).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_stats",
        annotations={"title": "Memory freshness & size", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def recall_stats() -> str:
        """Report how much memory exists and whether it is stale (freshness check).

        `stale` is True when the newest indexed content is older than 2 days.

        Returns:
            JSON of {chunks, newest_indexed_at, stale}.
        """
        state = _state()
        return await _to_thread(
            lambda: memory_stats(state["store"]).model_dump_json(indent=2)
        )

    return mcp


mcp = build_server()


def main() -> None:
    # stderr only, and propagate=False — stdout carries JSON-RPC, so a stray log line there
    # would corrupt the protocol.
    configure_logging()
    _log.info("starting stdio server", extra={"tenant": TENANT, "embedder": EMBEDDER_NAME})
    mcp.run()


if __name__ == "__main__":
    main()
