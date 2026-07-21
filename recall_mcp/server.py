from __future__ import annotations

import os
import sys
import traceback
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from recall.calibration import load_for
from recall.store import redacted_dsn
from recall_mcp.service import index_memory, make_embedder, memory_stats, search_memory

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")
EMBEDDER_NAME = os.environ.get("RECALL_EMBEDDER", "fastembed")


@asynccontextmanager
async def _lifespan(_server: FastMCP):
    """Open one PgVectorStore + embedder for the server's lifetime, reused by every tool."""
    from recall.store import PgVectorStore, warn_if_insecure_dsn

    warn_if_insecure_dsn(DEFAULT_DSN)  # loud stderr note if default creds target a remote host
    try:
        embedder = make_embedder(EMBEDDER_NAME)
        store = PgVectorStore(DEFAULT_DSN, dim=embedder.dim)
    except Exception:
        print(
            f"recall_mcp: startup failed (RECALL_DSN={redacted_dsn(DEFAULT_DSN)}, "
            f"RECALL_EMBEDDER={EMBEDDER_NAME!r}):\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        raise
    try:
        store.ensure_schema()
    except Exception:
        store.close()
        print(f"recall_mcp: schema check failed:\n{traceback.format_exc()}", file=sys.stderr)
        raise
    calibration = load_for(embedder.name)  # None -> uncalibrated fallback, flagged in results
    if calibration is None:
        print(
            f"recall_mcp: no calibration for embedder {embedder.name!r} — using the default "
            "threshold (results will say calibrated=false). Run `recall calibrate` to fix.",
            file=sys.stderr,
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
    def recall_search(query: str, source: str | None = None, k: int = 5) -> str:
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
        return search_memory(
            state["store"], state["embedder"], query, source=source, k=k,
            calibration=state.get("calibration"),
        ).model_dump_json(indent=2)

    @mcp.tool(
        name="recall_index",
        annotations={"title": "Add to agent memory", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def recall_index(path: str) -> str:
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
        return index_memory(state["store"], state["embedder"], path).model_dump_json(indent=2)

    @mcp.tool(
        name="recall_stats",
        annotations={"title": "Memory freshness & size", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def recall_stats() -> str:
        """Report how much memory exists and whether it is stale (freshness check).

        `stale` is True when the newest indexed content is older than 2 days.

        Returns:
            JSON of {chunks, newest_indexed_at, stale}.
        """
        state = _state()
        return memory_stats(state["store"]).model_dump_json(indent=2)

    return mcp


mcp = build_server()


def main() -> None:
    print("recall_mcp: starting stdio server", file=sys.stderr)  # stderr only — stdout is JSON-RPC
    mcp.run()


if __name__ == "__main__":
    main()
