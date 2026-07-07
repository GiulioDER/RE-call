from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from recall_mcp.service import make_embedder, search_memory

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")
EMBEDDER_NAME = os.environ.get("RECALL_EMBEDDER", "fastembed")


@asynccontextmanager
async def _lifespan(_server: FastMCP):
    """Open one PgVectorStore + embedder for the server's lifetime, reused by every tool."""
    from recall.store import PgVectorStore

    embedder = make_embedder(EMBEDDER_NAME)
    store = PgVectorStore(DEFAULT_DSN, dim=embedder.dim)
    store.ensure_schema()
    try:
        yield {"store": store, "embedder": embedder}
    finally:
        store.close()


def build_server() -> FastMCP:
    """Construct the recall_mcp FastMCP server with its three tools registered."""
    mcp = FastMCP("recall_mcp", lifespan=_lifespan)

    def _state() -> dict:
        return mcp.get_context().request_context.lifespan_context

    @mcp.tool(
        name="recall_search",
        annotations={"title": "Search agent memory", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def recall_search(query: str, source: str | None = None, k: int = 5) -> str:
        """Search the agent's OWN memory before acting, and get actionable guidance.

        Call this before proposing an idea, forming a hypothesis, or repeating past work:
        if a closed decision or falsified hypothesis surfaces, do not re-litigate it. The
        result's `gap_warning` marks when memory has no relevant answer (treat hits as noise
        rather than hallucinating), and `advice` states what to do.

        Args:
            query: what to recall (natural language).
            source: optional source filter (only search one file/source).
            k: max hits to return (default 5).

        Returns:
            JSON of {query, gap_warning, stale, advice, hits:[{source, score, text}]}.
        """
        state = _state()
        return search_memory(
            state["store"], state["embedder"], query, source=source, k=k
        ).model_dump_json(indent=2)

    @mcp.tool(
        name="recall_index",
        annotations={"title": "Add to agent memory", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def recall_index(path: str) -> str:
        """Index a file/folder of markdown into memory (stub — implemented in Task 3)."""
        return "not implemented"

    @mcp.tool(
        name="recall_stats",
        annotations={"title": "Memory freshness & size", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def recall_stats() -> str:
        """Report memory size and freshness (stub — implemented in Task 4)."""
        return "not implemented"

    return mcp


mcp = build_server()


def main() -> None:
    print("recall_mcp: starting stdio server", file=sys.stderr)  # stderr only — stdout is JSON-RPC
    mcp.run()


if __name__ == "__main__":
    main()
