from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


def build_server() -> FastMCP:
    """Construct the recall_mcp FastMCP server with its three tools registered."""
    mcp = FastMCP("recall_mcp")

    @mcp.tool(
        name="recall_search",
        annotations={"title": "Search agent memory", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def recall_search(query: str, source: str | None = None, k: int = 5) -> str:
        """Search the agent's own memory before acting (stub — implemented in Task 2)."""
        return "not implemented"

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
