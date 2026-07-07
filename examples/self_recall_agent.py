"""Example: an agent that consults its own memory before acting (anti-re-litigation).

Run standalone against the demo corpus, or import `decide` in tests. In a real MCP client
the same pattern is: call the `recall_search` tool before proposing, and back off if a closed
decision or falsified hypothesis surfaces (unless it's a `gap_warning`).
"""
from __future__ import annotations

from recall.embeddings import Embedder
from recall.store import PgVectorStore
from recall_mcp.service import search_memory


def decide(store: PgVectorStore, embedder: Embedder, proposal: str) -> dict:
    """Decide whether to proceed with `proposal`, consulting memory first.

    Proceeds only when memory has no strongly-relevant prior record. If a relevant memory
    exists (no gap), the agent backs off and cites it — the anti-re-litigation guard.
    """
    result = search_memory(store, embedder, proposal, k=3)
    if result.gap_warning or not result.hits:
        return {"proceed": True, "reason": "No relevant prior memory — safe to proceed."}
    top = result.hits[0]
    return {
        "proceed": False,
        "reason": (
            f"Found relevant memory ({top.source}): {top.text!r}. "
            "Do not re-litigate — review this first."
        ),
    }


def main() -> None:  # pragma: no cover - manual demo entry point
    import os

    from recall_mcp.service import make_embedder

    dsn = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")
    embedder = make_embedder(os.environ.get("RECALL_EMBEDDER", "fastembed"))
    with PgVectorStore(dsn, dim=embedder.dim) as store:
        store.ensure_schema()
        for proposal in [
            "let's inject retrieved context into the prompt to boost answers",
            "should we add a brand new telemetry dashboard",
        ]:
            d = decide(store, embedder, proposal)
            print(f"\nPROPOSAL: {proposal}\n  -> proceed={d['proceed']}\n  {d['reason']}")


if __name__ == "__main__":
    main()
