"""Example: an agent that consults its own memory before acting (anti-re-litigation).

Run standalone against the demo corpus, or import `decide` in tests. In a real MCP client
the same pattern is: call the `recall_search` tool before proposing, and back off only when a
still-TRUSTWORTHY prior record surfaces (a verdict-`ok` hit). An abstention or a `gap_warning`
means there is no valid prior record, so the proposal proceeds.
"""
from __future__ import annotations

from recall.embeddings import Embedder
from recall.store import PgVectorStore
from recall_mcp.service import search_memory


def decide(store: PgVectorStore, embedder: Embedder, proposal: str) -> dict:
    """Decide whether to proceed with `proposal`, consulting memory first.

    Backs off only on a memory that is BOTH relevant and still trustworthy — a verdict-`ok`
    hit. The two other outcomes both mean "no valid prior record", and both let the proposal
    through:

    - `abstained`: hits came back but none survived the trust check (superseded, expired,
      below the calibrated threshold). Quoting one of those as the blocker would be exactly
      the stale-memory failure this library exists to prevent, so the reason reports the
      abstention instead of the memory's text.
    - `gap_warning` / no hits: the corpus has nothing on this at all.

    When a memory was superseded, its successor is the hit carrying verdict `ok`, so the
    citation names the CURRENT version rather than the one that merely matched best.
    """
    result = search_memory(store, embedder, proposal, k=3)
    if result.abstained:
        return {
            "proceed": True,
            "reason": f"No trustworthy prior memory — {result.reason}. Safe to proceed.",
        }
    if result.gap_warning or not result.hits:
        return {"proceed": True, "reason": "No relevant prior memory — safe to proceed."}
    top = next((h for h in result.hits if h.verdict == "ok"), None)
    if top is None:  # belt-and-braces: `not abstained` implies an ok hit exists today
        return {"proceed": True, "reason": "No trustworthy prior memory — safe to proceed."}
    return {
        "proceed": False,
        "reason": (
            f"Found relevant memory ({top.source}, verdict={top.verdict}): {top.text!r}. "
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
