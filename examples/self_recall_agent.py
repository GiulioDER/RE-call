"""Example: an agent that consults its own memory before acting (anti-re-litigation).

Run standalone against the demo corpus, or import `decide` in tests. In a real MCP client
the same pattern is: call the `recall_search` tool before proposing, and back off only when a
trustworthy closed decision or falsified hypothesis surfaces (a verdict-`ok` hit).
"""
from __future__ import annotations

from recall.embeddings import Embedder
from recall.store import PgVectorStore
from recall.trust_policy import TrustRefusal
from recall_mcp.service import search_memory


def decide(
    store: PgVectorStore,
    embedder: Embedder,
    proposal: str,
    **search_kwargs: object,
) -> dict:
    """Decide whether to proceed with `proposal`, consulting memory first.

    Proceeds only when a WORKING, certified gate reports no strongly-relevant prior record. If a
    trustworthy memory exists, the agent backs off and cites it. A working-gate abstention
    allows the proposal to proceed without citing a rejected candidate.

    The `TrustRefusal` branch is the one worth copying. Without it this function has a silent
    worst case: the trust system goes down, the search returns nothing, "nothing" reads as "no
    relevant prior memory", and the agent proceeds to re-litigate a decision that was in fact
    closed. An empty answer from a working gate and an outage are the same shape on the wire and
    opposite in meaning, so the outage has to be caught as its own case and must NOT proceed.
    Failing closed here costs a retry; failing open costs the guard this example exists to show.

    `search_kwargs` is forwarded to `search_memory`. Local evaluations may pass an explicit
    `calibration` with `policy=TrustPolicy.development()`; leaving `search_kwargs`
    empty gives the production default, which is strict.
    """
    try:
        result = search_memory(store, embedder, proposal, k=3, **search_kwargs)  # type: ignore[arg-type]
    except TrustRefusal as refusal:
        return {
            "proceed": False,
            "reason": (
                f"Memory could not be consulted ({refusal.code.value}): {refusal.advice} "
                "Not proceeding: this is an unavailable guard, not an absence of prior work."
            ),
            "failure_code": refusal.code.value,
        }
    if result.abstained:
        return {
            "proceed": True,
            "reason": f"No trustworthy prior memory: {result.reason}. Safe to proceed.",
        }
    top = next((hit for hit in result.hits if hit.verdict == "ok"), None)
    if top is None:
        return {
            "proceed": False,
            "reason": "Memory could not be consulted: no trustworthy decision was established.",
            "failure_code": result.failure_code,
        }
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

    dsn = os.environ.get(
        "RECALL_SERVING_DSN",
        os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"),
    )
    embedder = make_embedder(os.environ.get("RECALL_EMBEDDER", "fastembed"))
    with PgVectorStore(dsn, dim=embedder.dim) as store:
        store.check_schema()
        for proposal in [
            "let's inject retrieved context into the prompt to boost answers",
            "should we add a brand new telemetry dashboard",
        ]:
            d = decide(store, embedder, proposal)
            print(f"\nPROPOSAL: {proposal}\n  -> proceed={d['proceed']}\n  {d['reason']}")


if __name__ == "__main__":
    main()
