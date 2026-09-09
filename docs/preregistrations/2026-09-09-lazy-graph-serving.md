# Pre-registration: lazy, query-scoped graph serving

**Date:** 2026-09-09   **Status:** predicted, not yet measured

## The question

Does one-hop graph expansion keep database query count and fetched text bounded as the synthetic
corpus grows from 1,000 to 100,000 chunks?

## What I predict

For each corpus size, the graph path will issue a constant number of store operations, fetch no
more than the configured graph candidate budget in one batch, and transfer text bytes only for
those admitted candidates. Trust evaluation will be called once for every fetched candidate.

## What would falsify this

Any query count that increases with corpus size, any text fetch or trust call above the graph
candidate budget, or any candidate admitted without a trust evaluation.

## How it will be measured

`python -m pytest tests/test_graph_lazy_loading.py -q`, with synthetic corpora of 1,000, 10,000,
and 100,000 chunks. The test records store operation count, fetched text bytes, batch size, and
trust evaluation IDs, and checks each against the configured budget.

## What I already know

The current one-hop path calls `project_store_graph(..., include_text=True)`, which streams the
entire generation before it selects bounded candidates. The persisted semantic graph itself has
no text column. This was confirmed by inspecting `recall_mcp.service._expand_semantic_graph` and
`recall.semantic_graph.load_semantic_graph` on 2026-09-09.

## Confounds I can name now

The synthetic store uses in-memory metadata and a deterministic embedder, so its absolute timings
are not a PostgreSQL or network latency measurement. The result tests scaling invariants only.
