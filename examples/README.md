# Examples

## `self_recall_agent.py` — anti-re-litigation

Demonstrates the self-recall pattern: an agent consults its own memory before acting and
backs off when a closed decision or falsified hypothesis surfaces.

Index the demo corpus first, then run:

    python -m recall.cli index corpus          # populate memory
    python -m examples.self_recall_agent       # run the two sample proposals

`decide(store, embedder, proposal)` returns `{"proceed": bool, "reason": str}`. In a real MCP
client the same pattern is: call the `recall_search` tool before proposing, and back off
(citing the memory) unless the result is a `gap_warning`.
