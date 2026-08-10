# Examples

## `basic_trusted_search.py` — five-minute product path

Indexes and searches the bundled demo corpus using the same trust-aware result surface the MCP tool
returns.

```bash
docker compose up -d --wait
python -m recall.cli --table recall_quickstart \
  --migration-dsn postgresql://recall:recall@localhost:5432/recall \
  schema --dim 384 apply
RECALL_TRUST_MODE=development python -m recall.cli --table recall_quickstart index corpus
python -m examples.basic_trusted_search
```

## `self_recall_agent.py` — anti-re-litigation

Demonstrates the self-recall pattern: an agent consults its own memory before acting and
backs off when a closed decision or falsified hypothesis surfaces.

Index the demo corpus first, then run:

    python -m recall.cli index corpus          # populate memory
    python -m examples.self_recall_agent       # run the two sample proposals

`decide(store, embedder, proposal)` returns `{"proceed": bool, "reason": str}`. In a real MCP
client the same pattern is: call the `recall_search` tool before proposing, and back off
(citing the memory) unless the result is a `gap_warning`.
