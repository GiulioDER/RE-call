# Using RE-call with Claude (MCP)

RE-call ships an MCP server so an agent can query its **own memory as a tool** — the self-recall
loop. This is how it's used in production: the agent calls `recall_search` *before* it acts, and a
surfaced closed decision (that isn't a `gap_warning`) tells it to back off instead of re-litigating.

Works with **Claude Code** and **Claude Desktop** — both take the same MCP server block.

## 1. Install & run

```bash
pip install -e ".[fastembed,mcp]"
python -m recall_mcp.server        # stdio server (Claude launches this for you via the config below)
```

## 2. Register the server

Both clients use the same `mcpServers` block; only the entry point differs.

```json
{
  "mcpServers": {
    "recall": {
      "command": "python",
      "args": ["-m", "recall_mcp.server"],
      "env": { "RECALL_DSN": "postgresql://recall:recall@localhost:5432/recall" }
    }
  }
}
```

- **Claude Code** — save this as `.mcp.json` in your project root, or run
  `claude mcp add recall -- python -m recall_mcp.server`.
- **Claude Desktop** — add the block to `claude_desktop_config.json`
  (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`), then restart.

> 🔒 **Credentials.** The DSN above is the **local Docker dev** default — not a secret. For any real
> database, supply the DSN (and the optional `VOYAGE_API_KEY` for the cloud embedder) through your
> shell environment or a **gitignored** `.env` — **never commit credentials to the config file or the
> repo.** The server reads them from the environment.

Optional env: `RECALL_EMBEDDER=hashing` for the fully-offline embedder (default `fastembed`);
`RECALL_INDEX_ROOT` bounds where `recall_index` may read (default: the server's working directory).

## 3. The three tools

| Tool | When the agent calls it |
|------|-------------------------|
| **`recall_search`** | *Before* proposing an idea, forming a hypothesis, or repeating past work — to check what memory already says. Every hit carries a trust `verdict` (`ok / superseded / expired / not_yet_valid / low_confidence`), a calibrated `confidence`, `superseded_by`, and `indexed_at`; the result adds `abstained` + `reason` + `gap_warning` + `advice`. When `abstained` is true, the advice is explicit: say you don't know, do not answer from the hits. |
| **`recall_index`** | To add a markdown file/folder to memory (bounded by `RECALL_INDEX_ROOT`). |
| **`recall_stats`** | To check how much memory exists and whether the index is stale. |

## 4. The self-recall loop (redacted)

A real interaction, with the domain scrubbed to placeholders — the shape is exact:

```text
You:     "Let's try <STRATEGY-X> on <MARKET-Y>."

Claude:  → recall_search("<STRATEGY-X> on <MARKET-Y>")
         recall → 1 hit, cosine 0.71 (NOT a gap):
            closed_hypotheses_index.md — "<STRATEGY-X> — CLOSED, FALSIFIED:
            <redacted> failed out-of-sample; re-entry needs a different data tier."

Claude:  "We already tested that and closed it — it didn't survive out-of-sample validation.
          Re-entering would need a different data tier. Want me to look at what's changed
          since, rather than re-run the dead version?"
```

And the opposite job — a genuine gap, where the guard stops Claude trusting noise:

```text
You:     "What did we conclude about <a topic the memory never covered>?"

Claude:  → recall_search(...)
         recall → [GAP] top match 0.41 < 0.50 — probable corpus gap; treat hits as noise.

Claude:  "Memory has no real answer on that — I'd be guessing. Want me to research it fresh?"
```

The agent-side glue is tiny — see [`examples/self_recall_agent.py`](../examples/self_recall_agent.py)
for the ~30-line pattern: search first; if a non-gap closed decision surfaces, back off.

— Back to the [README](../README.md) · the [engineering writeup](WRITEUP.md) · the
[case study](CASE_STUDY.md).
