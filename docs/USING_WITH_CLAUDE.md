# Using RE-call with Claude (MCP)

RE-call ships an MCP server so an agent can query its **own memory as a tool** — the self-recall
loop. This is how it's used in production: the agent calls `recall_search` *before* it acts, and a
surfaced closed decision (that isn't a `gap_warning`) tells it to back off instead of re-litigating.

Works with **Claude Code** and **Claude Desktop** — both take the same MCP server block.

## 1. Install & run

```bash
pip install -e ".[fastembed,mcp]"
python -m recall.cli --migration-dsn "$RECALL_MIGRATION_DSN" schema --dim 384 apply
python -m recall_mcp.server        # stdio server (Claude launches this for you via the config below)
```

The migration command is a deployment/provisioning step, not part of server startup. Use a
schema-owner DSN for it and an unprivileged `RECALL_SERVING_DSN` for the server; see
[MIGRATIONS.md](MIGRATIONS.md).

The MCP server opens the default `chunks` table. If you used a named table for a local CLI demo,
apply the default-table schema separately before starting MCP, or use an embedder whose dimension
matches the existing `chunks` table.

## 2. Register the server

Both clients use the same `mcpServers` block; only the entry point differs.

```json
{
  "mcpServers": {
    "recall": {
      "command": "python",
      "args": ["-m", "recall_mcp.server"],
      "env": {
        "RECALL_SERVING_DSN": "postgresql://recall:recall@localhost:5432/recall",
        "RECALL_TENANT": "default",
        "RECALL_TRUST_MODE": "development"
      }
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

Remove `RECALL_TRUST_MODE` for production after the tenant has an active generation with a
published certified calibration. It is included above only for local, uncalibrated stdio use.

Optional env: `RECALL_EMBEDDER=hashing` for the fully-offline embedder (default `fastembed`);
`RECALL_INDEX_ROOT` bounds where the development-only `recall_index` may read (default: the
server's working directory). V1 calibration is stored in PostgreSQL and resolved from the
authenticated tenant plus active generation on every search. `RECALL_CALIBRATION` and local
`calibration.json` files are never selected automatically; import a legacy file with
`recall --tenant T calibration import FILE` to retain it as unbound evidence.

### `RECALL_RERANK=1` — the single biggest quality lever

Set it and every `recall_search` reranks its candidate pool with a cross-encoder before truncating
to `k`. Measured on LOCOMO at n=1 536 ([`FINDINGS.md` §11](../results/FINDINGS.md)):

| hit@k | off | **on** |
|---|---|---|
| 1 | 0.398 | **0.553** |
| 5 | 0.671 | **0.777** |

Intervals disjoint from the baseline through k=10 — the largest single retrieval gain measured in
this project, roughly twice the best embedder effect, and it lifts every question category
including multi-hop.

**Off by default because it costs ~1 050 ms per query on CPU** (needs `pip install recall[rerank]`).
For an agent answering a person that is usually invisible next to the model call which follows, and
the answer is materially better. For high-volume automated retrieval, or constrained hardware,
leave it off.

`ms-marco-MiniLM-L-6-v2` is the default because it was *measured* to be right, not because it was
already there: `bge-reranker-base`, 12× the parameters and four years newer, is statistically
**indistinguishable** at 6.3× the cost. Override with `RECALL_RERANK_MODEL`, which then **requires**
`RECALL_RERANK_REVISION` — an unpinned Hub reference is mutable and the shipped pin belongs to the
shipped weights only.

⚠️ A value that is neither truthy nor falsey (`RECALL_RERANK=treu`) is **refused** rather than read
as "off". A server that quietly ignored the flag would be fast, silent, and indistinguishable from
one that honoured it.

## 3. The tools

| Tool | When the agent calls it |
|------|-------------------------|
| **`recall_search`** | *Before* proposing an idea, forming a hypothesis, or repeating past work — to check what memory already says. Every hit carries a trust `verdict` (`ok / superseded / expired / not_yet_valid / low_confidence / invalid_metadata` — plus `not_entailed` when the opt-in entailment stage is enabled; this MCP server keeps it off), the true dense cosine (`score`), a calibrated `confidence`, `superseded_by`, `valid_until`, and `indexed_at`; the result adds `abstained`, `reason`, `calibrated`, calibration status and ID, tenant/generation/pipeline/corpus/query-set identities, `stale`, `gap_warning`, and `advice`. `calibrated` is true only for a certified exact generation binding. When `abstained` is true, the advice is explicit: say you don't know, do not answer from the hits. |
| **`recall_evidence`** | When the agent is about to *answer* from memory rather than merely consult it. Same retrieval, returned as a citable bundle: only passages the trust layer cleared, in retrieval order, plus `system_prompt` and `user_message` — a fixed library-authored instruction and a delimited, JSON-escaped data payload. `decision: "abstain"` means the bundle is empty and the agent must not answer from memory. The server runs no generator; the client is the generator, which is why the prompt is handed back rather than consumed. Validate the answer with `recall.validate_answer`: it checks that every citation resolves to a supplied `chunk_id`, and deliberately does not check that a cited passage supports the answer. |
| **`recall_index`** | To add a markdown file/folder to memory (bounded by `RECALL_INDEX_ROOT`). |
| **`recall_forget`** | To permanently delete indexed memory for source values returned by `recall_search`. Requires `recall:forget` on authenticated transports; irreversible, so check the returned `sources_not_found` before assuming a request matched. |
| **`recall_stats`** | To check how much memory exists and whether the index is stale. |

## 4. The self-recall loop (redacted)

A real interaction, with the domain scrubbed to placeholders — the shape is exact:

```text
You:     "Let's try <STRATEGY-X> on <MARKET-Y>."

Claude:  → recall_search("<STRATEGY-X> on <MARKET-Y>")
         recall → 1 hit — verdict ok, abstained: false:
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
         recall → abstained: true — "no hit above the calibrated confidence threshold
                  (probable corpus gap)"; every hit verdict low_confidence.

Claude:  "Memory has no real answer on that — I'd be guessing. Want me to research it fresh?"
```

The agent-side glue is tiny — see [`examples/self_recall_agent.py`](../examples/self_recall_agent.py)
for the ~30-line pattern: search first; if a non-gap closed decision surfaces, back off.

— Back to the [README](../README.md) · the [engineering writeup](WRITEUP.md) · the
[case study](CASE_STUDY.md).

## Strict trust: what the tools return when the gate cannot certify an answer

The MCP service defaults to **strict**, like the library. A server that degraded by omission would
be a server that degrades in production.

`SearchResult` carries two additional fields:

- `trust_state` — `trusted` or `degraded`. A strict server never returns `degraded`; it refuses.
- `failure_code` — the stable code when the gate could not certify: `INDEX_NOT_READY`,
  `LINEAGE_MISMATCH`, `CALIBRATION_MISSING`, `CALIBRATION_UNCERTIFIED`, `CALIBRATION_STALE`,
  `DEPENDENCY_UNAVAILABLE`. `null` when trusted.

A strict refusal propagates as `TrustRefusal` rather than an empty `SearchResult`, deliberately. A
result object with no hits is indistinguishable from "the gate ran and found nothing", and keeping
those two apart is the entire purpose of this layer.

### What an agent should do with each

| Situation | What it means | What the agent should do |
|---|---|---|
| `abstained: true`, no failure code | A working, certified gate found nothing it would stand behind | Say you do not know. This is a real answer |
| `TrustRefusal` with any code | The gate could not run | Do **not** treat this as "no prior memory". Report the outage and do not proceed on the assumption that nothing was found |
| `trust_state: degraded` | A development server answered without a certified threshold | Treat every hit as unverified context. There is no abstention decision to rely on |

The failure between these is the one worth designing against: an agent that reads an outage as
"nothing found" will re-litigate a decision that was in fact settled, confidently. See
`examples/self_recall_agent.py`, whose refusal branch exists for exactly this.

### Development mode

Set `RECALL_TRUST_MODE=development` for local work against a corpus with no published calibration.
Anything other than the exact string `development` stays strict, so a typo cannot open the gate.
