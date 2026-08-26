# Using RE-call with Claude (MCP)

RE-call ships an MCP server so an agent can query its **own memory as a tool** — the self-recall
loop. This is how it's used in production: the agent calls `recall_search` *before* it acts, and a
surfaced closed decision (that isn't a `gap_warning`) tells it to back off instead of re-litigating.

Works with **Claude Code** and **Claude Desktop** — both take the same MCP server block.

**On Claude Code, the plugin is the short path** — it wires the server, the session hooks and a
skill that teaches Claude when to search, and keeps the DSN in your OS keychain:

```
/plugin marketplace add GiulioDER/RE-call
/plugin install recall@re-call
```

You still need a database first (`recall quickstart` makes a throwaway one). See
[plugin/README.md](../plugin/README.md). Everything below is the manual wiring, for other
clients and for anyone who wants to see what the plugin writes.

## 1. Install & run

```bash
pip install "recall-rag[fastembed,mcp]"
python -m recall.cli --migration-dsn "$RECALL_MIGRATION_DSN" schema --dim 384 apply
python -m recall_mcp.server        # stdio server (Claude launches this for you via the config below)
```

(Contributors working from a clone use `pip install -e ".[fastembed,mcp]"` instead.)

The migration command is a deployment/provisioning step, not part of server startup. Use a
schema-owner DSN for it and an unprivileged `RECALL_SERVING_DSN` for the server; see
[MIGRATIONS.md](MIGRATIONS.md).

The MCP server opens the table `RECALL_TABLE` names, defaulting to `chunks`; see the `RECALL_TABLE` note under the server block below for the production exception. If you used a named table for a local CLI demo, point `RECALL_TABLE` at it, or
apply the default-table schema separately before starting MCP, or use an embedder whose dimension
matches the existing `chunks` table.

## 2. Register the server

**`python -m recall.cli setup` does everything in this section for you**, including the session
hooks below, and it is the recommended path. It registers at local scope, sets `RECALL_TRUST_MODE`,
writes the entry under every spelling of your project path that the client already knows, and
prints the keys it used. The rest of this section is what it does, for anyone wiring it by hand or
debugging what the wizard wrote.

Both clients use the same `mcpServers` block; only the entry point differs.

```json
{
  "mcpServers": {
    "recall": {
      "command": "python",
      "args": ["-m", "recall_mcp.server"],
      "env": {
        "RECALL_SERVING_DSN": "postgresql://recall:recall@localhost:5432/recall",
        "RECALL_TABLE": "chunks",
        "RECALL_TENANT": "default",
        "RECALL_TRUST_MODE": "development"
      }
    }
  }
}
```

⚠️ **`RECALL_TABLE` and `RECALL_TENANT` must name the corpus you actually indexed, and getting
either wrong is SILENT.** The values above are what `recall setup` writes. `recall quickstart` uses
`quickstart_chunks` and `quickstart` instead, deliberately, so that its 22 documents of fiction can
never be retrieved beside real memory from the same database; it prints all four values when it
finishes. Point the server at the wrong one and it starts cleanly, answers, and reports
`0 relevant memory hit(s)` — indistinguishable from an empty corpus, because that is exactly what
it found. `RECALL_TRUST_MODE` at least names its own cause (`INDEX_NOT_READY`); these two do not.

`RECALL_TABLE` applies to the legacy single-tenant store only. Under `RECALL_ENV=production` or
authenticated tenant routing the store reads the generation table `recall_chunks_v1`, and a server
started with both refuses at startup rather than quietly serving a different corpus than the one it
was told to.

- **Claude Code** — register it at **local scope**, which is what
  `claude mcp add recall -- python -m recall_mcp.server` does by default. It writes the block into
  Claude Code's own `~/.claude.json` under this project, and loads only here.

  ⚠️ **Saving the block as `.mcp.json` in the project root also works, and is the worse option.**
  Claude Code gates project-scoped servers from that file behind an approval prompt, and until you
  answer it in an interactive session the tools are simply absent — no error, nothing naming the
  cause. It also puts a DSN in the repository, which the credentials note below says not to do.

  When one server name is defined in more than one scope, Claude Code uses **one** definition and
  does not merge fields across scopes. The order, **highest precedence first**, is:

  1. **local** — `~/.claude.json` under this project's entry
  2. **project** — `.mcp.json` in the repository
  3. **user** — `~/.claude.json` at the top level

  So a local entry **beats** an `.mcp.json` of the same name rather than losing to it. If the tools
  do not appear, read that order before deleting anything: a `.mcp.json` sitting under a local entry
  of the same name is already inert, and removing it changes nothing. `docs/WIZARD.md` has the full
  reasoning, and the installer does all of this for you.
- **Claude Desktop** — add the block to `claude_desktop_config.json`
  (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`), then restart.

### Two things the scope list above does not say

**Do not use user scope for this**, tempting though the "works everywhere" reading is. A recall
server carries one `RECALL_TENANT` and one DSN, so a user-scope entry follows you into every
unrelated checkout and answers confidently about a corpus belonging to a different repository. That
failure never raises an error, which is what makes it the expensive one.

⚠️ **A local entry is keyed by your project's path**, so moving or renaming the project orphans it
silently: no error, no tools. Re-run the registration after a move. The installer prints the keys
it registered under for this reason.

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

**Off by default because it costs ~1 050 ms per query on CPU**
(needs `pip install "recall-rag[rerank]"`).
For an agent answering a person that is usually invisible next to the model call which follows, and
the answer is materially better. For high-volume automated retrieval, or constrained hardware,
leave it off.

`ms-marco-MiniLM-L-6-v2` is the default because it was *measured* to be right, not because it was
already there: `bge-reranker-base`, 12× the parameters and four years newer, is statistically
**indistinguishable** at 6.3× the cost. Override with `RECALL_RERANK_MODEL`, which then **requires**
`RECALL_RERANK_REVISION` unless it names a built-in pinned alias such as `coreb-code`. An unpinned
Hub reference is mutable and the shipped pin belongs to the shipped weights only.

⚠️ A value that is neither truthy nor falsey (`RECALL_RERANK=treu`) is **refused** rather than read
as "off". A server that quietly ignored the flag would be fast, silent, and indistinguishable from
one that honoured it.

## 3. The tools

| Tool | When the agent calls it |
|------|-------------------------|
| **`recall_search`** | Before proposing an idea, forming a hypothesis, or repeating past work. Use it to check what memory already says, and follow the returned `advice`. |
| **`recall_evidence`** | When the agent is about to answer from memory. It returns a citable evidence bundle plus `system_prompt` and `user_message`; the client remains the generator. |
| **`recall_index`** | To add a markdown file/folder to memory (bounded by `RECALL_INDEX_ROOT`). |
| **`recall_forget`** | To permanently delete indexed memory for source values returned by `recall_search`. Requires `recall:forget` on authenticated transports; irreversible, so check the returned `sources_not_found` before assuming a request matched. |
| **`recall_stats`** | To check how much memory exists and whether the index is stale. |
| **`recall_reasoning_query`** | When the agent explicitly needs the reasoning layer. It runs over trusted retrieval, obeys the configured policy and budget, and returns cited output, review state, clarification, or abstention. |
| **`recall_reasoning_projection`** | To inspect the generation-bound reasoning graph projection without answering. |
| **`recall_reasoning_proposals`** | To inspect inference proposals as review candidates, not trusted memory. |
| **`recall_reasoning_audit`** | To verify reasoning integration state and diagnostics before relying on the reasoning layer. |

`recall_search` returns `abstained`, `reason`, `calibrated`, calibration identity, tenant and
generation identity, `stale`, `gap_warning`, `advice`, and hits carrying `verdict`, `score`,
`confidence`, `superseded_by`, `valid_until`, and `indexed_at`. `calibrated` is true only for a
certified exact generation binding. When `abstained` is true, say you do not know and do not answer
from the hits.

For non-English presentation, pass `locale` to `recall_search` or `recall_evidence` after enabling
the optional translation endpoint; localized text is additive and never replaces canonical
evidence (configuration in [ENVIRONMENT.md](ENVIRONMENT.md)).

`recall_evidence` uses the same retrieval path, but admits only passages cleared by the trust layer.
Reasoning tools are additive and opt in; `recall_search` and `recall_evidence` keep the same
retrieval behavior when they are used directly.
`decision: "abstain"` means the bundle is empty and the agent must not answer from memory. Validate
generated answers with `recall.validate_answer`, which checks that every citation resolves to a
supplied `chunk_id`.

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

## Session hooks: what makes the tools get used

Registering the server makes the tools *available*. It does not make Claude reach for them, and it
does not keep the corpus current. `recall setup` also offers three hooks, written into
`~/.claude/settings.json`:

| Event | What it does | Why there |
|---|---|---|
| `SessionStart` | Injects a one-line digest naming the indexed chunk count and the standing instruction | The only event that can add context before the first turn |
| `PreCompact` | Indexes `memory/` | Compaction is where a long session loses the detail behind its conclusions |
| `SessionEnd` | Indexes `memory/` and refreshes the cached count | Closes the write-to-searchable loop |

Three properties are deliberate and worth knowing before you edit them:

- **`SessionStart` touches no database.** It reads a count cached by the other two hooks, so the
  digest still appears when your database is not running, and it adds about 66 ms to a session
  start rather than the ~1.2 s a database round trip and an embedder import would cost.
- **`PreCompact` never blocks.** Exit code 2 on that event *blocks compaction*, so every path
  returns 0 and the handler runs `async`. A memory tool must not be able to wedge a session whose
  context window is already full.
- **They run out of `recall_hooks`, not `recall`.** Importing the `recall` package costs about a
  second, and a session-start hook pays that on every launch.

The hooks are removable exactly, without disturbing anything else in the file:

```bash
python -c "from recall.claude_code import uninstall; uninstall()"
```

## 5. Configure your project's memory files

`recall setup` offers to scaffold two files after the embedder/reranker/entailment prompts:

- A `<!-- recall setup begin -->` / `<!-- recall setup end -->` block appended to `CLAUDE.md`
  (created if missing) telling Claude when to call `recall_search`/`recall_evidence` and how to
  write new facts to `memory/`. Re-running `recall setup` only replaces this block — everything
  else in `CLAUDE.md` is left alone.
- A starter `memory/MEMORY.md`, created only if one does not already exist, documenting the
  frontmatter convention (`name`, `description`, `metadata.type`) and the one-line-per-fact index
  format. `memory/*.md` files you add later are where individual facts live.

The wizard then tries to index `memory/` immediately, so `recall_search` can find it from your
first turn with Claude. This auto-index step requires the schema to already be applied at a
dimension matching the chosen embedder; if the schema has not been applied yet (or is applied at a
different dimension), indexing fails and the wizard prints remediation instead of blocking setup;
run `python -m recall.cli index memory/` yourself once the schema is ready. Under
`RECALL_ENV=production` this indexing step is skipped entirely (local filesystem indexing is
development-only there — see [PRODUCTION.md](PRODUCTION.md)); index it through your production
build pipeline instead.

Auto-index always targets the default table and tenant (`DEFAULT_TABLE`/`DEFAULT_TENANT`), even if
you ran `recall setup` with a non-default `--tenant` or `--table`. If your project uses a
non-default tenant or table, the rows written by auto-index will not show up in that tenant's later
searches; index `memory/` yourself against the correct table/tenant instead.

The first auto-index run indexes `memory/MEMORY.md` itself along with any facts already present,
since it walks the whole directory. On a fresh scaffold that means the starter file's own
frontmatter/format instructions are indexed as a chunk — harmless (the trust layer scores generic
boilerplate low), but if `recall_search` surfaces it, that's what happened.

The scaffold/index step runs only after `.env` is written, so an interruption during a slow model
download or DB connection never loses the answers you already gave earlier in the interview.

Decline the prompt to skip scaffolding entirely, or answer it again on a later `recall setup` run
to refresh the `CLAUDE.md` block.

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
