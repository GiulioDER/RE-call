# Compiled graph (opt-in, LLM-built)

The compiled graph is an evidence graph that a language model builds from a tenant's sources. It
is **off by default** and changes nothing until a tenant's mode is set. It works the same way on
PostgreSQL and on a Lite (SQLite) store.

It carries **no quality claim**. A grounded graph of this design showed no measured gain on the
benchmark service it was first built for. Measure it on your own corpus with `shadow` before you
turn it to `active`.

## What it does

1. **At build time**, the model reads each source as numbered excerpts (1,600 characters, 160
   overlapping) and proposes up to eight typed records per source (fact, decision, procedure,
   problem, outcome, constraint, event). Each record cites excerpts by id.
2. **The server, not the model, decides what is kept.** A record keeps a field or an entity only
   when that text is copied verbatim from an excerpt it cites. An edge links a record to every
   chunk of the same source whose offsets overlap a cited excerpt. The model never names a graph
   endpoint.
3. **At search time**, the query is compared with the records. Chunks linked from the 20 nearest
   records may be promoted, at most two of them, into **ranks 9 and 10** of an unscoped search.
   Ranks 1 to 8 and every score never change; the two chunks promoted come from further down the
   candidate pool, so with 9 or more results they replace the chunks that held ranks 9 and 10. A
   scoped search is never touched. Search never calls the model.
4. **For `recall_reasoning_query`**, in `active` mode only, an entity a record copied becomes a
   mention on each linked chunk whose own text holds it, but only onto an entity the authored graph
   already has (matched by normalized name, never an ambiguous one). Graph expansion still walks
   authored relations only, and these mentions only widen which chunks a relation's neighbor
   reaches: the seed entities, the hub counts and the order in which the candidate budget is spent
   all come from authored mentions, and a chunk an authored mention reaches keeps only its authored
   relations. So every candidate `off` admits, `active` admits too, unchanged. Every chunk the
   expansion adds that no authored mention reaches carries `graph_link: llm_compiled` on its hit; a
   chunk retrieval already returned is never labelled. Such a chunk still competes on score for the
   result's limited places. A scoped request admits mentions only on chunks its scoped graph
   already mentions, which are a subset of the chunks its source policy lets it see.

With the default `k=5`, a plain search returns the same results with the graph on. The effect is
visible when a caller asks for 9 or more results, and in the reasoning query.

## Modes

| `RECALL_COMPILED_GRAPH_MODE` | Build | Search | Reasoning query |
|---|---|---|---|
| `off` (default) | nothing | nothing | nothing |
| `shadow` | builds the artifact | reports `shadow:would_promote=N`, order unchanged | nothing |
| `active` | builds the artifact | promotes, reports `active:promoted=N` | admits LLM mentions in graph expansion |

`RECALL_COMPILED_GRAPH_MODE_BY_TENANT=tenant-a:active,tenant-b:off` overrides the mode per tenant;
each entry splits on its last colon. A malformed entry, a tenant named twice, or an unknown mode in
any entry is refused: the MCP server does not start, a Lite search raises, and a build reports
`compiled graph not built` without failing the write or promotion that called it.
Every search result carries the outcome in its diagnostics as `compiled_graph`, including
`skipped:` and why (`scoped`, `missing`, `unconfigured`, `lineage`, `damaged`, `error`).

## Configuration

A build needs every one of these; there is no default model and no default price. A search in
`shadow` or `active` also needs `RECALL_COMPILED_GRAPH_ARTIFACT_ROOT` and the
`RECALL_COMPILED_GRAPH_MODEL` the artifact was built with; without either it reports
`skipped:unconfigured`.

| Variable | Meaning |
|---|---|
| `RECALL_COMPILED_GRAPH_ARTIFACT_ROOT` | absolute folder for artifacts and the compile cache |
| `RECALL_COMPILED_GRAPH_MODEL` | for example `openai/gpt-6-luna-pro` or `openai/gpt-4o-mini` |
| `RECALL_COMPILED_GRAPH_API_KEY` | key of the provider |
| `RECALL_COMPILED_GRAPH_BASE_URL` | OpenAI-compatible endpoint, default `https://openrouter.ai/api/v1` |
| `RECALL_COMPILED_GRAPH_MAX_USD` | USD cap of one build |
| `RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT`, `_OUTPUT` | the model's prices, which the cap is computed from |
| `RECALL_COMPILED_GRAPH_TIMEOUT` | seconds per request, default 60 |

## Cost and failure

- **The cap.** Before every attempt, its worst case (one token per request byte, plus the 2,400
  token output cap) must fit what is left of `RECALL_COMPILED_GRAPH_MAX_USD`, or nothing is sent and
  the build stops. A reply that was billed is never sent again.
- **The cache.** Accepted records are cached by provider, model, prompt and source content, so an
  unchanged source is never sent twice and a stopped build resumes where it stopped.
- **Credit.** A credit or authorization refusal (401, 402, a spending or key limit) stops the build
  at once. On Lite it also sets a backoff, so writes stop calling the provider until
  `recall graph compile --force`.
- **Other failures.** The same refusal on three sources in a row (an unknown model, an unsupported
  parameter) stops the build. A single rejected source is skipped.
- **What a stopped build leaves.** No artifact is written. Searches run without the graph and say
  `skipped:missing`. A build never makes a write, an index run or a promotion fail.

## When it is built

- **Lite:** after every write, beside the atomic rescue. Only sources the cache has not seen cost a
  call.
- **PostgreSQL:** before a generation is promoted.
- **On demand:** `recall graph compile [--generation G] [--force] [--dry-run]`. `--dry-run` prints
  the worst-case spend and sends nothing.

An artifact is bound to the tenant, generation, corpus fingerprint, embedder, model and prompt. It
is written once, refused when any of those differ, and refused when a file fails its checksum.

## Forget

`recall forget` and `recall withdraw` remove every artifact of the forgetting tenant that quotes a
forgotten source, and the source's rows in the compile cache (deleted with SQLite's
`secure_delete`). The tenant searches without the graph until the next build, which no longer sees
the source. Another tenant's artifacts are kept, even where a source has the same name.

## Measuring it

Run with `shadow`, compare searches with `k` of 10 or more against `off`, and write down what you
expect before you look. Turn it to `active` only on evidence from your own corpus.
