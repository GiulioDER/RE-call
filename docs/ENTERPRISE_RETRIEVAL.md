# Enterprise retrieval and evidence deployment

This implementation keeps the existing chunk table and retrieval flow. It adds immutable embedding identities, fixed process retrieval profiles, deterministic contextual passage text, a generator neutral evidence boundary, and PostgreSQL generation routing.

## Embedding profile registry

`recall/embedding_registry.py` is the single place a profile is defined. It owns the profile identifier, the model, the artifact digest, the dimension, the query and passage encoder modes, the normalization, the instruction version, the chunker version and the context version. Nothing else may hold a second copy of that vocabulary.

Three properties follow from that and are enforced by tests:

* `context_version` is derived from `context_mode`, never declared next to it. `Indexer` refuses an embedder whose profile does not spell its context exactly `raw-v1` or `context-<mode>-<policy version>`, so the derivation in the registry is that contract written once.

* `query_mode` and `passage_mode` name the encoder that is actually called. A profile declaring `query_embed` gets `TextEmbedding.query_embed`, and a backend without that encoder refuses to start rather than falling back to the symmetric one.

* The declared dimension is checked against the artifact at startup. An artifact that embeds at a different width is not that profile, and the process refuses rather than writing vectors no other process can interpret.

Registered identifiers: `bge-small-symmetric-v1`, `bge-small-asymmetric-v1`, `bge-small-context-document-v1`, `bge-small-context-section-v1`, `bge-small-context-neighbor-v1`, and the rejected `qwen3-embedding-0.6b-384-v1`.

Passage encoding is used for indexing and for dimension discovery. Query encoding is used for retrieval, calibration, semantic lint, evaluation and the timing wrappers. An embedder that implements only `embed` keeps working: both helpers fall back to it, and its cached vectors are keyed under a legacy descriptor that no verified profile can collide with.

## Embedding cache identity

The embedding cache is keyed by the complete immutable profile identity, not by the profile identifier. `EmbeddingProfile.fingerprint` covers every identity field, including `artifact_digest`, `context_version` and the pinned inference library version, and the cache key adds the purpose (`query`, `passage` or `legacy`), the dimension and the text.

The identifier alone is not an identity. A re-provisioned artifact and a context mode change both move the stored vectors while the identifier stays fixed, and a key that misses either serves a vector computed from different weights or from different text. A cache hit is a plausible vector of the right width, so nothing downstream can detect it. Cross identity reuse therefore fails closed: the key misses and the text is embedded again.

Changing the fingerprint encoding invalidates every cache in existence at once. If that is ever wanted, bump the domain tag inside `EmbeddingProfile.fingerprint` so the change is legible.

## Deterministic context modes

Three context modes build the text handed to the embedder. They are declared by the profile (`bge-small-context-document-v1`, `bge-small-context-section-v1`, `bge-small-context-neighbor-v1`) and implemented in `recall/context.py`. `mode="none"` is the symmetric baseline and embeds the chunk as stored.

**Embedding text is built separately from stored text, and the stored text never moves.** `chunk_text()` and the chunk row are untouched by every mode. The rendered passage is assembled from `StructuredChunk`, which carries source offsets and the heading hierarchy alongside the chunk's own bytes.

| Rule | Behaviour |
|---|---|
| Title precedence | frontmatter `title`, then the first H1, then the root-relative basename. The frontmatter key must be **top level**; an indented `title:` belongs to a sub-object and is skipped. The basename is taken from the whole path, before any cap |
| Paths | root-relative only. An absolute path, a drive letter, a UNC path or any `..` segment is **refused**, in every mode including `none`. `root_relative_source` validates and **does not truncate**: the cap belongs to the rendered field, because a cap applied inside the guard runs after its own checks and can reintroduce what they refused. The refusal names the rule, never the path, since the value it fires on is an absolute host path |
| Control characters | stripped from every structural field (title, source, section hierarchy). The chunk is content and is left exactly as stored |
| Caps | title 256 characters, source 256, section hierarchy 512 |
| Neighbour context | at most 200 characters from each adjacent chunk: the tail of the preceding one, the head of the following one. Folded to one line **before** the 200 is counted, so the neighbour budget is in the same unit as the other caps and an adjacent chunk cannot put a second `source:` line into this chunk's passage. Folding also collapses whitespace runs, so a neighbour excerpt is normalised **more** than the other structural fields, which keep theirs. None is invented at a document's first or last chunk |
| Degradation under a token limit | drop neighbour context first, shorten then drop section detail second, drop title detail last. **The complete current chunk is preserved at every rung**; it is never shortened to make room, and the last resort is the bare chunk |
| Recorded identity | the mode and the policy version are written into each chunk's metadata (`context_mode`, `context_version`) and into the profile identity, where `context_version` is derived from the mode and is part of the cache fingerprint |

**The load-bearing invariant: raw chunk content and raw content hashes are byte-identical across generations and across all three modes.** A context mode changes what is embedded and nothing else, so a cutover between generations built under different modes changes how the corpus is retrieved, never what it says. `tests/test_context_modes.py` asserts it over five corpus shapes (with frontmatter, without, no headings, nested headings, and across chunker boundaries) and `tests/test_context_modes_index.py` asserts it again against stored PostgreSQL rows, including a real dual write with the two generations on different modes.

The one field that deliberately **does** change with the mode is `index_fingerprint`, the value the indexer compares to decide whether a file needs re-indexing. If it did not move, switching a generation's context mode would skip every unchanged file and leave vectors built under the old mode in place.

The rendered form is one `field: value` per line. It is text to EMBED and never text to parse: the chunk itself is interpolated verbatim, because rule 5 preserves it, so a document containing a line that looks like a field will render one. Structural fields cannot forge a line (control characters are stripped) and neither can a neighbour excerpt (it is folded), but do not write a parser against this format.

**The 256-character cap on the rendered `source:` field applies to a path the guard has already accepted, so the FIELD can still end in a truncated `..`.** `root_relative_source` refuses traversal and its return value carries none; the cap is applied afterwards, where the field is built, and truncating any path at a fixed length can produce that shape. It is inert — the field is embedding text and is never resolved — but a future consumer of it must not read "traversal is refused" as a property of the rendered string. The cap also keeps the HEAD of a long path, so two paths sharing 256 characters render identically; keeping the tail would identify a document better and is a deliberate open decision, not an oversight.

Which mode retrieves best is not decided here, and no measurement in this repository claims it.

## Security boundary

All model artifacts must exist locally before startup. An explicit embedding profile verifies the configured artifact tree against its SHA256 digest and requests local only loading. The artifact is verified before the backend library is even imported, so a missing or tampered tree fails the same way whether or not the optional extra is installed. The quality retrieval profile also requires a local reranker path and digest. Production should block outbound network access at the workload boundary.

Runtime model downloads are prohibited. Startup is proven to complete with every socket entry point blocked, and to refuse when the artifact is missing or its checksum does not match.

Tenant routes never accept a physical table from a client. The runtime resolves table names only from validated control plane rows. Chunk tables, tenant routes, and migration events use row level security. The runtime database role must be neither superuser nor `BYPASSRLS`.

## Operator sequence

`recall-enterprise` picks its credential by subcommand, so the operator no longer has to.
`migrate` and `create-generation` perform DDL and read `RECALL_MIGRATION_DSN`; `readiness`,
`status`, `parity` and `replay` only read and take `RECALL_SERVING_DSN`; `mark-ready`, `set-route`,
`cutover` and `retire` are DML against the control-plane tables and take the migration credential.
All of them fall back to `RECALL_DSN`, so a single-variable deployment keeps working.

The split matters most for `readiness`, which reports whether row level security constrains "the
runtime database role". That check reads `current_user` of the connection it was handed, so on the
migration role a green verdict would certify a credential that never serves a request. The command
prints the role it evaluated, so the verdict names its own subject.

```console
RECALL_MIGRATION_DSN="$RECALL_MIGRATION_DSN" recall-enterprise migrate
```

> ⚠️ `RECALL_DSN` is also the deprecated fallback the serving process and the MCP server
> read when `RECALL_SERVING_DSN` is unset (see [MIGRATIONS.md](MIGRATIONS.md#configuration)).
> Exporting it globally as the migration role therefore hands a schema-owner credential to
> every serving process, which is exactly what the role split in
> [SECURITY.md](../SECURITY.md) forbids. Set it per command, never in the serving
> environment.

The database operator must install pgvector once in a new database before the restricted
migration role creates a generation:

```sql
CREATE EXTENSION vector;
```

Do not grant superuser or `BYPASSRLS` to the migration or runtime roles.

Create an empty generation table and register its profile identity:

```console
recall-enterprise create-generation g2026_08 chunks_g2026_08 bge-small-asymmetric-v1 384
```

Build and validate the shadow corpus, then mark it ready with measured counts:

```console
recall-enterprise mark-ready g2026_08 --chunks 1000000 --sources 120000
recall-enterprise set-route acme g2026_07 --shadow-generation g2026_08
```

While a shadow route exists, indexing prepares both vector sets before either generation changes. It records a durable ordered event, applies the active and shadow writes, then clears the event payload on completion. A crash leaves an idempotent replay record. The `recall_forget` MCP tool deletes from both generation tables in one database transaction, then scrubs the erased sources out of any pending replay record, so a later replay cannot restore them. The scrub is keyed on the sources the caller named, not on what still had rows, because the case that most needs it is the one where a crash left the text in the outbox and nowhere else. One window remains and is reported rather than hidden: the deletes commit before the scrub, so a crash between them leaves the outbox entry, and the result carries `outbox_events_scrubbed = -1` when the scrub failed after the deletion succeeded. ⚠️ The `recall forget` CLI is single-generation and does NOT scrub the outbox; on an enterprise deployment use the MCP tool.

Cutover refuses to proceed while any migration event is pending or while the shadow is not ready:

```console
recall-enterprise cutover acme
```

A crash that left an event pending blocks cutover until the outbox is drained. Drain it, then compare the two generations before promoting:

```console
recall-enterprise status --tenant acme
recall-enterprise replay acme
recall-enterprise parity acme
```

`replay` opens only the generations the pending events name, resolving each physical table from `recall_index_generations`, and exits non-zero if anything is still pending afterwards. `parity` exits non-zero when the generations disagree on sources, raw content hashes or chunk counts, and also when either generation has an invalid required index or does not have row level security forced. `status` reports generations, the tenant's route and the outbox depth; it never prints a pending event's payload, which holds corpus text and vectors. It also lists any registry row whose `physical_table` the identifier allowlist rejects, rather than failing on it: such a row cannot serve, and the command an operator uses to find it must not be the command that dies on it. Run `recall-enterprise status` before upgrading.

`readiness` runs the startup checks for one tenant without starting a server, and exits non-zero when any of them fails. Run it with `RECALL_SERVING_DSN` set: its row level security verdict is about the role it connects as, and it prints that role so the result names its own subject.

```console
recall-enterprise readiness acme
```

The route update is transactional and sends a content free PostgreSQL notification. Service processes invalidate their cached route immediately. The fallback is a five second cache TTL on the route, not a poll: a process whose notification never arrives picks the new route up within that window on its next request. Existing requests keep their acquired store object. New requests use the new generation.

## Runtime configuration

Set `RECALL_ENTERPRISE_CONTROL_PLANE=1` only on authenticated HTTP deployments. Enterprise readiness then fails startup when a route is missing, the control plane is unreachable, the profile or dimension differs, the active generation is not `ready` or `active`, either schema ledger is not current, required indexes are invalid, row level security is ineffective, model identity is unverified, a loaded calibration names a different embedding profile, or stored rows lack profile metadata. A database carrying migrations this package does not ship is reported as degraded rather than fatal (readiness returns `degraded=true` with a warning the server logs), so migrating forward and then rolling the application back does not refuse to boot.

Choose one service cost profile per process:

* `RECALL_RETRIEVAL_PROFILE=fast` uses twenty candidates per retrieval leg and no reranker.

* `RECALL_RETRIEVAL_PROFILE=quality` uses the same candidate pool and the local pinned reranker.

Run separate deployments when both profiles are required. Clients cannot select the expensive path per request. `RECALL_SEARCH_CONCURRENCY` and `RECALL_SEARCH_QUEUE` bound CPU admission before query embedding begins.

## Evidence integration

Use `build_evidence_bundle`, `render_evidence_prompt`, and `validate_answer`. They are exported from the package root (`from recall import build_evidence_bundle`) as well as from `recall.evidence`. `generate_from_evidence` is the optional orchestration helper. It never invokes its generator when retrieval abstains. The fixed system prompt contains no corpus controlled value. Evidence is JSON escaped inside the user data message, and successful answers require citations that resolve to supplied chunk IDs.

Validation is structural. It does not claim that a cited passage entails an answer.

### What enters a bundle, and what never does

Only `ok` verdicts. A DEGRADED result — the trust gate could not run, every verdict is `unverified`, and `abstained` is forced False — produces an EMPTY bundle with `reason_code="no_trusted_evidence"`, not an unjudged one. Retrieval order is preserved: no newest wins, no re-sort by score. There is no semantic deduplication, so two chunks with identical text remain two citable identifiers. There is no neighbour retrieval: the module holds no store and `build_evidence_bundle` takes no argument through which one could be supplied, so a passage that was not retrieved cannot appear.

An abstained retrieval produces an empty bundle and bypasses the generator entirely — `generate_from_evidence` returns `insufficient_evidence=true` with `generator_invoked=False` without constructing or calling anything.

### The prompt boundary

`render_evidence_prompt` returns the module constant `SYSTEM_PROMPT` itself. There is no format string and no argument on that path, so there is no site at which a corpus controlled value could be interpolated. Every corpus byte lives inside `<evidence_data>…</evidence_data>` in the second message, JSON escaped — including both angle brackets, which `json.dumps` does not escape and which the delimiter is made of. A frozen adversarial suite (`benchmarks/evidence_injection.py`) runs thirteen payloads through three carriers (file name, chunk metadata, memory text) and records the escape rate in `results/evidence_injection_baseline.json`, with a positive control against the previous renderer so that a zero cannot be produced by an inert detector.

### Citations

At least one per answer, and every one must resolve to a chunk ID in the bundle. Duplicates are collapsed deterministically by `normalize_citations` — first occurrence order, idempotent, and only ever subtractive, so normalisation cannot mint an identifier that would then satisfy the resolution check. `GenerationResult.citations_normalized` reports whether that edit happened. `validate_answer` on its own remains strict and reports a duplicate as an error.

A token budget requires an injected tokenizer: `EvidencePolicy(max_tokens=…)` with no `tokenizer` raises rather than estimating.

### Reaching it from the four integrations

| Surface | Entry point |
|---|---|
| Library | `from recall import build_evidence_bundle, render_evidence_prompt, validate_answer` |
| CLI | `recall search "<query>" --evidence` prints the bundle and the rendered prompt as JSON |
| MCP | the `recall_evidence` tool returns the bundle plus `system_prompt` and `user_message` |
| LangChain | `RecallRetriever.evidence(query)` / `.evidence_prompt(query)` |
| LlamaIndex | `RecallRetriever.evidence(query)` / `.evidence_prompt(query)` |

All five are additive; every pre-existing field, metadata key and tool is unchanged.

`recall_evidence` runs no generator. This deployment chooses none and ships none, so the tool stops one step short and hands back the two messages for the client to run its own model against. That is what generator neutrality means here, and it is also why the end-to-end path with a real generator remains unexercised: no approved local generator has been confirmed for this program. The neutral flow is tested against a stub.

The retriever adapters deliberately do NOT honour their `include_untrusted` escape hatch in `evidence()`. That flag exists so a caller can inspect what the trust layer refused; what a generator may cite is a rule, not a constructor setting.

## Promotion

`recall.promotion.evaluate_retrieval_promotion` implements the paired macro bootstrap interval, per corpus regression limit, paired sign tests with Holm correction, safety parity checks, security gate, and latency budget. Experiments remain opt in until the decision reports `promoted=true`. Negative artifacts should be retained with fixed question identifiers and model digests.

## Rejected profile: Qwen3-Embedding-0.6B truncated to 384 dimensions

`qwen3-embedding-0.6b-384-v1` is registered and **rejected**. It is kept, with its measurement, so that the decision is reproducible and so that no later session re-measures it by accident. It is not a candidate and it is not gated on anything that could still open.

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-Embedding-0.6B` |
| Revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Artifact SHA256 | `0e9f06588b7e661b8d8e6d393b5936750e428ec422f9971c7f02838dbe70fc9f` |
| Dimension | 384 (truncated, renormalized after truncation) |
| Licence | Apache 2.0 |
| Verdict | rejected on CPU serving latency, 2026-08-03 |

Measured offline on the provisioned artifact at a four thread budget:

| Measurement | Value |
|---|---|
| Query p50 | 4638.83 ms |
| Query p95 | 5816.34 ms |
| Passage batch of 20, p50 | 41016.64 ms |
| Model load | 24558.4 ms |
| Peak RSS | 1739.47 MB |

The fast retrieval profile budgets 250 ms and the quality profile 1500 ms. A query p95 of 5.8 seconds is more than three times the quality budget for the embedding step alone, before any store or reranker cost, and a 41 second batch of twenty passages makes bulk indexing impractical on the same hardware.

Two limits on what this says. It is a latency verdict, not a quality one: retrieval quality was never measured against `bge-small-asymmetric-v1`, so nothing here claims the model retrieves worse. And it was measured on CPU, at four threads, on the host described under the latency blocker in `ENTERPRISE_PROGRAM_STATUS.md`. GPU requirements are out of scope for this program, so a GPU number would not change the decision.

The registry pins the artifact digest for this profile. A different artifact tree is a different experiment and is refused rather than inheriting this verdict.

## Rollback and retirement

Cutover swaps the previous active generation into the shadow route. Restore it with `set-route` if rollback is required. Keep the old table for seven days and two successful backup cycles. Removal is an explicit operator migration after the rollback period. Never allow a request field to name a retired table.

After the rollback window, retire the old generation:

```console
recall-enterprise retire g2026_07 --tenant acme
```

Retirement is confirmed one tenant at a time, and the reason is the isolation model rather than convenience: `recall_tenant_routes` carries forced row level security keyed on the tenant, and neither the migration role nor the runtime role may enumerate every tenant's routes to prove a generation is globally unrouted. The command therefore refuses while the named tenant's route references the generation, and the serving path refuses a retired or failed generation independently, per request. That second refusal is the one that protects a request; weakening the isolation model to make a single global check possible would have cost more than it bought.
