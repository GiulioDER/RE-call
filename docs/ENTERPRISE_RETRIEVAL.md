# Enterprise retrieval and evidence deployment

This implementation keeps the existing chunk table and retrieval flow. It adds immutable embedding identities, fixed process retrieval profiles, deterministic contextual passage text, a generator neutral evidence boundary, and PostgreSQL generation routing.

## Embedding profile registry

`recall/embedding_registry.py` is the single place a profile is defined. It owns the profile identifier, the model, the artifact digest, the dimension, the query and passage encoder modes, the normalization, the instruction version, the chunker version and the context version. Nothing else may hold a second copy of that vocabulary.

Three properties follow from that and are enforced by tests:

* `context_version` is derived from `context_mode`, never declared next to it. `Indexer` refuses an embedder whose profile does not spell its context exactly `raw-v1` or `context-<mode>-<policy version>`, so the derivation in the registry is that contract written once.

* `query_mode` and `passage_mode` name the encoder that is actually called. A profile declaring `query_embed` gets `TextEmbedding.query_embed`, and a backend without that encoder refuses to start rather than falling back to the symmetric one.

  ⚠️ **Naming a distinct encoder is not the same as getting distinct vectors, and on this deployment it does not.** Under fastembed 0.8.0, `BAAI/bge-small-en-v1.5` returns byte-identical vectors from `embed`, `query_embed` and `passage_embed`: the model resolves to `OnnxTextEmbedding`, which overrides neither method, so both come from `TextEmbeddingBase` and `yield from self.embed(...)` with no instruction. `bge-small-symmetric-v1` and `bge-small-asymmetric-v1` share every other identity INPUT to a stored vector (model, dimension, context mode, normalization, instruction version, chunker version, backend) and one provisioned artifact tree, so they cannot produce different vectors here, and **a paired comparison of the two is a null by construction rather than an experiment**. Their FINGERPRINTS still differ, because `EmbeddingProfile.fingerprint` covers `profile_id` and both mode fields, so the two index into separate physical tables and a comparison of them would carry tie-break noise rather than exact zeros. Reproduce with `benchmarks/check_profile_encoder_distinctness.py` on a host with the tree provisioned (elsewhere pass `--cache-dir`); the delegation half needs no weights at all and is checkable from library source anywhere fastembed is installed. The dispatch still has to be correct, because `qwen3-embedding-0.6b-384-v1` does use a distinct instruction-prefixed query encoder and a future backend could add one without changing a profile identifier. Giving BGE a query instruction would be a new experiment and needs registering.

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

## Operator runbook

The whole sequence, in order, with the credential each step takes, what it refuses, and how to
verify that it did what it says. Every command exits non-zero on failure, so each step is a gate:
do not proceed past a red one.

⚠️ **Steps 2 and 3 are two different things and skipping the first is the failure this deployment
actually hit.** `create-generation` registers a generation and creates its table; it does not write
that table's rows into the per-table migration ledger in every path, and `GenerationStore` refuses
to migrate at all by design (`ImmutableGenerationError: GenerationStore never migrates; run
"recall schema apply" with the migration role`). A generation whose ledger rows were never written
looks completely healthy (table present, all indexes valid, RLS forced) and `readiness` reports
`SchemaTooOld` for it. Run step 2 for every chunk table, including each generation table.

| # | Step | Credential | Exits non-zero when |
|---|---|---|---|
| 0 | Preconditions | operator | see below; these are checks, not commands |
| 1 | `recall-enterprise migrate` | `RECALL_MIGRATION_DSN` | control-plane DDL fails |
| 2 | `recall schema apply` per chunk table | `RECALL_MIGRATION_DSN` | a migration fails, drifted, or is unknown |
| 3 | `recall-enterprise create-generation` | `RECALL_MIGRATION_DSN` | pgvector absent, or table DDL fails |
| 4 | index the shadow corpus | serving | (application step) |
| 5 | `recall-enterprise mark-ready` | `RECALL_MIGRATION_DSN` | counts do not match what was built |
| 6 | `recall-enterprise set-route --shadow-generation` | `RECALL_MIGRATION_DSN` | the generation is not servable |
| 7 | `recall-enterprise replay` | `RECALL_SERVING_DSN` | anything is still pending afterwards |
| 8 | `recall-enterprise parity` | `RECALL_SERVING_DSN` | sources, hashes or counts disagree; an index is invalid; RLS is not forced |
| 9 | `recall-enterprise readiness` | `RECALL_SERVING_DSN` | any startup check fails |
| 10 | `recall-enterprise cutover` | `RECALL_MIGRATION_DSN` | an event is pending, or the shadow is not ready |
| 11 | `recall-enterprise retire` | `RECALL_MIGRATION_DSN` | the named tenant still routes at that generation |
| R | rollback: `recall-enterprise set-route` | `RECALL_MIGRATION_DSN` | the previous generation is not servable |

### 0. Preconditions

Check these before step 1. None of them is a command this package ships, and each has been an
actual failure.

* **Roles.** A migration role and a runtime role, neither `SUPERUSER` nor `BYPASSRLS`. Verify with
  `select rolname, rolsuper, rolbypassrls from pg_roles`. A superuser bypasses row level security,
  so every isolation check would pass vacuously.
* **The runtime role needs `SELECT` on `recall_schema_versions`.** `status` and `readiness` are
  documented to take the serving credential and both read the control-plane ledger. Without that
  grant they fail with `InsufficientPrivilege: permission denied for table recall_schema_versions`,
  which reads like a database outage rather than a missing grant.
* **pgvector**, installed once by a database operator: `CREATE EXTENSION vector;`. The migration
  role must not be elevated to do this. `sparsevec`, which migration 0012 needs, requires pgvector
  0.7 or later; check with `select count(*) from pg_type where typname = 'sparsevec'`.
* **Model artifacts present and verified locally**, with their digests recorded. Recompute rather
  than trust the record: a tree that agrees with its own manifest proves nothing, so hash it with an
  independent implementation of `artifact_tree_sha256` and compare against the value pinned in the
  package.
* **Licence recorded for every artifact.** A digest says which bytes; it does not say whether you
  may ship them.
* **Outbound network blocked at the workload boundary.** Runtime model downloads are prohibited and
  startup is proven to complete with every socket entry point blocked, but the package cannot
  enforce the boundary. `ufw` defaulting to `allow (outgoing)` satisfies nothing here.
* **Disk headroom at least 2.2x the active index size** before any build, since the shadow is built
  alongside the active generation rather than in place. Measure with
  `pg_indexes_size('<active table>')` against the free bytes on the data directory's mount, not
  against total capacity.

### 1. Migrate the control plane

```console
RECALL_MIGRATION_DSN="$RECALL_MIGRATION_DSN" recall-enterprise migrate
```

Verify: `recall-enterprise status` prints `control plane ledger is current`.

### 2. Apply the chunk-table migrations, per table

Run this for the legacy chunk table and for **every** generation table. It is the step that writes
the per-table ledger rows, and `readiness` reports `SchemaTooOld` for any table missing them.

Verify: `schema_status(dsn, table=…, dim=…).compatible` is true for each table, and every index on
it is still `indisvalid`.

> ⚠️ **A migration whose bytes changed after you applied it is a hard stop, by design.** The ledger
> stores the checksum of what was applied, and any schema call then raises
> `MigrationChecksumMismatch` rather than migrating forward. There is no flag for this and there
> should not be. The remedy is to clear that ledger row and re-apply, and it is only defensible when
> you can show the two versions are equivalent **on this database** (read the diff, and check the
> tables it touches are empty if the difference is data-dependent). Record the row before deleting
> it so the step is reversible.

### 3. Create the shadow generation

```console
recall-enterprise create-generation g2026_08 chunks_g2026_08 bge-small-asymmetric-v1 384
```

The declared dimension is checked against the artifact at startup, so a generation registered at the
wrong width refuses to serve rather than writing vectors nothing can interpret. Then run step 2
against `chunks_g2026_08`.

### 4 and 5. Build, then mark ready with measured counts

```console
recall-enterprise mark-ready g2026_08 --chunks 1000000 --sources 120000
```

The counts are measured, not estimated: they are what `parity` and `cutover` later compare against.

### 6. Attach the shadow route

```console
recall-enterprise set-route acme g2026_07 --shadow-generation g2026_08
```

### 7 to 9. Drain, compare, then check readiness

```console
recall-enterprise status --tenant acme
recall-enterprise replay acme
recall-enterprise parity acme
recall-enterprise readiness acme
```

⚠️ **`parity` on two empty generations exits 0 and prints `parity: OK`.** That is a vacuous pass,
not a comparison. Read the chunk counts it prints and treat a zero on both sides as "nothing was
compared". The same applies to a shadow attached midway through a corpus: a dual-write re-index
reads its skip set from the active store, so a file already indexed there is skipped and never
reaches the shadow, and `cutover`'s emptiness check only catches a **totally** empty shadow.

Run `readiness` with `RECALL_SERVING_DSN` set. Its row level security verdict is about the role it
connects as, and it prints that role, so the verdict names its own subject. On the migration role a
green verdict would certify a credential that never serves a request.

### 10 and 11. Cutover, then retire after the rollback window

```console
recall-enterprise cutover acme
recall-enterprise retire g2026_07 --tenant acme
```

Keep the old table for seven days and two successful backup cycles before retiring.

### R. Rollback

Cutover swaps the previous active generation into the shadow route, so rolling back is
`set-route` naming the previous generation as active. The serving path independently refuses a
retired or failed generation, per request, so a retired table cannot be reached even if a route
still names it.

### Credentials by subcommand

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

* `RECALL_RETRIEVAL_PROFILE=fast` uses twenty candidates per retrieval leg and no reranker. It returns five hits and budgets 250 ms.

* `RECALL_RETRIEVAL_PROFILE=quality` uses the same candidate pool and the local pinned reranker. It returns five hits and budgets 1500 ms. The reranker scores the COMPLETE fused candidate pool before truncation, so a relevant passage sitting just below the fused cutoff can still be rescued, which is the only reason the stage exists.

Run separate deployments when both profiles are required. Clients cannot select the expensive path per request: the profile is read from the process environment and a request's `k` is clamped down to the profile's returned count, never raised. Leaving `RECALL_RETRIEVAL_PROFILE` unset preserves the legacy `RECALL_RERANK` switch exactly; setting the two to values that contradict each other refuses **startup**, not the first search.

### The latency budget at request time

`latency_budget_ms` means two enforced, observable things.

It **bounds the admission wait**. A request that cannot acquire a running slot within the budget is shed with `RetrievalOverloaded` before the query is embedded. That ordering is the point: admission is taken ahead of the embedder, so a refused request costs nothing. Without the bound, `queue_capacity` caps how many threads may be parked and says nothing about how long any of them waits, and a process can hold a client for minutes behind a slow reranked query while every counter reads healthy. `RetrievalOverloaded` carries a `reason` (`queue_full` or `budget_exhausted`) and a `retry_after_seconds`, matching how `RateLimited` reports a retryable refusal.

It **labels an overrun**. A request that completes over budget still returns its answer, and reports `total_ms`, `latency_budget_ms` and `budget_exceeded` on the response. Aborting in flight was rejected: there is no cancellation point inside a blocking cross-encoder `predict`, so the process would pay the whole cost and then discard the answer, converting a latency regression into an availability incident.

The budget is charged **once**. `budget_exceeded` is computed on the work a request actually did (`total_ms` minus `admission_wait`), not on its end-to-end latency. Since the budget is already spent as the admission timeout, a request may legitimately wait almost all of it before starting; comparing the same allowance against the total as well would label a fast retrieval slow because another request was ahead of it, and would saturate the counter under any queueing. `total_ms` still reports client-visible latency, which is a different and also necessary number.

The **legacy profile enforces no budget**, and reports `latency_budget_ms` as `null` rather than as the 24-day sentinel the code uses internally. `budget_exceeded` is then always false.

Each profile carries its **own** concurrency budget rather than one shared default: fast admits 8 concurrent with 32 queued, quality 2 with 8, legacy 4 with 16. Quality's per-request budget is six times fast's, so an equal queue depth would make its clients wait roughly six times as long; the numbers hold `queue_capacity * latency_budget_ms` within one order of magnitude (fast 8000 slot-ms, quality 12000). These values are a policy choice, not a measurement, and the latency blocker in `ENTERPRISE_PROGRAM_STATUS.md` is why. `RECALL_SEARCH_CONCURRENCY` and `RECALL_SEARCH_QUEUE` override them for the selected profile.

The admission gate is entered inside a worker thread, so its capacity is denominated in threads whether or not it says so. The server therefore **sizes the worker pool from the profile** at startup (`worker_thread_budget`: admission capacity plus eight reserved threads), and only ever raises it. Without that, fast's 8 + 32 would exactly equal anyio's 40-token default: the request that should be shed would never reach the gate at all, it would wait in anyio's limiter, which has no timeout and no counter, and `recall_retrieval_rejected_total` would read zero while clients waited unboundedly. The reserved headroom keeps queued searches from starving `recall_index`, `recall_forget`, `recall_stats` and token validation.

`RECALL_RERANK_THREADS` bounds inference threads **on the quality profile only**; it is not read on the legacy `RECALL_RERANK` path. One reranker is built per worker process, under a construction lock: a cache lookup is not a lock, and a cold start under load would otherwise have every concurrent first request load its own copy of the model. A construction that fails is cached too, so a broken artifact fails immediately instead of re-hashing the model tree on every request.

### The quality profile's reranker is pinned by digest

`RECALL_RERANK_PATH` is deployment specific. The artifact digest is not: `recall/rerank.py` pins artifact SHA256 `db6ad87969c7dc78320152e68a16118aeb4b2a6f7d8cc979c57f61ddb5e2ab2a`, and `RECALL_RERANK_SHA256` must equal it. Verifying the tree against a digest the operator supplied would prove only that the tree hashes to its own hash; the pin is the value chosen elsewhere that makes the comparison mean something.

Two limits on what that pin says, both deliberate.

The model name `cross-encoder/ms-marco-MiniLM-L-6-v2` and revision `c5ee24cb16019beea0893ab7796b1df96625c6b8` are recorded beside it as **provenance, not as a runtime check**. Nothing reads them at load time: the quality profile loads from a local tree with `local_files_only`, where the Hub revision is unused.

And the digest is a hash of a whole provisioned **tree**, path names included, so it identifies one provisioned directory rather than the model in general. A differently laid out copy of the same weights (a Hugging Face `blobs`/`snapshots` cache, a `snapshot_download` that left a lock file behind) hashes differently and is refused. This deployment's tree is the one recorded in `/opt/recall-enterprise/manifest.json`. There is no shipped command that reproduces it elsewhere, which is a real gap for any operator outside that host and is recorded as such in `ENTERPRISE_PROGRAM_STATUS.md`.

### What every result reports

Every search response carries, additively: the embedding profile identity, the retrieval profile, the index generation, the candidate pool size, whether reranking ran, `total_ms`, `latency_budget_ms`, `budget_exceeded`, and per-stage wall time for `admission_wait`, `query_embedding`, `dense_retrieval`, `sparse_retrieval`, `fusion`, `reranking`, `trust_evaluation` and `evidence_assembly`. The same stage timings are observed into `METRICS` under `recall_retrieval_stage_ms`, labelled by profile and stage, so per-stage percentiles exist across a population of queries rather than only per response. `recall_retrieval_total_ms` is observed on failures as well as successes, because a timer that only records on success hides the slow path worth finding. It deliberately excludes requests that were **shed**: those did no work by construction, so booking them would make healthy load shedding indistinguishable from an outage and would contaminate the served-latency population with rejections in exactly the overload regime where a p95 matters most. A shed request appears in `recall_retrieval_rejected_total{profile,reason}` and nowhere else.

Every hit's `score` is its dense cosine, including hits that arrived through the sparse leg and hits the reranker moved. Reranking reorders and never rewrites the score, because calibration thresholds are stated in cosine units and a cross-encoder logit is not one.

Metric labels and log records are library-authored throughout. No query text and no corpus text reaches a log record, a metric label or an exception message; a test drives a distinctive sentinel through the search path and asserts its absence from every captured record, with a positive control proving the detector can fire.

## Evidence integration

Use `build_evidence_bundle`, `render_evidence_prompt`, and `validate_answer`. They are exported from the package root (`from recall import build_evidence_bundle`) as well as from `recall.evidence`. `generate_from_evidence` is the optional orchestration helper. It never invokes its generator when retrieval abstains. The fixed system prompt contains no corpus controlled value. Evidence is JSON escaped inside the user data message, and successful answers require citations that resolve to supplied chunk IDs.

Validation is structural. It does not claim that a cited passage entails an answer.

### What enters a bundle, and what never does

Only `ok` verdicts. A DEGRADED result — the trust gate could not run, every verdict is `unverified`, and `abstained` is forced False — produces an EMPTY bundle with `reason_code="no_trusted_evidence"`, not an unjudged one. Retrieval order is preserved: no newest wins, no re-sort by score. There is no semantic deduplication, so two chunks with identical text remain two citable identifiers. There is no neighbour retrieval: the module holds no store and `build_evidence_bundle` takes no argument through which one could be supplied, so a passage that was not retrieved cannot appear.

An abstained retrieval produces an empty bundle and bypasses the generator entirely — `generate_from_evidence` returns `insufficient_evidence=true` with `generator_invoked=False` without constructing or calling anything.

### The prompt boundary

`render_evidence_prompt` returns the module constant `SYSTEM_PROMPT` itself. There is no format string and no argument on that path, so there is no site at which a corpus controlled value could be interpolated. Every corpus byte lives inside `<evidence_data>…</evidence_data>` in the second message, JSON escaped — including both angle brackets, which `json.dumps` does not escape and which the delimiter is made of. A frozen adversarial suite (`benchmarks/evidence_injection.py`) runs thirteen payloads through four carriers (file name, chunk metadata, memory text, chunk id) and records the escape rate in `results/evidence_injection_baseline.json`, with a positive control against the previous renderer so that a zero cannot be produced by an inert detector, and a negative control against a renderer that ships no evidence at all so that a perfect score cannot be bought by deleting the corpus text. The chunk-metadata arm carries no payload into the prompt at all — an evidence item has no corpus metadata dict — so its zero is structural rather than earned, and it is excluded from the rate's denominator. The suite and its baseline ship in the source distribution and the repository, not in the wheel.

### Citations

At least one per answer, and every one must resolve to a chunk ID in the bundle. Duplicates are collapsed deterministically by `normalize_citations` — first occurrence order, idempotent, and only ever subtractive, so normalisation cannot mint an identifier that would then satisfy the resolution check. `GenerationResult.citations_normalized` reports whether that edit happened. `validate_answer` on its own remains strict and reports a duplicate as an error.

A token budget requires an injected tokenizer: `EvidencePolicy(max_tokens=…)` with no `tokenizer` raises rather than estimating.

### Reaching it from the library and the four integrations

| Surface | Entry point |
|---|---|
| Library | `from recall import build_evidence_bundle, render_evidence_prompt, validate_answer` |
| CLI | `recall search "<query>" --evidence` prints the bundle and the rendered prompt as JSON |
| MCP | the `recall_evidence` tool returns the bundle plus `system_prompt` and `user_message` |
| LangChain | `RecallRetriever.evidence(query)` / `.evidence_prompt(query)` |
| LlamaIndex | `RecallRetriever.evidence(query)` / `.evidence_prompt(query)` |

All five surfaces are additive: every pre-existing field, metadata key and tool is unchanged, and each of the four integrations carries a test asserting a frozen list of its pre-existing keys. (Four integrations plus the library import itself, which is why the table has five rows.)

`recall_evidence` runs no generator. This deployment chooses none and ships none, so the tool stops one step short and hands back the two messages for the client to run its own model against. That is what generator neutrality means here, and it is also why the end-to-end path with a real generator remains unexercised: no approved local generator has been confirmed for this program. The neutral flow is tested against a stub.

The retriever adapters deliberately do NOT honour their `include_untrusted` escape hatch in `evidence()`. That flag exists so a caller can inspect what the trust layer refused; what a generator may cite is a rule, not a constructor setting.

## Promotion

`recall.promotion.evaluate_retrieval_promotion` implements the paired macro bootstrap interval, per corpus regression limit, paired sign tests with Holm correction, safety parity checks, security gate, and latency budget. Experiments remain opt in until the decision reports `promoted=true`. Negative artifacts should be retained with fixed question identifiers and model digests.

### Producing a decision

`recall/eval/promotion/` builds the gate's input. Run it in three steps, in this order:

```bash
python -m recall.eval.promotion freeze --corpus labelled \
    --out results/promotion/labelled.manifest.jsonl

python -m recall.eval.promotion run --manifest results/promotion/labelled.manifest.jsonl \
    --expected-digest <the digest freeze printed> --corpus labelled --arm baseline \
    --dsn "$RECALL_DSN"

python -m recall.eval.promotion decide --manifest results/promotion/labelled.manifest.jsonl \
    --baseline <baseline ledger> --candidate <candidate ledger> \
    --out results/promotion/decision.json
```

`freeze` must run **before** either arm. It fixes question ids and input hashes while no candidate result exists, which is what makes "the same questions" checkable rather than assumed. It refuses to overwrite an existing manifest, and the reader refuses a body that no longer matches its digest. Each arm's rows carry the frozen `input_hash`, and `decide` refuses a pair of arms that do not cover the manifest exactly.

Supported corpora: `labelled` (in-repo), `peps`, `locomo`, `ladder`, `longmemeval`, `mtrag`. Every adapter declares which id space its labels live in, with no default, because a wrong one scores an entire corpus as a total miss and reports a successful run.

**Latency is PENDING by default, and PENDING blocks promotion.** `decide` emits `latency_p95_ms=None` unless `--certified-latency-p95-ms` is passed, and that flag is for a number measured on an idle reference host. This program has no such host, so every decision it can produce today is blocked on latency; the observed p95 is recorded in the artifact under `observed_diagnostic_only` and is not a gate input.

**Calibration is scoped to the complete embedding identity.** Use `recall.calibration.save_for_profile` and `load_for_profile`, not `save` and `load_for`, for anything a gate reads. The latter pair keys on the profile ID alone, and two runs sharing a profile ID can differ in artifact digest, dimension, encoder modes or chunker version, each of which moves the cosine regime a threshold was fitted in. `load_for_profile` fails closed on a calibration that cannot show which identity it belongs to.

**A run against a plain store is DEGRADED.** Strict trust mode refuses a store with no generation-bound certified calibration, which is correct. `--trust-policy development` runs anyway and records every hit's verdict as `unverified`, which carries no trust claim at all: the safety metrics of such a run describe a degraded system and are not a measurement of the trust layer. The decision artifact records the verdict distribution so a reader cannot mistake one for the other.

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
