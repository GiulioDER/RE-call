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

Use `build_evidence_bundle`, `render_evidence_prompt`, and `validate_answer` from `recall.evidence`. `generate_from_evidence` is the optional orchestration helper. It never invokes its generator when retrieval abstains. The fixed system prompt contains no corpus controlled value. Evidence is JSON escaped inside the user data message, and successful answers require unique citations that resolve to supplied chunk IDs.

Validation is structural. It does not claim that a cited passage entails an answer.

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
