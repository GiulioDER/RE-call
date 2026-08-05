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

`recall-enterprise` reads its connection from `RECALL_DSN`. Of its subcommands, `migrate` and
`create-generation` perform DDL; `mark-ready`, `set-route` and `cutover` are ordinary DML against
the control-plane tables. Export the migration credential **only for the duration of the commands
that need it**:

```console
RECALL_DSN="$RECALL_MIGRATION_DSN" recall-enterprise migrate
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

While a shadow route exists, indexing prepares both vector sets before either generation changes. It records a durable ordered event, applies the active and shadow writes, then clears the event payload on completion. A crash leaves an idempotent replay record. Forget operations delete from both generation tables in one database transaction.

Cutover refuses to proceed while any migration event is pending or while the shadow is not ready:

```console
recall-enterprise cutover acme
```

The route update is transactional and sends a content free PostgreSQL notification. Service processes invalidate their cached route immediately, with a five second polling fallback. Existing requests keep their acquired store object. New requests use the new generation.

## Runtime configuration

Set `RECALL_ENTERPRISE_CONTROL_PLANE=1` only on authenticated HTTP deployments. Enterprise readiness then fails startup when a route is missing, the profile or dimension differs, required indexes are invalid, row level security is ineffective, model identity is unverified, or stored rows lack profile metadata.

Choose one service cost profile per process:

* `RECALL_RETRIEVAL_PROFILE=fast` uses twenty candidates per retrieval leg and no reranker.

* `RECALL_RETRIEVAL_PROFILE=quality` uses the same candidate pool and the local pinned reranker.

Run separate deployments when both profiles are required. Clients cannot select the expensive path per request. `RECALL_SEARCH_CONCURRENCY` and `RECALL_SEARCH_QUEUE` bound CPU admission before query embedding begins.

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
