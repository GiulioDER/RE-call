# Enterprise retrieval and evidence deployment

This implementation keeps the existing chunk table and retrieval flow. It adds immutable embedding identities, fixed process retrieval profiles, deterministic contextual passage text, a generator neutral evidence boundary, and PostgreSQL generation routing.

## Security boundary

All model artifacts must exist locally before startup. An explicit embedding profile verifies the configured artifact tree against its SHA256 digest and requests local only loading. The quality retrieval profile also requires a local reranker path and digest. Production should block outbound network access at the workload boundary.

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

Use `build_evidence_bundle`, `render_evidence_prompt`, and `validate_answer` from `recall.evidence`. `generate_from_evidence` is the optional orchestration helper. It never invokes its generator when retrieval abstains. The fixed system prompt contains no corpus controlled value. Evidence is JSON escaped inside the user data message, and successful answers require unique citations that resolve to supplied chunk IDs.

Validation is structural. It does not claim that a cited passage entails an answer.

## Promotion

`recall.promotion.evaluate_retrieval_promotion` implements the paired macro bootstrap interval, per corpus regression limit, paired sign tests with Holm correction, safety parity checks, security gate, and latency budget. Experiments remain opt in until the decision reports `promoted=true`. Negative artifacts should be retained with fixed question identifiers and model digests.

## Rollback and retirement

Cutover swaps the previous active generation into the shadow route. Restore it with `set-route` if rollback is required. Keep the old table for seven days and two successful backup cycles. Removal is an explicit operator migration after the rollback period. Never allow a request field to name a retired table.

After the rollback window, retire the old generation:

```console
recall-enterprise retire g2026_07 --tenant acme
```

Retirement is confirmed one tenant at a time, and the reason is the isolation model rather than convenience: `recall_tenant_routes` carries forced row level security keyed on the tenant, and neither the migration role nor the runtime role may enumerate every tenant's routes to prove a generation is globally unrouted. The command therefore refuses while the named tenant's route references the generation, and the serving path refuses a retired or failed generation independently, per request. That second refusal is the one that protects a request; weakening the isolation model to make a single global check possible would have cost more than it bought.
