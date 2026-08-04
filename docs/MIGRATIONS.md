# Database migrations and roles

RE-call v1 never executes DDL from normal library data operations, CLI data commands, or MCP
startup. Schema changes are ordered SQL files shipped in `recall/migrations/sql`; their SHA-256
values are committed in
`recall/migrations/checksums.json` and recorded in `recall_schema_migrations` when applied.

## Configuration

- `RECALL_SERVING_DSN`: unprivileged credential used by indexing, search, forget, and MCP.
- `RECALL_MIGRATION_DSN`: schema-owner credential used only by `recall schema apply`.
- `RECALL_DSN`: deprecated development fallback for the serving DSN.
- `RECALL_ENV`: `development` (default) | `test` | `production`.

Keep the two credentials distinct outside a disposable local database. The MCP process never reads
the migration DSN.

`RECALL_ENV` is what selects the production code paths, and it **fails open**. Set it explicitly on
every production process: left unset, it resolves to `development` and silently disables all of the
following. A misspelling behaves worse than either, because it is not handled consistently — `prod`
or a stray trailing space degrades the seven `== "production"` comparisons below to development
silently, while `GenerationManager` validates its value and raises `ValueError`, so the same typo
is loud in one place and mute in the rest.

| Set to `production` | Left at the default |
|---|---|
| `recall search` / `recall forget` use the v1 `GenerationStore` | they use the legacy v0.8 `chunks` table |
| MCP server serves generation-routed reads | serves the legacy table |
| `recall index` / `demo` / `code` and the MCP `recall_index` tool refuse local-filesystem indexing | accepted |
| generations require a pinned, verified embedder identity | an unverified embedder can build one |
| `generation promote` is blocked | promotion is permitted |

```bash
recall --serving-dsn "$RECALL_SERVING_DSN" --table chunks schema --dim 384 status
recall --serving-dsn "$RECALL_SERVING_DSN" --table chunks schema --dim 384 plan
recall --migration-dsn "$RECALL_MIGRATION_DSN" --table chunks schema --dim 384 apply
```

`status` and `plan` execute SELECT statements only. `apply` takes a PostgreSQL advisory lock,
rejects changed checksums, runs ordinary DDL transactionally, and records concurrent index phases
so an interrupted `CREATE INDEX CONCURRENTLY` can be validated and resumed.

`PgVectorStore.ensure_schema()` remains as a deprecated, explicit v0.8 compatibility wrapper for
disposable test/evaluation stores; it delegates to this same migrator. Production code should call
`check_schema()` and keep the migration credential out of the serving process.

The first migration adopts a v0.8 table in place: existing rows stay in the same table, are assigned
to tenant `default`, and retain their text, metadata, vectors, and timestamps. The legacy table is
not renamed or converted into a generation table. Migration 0008 records its tenants as
`legacy_unverified` evidence, but never copies or activates those rows. Migrations 0008 through
0011 are database-global and are recorded once under the `__global__` ledger target. Apply them
through the default `chunks` target before provisioning custom evaluation tables.

## Role split

The exact bootstrap syntax varies on managed PostgreSQL. The intended privileges are equivalent to:

```sql
CREATE ROLE recall_migrator LOGIN NOINHERIT NOBYPASSRLS;
CREATE ROLE recall_server LOGIN NOINHERIT NOBYPASSRLS;

-- Run as the database/schema owner. Install pgvector through the provider's supported admin path
-- if CREATE EXTENSION is reserved to a managed-service administrator.
GRANT CONNECT ON DATABASE recall TO recall_migrator, recall_server;
GRANT USAGE, CREATE ON SCHEMA public TO recall_migrator;
GRANT USAGE ON SCHEMA public TO recall_server;
REVOKE CREATE ON SCHEMA public FROM recall_server;
```

After `recall schema apply`, grant the serving role only the objects it uses:

```sql
GRANT SELECT ON recall_schema_migrations TO recall_server;
GRANT SELECT, INSERT, UPDATE, DELETE ON chunks TO recall_server;
GRANT SELECT, INSERT, UPDATE, DELETE ON
  recall_generations, recall_tenant_state, recall_chunks_v1, recall_ingest_jobs,
  recall_audit_events, recall_source_tombstones, recall_calibration_query_sets,
  recall_calibrations TO recall_server;
-- These tables use application-generated text IDs and no sequence.
```

The migration role must own the managed objects (or be a member of their owner role) and have
`CREATE` on the target schema. The serving role must not own the table, be a superuser, carry
`BYPASSRLS`, or receive schema `CREATE`.

## Startup and readiness

MCP startup checks the ledger before constructing the pgvector store. A missing ledger, pending or
failed phase, unknown future version, or checksum mismatch fails startup/readiness; startup never
tries to repair it. Apply migrations as a separate deployment job, then start or roll the serving
pods.

## Failure recovery

- `another RE-call schema migrator is already running`: wait for the active migration job. Do not
  run multiple jobs against the same database.
- `checksum drift`: restore the released migration bytes. Never edit an applied SQL file; add a new
  ordered migration. One pre-release exception has already been taken: `0008_generation_foundation.sql`
  was corrected in place before v1 shipped, because the bug it carried aborted the migration on any
  database that held v0.8 data, so no populated install could have applied it and a later migration
  could never have been reached to repair it. A database that applied the *earlier* 0008 (which means
  an empty local or CI database) fails here and must be recreated. There is no in-place repair,
  because the drift check runs before any work.
- failed/interrupted concurrent index: rerun `schema apply`. An invalid index is dropped
  concurrently and rebuilt; a completed-but-unrecorded index is validated and adopted.
- schema too new: deploy application code that knows the recorded versions. Do not delete ledger
  rows to force an older binary to start.
