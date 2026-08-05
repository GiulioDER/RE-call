# Database migrations and roles

RE-call v1 never executes DDL from normal library data operations, CLI data commands, or MCP
startup. Schema changes are ordered SQL files shipped in `recall/migrations/sql`; their SHA-256
values are committed in
`recall/migrations/checksums.json` and recorded in `recall_schema_migrations` when applied.

## Configuration

- `RECALL_SERVING_DSN`: unprivileged credential used by indexing, search, forget, and MCP.
- `RECALL_MIGRATION_DSN`: schema-owner credential used only by `recall schema apply`.
- `RECALL_DSN`: deprecated development fallback for BOTH, kept so single-variable
  deployments keep working.

`recall-enterprise` follows the same split. Its DDL subcommands (`migrate`,
`create-generation`) read `RECALL_MIGRATION_DSN`; its read-only ones (`readiness`,
`status`, `parity`, `replay`) read `RECALL_SERVING_DSN`; both fall back to `RECALL_DSN`.
That split is load-bearing for `readiness`, which reports whether row level security
constrains "the runtime database role": the check reads `current_user` of the connection
it was given, so run on the migration role it would certify a credential that never
serves a request. The command prints the role it evaluated, so the verdict names its own
subject.

Keep the two credentials distinct outside a disposable local database. The MCP process never reads
the migration DSN.

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

## The second ledger: `recall_schema_versions`

There are **two** migration ledgers, deliberately, and a deployment running the optional
enterprise control plane has both.

| | `recall_schema_migrations` | `recall_schema_versions` |
|---|---|---|
| SQL | `recall/migrations/sql/0001…0011` | `recall/sql/001_enterprise_control_plane.sql` |
| Applied by | `recall schema apply` (`RECALL_MIGRATION_DSN`) | `recall-enterprise migrate` (`RECALL_MIGRATION_DSN`) |
| Scope | per target table, plus a `__global__` bucket | database-global |
| Checksums | committed in `recall/migrations/checksums.json` | computed from the shipped file |
| Verified at startup | `check_schema` | `ControlPlane.ledger_state`, through readiness |

They stay separate for three reasons. Merging is a one-way door: both sets are checksum-immutable
by design, so renumbering the control-plane SQL into the `0001…0011` sequence changes what every
already-migrated database must agree with, with `MigrationChecksumMismatch` waiting at the end of
it. They have different lifecycles: the control-plane tables must exist before any generation
does, while generation chunk tables are created per generation by `recall-enterprise
create-generation`, which is exactly why the first ledger is scoped per target table. And the
enterprise deployment is opt in (`RECALL_ENTERPRISE_CONTROL_PLANE` defaults off), so merging would
impose control-plane tables on every deployment that does not want them.

Two ledgers need two things a single one would have given for free, and both are now present
rather than assumed:

- **Both are locked.** `recall-enterprise migrate` takes a PostgreSQL advisory lock
  (`recall-control-plane-migrations-v1`) for the same reason `recall schema apply` takes
  `recall-schema-migrations-v1`, and refuses rather than waiting when another migrator holds it.
  Two concurrent `migrate` jobs previously interleaved, and `CREATE TABLE IF NOT EXISTS` followed
  by a ledger `INSERT` is not atomic across sessions.
- **Both are verified.** Enterprise readiness checks `recall_schema_versions` as well as
  `recall_schema_migrations`. Verifying only one is how a process boots against a control plane
  that is behind, or whose applied SQL no longer matches the bytes the installed package ships.

`recall-enterprise status` prints the control-plane ledger's state, and
`recall-enterprise readiness <tenant>` exits non-zero when either ledger is not current.

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

An enterprise deployment (`RECALL_ENTERPRISE_CONTROL_PLANE=1`) needs four more, and this is a
**required upgrade step**, not a nicety: enterprise readiness verifies BOTH ledgers, so the serving
role now reads `recall_schema_versions` at startup and a role provisioned to the block above will
refuse to boot with `control plane ledger query failed: InsufficientPrivilege`. Grant them before
rolling the new image.

```sql
GRANT SELECT ON recall_schema_versions, recall_index_generations TO recall_server;
GRANT SELECT, INSERT, UPDATE, DELETE ON recall_tenant_routes, recall_migration_events
  TO recall_server;
GRANT USAGE, SELECT ON SEQUENCE recall_migration_events_sequence_id_seq TO recall_server;
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
  ordered migration.
- failed/interrupted concurrent index: rerun `schema apply`. An invalid index is dropped
  concurrently and rebuilt; a completed-but-unrecorded index is validated and adopted.
- schema too new: deploy application code that knows the recorded versions. Do not delete ledger
  rows to force an older binary to start.
