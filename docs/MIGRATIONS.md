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

Do not copy a list from this page. Generate it, so it cannot drift out of step with the tables
the code actually creates:

```bash
recall schema grants --role recall_server
recall schema grants --role recall_server --enterprise   # if RECALL_ENTERPRISE_CONTROL_PLANE is on
```

The command prints SQL and runs nothing, so it needs no DSN. Run the output as the object owner.

`--enterprise` adds the four control-plane tables (`recall_index_generations`,
`recall_schema_versions`, `recall_tenant_routes`, `recall_migration_events`) and, critically,
`GRANT USAGE ON SEQUENCE recall_migration_events_sequence_id_seq`. The serving process reads the
first two on every routed request and appends to `recall_migration_events` on every shadow flush.
That table is the one object in the schema with a `bigserial` key, so table privileges alone are
not enough: the INSERT fails with `permission denied for sequence` until the sequence is granted.
An earlier version of this section listed ten objects and omitted all of these, which meant an
operator who followed it exactly got `permission denied` at startup readiness.

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
  run multiple jobs against the same database. The migrator keeps trying for
  `MIGRATION_LOCK_WAIT_SECONDS` (2s) before reporting this, because the advisory lock is released
  when PostgreSQL reaps the holding backend rather than when the holding process exits: a migrator
  restarted straight after a kill, a Ctrl-C or a container restart would otherwise be refused on
  account of its own predecessor. Seeing the error after that wait means a migrator really is
  holding the lock.
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
# Migration 0015: Evidence Graph V1

Migration `0015_semantic_graph_foundation.sql` creates the tenant and generation scoped semantic
graph tables used by Evidence Graph V1. It adds entities, mentions, relations, and normalized
relation evidence links. Every graph row is protected by row level security and linked to the
corresponding generation and chunk. The relation evidence table stores chunk identifiers only and
never duplicates source text.

The migration is additive. Existing generations remain valid for ordinary retrieval. Run
`recall graph rebuild --generation <generation_id>` after applying the migration to opt an existing
generation into graph expansion. New generations build the deterministic graph before validation
and promotion. A fingerprint mismatch or missing graph marker returns `GRAPH_NOT_READY` for graph
expansion and does not affect `recall_search` or `recall_evidence`.
