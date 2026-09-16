# Compatibility policy

This document defines what RE-call treats as a supported compatibility boundary. The supported
surface is listed in [API.md](API.md). The migration procedure is specified in
[MIGRATIONS.md](MIGRATIONS.md), and release-specific behavior changes are recorded in
[PRODUCTION.md](PRODUCTION.md) and [CHANGELOG.md](../CHANGELOG.md).

## Versioning

RE-call is pre-1.0 and uses `0.MINOR.PATCH` versions.

- A patch release is for compatible bug fixes, documentation, and dependency maintenance. It
  must not intentionally remove a documented API, change a persisted schema contract, or alter a
  wire contract in a way that requires caller changes.
- A minor release may add APIs, fields, commands, MCP tools, and forward-only migrations. It may
  also make documented behavior or security changes that require an upgrade note. A minor release
  may remove a deprecated surface when the release notes and upgrade notes say so.
- Any change that can make a working deployment refuse to start, refuse a request, return a
  materially different result, or require a caller to change is called out in the changelog and
  the upgrade section of [PRODUCTION.md](PRODUCTION.md), even when the version remains pre-1.0.

The package version is authoritative for a release. The package metadata, `recall.__version__`,
shipped plugin manifests, and release artifacts must agree. A version change is incomplete until
the version consistency tests and the release checks pass.

## Python API

The Python symbols in [API.md](API.md) are the supported library surface. Within that surface:

- Adding optional parameters, enum values, response fields, or new symbols is additive when
  existing calls and consumers continue to work.
- Renaming or removing a documented symbol, changing a required parameter, changing a documented
  exception family, or changing return semantics is breaking and needs an upgrade note.
- Callers should handle documented exception base classes and ignore unknown fields in structured
  responses. They should not depend on private modules, dataclass field ordering, incidental error
  text, or undocumented serialization details.
- Deprecated symbols remain importable for the compatibility window stated in the release notes.
  Because RE-call is pre-1.0, deprecation does not guarantee retention across every future minor
  release.

Anything outside the supported table is experimental, benchmark-only, migration-internal, or an
implementation detail. It may change without API compatibility guarantees.

## CLI and MCP contracts

The commands and MCP tools listed in [API.md](API.md) are supported integration surfaces.

- Existing CLI commands, required options, exit meanings, and machine-readable output keys are
  preserved unless a release note marks a breaking change.
- MCP tool names and required input fields are preserved. New object fields are additive, and MCP
  clients must ignore unknown fields. Removing or renaming a field, changing its type, or changing
  a refusal into a success is breaking.
- Human-readable text, ordering where not documented, diagnostic detail, and metrics may change.

## PostgreSQL schema and upgrades

Schema compatibility is forward-only. A newer package upgrades an older supported schema by
applying the ordered migrations in [MIGRATIONS.md](MIGRATIONS.md). Operators must:

1. Read the release changelog and upgrade notes.
2. Take the deployment's normal database backup or snapshot.
3. Run `recall schema status` with the serving DSN to inspect state.
4. Run `recall schema apply` with the migration DSN before starting or rolling out the new
   serving package.
5. Run readiness and a representative tenant smoke test before routing traffic.

Applied migration bytes are immutable. Their committed SHA256 checksums must remain unchanged. A
schema correction is a new ordered migration, never an edit to an applied SQL file. The migrator
uses an advisory lock, records transactional and concurrent-index progress, and can resume a
failed upgrade. Serving startup is read-only and refuses pending, unknown, failed, or checksum
drifted migration state.

The supported upgrade path retains populated v0.8 tables through the generation migration. The
regression suite must keep a populated legacy install test and a test that upgrades the previous
release checkpoint without losing existing rows.

Database rollback means restoring the tested backup or snapshot and deploying the previously
supported package. SQL migrations are not rolled back in place. If an application rollback is
needed after a forward migration, the older package must be proven to refuse or safely tolerate
the resulting schema before traffic is shifted.

## Lineage and calibration

An index generation is bound to its pipeline identity, corpus identity, and calibration profile.
Changing the embedder, chunker, context mode, reranker, generation policy, or relevant schema
contract requires a new generation and a fresh calibration or a documented carry-forward decision.
Do not treat a successful SQL upgrade as proof that old retrieval thresholds remain valid.

## Release evidence

Every compatibility-relevant change must include a regression test at the affected boundary.
Schema changes use [tests/test_schema_migrations.py](../tests/test_schema_migrations.py). API
surface changes use [tests/test_api_doc_drift.py](../tests/test_api_doc_drift.py) and the relevant
behavioral tests. Release CI must pass before merge, and the release artifact checks must pass
before publication.
