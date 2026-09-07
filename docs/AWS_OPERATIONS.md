# AWS production operations

The supported reference topology is two or more private ECS tasks across two Availability Zones,
an ALB, Aurora PostgreSQL with pgvector and RDS Proxy, Valkey for centralized rate limiting, KMS,
Secrets Manager, and versioned S3 backup receipts. Terraform lives in `infra/aws/`.

## Recovery objectives

The production target is an RPO of at most five minutes and an RTO of at most thirty minutes.
Aurora continuous backups retain thirty five days. Daily snapshots, weekly retained snapshots, and
cross region copies are environment policies. Every backup receipt records only identifiers,
timestamps, schema and generation state, counts, checksums, and configuration fingerprints.

## Restore runbook

1. Run `recall backup status` and record the latest restorable time.
2. Run `recall backup restore` with a new cluster identifier and the explicit
   `RESTORE_NEW_CLUSTER` confirmation.
3. Attach a temporary ECS restore service to the new cluster and run schema, extension, role, grant,
   RLS, generation, calibration, checksum, index, and authenticated representative search checks.
4. Freeze writes, keep the old production target, and cut over only after the operator confirms
   `CUTOVER_RESTORED_CLUSTER`.
5. If smoke tests or monitoring fail, restore the previous target with `ROLLBACK_RESTORE`.
6. Record measured restore duration, effective RPO, smoke result, and rollback result in the drill
   receipt. Alert when no successful drill exists in seven days.

## Secret rotation

Publish a new Secrets Manager version, deploy replacement tasks, verify `/readyz`, verify the task
secret version, and drain the old tasks. Rotate database credentials through RDS Proxy before
revoking the old version. Rotate Valkey credentials with the overlapping token sequence. Rotate
provider and MCP credentials by validating a real authenticated call before revocation. Never put a
secret value in Terraform state, task definition JSON, logs, backup receipts, or this repository.
