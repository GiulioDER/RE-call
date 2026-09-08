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

1. Run `recall backup status --cluster <production-cluster> --region <aws-region>` and record the
   latest restorable time.
2. Run `recall backup restore --source-cluster <production-cluster> --target-cluster <new-cluster>
   --subnet-group <isolated-subnet-group> --kms-key-id <restore-key> --confirm RESTORE_NEW_CLUSTER`.
3. Attach a temporary ECS restore service to the new cluster and run schema, extension, role, grant,
   RLS, generation, calibration, checksum, index, and authenticated representative search checks.
   The validation task must set `RECALL_RESTORE_TENANT`, `RECALL_RESTORE_EXPECTED_GENERATION`,
   `RECALL_RESTORE_EXPECTED_ROLE`, and `RECALL_RESTORE_REPRESENTATIVE_CHUNK_ID`. The latter names
   a known chunk in that tenant and makes the search check prove both authenticated vector search
   and representative retrieval. The calibration check resolves the published artifact against
   the tenant and generation, including its query set and artifact checksum.
   `RECALL_RESTORE_EXPECTED_CHECKSUMS` must contain checksums for both `recall_chunks_v1` and
   `recall_generations`. The validation role must be a non-superuser role without `BYPASSRLS`,
   with `SELECT` on the restored serving tables.
4. Freeze writes, keep the old production target, and cut over only after the operator confirms
   `CUTOVER_RESTORED_CLUSTER`.
5. If smoke tests or monitoring fail, restore the previous target with `ROLLBACK_RESTORE`.
6. Record measured restore duration, effective RPO, smoke result, and rollback result in the drill
   receipt. Alert when no successful drill exists in seven days.

The scheduled restore drill runs a dedicated ECS task definition and task role. The serving task
role has no permission to create, restore, tag, or delete ECS or RDS infrastructure.

## Secret rotation

Publish a new Secrets Manager version, deploy replacement tasks, verify `/readyz`, verify the task
secret version, and drain the old tasks. Rotate database credentials through RDS Proxy before
revoking the old version. Rotate Valkey credentials with the overlapping token sequence. Rotate
provider and MCP credentials by validating a real authenticated call before revocation. Never put a
secret value in Terraform state, task definition JSON, logs, backup receipts, or this repository.

Verify the rollout with `recall secret verify --cluster <cluster> --service <service> --versions
'{"RECALL_SERVING_DSN":"<version-id>","RECALL_REDIS_URL":"<version-id>"}'`. The command rejects
an empty expected map and exits nonzero if any running task is missing a version.
