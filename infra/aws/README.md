# RE-call AWS production foundation

This stack provides two private ECS task placements, an ALB, Aurora PostgreSQL with pgvector,
RDS Proxy, Valkey, KMS, versioned immutable S3 receipts, IAM, and baseline alarms.

The image variable must be an immutable digest. Secret values are not committed and should not be
put into task definition JSON or Terraform files. Populate Secrets Manager and use the task secret
base ARNs and matching nonsecret `serving_secret_version_ids`, then roll ECS. Every configured
serving secret needs a version pin keyed by its container environment name. ECS injects that exact
version and propagates the matching version tags without an application task role. Update the pins
for each rotation; see [the rotation procedure](../../docs/AWS_OPERATIONS.md#secret-rotation).
The optional Redis bootstrap token is only for initial provisioning; use the
documented overlapping token rotation procedure before production service.

Apply this stack in a staging account first and provide the required production inputs, including
the restore drill identifiers and both restore table checksums. Native Aurora continuous backups
use 35 day retention. This directory includes the scheduled restore drill, TLS listener, OIDC
configuration, an ALB WAF, source allowlist and blocklist inputs, per source MCP request limits,
an SNS alert topic, and baseline alarms. Set `waf_allowed_source_cidrs` when the deployment has a
fixed client network. If it is empty, the WAF remains public but still applies managed rules,
configured source blocks, and the per IP request limit. Set `alert_email` to create an email
subscription, then confirm it from the recipient mailbox. Production approval still requires daily
and weekly snapshot retention, cross region copies, and measured restore evidence in accordance
with the operating runbook.

WAF source CIDRs must be IPv4 with prefixes from 1 to 32. The body-size managed rule counts
bodies above the ALB's 8 KiB inspection limit to preserve the application's Unicode query contract;
the SDK's 4 MiB HTTP body cap and remaining managed blocking rules still apply. An external `kms_key_arn` must separately grant the
scoped CloudWatch publisher permissions described in the [alarm runbook](../../docs/AWS_OPERATIONS.md#edge-limits-and-alarm-delivery).

The operational CLI is:

```text
recall backup status --cluster recall-production
recall backup create --cluster recall-production --snapshot recall-manual-20260907
recall backup verify --cluster recall-production --snapshot recall-manual-20260907
recall backup restore --source-cluster recall-production --target-cluster recall-restore-20260907 --subnet-group recall-production --kms-key-id alias/recall-production --confirm RESTORE_NEW_CLUSTER
```
