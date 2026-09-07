# RE-call AWS production reference

This stack provides two private ECS task placements, an ALB, Aurora PostgreSQL with pgvector,
RDS Proxy, Valkey, KMS, versioned immutable S3 receipts, IAM, and baseline alarms.

The image variable must be an immutable digest. Secret values are not committed and should not be
put into task definition JSON or Terraform files. Populate Secrets Manager and use the task secret
ARNs, then roll ECS. The optional Redis bootstrap token is only for initial provisioning; use the
documented overlapping token rotation procedure before production service.

Apply this stack in a staging account first. Native Aurora continuous backups use 35 day retention.
Daily and weekly snapshot retention, cross region copies, restore drills, TLS certificates, WAF,
OIDC issuer configuration, and alert destinations should be added in the environment specific
root module before production approval.

The operational CLI is:

```text
recall backup status --cluster recall-production
recall backup create --cluster recall-production --snapshot recall-manual-20260907
recall backup verify --cluster recall-production --snapshot recall-manual-20260907
recall backup restore --source-cluster recall-production --target-cluster recall-restore-20260907 --subnet-group recall-production --kms-key-id alias/recall-production --confirm RESTORE_NEW_CLUSTER
```
