aws_region         = "eu-west-1"
offline_plan       = true
aws_account_id     = "123456789012"
availability_zones = ["eu-west-1a", "eu-west-1b"]
environment        = "ci"
name               = "recall-ci"

# Synthetic digest only. CI never pulls or deploys this image.
image = "123456789012.dkr.ecr.eu-west-1.amazonaws.com/recall@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

oidc_issuer          = "https://issuer.example.test"
oidc_audience        = "recall-ci"
oidc_tenants         = "tenant-a"
oidc_subject_tenants = "service-account:tenant-a"
auth_resource_url    = "https://recall.example.test"
certificate_arn      = "arn:aws:acm:eu-west-1:123456789012:certificate/00000000-0000-0000-0000-000000000000"

restore_source_cluster            = "recall-ci-source"
restore_subnet_group              = "recall-ci-private"
restore_kms_key_id                = "alias/recall-ci"
restore_validation_dsn_secret_arn = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:recall-ci-restore"
restore_schema_version            = "ci"
restore_tenant                    = "tenant-a"
restore_expected_generation       = "ci-generation"
restore_expected_role             = "recall_server"
restore_representative_chunk_id   = "ci-chunk"
restore_expected_checksums        = "{\"recall_chunks_v1\":\"fixture-chunks\",\"recall_generations\":\"fixture-generations\"}"
