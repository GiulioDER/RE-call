output "alb_dns_name" { value = aws_lb.this.dns_name }
output "rds_proxy_endpoint" { value = aws_db_proxy.this.endpoint }
output "aurora_cluster_identifier" { value = aws_rds_cluster.this.cluster_identifier }
output "redis_primary_endpoint" { value = aws_elasticache_replication_group.this.primary_endpoint_address }
output "backup_receipt_bucket" { value = aws_s3_bucket.receipts.bucket }
output "kms_key_arn" { value = aws_kms_key.this.arn }
