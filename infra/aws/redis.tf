resource "aws_elasticache_subnet_group" "this" {
  name       = "${var.name}-${var.environment}"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id       = "${var.name}-${var.environment}"
  description                = "RE-call centralized rate limiter"
  node_type                  = var.redis_node_type
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled           = true
  engine                     = "valkey"
  engine_version             = "7.2"
  transit_encryption_enabled = true
  at_rest_encryption_enabled = true
  kms_key_id                 = coalesce(var.kms_key_arn, aws_kms_key.this.arn)
  subnet_group_name          = aws_elasticache_subnet_group.this.name
  security_group_ids         = [aws_security_group.data.id]
  auth_token                 = var.redis_auth_token
  snapshot_retention_limit   = 7
  snapshot_window            = "02:00-03:00"
}
