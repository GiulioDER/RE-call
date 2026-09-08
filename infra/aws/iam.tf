resource "aws_iam_role" "ecs_execution" {
  name = "${var.name}-${var.environment}-ecs-execution"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = compact([var.oidc_secret_arn, var.provider_secret_arn, var.serving_dsn_secret_arn, var.redis_url_secret_arn, var.restore_validation_dsn_secret_arn]) },
    { Effect = "Allow", Action = ["kms:Decrypt"], Resource = [coalesce(var.kms_key_arn, aws_kms_key.this.arn)] }
  ] })
}

resource "aws_iam_role" "ecs_task" {
  name = "${var.name}-${var.environment}-ecs-task"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy" "ecs_task" {
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], Resource = [aws_s3_bucket.receipts.arn, "${aws_s3_bucket.receipts.arn}/*"] },
    { Effect = "Allow", Action = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"], Resource = compact([var.provider_secret_arn, var.serving_dsn_secret_arn, var.redis_url_secret_arn, var.restore_validation_dsn_secret_arn, var.db_proxy_secret_arn, aws_rds_cluster.this.master_user_secret[0].secret_arn]) },
    { Effect = "Allow", Action = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"], Resource = [coalesce(var.kms_key_arn, aws_kms_key.this.arn)] },
    { Effect = "Allow", Action = ["ecs:TagResource"], Resource = [local.ecs_task_arn_pattern] },
    { Effect = "Allow", Action = ["rds:RestoreDBClusterToPointInTime", "rds:DescribeDBClusters", "rds:DescribeDBClusterSnapshots", "rds:DeleteDBCluster", "rds:CreateDBInstance"], Resource = [local.rds_cluster_arn_pattern] },
    { Effect = "Allow", Action = ["rds:CreateDBInstance", "rds:DescribeDBInstances", "rds:DeleteDBInstance"], Resource = [local.rds_instance_arn_pattern] }
  ] })
}
