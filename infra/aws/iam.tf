resource "aws_iam_role" "ecs_execution" {
  name               = "${var.name}-${var.environment}-ecs-execution"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = compact([var.oidc_secret_arn, var.provider_secret_arn, var.serving_dsn_secret_arn, var.redis_url_secret_arn]) },
    { Effect = "Allow", Action = ["kms:Decrypt"], Resource = [coalesce(var.kms_key_arn, aws_kms_key.this.arn)] }
  ] })
}

resource "aws_iam_role" "restore_drill_execution" {
  name               = "${var.name}-${var.environment}-restore-drill-execution"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy_attachment" "restore_drill_execution" {
  role       = aws_iam_role.restore_drill_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "restore_drill_execution_secrets" {
  role = aws_iam_role.restore_drill_execution.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = [var.restore_validation_dsn_secret_arn] },
    { Effect = "Allow", Action = ["kms:Decrypt"], Resource = [coalesce(var.kms_key_arn, aws_kms_key.this.arn)] }
  ] })
}

resource "aws_iam_role" "restore_drill_task" {
  name               = "${var.name}-${var.environment}-restore-drill-task"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy" "restore_drill_task" {
  role = aws_iam_role.restore_drill_task.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["rds:RestoreDBClusterToPointInTime", "rds:DescribeDBClusters", "rds:DescribeDBClusterSnapshots", "rds:DeleteDBCluster", "rds:CreateDBInstance"], Resource = [local.rds_cluster_arn_pattern] },
    { Effect = "Allow", Action = ["rds:CreateDBInstance", "rds:AddTagsToResource", "rds:DescribeDBInstances", "rds:DeleteDBInstance"], Resource = [local.rds_instance_arn_pattern] }
  ] })
}
