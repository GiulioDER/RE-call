resource "aws_cloudwatch_event_rule" "restore_drill" {
  name                = "${var.name}-${var.environment}-restore-drill"
  description         = "Run an isolated Aurora restore drill at least every seven days"
  schedule_expression = "rate(7 days)"
}

resource "aws_iam_role" "restore_drill" {
  name               = "${var.name}-${var.environment}-restore-drill"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "events.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy" "restore_drill" {
  role = aws_iam_role.restore_drill.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["ecs:RunTask"], Resource = [aws_ecs_task_definition.restore_drill.arn] },
    { Effect = "Allow", Action = ["iam:PassRole"], Resource = [aws_iam_role.restore_drill_execution.arn, aws_iam_role.restore_drill_task.arn] }
  ] })
}

resource "aws_ecs_task_definition" "restore_drill" {
  family                   = "${var.name}-${var.environment}-restore-drill"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.restore_drill_execution.arn
  task_role_arn            = aws_iam_role.restore_drill_task.arn
  container_definitions = jsonencode([{ name = "restore-drill", image = var.image, essential = true, command = ["python", "-m", "recall.ops.restore_drill"], environment = [
    { name = "AWS_REGION", value = var.aws_region },
    { name = "RECALL_AWS_REGION", value = var.aws_region },
    { name = "RECALL_RESTORE_SOURCE_CLUSTER", value = var.restore_source_cluster },
    { name = "RECALL_RESTORE_SUBNET_GROUP", value = var.restore_subnet_group },
    { name = "RECALL_RESTORE_KMS_KEY_ID", value = var.restore_kms_key_id },
    { name = "RECALL_RESTORE_SCHEMA_VERSION", value = var.restore_schema_version },
    { name = "RECALL_RESTORE_TENANT", value = var.restore_tenant },
    { name = "RECALL_RESTORE_EXPECTED_GENERATION", value = var.restore_expected_generation },
    { name = "RECALL_RESTORE_EXPECTED_ROLE", value = var.restore_expected_role },
    { name = "RECALL_RESTORE_REPRESENTATIVE_CHUNK_ID", value = var.restore_representative_chunk_id },
    { name = "RECALL_RESTORE_EXPECTED_CHECKSUMS", value = var.restore_expected_checksums },
    { name = "RECALL_RESTORE_CHECKSUM_MODE", value = var.restore_checksum_mode },
    { name = "RECALL_RESTORE_CHECKSUM_LIMIT", value = tostring(var.restore_checksum_limit) },
    { name = "RECALL_RESTORE_INSTANCE_CLASS", value = var.db_instance_class },
  ], secrets = [{ name = "RECALL_RESTORE_VALIDATION_DSN", valueFrom = var.restore_validation_dsn_secret_arn }], logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.this.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "restore-drill" } } }])
}

resource "aws_cloudwatch_event_target" "restore_drill" {
  rule      = aws_cloudwatch_event_rule.restore_drill.name
  target_id = "restore-drill"
  arn       = aws_ecs_cluster.this.arn
  role_arn  = aws_iam_role.restore_drill.arn
  ecs_target {
    task_definition_arn = aws_ecs_task_definition.restore_drill.arn
    launch_type         = "FARGATE"
    task_count          = 1
    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.ecs.id]
      assign_public_ip = false
    }
  }
}

resource "aws_cloudwatch_log_metric_filter" "restore_drill_success" {
  name           = "${var.name}-${var.environment}-restore-drill-success"
  log_group_name = aws_cloudwatch_log_group.this.name
  pattern        = "{ $.drill = true }"
  metric_transformation {
    name      = "RestoreDrillSuccess"
    namespace = "RECALL/Operations"
    value     = "1"
  }
}

resource "aws_cloudwatch_metric_alarm" "restore_drill_missing" {
  alarm_name          = "${var.name}-${var.environment}-restore-drill-missing"
  namespace           = "RECALL/Operations"
  metric_name         = "RestoreDrillSuccess"
  statistic           = "Sum"
  period              = 604800
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"
  alarm_description   = "No successful isolated restore drill was recorded in seven days"
}
