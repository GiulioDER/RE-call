resource "aws_cloudwatch_event_rule" "restore_drill" {
  name                = "${var.name}-${var.environment}-restore-drill"
  description         = "Run an isolated Aurora restore drill at least every seven days"
  schedule_expression = "rate(7 days)"
}

resource "aws_iam_role" "restore_drill" {
  name = "${var.name}-${var.environment}-restore-drill"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "events.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy" "restore_drill" {
  role = aws_iam_role.restore_drill.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["ecs:RunTask"], Resource = [aws_ecs_task_definition.this.arn] },
    { Effect = "Allow", Action = ["iam:PassRole"], Resource = [aws_iam_role.ecs_execution.arn, aws_iam_role.ecs_task.arn] }
  ] })
}

resource "aws_cloudwatch_event_target" "restore_drill" {
  rule      = aws_cloudwatch_event_rule.restore_drill.name
  target_id = "restore-drill"
  arn       = aws_ecs_cluster.this.arn
  role_arn  = aws_iam_role.restore_drill.arn
  ecs_target {
    task_definition_arn = aws_ecs_task_definition.this.arn
    launch_type         = "FARGATE"
    task_count          = 1
    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.ecs.id]
      assign_public_ip = false
    }
  }
  input = jsonencode({ containerOverrides = [{ name = "recall", command = ["python", "-m", "recall.ops.restore_drill"] }] })
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
