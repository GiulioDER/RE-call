resource "aws_cloudwatch_log_group" "this" {
  name              = "/ecs/${var.name}/${var.environment}"
  retention_in_days = 30
}

resource "aws_ecs_cluster" "this" {
  name = "${var.name}-${var.environment}"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_task_definition" "this" {
  family                   = "${var.name}-${var.environment}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  container_definitions = jsonencode([{ name = "recall", image = var.image, essential = true, portMappings = [{ containerPort = 8000, protocol = "tcp" }], environment = [
    { name = "RECALL_TRANSPORT", value = "streamable-http" },
    { name = "RECALL_ENV", value = lower(trimspace(var.environment)) },
    { name = "RECALL_RATE_LIMIT_BACKEND", value = "redis" },
    { name = "RECALL_DEPLOYMENT", value = lower(trimspace(var.environment)) },
    { name = "RECALL_AWS_REGION", value = var.aws_region },
    { name = "RECALL_AUTH_MODE", value = "oidc" },
    { name = "RECALL_OIDC_ISSUER", value = var.oidc_issuer },
    { name = "RECALL_OIDC_AUDIENCE", value = var.oidc_audience },
    { name = "RECALL_OIDC_TENANTS", value = var.oidc_tenants },
    { name = "RECALL_OIDC_SUBJECT_TENANTS", value = var.oidc_subject_tenants },
    { name = "RECALL_AUTH_RESOURCE_URL", value = var.auth_resource_url },
    { name = "RECALL_SECRET_VERSION_SECRETS", value = jsonencode({ for name, arn in {
      RECALL_SERVING_DSN      = var.serving_dsn_secret_arn,
      RECALL_REDIS_URL        = var.redis_url_secret_arn,
      (var.provider_env_name) = var.provider_secret_arn,
    } : name => arn if arn != null }) },
  ], secrets = concat(var.provider_secret_arn == null ? [] : [{ name = var.provider_env_name, valueFrom = var.provider_secret_arn }], var.serving_dsn_secret_arn == null ? [] : [{ name = "RECALL_SERVING_DSN", valueFrom = var.serving_dsn_secret_arn }], var.redis_url_secret_arn == null ? [] : [{ name = "RECALL_REDIS_URL", valueFrom = var.redis_url_secret_arn }]), healthCheck = { command = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=2)\""], interval = 10, timeout = 5, retries = 3, startPeriod = 30 }, logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.this.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "recall" } } }])
}

resource "aws_ecs_service" "this" {
  name                               = "${var.name}-${var.environment}"
  cluster                            = aws_ecs_cluster.this.id
  task_definition                    = aws_ecs_task_definition.this.arn
  desired_count                      = max(2, var.desired_count)
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  health_check_grace_period_seconds  = 60
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.this.arn
    container_name   = "recall"
    container_port   = 8000
  }
  ordered_placement_strategy {
    type  = "spread"
    field = "attribute:ecs.availability-zone"
  }
  lifecycle {
    ignore_changes = [desired_count]
  }
}
