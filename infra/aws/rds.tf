resource "aws_db_subnet_group" "this" {
  name       = "${var.name}-${var.environment}"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_rds_cluster" "this" {
  cluster_identifier              = "${var.name}-${var.environment}"
  engine                          = "aurora-postgresql"
  engine_version                  = var.db_engine_version
  database_name                   = var.db_name
  master_username                 = var.db_username
  manage_master_user_password     = true
  db_subnet_group_name            = aws_db_subnet_group.this.name
  vpc_security_group_ids          = [aws_security_group.data.id]
  backup_retention_period         = 35
  preferred_backup_window         = "03:00-03:30"
  preferred_maintenance_window    = "sun:04:00-sun:04:30"
  storage_encrypted               = true
  kms_key_id                      = coalesce(var.kms_key_arn, aws_kms_key.this.arn)
  copy_tags_to_snapshot            = true
  deletion_protection              = true
  enabled_cloudwatch_logs_exports  = ["postgresql"]
  skip_final_snapshot              = false
  final_snapshot_identifier        = "${var.name}-${var.environment}-final"
}

resource "aws_rds_cluster_instance" "this" {
  count              = 2
  identifier         = "${var.name}-${var.environment}-${count.index}"
  cluster_identifier = aws_rds_cluster.this.id
  instance_class     = var.db_instance_class
  engine             = aws_rds_cluster.this.engine
  engine_version     = aws_rds_cluster.this.engine_version
  db_subnet_group_name = aws_db_subnet_group.this.name
  publicly_accessible = false
}

resource "aws_iam_role" "rds_proxy" {
  name = "${var.name}-${var.environment}-rds-proxy"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "rds.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy" "rds_proxy" {
  role = aws_iam_role.rds_proxy.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = [aws_rds_cluster.this.master_user_secret[0].secret_arn] }, { Effect = "Allow", Action = ["kms:Decrypt"], Resource = [coalesce(var.kms_key_arn, aws_kms_key.this.arn)] }] })
}

resource "aws_db_proxy" "this" {
  name                   = "${var.name}-${var.environment}"
  engine_family          = "POSTGRESQL"
  idle_client_timeout    = 1800
  require_tls            = true
  role_arn               = aws_iam_role.rds_proxy.arn
  vpc_security_group_ids = [aws_security_group.data.id]
  vpc_subnet_ids         = aws_subnet.private[*].id
  auth {
    auth_scheme = "SECRETS"
    description = "RDS managed master secret, replace with application secret in production"
    iam_auth    = "DISABLED"
    secret_arn  = aws_rds_cluster.this.master_user_secret[0].secret_arn
  }
}

resource "aws_db_proxy_default_target_group" "this" {
  db_proxy_name = aws_db_proxy.this.name
  connection_pool_config {
    connection_borrow_timeout   = 30
    max_connections_percent     = 90
    max_idle_connections_percent = 50
  }
}

resource "aws_db_proxy_target" "this" {
  db_proxy_name         = aws_db_proxy.this.name
  target_group_name     = aws_db_proxy_default_target_group.this.name
  db_cluster_identifier = aws_rds_cluster.this.id
}
