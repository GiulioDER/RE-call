resource "aws_kms_key" "this" {
  description             = "RE-call production data, backup, and receipt encryption"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "EnableAccountKeyAdministration"
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${local.aws_account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "AllowStackAlarmEncryption"
        Effect    = "Allow"
        Principal = { Service = "cloudwatch.amazonaws.com" }
        Action    = ["kms:GenerateDataKey*", "kms:Decrypt"]
        Resource  = "*"
        Condition = {
          StringEquals = { "aws:SourceAccount" = local.aws_account_id }
          ArnLike      = { "aws:SourceArn" = local.alert_alarm_arn_pattern }
        }
      }
    ]
  })
}

resource "aws_kms_alias" "this" {
  name          = "alias/${var.name}-${var.environment}"
  target_key_id = aws_kms_key.this.key_id
}
