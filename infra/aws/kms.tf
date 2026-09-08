resource "aws_kms_key" "this" {
  description             = "RE-call production data, backup, and receipt encryption"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "this" {
  name          = "alias/${var.name}-${var.environment}"
  target_key_id = aws_kms_key.this.key_id
}
