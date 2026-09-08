resource "aws_s3_bucket" "receipts" {
  bucket              = coalesce(var.backup_receipt_bucket, "${var.name}-${var.environment}-backup-receipts")
  object_lock_enabled = true
}

resource "aws_s3_bucket_versioning" "receipts" {
  bucket = aws_s3_bucket.receipts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_object_lock_configuration" "receipts" {
  bucket = aws_s3_bucket.receipts.id
  rule {
    default_retention {
      mode = "COMPLIANCE"
      days = 35
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "receipts" {
  bucket = aws_s3_bucket.receipts.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = coalesce(var.kms_key_arn, aws_kms_key.this.arn)
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "receipts" {
  bucket                  = aws_s3_bucket.receipts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
