variable "aws_region" { type = string }
variable "environment" {
  type    = string
  default = "production"
}
variable "name" {
  type    = string
  default = "recall"
}
variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}
variable "image" {
  type        = string
  description = "Immutable ECR image digest for recall-mcp"
}
variable "desired_count" {
  type    = number
  default = 2
  validation {
    condition     = var.desired_count >= 2
    error_message = "desired_count must be at least 2 for multi-AZ availability."
  }
}
variable "db_name" {
  type    = string
  default = "recall"
}
variable "db_engine_version" {
  type    = string
  default = "16.6"
}
variable "db_instance_class" {
  type    = string
  default = "db.r7g.large"
}
variable "db_username" {
  type    = string
  default = "recall_app"
}
variable "redis_node_type" {
  type    = string
  default = "cache.r7g.large"
}
variable "redis_auth_token" {
  type      = string
  sensitive = true
  default   = null
  description = "Optional bootstrap token. Prefer out of band AUTH rotation after apply."
}

variable "db_proxy_secret_arn" {
  type        = string
  default     = null
  description = "Secrets Manager ARN containing the application username and password accepted by RDS Proxy"
}

variable "oidc_issuer" {
  type        = string
  description = "HTTPS issuer URL accepted by the production OIDC validator"
  validation {
    condition     = can(regex("^https://", var.oidc_issuer))
    error_message = "oidc_issuer must use HTTPS."
  }
}

variable "oidc_audience" { type = string }
variable "oidc_tenants" { type = string }
variable "oidc_subject_tenants" {
  type        = string
  description = "Comma separated OIDC subject to tenant bindings, for example service-account:tenant-a"
}
variable "provider_env_name" {
  type        = string
  default     = "OPENROUTER_API_KEY"
  description = "Environment variable consumed by the selected hosted provider"
  validation {
    condition     = contains(["VOYAGE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"], var.provider_env_name)
    error_message = "provider_env_name must be a supported provider key environment variable."
  }
}
variable "auth_resource_url" {
  type        = string
  description = "Protected resource URL used by the MCP auth metadata"
  validation {
    condition     = can(regex("^https://", var.auth_resource_url))
    error_message = "auth_resource_url must use HTTPS."
  }
}
variable "kms_key_arn" {
  type    = string
  default = null
}
variable "oidc_secret_arn" {
  type    = string
  default = null
}
variable "provider_secret_arn" {
  type    = string
  default = null
}
variable "serving_dsn_secret_arn" {
  type        = string
  default     = null
  description = "Secrets Manager ARN whose value is the serving DSN through RDS Proxy"
}
variable "redis_url_secret_arn" {
  type        = string
  default     = null
  description = "Secrets Manager ARN whose value is the authenticated rediss URL"
}
variable "backup_receipt_bucket" {
  type    = string
  default = null
}
variable "certificate_arn" {
  type        = string
  default     = null
  description = "ACM certificate ARN for the production HTTPS listener"
}

variable "restore_source_cluster" { type = string }
variable "restore_subnet_group" { type = string }
variable "restore_kms_key_id" { type = string }
variable "restore_validation_dsn_secret_arn" { type = string }
variable "restore_schema_version" { type = string }
variable "restore_expected_checksums" {
  type        = string
  description = "JSON object of restore table checksum expectations"
}

check "production_secrets" {
  assert {
    condition = lower(trimspace(var.environment)) != "production" || (
      var.serving_dsn_secret_arn != null &&
      var.redis_url_secret_arn != null &&
      var.certificate_arn != null &&
      var.redis_auth_token != null &&
      var.restore_validation_dsn_secret_arn != null &&
      var.db_proxy_secret_arn != null &&
      trimspace(var.oidc_subject_tenants) != ""
    )
    error_message = "Production requires serving, Redis, restore validation, proxy credentials, OIDC subject bindings, Redis AUTH, and an ACM certificate."
  }
}
