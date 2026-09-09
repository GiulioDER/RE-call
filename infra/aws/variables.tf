variable "aws_region" { type = string }
variable "offline_plan" {
  type        = bool
  default     = false
  description = "Skip AWS credential and metadata checks for offline validation plans."
}
variable "aws_account_id" {
  type        = string
  default     = null
  description = "Optional AWS account ID override for plans that do not contact AWS."
  validation {
    condition     = var.aws_account_id == null || can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "aws_account_id must be a 12 digit AWS account ID."
  }
}
variable "availability_zones" {
  type        = list(string)
  default     = null
  description = "Optional explicit availability zones for plans that do not contact AWS."
  validation {
    condition     = var.availability_zones == null || length(var.availability_zones) >= 2
    error_message = "availability_zones must contain at least two zones."
  }
}
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
  validation {
    condition     = can(regex("^[^@[:space:]]+@sha256:[0-9a-f]{64}$", lower(trimspace(var.image))))
    error_message = "image must be a nonempty container image reference pinned by a 64 character sha256 digest."
  }
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
  type        = string
  sensitive   = true
  default     = null
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

variable "waf_allowed_source_cidrs" {
  type        = list(string)
  default     = []
  description = "Optional IPv4 source allowlist. When nonempty, every other source is blocked at the ALB WAF."
  validation {
    condition     = alltrue([for cidr in var.waf_allowed_source_cidrs : can(cidrhost(cidr, 0))])
    error_message = "waf_allowed_source_cidrs must contain valid IPv4 CIDR ranges."
  }
}

variable "waf_blocked_source_cidrs" {
  type        = list(string)
  default     = []
  description = "IPv4 source ranges blocked before authentication at the ALB WAF."
  validation {
    condition     = alltrue([for cidr in var.waf_blocked_source_cidrs : can(cidrhost(cidr, 0))])
    error_message = "waf_blocked_source_cidrs must contain valid IPv4 CIDR ranges."
  }
}

variable "waf_rate_limit_per_5m" {
  type        = number
  default     = 1000
  description = "Maximum requests per source IP in a five minute WAF window."
  validation {
    condition     = var.waf_rate_limit_per_5m >= 100 && var.waf_rate_limit_per_5m == floor(var.waf_rate_limit_per_5m)
    error_message = "waf_rate_limit_per_5m must be an integer of at least 100."
  }
}

variable "alert_email" {
  type        = string
  default     = null
  description = "Optional email subscription for the stack alert SNS topic. The recipient must confirm the subscription."
  validation {
    condition     = var.alert_email == null || can(regex("^[^@[:space:]]+@[^@[:space:]]+\\.[^@[:space:]]+$", var.alert_email))
    error_message = "alert_email must be a valid email address when provided."
  }
}

variable "restore_source_cluster" { type = string }
variable "restore_subnet_group" { type = string }
variable "restore_kms_key_id" { type = string }
variable "restore_validation_dsn_secret_arn" { type = string }
variable "restore_schema_version" { type = string }
variable "restore_tenant" { type = string }
variable "restore_expected_generation" { type = string }
variable "restore_expected_role" { type = string }
variable "restore_representative_chunk_id" { type = string }
variable "restore_expected_checksums" {
  type        = string
  description = "JSON object containing nonempty checksums for recall_chunks_v1 and recall_generations"
  validation {
    condition = try(
      toset(keys(jsondecode(trimspace(var.restore_expected_checksums)))) == toset(["recall_chunks_v1", "recall_generations"]) &&
      alltrue([for checksum in values(jsondecode(trimspace(var.restore_expected_checksums))) : trimspace(checksum) != ""]),
      false
    )
    error_message = "restore_expected_checksums must be a JSON object with nonempty recall_chunks_v1 and recall_generations values."
  }
}

variable "restore_checksum_mode" {
  type        = string
  default     = "bounded"
  description = "Restore checksum coverage. Bounded hashes a deterministic primary key prefix; full hashes every visible row."
  validation {
    condition     = contains(["bounded", "full"], lower(trimspace(var.restore_checksum_mode)))
    error_message = "restore_checksum_mode must be bounded or full."
  }
}

variable "restore_checksum_limit" {
  type        = number
  default     = 10000
  description = "Maximum rows hashed per restore table when restore_checksum_mode is bounded."
  validation {
    condition     = var.restore_checksum_limit >= 1 && var.restore_checksum_limit <= 1000000 && var.restore_checksum_limit == floor(var.restore_checksum_limit)
    error_message = "restore_checksum_limit must be an integer between 1 and 1000000."
  }
}

variable "restore_smoke_embedder" {
  type        = string
  default     = "fastembed"
  description = "Embedding backend used by the required application recovery smoke test."
  validation {
    condition     = trimspace(var.restore_smoke_embedder) != ""
    error_message = "restore_smoke_embedder must be nonempty."
  }
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

check "production_restore_validation" {
  assert {
    condition = lower(trimspace(var.environment)) != "production" || (
      alltrue([
        for value in [
          var.restore_source_cluster,
          var.restore_subnet_group,
          var.restore_kms_key_id,
          var.restore_tenant,
          var.restore_expected_generation,
          var.restore_expected_role,
          var.restore_representative_chunk_id,
          var.restore_expected_checksums,
          var.restore_smoke_embedder,
        ] : trimspace(value) != ""
      ]) &&
      try(
        toset(keys(jsondecode(trimspace(var.restore_expected_checksums)))) == toset(["recall_chunks_v1", "recall_generations"]) &&
        alltrue([for checksum in values(jsondecode(trimspace(var.restore_expected_checksums))) : trimspace(checksum) != ""]),
        false
      )
    )
    error_message = "Production requires nonempty restore source, subnet group, KMS key, tenant, generation, role, representative chunk, and both restore table checksums."
  }
}

check "offline_plan_safety" {
  assert {
    condition     = !var.offline_plan || lower(trimspace(var.environment)) != "production"
    error_message = "offline_plan is for non-production validation only."
  }
}
