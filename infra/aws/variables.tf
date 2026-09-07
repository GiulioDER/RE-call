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
