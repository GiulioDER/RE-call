terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = local.tags
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  azs  = slice(data.aws_availability_zones.available.names, 0, 2)
  tags = { Service = "recall", Environment = var.environment, ManagedBy = "terraform" }
  rds_cluster_arn_pattern = "arn:aws:rds:${var.aws_region}:${data.aws_caller_identity.current.account_id}:cluster:${var.name}-${lower(trimspace(var.environment))}-*"
  rds_instance_arn_pattern = "arn:aws:rds:${var.aws_region}:${data.aws_caller_identity.current.account_id}:db:${var.name}-${lower(trimspace(var.environment))}-*"
  ecs_task_arn_pattern = "arn:aws:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:task/${var.name}-${lower(trimspace(var.environment))}/*"
}
