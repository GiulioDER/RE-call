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
  region                      = var.aws_region
  skip_credentials_validation = var.offline_plan
  skip_metadata_api_check     = var.offline_plan
  skip_region_validation      = var.offline_plan
  skip_requesting_account_id  = var.offline_plan
  default_tags {
    tags = local.tags
  }
}

data "aws_availability_zones" "available" {
  count = var.availability_zones == null ? 1 : 0
  state = "available"
}

data "aws_caller_identity" "current" {
  count = var.aws_account_id == null ? 1 : 0
}

locals {
  azs                      = var.availability_zones != null ? slice(var.availability_zones, 0, 2) : slice(data.aws_availability_zones.available[0].names, 0, 2)
  tags                     = { Service = "recall", Environment = var.environment, ManagedBy = "terraform" }
  aws_account_id           = var.aws_account_id != null ? var.aws_account_id : data.aws_caller_identity.current[0].account_id
  rds_cluster_arn_pattern  = "arn:aws:rds:${var.aws_region}:${local.aws_account_id}:cluster:${var.name}-${lower(trimspace(var.environment))}-*"
  rds_instance_arn_pattern = "arn:aws:rds:${var.aws_region}:${local.aws_account_id}:db:${var.name}-${lower(trimspace(var.environment))}-*"
  ecs_task_arn_pattern     = "arn:aws:ecs:${var.aws_region}:${local.aws_account_id}:task/${var.name}-${lower(trimspace(var.environment))}/*"
}
