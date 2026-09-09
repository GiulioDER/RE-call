# The ALB is public by design, so abuse controls must run before traffic reaches ECS.
resource "aws_wafv2_ip_set" "alb_allowed_sources" {
  count              = length(var.waf_allowed_source_cidrs) == 0 ? 0 : 1
  name               = "${var.name}-${var.environment}-allowed-sources"
  description        = "Optional source allowlist for the public RE-call ALB"
  scope              = "REGIONAL"
  ip_address_version = "IPV4"
  addresses          = var.waf_allowed_source_cidrs
}

resource "aws_wafv2_ip_set" "alb_blocked_sources" {
  count              = length(var.waf_blocked_source_cidrs) == 0 ? 0 : 1
  name               = "${var.name}-${var.environment}-blocked-sources"
  description        = "Source ranges blocked before authentication"
  scope              = "REGIONAL"
  ip_address_version = "IPV4"
  addresses          = var.waf_blocked_source_cidrs
}

resource "aws_wafv2_web_acl" "alb" {
  name        = "${var.name}-${var.environment}-alb"
  description = "Pre authentication controls for the public RE-call ALB"
  scope       = "REGIONAL"

  default_action {
    dynamic "block" {
      for_each = length(var.waf_allowed_source_cidrs) == 0 ? [] : [1]
      content {}
    }
    dynamic "allow" {
      for_each = length(var.waf_allowed_source_cidrs) == 0 ? [1] : []
      content {}
    }
  }

  dynamic "rule" {
    for_each = length(var.waf_blocked_source_cidrs) == 0 ? [] : [1]
    content {
      name     = "blocked-source-ranges"
      priority = 0
      action {
        block {}
      }
      statement {
        ip_set_reference_statement {
          arn = aws_wafv2_ip_set.alb_blocked_sources[0].arn
        }
      }
      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "${var.name}-${var.environment}-blocked-sources"
        sampled_requests_enabled   = true
      }
    }
  }

  rule {
    name     = "per-source-request-limit"
    priority = 10
    action {
      block {}
    }
    statement {
      rate_based_statement {
        limit              = var.waf_rate_limit_per_5m
        aggregate_key_type = "IP"
        scope_down_statement {
          byte_match_statement {
            field_to_match {
              uri_path {}
            }
            positional_constraint = "STARTS_WITH"
            search_string         = "/mcp"
            text_transformation {
              priority = 0
              type     = "NONE"
            }
          }
        }
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-${var.environment}-per-source-limit"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "aws-common-rules"
    priority = 20
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-${var.environment}-common-rules"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "aws-known-bad-inputs"
    priority = 30
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-${var.environment}-known-bad-inputs"
      sampled_requests_enabled   = true
    }
  }

  dynamic "rule" {
    for_each = length(var.waf_allowed_source_cidrs) == 0 ? [] : [1]
    content {
      name     = "trusted-source-ranges"
      priority = 40
      action {
        allow {}
      }
      statement {
        ip_set_reference_statement {
          arn = aws_wafv2_ip_set.alb_allowed_sources[0].arn
        }
      }
      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "${var.name}-${var.environment}-trusted-sources"
        sampled_requests_enabled   = true
      }
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.name}-${var.environment}-alb-waf"
    sampled_requests_enabled   = true
  }
}

resource "aws_wafv2_web_acl_association" "alb" {
  resource_arn = aws_lb.this.arn
  web_acl_arn  = aws_wafv2_web_acl.alb.arn
}
