"""Inspect real offline Terraform plans, without applying resources or querying AWS.

Set RECALL_TERRAFORM to the CI-pinned binary and RECALL_TERRAFORM_DATA_DIR to a directory
initialized with the committed provider lock. The fixture uses synthetic CI inputs and
dummy credentials only. No cloud operation is authorized by these tests.

Red proof on 3ada1afd0: WAF Unicode, rejected CIDRs, credential-free serving and alarm policy
nodes fail their intended assertions. New pin-input nodes use fixed-source mutations:
drop the valueFrom version suffix, or replace the pin-name precondition with true.
Valid CIDR controls use a mutation making the relevant IPv4 validator always false.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from recall_mcp.retrieval import MAX_QUERY_CHARS


SECRET_ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:serving-ABC123"
VERSION_ID = "12345678-1234-1234-1234-123456789012"


@pytest.fixture(scope="module")
def offline_plan(tmp_path_factory):
    terraform = os.environ.get("RECALL_TERRAFORM")
    data_dir = os.environ.get("RECALL_TERRAFORM_DATA_DIR")
    if not terraform or not data_dir:
        pytest.skip("offline Terraform plans require the pinned binary and initialized provider directory")
    directory = tmp_path_factory.mktemp("aws-plan")
    source = Path(__file__).resolve().parents[1] / "infra" / "aws"
    for path in source.iterdir():
        if path.suffix == ".tf" or path.name in {"ci.tfvars", ".terraform.lock.hcl"}:
            shutil.copyfile(path, directory / path.name)
    # An isolated, version-identical lock may add the running platform's package checksum.
    platform_lock = os.environ.get("RECALL_TERRAFORM_LOCKFILE")
    if platform_lock:
        shutil.copyfile(platform_lock, directory / ".terraform.lock.hcl")
    env = {**os.environ, "TF_DATA_DIR": data_dir, "TF_IN_AUTOMATION": "1",
           "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
           "AWS_EC2_METADATA_DISABLED": "true"}
    cache = {}

    def plan(**variables):
        key = json.dumps(variables, sort_keys=True)
        if key in cache:
            return cache[key]
        target = directory / "plan.out"
        command = [terraform, f"-chdir={directory}", "plan", "-no-color", "-input=false",
                   "-lock=false", "-refresh=false", "-var-file=ci.tfvars", f"-out={target}"]
        command.extend(f"-var={name}={value if isinstance(value, str) else json.dumps(value)}"
                       for name, value in variables.items())
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        document = None
        if result.returncode == 0:
            shown = subprocess.run([terraform, f"-chdir={directory}", "show", "-json", str(target)],
                                   env=env, capture_output=True, text=True, timeout=60, check=True)
            document = json.loads(shown.stdout)
        cache[key] = result, document
        return result, document

    return plan


def _resources(document):
    return {resource["address"]: resource.get("values", {})
            for resource in document["planned_values"]["root_module"]["resources"]}


def _successful(offline_plan, **variables):
    result, document = offline_plan(**variables)
    assert result.returncode == 0, result.stdout + result.stderr
    return _resources(document)


def test_waf_preserves_valid_unicode_query_bodies(offline_plan) -> None:
    query = "界" * 3000
    assert len(query) <= MAX_QUERY_CHARS and len(query.encode("utf-8")) > 8192
    resources = _successful(offline_plan)
    common = next(rule for rule in resources["aws_wafv2_web_acl.alb"]["rule"]
                  if rule["name"] == "aws-common-rules")
    managed = common["statement"][0]["managed_rule_group_statement"][0]
    overrides = managed.get("rule_action_override", [])
    assert len(overrides) == 1, "only the incompatible body-size block may be overridden"
    assert overrides[0]["name"] == "SizeRestrictions_BODY"
    assert len(overrides[0]["action_to_use"][0]["count"]) == 1
    assert common["override_action"][0]["none"] == [{}]


@pytest.mark.parametrize("name", ["waf_allowed_source_cidrs", "waf_blocked_source_cidrs"])
@pytest.mark.parametrize("cidr", ["::/1", "0.0.0.0/0"])
def test_waf_rejects_unusable_cidr_ranges(offline_plan, name, cidr) -> None:
    result, _ = offline_plan(**{name: [cidr]})
    assert result.returncode != 0, "the input validator admitted an unusable WAF IP-set range"
    assert name in result.stdout + result.stderr
    assert "Invalid value for variable" in result.stdout + result.stderr


@pytest.mark.parametrize("name", ["waf_allowed_source_cidrs", "waf_blocked_source_cidrs"])
@pytest.mark.parametrize("cidr", ["0.0.0.0/1", "192.0.2.1/32"])
def test_waf_accepts_valid_ipv4_prefix_boundaries(offline_plan, name, cidr) -> None:
    _successful(offline_plan, **{name: [cidr]})


def test_serving_task_needs_no_application_aws_credentials(offline_plan) -> None:
    resources = _successful(offline_plan)
    task = resources["aws_ecs_task_definition.this"]
    container = json.loads(task["container_definitions"])[0]
    assert "RECALL_SECRET_VERSION_SECRETS" not in {item["name"] for item in container["environment"]}
    assert not task.get("task_role_arn")


def test_secret_pins_and_reported_task_versions_agree(offline_plan) -> None:
    resources = _successful(offline_plan, serving_dsn_secret_arn=SECRET_ARN,
                            serving_secret_version_ids={"RECALL_SERVING_DSN": VERSION_ID})
    task = resources["aws_ecs_task_definition.this"]
    container = json.loads(task["container_definitions"])[0]
    assert container["secrets"] == [{"name": "RECALL_SERVING_DSN", "valueFrom": f"{SECRET_ARN}:::{VERSION_ID}"}]
    assert task["tags"]["recall:secret-version:RECALL_SERVING_DSN"] == VERSION_ID
    assert resources["aws_ecs_service.this"]["propagate_tags"] == "TASK_DEFINITION"


@pytest.mark.parametrize("pins", [{}, {"RECALL_SERVING_DSN": VERSION_ID, "OTHER": VERSION_ID}])
def test_secret_pin_names_exactly_match_configured_secrets(offline_plan, pins) -> None:
    result, _ = offline_plan(serving_dsn_secret_arn=SECRET_ARN, serving_secret_version_ids=pins)
    assert result.returncode != 0, "a missing or unrelated version pin reached a deployable task"
    assert "exactly one version pin" in result.stdout + result.stderr


def test_encrypted_alarm_policies_authorize_only_stack_publishers(offline_plan) -> None:
    resources = _successful(offline_plan)
    key_policy = json.loads(resources["aws_kms_key.this"].get("policy") or "{}")
    grants = [statement for statement in key_policy.get("Statement", [])
              if statement.get("Principal") == {"Service": "cloudwatch.amazonaws.com"}]
    assert len(grants) == 1, "CloudWatch cannot use the encrypted alarm topic without a key grant"
    assert set(grants[0]["Action"]) == {"kms:GenerateDataKey*", "kms:Decrypt"}
    condition = {"StringEquals": {"aws:SourceAccount": "123456789012"},
                 "ArnLike": {"aws:SourceArn": "arn:aws:cloudwatch:eu-west-1:123456789012:alarm:recall-ci-ci-*"}}
    assert grants[0]["Condition"] == condition
    assert any(statement.get("Principal") == {"AWS": "arn:aws:iam::123456789012:root"}
               and statement.get("Action") == "kms:*" for statement in key_policy["Statement"])
    topic_policy = json.loads(resources["aws_sns_topic_policy.alerts"]["policy"])
    publish = next(statement for statement in topic_policy["Statement"]
                   if statement.get("Principal") == {"Service": "cloudwatch.amazonaws.com"})
    assert publish["Action"] == "sns:Publish"
    assert publish["Resource"] == "arn:aws:sns:eu-west-1:123456789012:recall-ci-ci-alerts"
    assert publish["Condition"] == condition
