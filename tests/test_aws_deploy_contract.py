"""Deploy contract of the AWS reference stack and the restore drill (CCA run 20261030T101709Z).

Each test pins one confirmed finding from the audit of PRs 600 to 609. The Terraform tests read the
configuration as text, because CI validates and plans it but never applies it, so a missing setting
is visible only as a broken deployment. Red proof for each test is recorded in its docstring.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.ops import restore_drill
from recall.ops.backup import BackupManager

_INFRA = Path(__file__).parents[1] / "infra" / "aws"


def _tf(name: str) -> str:
    return (_INFRA / name).read_text(encoding="utf-8")


def _serving_task() -> str:
    return _tf("ecs.tf").split('resource "aws_ecs_task_definition" "this"', 1)[1].split(
        'resource "aws_ecs_service" "this"', 1
    )[0]


def _block(text: str, header: str) -> str:
    """Return the brace-balanced body that follows ``header``."""
    start = text.index(header) + len(header)
    start = text.index("{", start)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise AssertionError(f"unbalanced block after {header!r}")


def test_serving_task_binds_all_interfaces_for_the_alb() -> None:
    """DEPLOY-002: the ALB health check reaches the task ENI, never its loopback.

    Red proof: failed on 3ada1afd0 (no RECALL_HOST in infra/aws/ecs.tf, so the server keeps the
    127.0.0.1 default of recall_mcp/settings.py) with AssertionError on the RECALL_HOST check.
    """
    assert '{ name = "RECALL_HOST", value = "0.0.0.0" }' in _serving_task()


def test_serving_task_does_not_ask_to_tag_itself() -> None:
    """DEPLOY-001: the serving role holds no ecs:TagResource, so startup must not need it.

    recall_mcp.settings.bootstrap_settings tags the task whenever RECALL_SECRET_VERSION_SECRETS is
    set, and recall.ops.secrets re-raises the failure in production.
    Red proof: failed on 3ada1afd0, where ecs.tf still set RECALL_SECRET_VERSION_SECRETS, with
    AssertionError on the absence check.
    """
    assert "RECALL_SECRET_VERSION_SECRETS" not in _serving_task()
    assert 'Action = ["ecs:TagResource"]' not in _tf("iam.tf")


def test_data_group_lets_rds_proxy_reach_aurora() -> None:
    """DEPLOY-003: RDS Proxy shares the data group and opens its own connections to Aurora.

    Red proof: failed on 3ada1afd0 (data group had `egress = []` and no self rule) with
    AssertionError on the egress check.
    """
    data = _block(_tf("network.tf"), 'resource "aws_security_group" "data"')
    self_rules = re.findall(
        r"(ingress|egress) \{\s*from_port = 5432\s*to_port\s*= 5432\s*protocol\s*= \"tcp\"\s*self\s*= true",
        data,
    )
    assert "egress" in self_rules
    assert "ingress" in self_rules


def test_drill_role_can_delete_only_drill_resources() -> None:
    """SEC-001: the drill delete grants must not match the production instances.

    Production instances are `${name}-${env}-<n>` (rds.tf); the drill creates only
    `<source>-drill-*`. Red proof: failed on 3ada1afd0, where both ARN patterns ended in
    `${env}-*`, with AssertionError on the pattern check.
    """
    versions = _tf("versions.tf")
    for local in ("rds_cluster_arn_pattern", "rds_instance_arn_pattern"):
        line = next(line for line in versions.splitlines() if line.strip().startswith(local))
        assert line.rstrip().endswith('-drill-*"'), line


def test_drill_role_may_use_the_data_key_through_rds() -> None:
    """DEPLOY-005: an encrypted restore needs kms:CreateGrant and kms:DescribeKey via RDS.

    Red proof: failed on 3ada1afd0 (no KMS statement in the restore_drill_task policy) with
    AssertionError on the action check.
    """
    policy = _tf("iam.tf").split('resource "aws_iam_role_policy" "restore_drill_task"', 1)[1]
    assert '"kms:CreateGrant", "kms:DescribeKey"' in policy
    assert '"kms:ViaService" = "rds.${var.aws_region}.amazonaws.com"' in policy


def test_drill_role_may_place_the_restore_in_its_subnet_group() -> None:
    """DEPLOY-005: RDS authorizes the restore and the writer against the named subnet group too.

    Red proof: failed on 3ada1afd0 (no subgrp resource in the restore_drill_task policy) with
    AssertionError.
    """
    policy = _tf("iam.tf").split('resource "aws_iam_role_policy" "restore_drill_task"', 1)[1]
    assert (
        '{ Effect = "Allow", Action = ["rds:RestoreDBClusterToPointInTime", "rds:CreateDBInstance"], '
        'Resource = ["arn:aws:rds:${var.aws_region}:${local.aws_account_id}:subgrp:${var.restore_subnet_group}"] }'
    ) in policy


def test_restore_drill_attaches_the_data_security_group() -> None:
    """DEPLOY-004: the restored cluster must admit the drill task, which runs in the ecs group.

    Red proof: failed on 3ada1afd0 (no RECALL_RESTORE_SECURITY_GROUP_IDS in restore_drill.tf) with
    AssertionError.
    """
    assert (
        '{ name = "RECALL_RESTORE_SECURITY_GROUP_IDS", value = aws_security_group.data.id }'
        in _tf("restore_drill.tf")
    )


def test_production_requirements_block_the_plan() -> None:
    """ENV-003, ENV-004: a Terraform check block only warns, so the guards must be preconditions.

    Red proof: failed on 3ada1afd0 (production_secrets, production_restore_validation and
    offline_plan_safety were `check` blocks in variables.tf) with AssertionError on the
    `check "` absence check.
    """
    assert 'check "' not in _tf("variables.tf")
    serving = _block(
        _tf("ecs.tf").split('resource "aws_ecs_service" "this"', 1)[1], "lifecycle"
    )
    assert "var.certificate_arn != null" in serving
    assert "!var.offline_plan" in serving
    drill = _block(_tf("restore_drill.tf"), "lifecycle")
    assert "var.restore_schema_version" in drill


def test_environment_is_one_the_server_accepts() -> None:
    """ENV-002: `environment` becomes RECALL_ENV, which only accepts development or production.

    Red proof: failed on 3ada1afd0 (variable "environment" had no validation) with AssertionError.
    """
    variable = _block(_tf("variables.tf"), 'variable "environment"')
    assert 'contains(["development", "production"], lower(trimspace(var.environment)))' in variable
    match = re.search(r'^environment\s*=\s*"([^"]+)"', _tf("ci.tfvars"), re.M)
    assert match is not None and match.group(1) in {"development", "production"}


class _Rds:
    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def restore_db_cluster_to_point_in_time(self, **params: object) -> dict[str, object]:
        self.params = params
        return {"DBCluster": {"DBClusterIdentifier": params["DBClusterIdentifier"]}}


def _restore(rds: _Rds, **kwargs: object) -> None:
    BackupManager(rds_client=rds, s3_client=object()).restore_pitr(
        "src",
        "src-drill-1",
        subnet_group_name="subnets",
        kms_key_id="key",
        confirmation="RESTORE_NEW_CLUSTER",
        **kwargs,  # type: ignore[arg-type]
    )


def test_latest_restore_asks_rds_for_the_latest_restorable_time() -> None:
    """STAKES-002: RDS needs exactly one of RestoreToTime and UseLatestRestorableTime.

    Red proof: failed on 3ada1afd0 (recall.ops.backup.BackupManager.restore_pitr sent neither)
    with AssertionError on UseLatestRestorableTime.
    """
    rds = _Rds()
    _restore(rds)
    assert rds.params.get("UseLatestRestorableTime") is True
    assert "RestoreToTime" not in rds.params
    assert "RestoreType" not in rds.params


def test_point_in_time_restore_is_a_full_copy() -> None:
    """STAKES-002: RDS refuses RestoreToTime with a copy-on-write clone.

    Red proof: failed on 3ada1afd0 (restore_pitr set RestoreType=copy-on-write beside
    RestoreToTime) with AssertionError on RestoreType.
    """
    rds = _Rds()
    _restore(rds, restore_time=datetime(2026, 10, 1, tzinfo=UTC))
    assert rds.params["RestoreToTime"] == datetime(2026, 10, 1, tzinfo=UTC)
    assert "RestoreType" not in rds.params
    assert "UseLatestRestorableTime" not in rds.params


def test_restore_passes_the_requested_security_groups() -> None:
    """DEPLOY-004: without VpcSecurityGroupIds RDS attaches the VPC default group.

    Red proof: failed on 3ada1afd0 with TypeError, because restore_pitr accepted no
    vpc_security_group_ids; the behaviour is pinned by the drill test below as well.
    """
    rds = _Rds()
    _restore(rds, vpc_security_group_ids=["sg-data"])
    assert rds.params["VpcSecurityGroupIds"] == ["sg-data"]


class _Manager:
    calls: list[tuple[str, dict[str, object]]] = []
    stale: list[dict[str, object]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def stale_drill_clusters(self, prefix: str, **kwargs: object) -> list[dict[str, object]]:
        type(self).calls.append(("stale_drill_clusters", {"prefix": prefix, **kwargs}))
        return list(type(self).stale)

    def delete_instance(self, instance: str, **kwargs: object) -> None:
        type(self).calls.append(("delete_instance", {"instance": instance, **kwargs}))

    def restore_pitr(self, source: str, target: str, **kwargs: object) -> dict[str, object]:
        type(self).calls.append(("restore_pitr", {"source": source, "target": target, **kwargs}))
        raise RuntimeError("stop after the restore call")

    def delete_cluster(self, target: str, **kwargs: object) -> None:
        type(self).calls.append(("delete_cluster", {"target": target, **kwargs}))


_DRILL_ENV = {
    "RECALL_RESTORE_SOURCE_CLUSTER": "recall-production",
    "RECALL_RESTORE_SUBNET_GROUP": "subnets",
    "RECALL_RESTORE_KMS_KEY_ID": "alias/recall",
    "RECALL_RESTORE_INSTANCE_CLASS": "db.r6g.large",
    "RECALL_RESTORE_SECURITY_GROUP_IDS": "sg-data",
    "RECALL_RESTORE_VALIDATION_DSN": "postgresql://u:p@{host}:{port}/recall",
    "RECALL_RESTORE_EXPECTED_CHECKSUMS": '{"recall_chunks_v1": "a", "recall_generations": "b"}',
    "RECALL_RESTORE_SCHEMA_VERSION": "0031",
    "RECALL_RESTORE_EXPECTED_GENERATION": "gen-1",
    "RECALL_RESTORE_EXPECTED_ROLE": "recall_server",
    "RECALL_RESTORE_TENANT": "tenant-a",
    "RECALL_RESTORE_REPRESENTATIVE_CHUNK_ID": "chunk-1",
}


@pytest.fixture
def drill_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name, value in _DRILL_ENV.items():
        monkeypatch.setenv(name, value)
    for name in ("RECALL_RESTORE_TARGET_CLUSTER", "RECALL_RESTORE_TARGET_INSTANCE"):
        monkeypatch.delenv(name, raising=False)
    _Manager.calls = []
    _Manager.stale = []
    monkeypatch.setattr(restore_drill, "BackupManager", _Manager)
    return monkeypatch


def test_drill_refuses_missing_settings_before_restoring(drill_env: pytest.MonkeyPatch) -> None:
    """ENV-005: a misconfigured drill must fail before it pays for an Aurora restore.

    Red proof: failed on 3ada1afd0 (recall.ops.restore_drill.run checked the schema version only
    after restore_pitr) with pytest's `Regex pattern did not match`: the fake restore was reached.
    """
    drill_env.setenv("RECALL_RESTORE_SCHEMA_VERSION", " ")
    with pytest.raises(RuntimeError, match="RECALL_RESTORE_SCHEMA_VERSION"):
        restore_drill.run()
    assert _Manager.calls == []


def test_drill_refuses_a_target_outside_the_drill_prefix(drill_env: pytest.MonkeyPatch) -> None:
    """SEC-001: cleanup deletes the target, so it must never be a serving resource.

    Red proof: failed on 3ada1afd0 (run accepted any RECALL_RESTORE_TARGET_INSTANCE) with pytest's
    `Regex pattern did not match`: the fake restore was reached for recall-production-0.
    """
    drill_env.setenv("RECALL_RESTORE_TARGET_INSTANCE", "recall-production-0")
    with pytest.raises(RuntimeError, match="drill-"):
        restore_drill.run()
    assert _Manager.calls == []


def test_drill_restores_into_the_configured_security_groups(drill_env: pytest.MonkeyPatch) -> None:
    """DEPLOY-004: the drill passes RECALL_RESTORE_SECURITY_GROUP_IDS to the restore.

    Red proof: failed on 3ada1afd0 with KeyError on vpc_security_group_ids, because run passed no
    security groups to restore_pitr.
    """
    with pytest.raises(RuntimeError, match="stop after the restore call"):
        restore_drill.run()
    name, call = next(item for item in _Manager.calls if item[0] == "restore_pitr")
    assert name == "restore_pitr"
    assert str(call["target"]).startswith("recall-production-drill-")
    assert call["vpc_security_group_ids"] == ["sg-data"]


def test_drill_refuses_an_empty_security_group_list(drill_env: pytest.MonkeyPatch) -> None:
    """DEPLOY-004: a list of separators must not silently drop VpcSecurityGroupIds.

    Red proof: on 3ada1afd0 the drill read no security groups at all, so this fails there with
    pytest's `Regex pattern did not match`: the fake restore was reached.
    """
    drill_env.setenv("RECALL_RESTORE_SECURITY_GROUP_IDS", " , ")
    with pytest.raises(RuntimeError, match="at least one security group"):
        restore_drill.run()
    assert _Manager.calls == []


def test_validation_dsn_keeps_braces_in_the_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-007: a generated password may contain braces; only {host} and {port} are placeholders.

    Red proof: failed on 4fc690924 (recall.ops.restore_drill._validation_dsn used str.format) with
    KeyError on the brace in the password.
    """
    monkeypatch.setenv("RECALL_RESTORE_VALIDATION_DSN", "postgresql://u:p{x}y@{host}:{port}/recall")
    assert restore_drill._validation_dsn({"endpoint": "writer.example", "port": 5433}) == (
        "postgresql://u:p{x}y@writer.example:5433/recall"
    )


def test_drill_turns_sigterm_into_an_exit_that_runs_cleanup(
    drill_env: pytest.MonkeyPatch,
) -> None:
    """STAKES-005: ECS stops the drill with SIGTERM, whose default action skips the cleanup.

    Red proof: failed on 4fc690924 (recall.ops.restore_drill.run installed no handler) with
    AssertionError: no SIGTERM handler was installed.
    """
    import signal

    installed: list[tuple[int, object]] = []

    def record(signum: int, handler: object) -> object:
        installed.append((signum, handler))
        return signal.SIG_DFL

    drill_env.setattr(signal, "signal", record)
    with pytest.raises(RuntimeError, match="stop after the restore call"):
        restore_drill.run()
    handlers = [handler for signum, handler in installed if signum == signal.SIGTERM]
    assert handlers, "no SIGTERM handler was installed"
    with pytest.raises(SystemExit) as stopped:
        handlers[0](signal.SIGTERM, None)  # type: ignore[operator]
    assert stopped.value.code == 143
    assert handlers[-1] is signal.SIG_DFL, "the previous handler must be restored after cleanup"


def test_drill_validates_on_an_autocommit_connection(drill_env: pytest.MonkeyPatch) -> None:
    """DAT-005: one failing guarded check must not abort the transaction for every later check.

    Red proof: failed on 4fc690924 (run opened the validation connection with autocommit off) with
    AssertionError on the autocommit argument.
    """
    import psycopg

    class _Ready(_Manager):
        def restore_pitr(self, source: str, target: str, **kwargs: object) -> dict[str, object]:
            return {"cluster_identifier": target}

        def wait_for_cluster_available(self, target: str) -> None:
            return None

        def create_restore_instance(self, *_args: object, **_kwargs: object) -> None:
            return None

        def wait_for_instance_available(self, _instance: str) -> dict[str, object]:
            return {"endpoint": "writer.example", "port": 5432}

        def delete_instance(self, *_args: object, **_kwargs: object) -> None:
            return None

    seen: dict[str, object] = {}

    def connect(_dsn: str, **kwargs: object) -> object:
        seen.update(kwargs)
        raise RuntimeError("stop at the validation connection")

    drill_env.setattr(restore_drill, "BackupManager", _Ready)
    drill_env.setattr(psycopg, "connect", connect)
    with pytest.raises(RuntimeError, match="stop at the validation connection"):
        restore_drill.run()
    assert seen.get("autocommit") is True


def test_restore_drill_runs_more_often_than_its_missing_alarm_window() -> None:
    """NUM-002: a schedule equal to the alarm window leaves windows with no success after a slow run.

    Red proof: failed on 4fc690924 (rate(7 days) against a 604800 s period) with AssertionError.
    """
    drill = _tf("restore_drill.tf")
    days = int(re.search(r'schedule_expression = "rate\((\d+) days\)"', drill).group(1))  # type: ignore[union-attr]
    period = int(re.search(r"period\s*=\s*(\d+)", drill).group(1))  # type: ignore[union-attr]
    assert days * 86400 < period


def test_restore_drill_has_time_to_clean_up_after_a_stop() -> None:
    """STAKES-005: the SIGTERM handler starts cleanup, which needs more than Fargate's 30 s default.

    Red proof: failed on 4fc690924 (no stopTimeout in the restore-drill container definition) with
    AssertionError.
    """
    assert "stopTimeout = 120" in _tf("restore_drill.tf")


def test_drill_deletes_left_behind_drill_clusters_before_restoring(
    drill_env: pytest.MonkeyPatch,
) -> None:
    """Drill janitor (owner-approved 2026-10-30): a stopped drill's cluster is removed by the next.

    Red proof: failed on e60e79883 (recall.ops.restore_drill.run neither looked for nor deleted
    drill clusters) with AssertionError on the delete calls.
    """
    _Manager.stale = [
        {
            "cluster_identifier": "recall-production-drill-old",
            "instances": ["recall-production-drill-old-writer"],
        }
    ]
    with pytest.raises(RuntimeError, match="stop after the restore call"):
        restore_drill.run()
    names = [name for name, _call in _Manager.calls]
    assert names[:3] == ["stale_drill_clusters", "delete_instance", "delete_cluster"]
    assert _Manager.calls[0][1]["prefix"] == "recall-production-drill-"
    assert _Manager.calls[1][1]["instance"] == "recall-production-drill-old-writer"
    assert _Manager.calls[2][1]["target"] == "recall-production-drill-old"
    restore = next(call for name, call in _Manager.calls if name == "restore_pitr")
    assert restore["tags"] == [{"Key": "recall:restore-drill", "Value": "true"}]


class _Pages:
    def __init__(self, clusters: list[dict[str, object]]) -> None:
        self.clusters = clusters

    def get_paginator(self, name: str) -> "_Pages":
        assert name == "describe_db_clusters"
        return self

    def paginate(self) -> list[dict[str, object]]:
        return [{"DBClusters": self.clusters[:2]}, {"DBClusters": self.clusters[2:]}]


def test_only_old_drill_tagged_drill_clusters_are_stale() -> None:
    """Drill janitor: the name prefix, the drill tag and the age must all hold.

    Red proof: by mutation, since the method is new: removing the tag condition from
    recall.ops.backup.BackupManager.stale_drill_clusters failed this test with AssertionError,
    because the untagged drill-named cluster was returned.
    """
    tag = [{"Key": "recall:restore-drill", "Value": "true"}]
    old = datetime(2026, 10, 1, tzinfo=UTC)
    young = datetime(2026, 10, 30, 12, tzinfo=UTC)
    clusters: list[dict[str, object]] = [
        {"DBClusterIdentifier": "src-drill-old", "TagList": tag, "ClusterCreateTime": old,
         "DBClusterMembers": [{"DBInstanceIdentifier": "src-drill-old-writer"}]},
        {"DBClusterIdentifier": "src-drill-young", "TagList": tag, "ClusterCreateTime": young},
        {"DBClusterIdentifier": "src-drill-untagged", "TagList": [], "ClusterCreateTime": old},
        {"DBClusterIdentifier": "src", "TagList": tag, "ClusterCreateTime": old},
    ]
    manager = BackupManager(rds_client=_Pages(clusters), s3_client=object())
    stale = manager.stale_drill_clusters("src-drill-", older_than=datetime(2026, 10, 30, tzinfo=UTC))
    assert stale == [
        {"cluster_identifier": "src-drill-old", "instances": ["src-drill-old-writer"]}
    ]


def test_drill_role_may_tag_its_cluster_and_list_clusters() -> None:
    """Drill janitor: restoring with Tags needs AddTagsToResource on the drill cluster, and the
    janitor lists clusters; deletion stays limited to drill names.

    Red proof: failed on e60e79883 (no AddTagsToResource on the cluster statement, no list grant)
    with AssertionError.
    """
    policy = _tf("iam.tf").split('resource "aws_iam_role_policy" "restore_drill_task"', 1)[1]
    cluster = next(line for line in policy.splitlines() if "local.rds_cluster_arn_pattern" in line)
    assert '"rds:AddTagsToResource"' in cluster
    assert (
        '{ Effect = "Allow", Action = ["rds:DescribeDBClusters"], '
        'Resource = ["arn:aws:rds:${var.aws_region}:${local.aws_account_id}:cluster:*"] }'
    ) in policy


def test_a_failed_drill_listing_does_not_stop_the_drill(drill_env: pytest.MonkeyPatch) -> None:
    """Drill janitor: cleanup must never cost the drill, even when the listing itself fails.

    Red proof: by mutation: calling stale_drill_clusters outside the try in
    recall.ops.restore_drill._reap_stale_drills failed this test with pytest's
    `Regex pattern did not match`, because the listing error ended the drill before the restore.
    """

    def refuse(self: _Manager, prefix: str, **_kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("AccessDenied")

    drill_env.setattr(_Manager, "stale_drill_clusters", refuse)
    with pytest.raises(RuntimeError, match="stop after the restore call"):
        restore_drill.run()
    assert [name for name, _call in _Manager.calls] == ["restore_pitr"]
