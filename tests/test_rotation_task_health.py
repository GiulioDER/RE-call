"""Propagated version tags prove identity at creation, not task health.

Red proof: 3ada1afd0 EcsSecretRotator.verify_tasks treats all matching tags as healthy.
Each unhealthy status node fails its healthy assertion. The mismatched-version case is
proven with a deliberate mutation removing the ``not missing`` term from the fixed guard.
"""
import pytest

from recall.ops.rotation import EcsSecretRotator


@pytest.mark.parametrize(
    ("last_status", "health_status", "version"),
    [("PENDING", "HEALTHY", "v1"), ("RUNNING", "UNHEALTHY", "v1"),
     ("RUNNING", "UNKNOWN", "v1"), ("RUNNING", None, "v1"),
     ("RUNNING", "HEALTHY", "old")],
)
def test_version_tags_do_not_bypass_task_health(last_status, health_status, version) -> None:
    class Ecs:
        def list_tasks(self, **kwargs):
            return {"taskArns": ["task-a"]}

        def describe_tasks(self, **kwargs):
            return {"tasks": [{"taskArn": "task-a", "lastStatus": last_status,
                               "healthStatus": health_status}]}

        def list_tags_for_resource(self, **kwargs):
            return {"tags": [{"key": "recall:secret-version:RECALL_SERVING_DSN", "value": version}]}

    rotator = EcsSecretRotator(ecs_client=Ecs(), secrets_client=object())
    checks = rotator.verify_tasks("cluster", "service", {"RECALL_SERVING_DSN": "v1"})
    assert checks and not checks[0].healthy, "version identity cannot replace a healthy running task"
    with pytest.raises(RuntimeError, match="not every replacement"):
        rotator.require_all_tasks_healthy(checks)
    last_status, health_status, version = "RUNNING", "HEALTHY", "v1"
    healthy = rotator.verify_tasks("cluster", "service", {"RECALL_SERVING_DSN": "v1"})
    assert healthy[0].healthy
    rotator.require_all_tasks_healthy(healthy)
