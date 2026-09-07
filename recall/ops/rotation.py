"""Safe staged secret rotation helpers for ECS deployments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recall.ops.secrets import RotationReceipt, rotation_receipt


@dataclass(frozen=True)
class TaskSecretCheck:
    task_arn: str
    healthy: bool
    version_ids: dict[str, str]
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_arn": self.task_arn,
            "healthy": self.healthy,
            "version_ids": self.version_ids,
            "missing": list(self.missing),
        }


class EcsSecretRotator:
    """Deploy first, verify replacement tasks, and revoke only after explicit confirmation."""

    def __init__(self, *, ecs_client: Any | None = None, secrets_client: Any | None = None, region: str | None = None) -> None:
        self._ecs = ecs_client
        self._secrets = secrets_client
        self._region = region

    def _clients(self) -> tuple[Any, Any]:
        if self._ecs is None or self._secrets is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover, optional AWS extra
                raise RuntimeError("boto3 is required for ECS secret rotation") from exc
            session = boto3.session.Session(region_name=self._region)
            self._ecs = self._ecs or session.client("ecs")
            self._secrets = self._secrets or session.client("secretsmanager")
        return self._ecs, self._secrets

    def verify_tasks(self, cluster: str, service: str, expected_versions: dict[str, str]) -> list[TaskSecretCheck]:
        ecs, _ = self._clients()
        task_arns = ecs.list_tasks(cluster=cluster, serviceName=service, desiredStatus="RUNNING").get("taskArns", [])
        tasks = ecs.describe_tasks(cluster=cluster, tasks=task_arns).get("tasks", [])
        checks: list[TaskSecretCheck] = []
        for task in tasks:
            # The deployment controller copies version identifiers, never secret values, into ECS
            # task tags after the startup health check succeeds.
            tags = ecs.list_tags_for_resource(resourceArn=task["taskArn"]).get("tags", [])
            observed = {
                str(tag["key"]).removeprefix("recall:secret-version:"): str(tag["value"])
                for tag in tags
                if str(tag.get("key", "")).startswith("recall:secret-version:")
            }
            missing = tuple(name for name, version in expected_versions.items() if observed.get(name) != version)
            checks.append(TaskSecretCheck(task_arn=task["taskArn"], healthy=not missing, version_ids=observed, missing=missing))
        return checks

    def require_all_tasks_healthy(self, checks: list[TaskSecretCheck]) -> None:
        if not checks or any(not check.healthy for check in checks):
            raise RuntimeError("not every replacement ECS task reports the intended secret versions")

    def start_rolling_deployment(self, cluster: str, service: str) -> dict[str, object]:
        ecs, _ = self._clients()
        result = ecs.update_service(cluster=cluster, service=service, forceNewDeployment=True)
        return {"cluster": cluster, "service": service, "deployment": result["service"].get("deployments", [])}

    def revoke_previous(self, secret_id: str, previous_version_id: str, *, confirmation: str) -> None:
        if confirmation != "REVOKE_PREVIOUS_SECRET":
            raise ValueError("revocation requires confirmation=REVOKE_PREVIOUS_SECRET")
        _, secrets = self._clients()
        secrets.update_secret_version_stage(
            SecretId=secret_id,
            VersionStage="AWSPREVIOUS",
            RemoveFromVersionId=previous_version_id,
        )

    def receipt(
        self,
        secret_name: str,
        new_version_id: str,
        previous_version_id: str | None,
        *,
        verified: bool,
    ) -> RotationReceipt:
        return rotation_receipt(
            secret_name,
            new_version_id,
            previous_version_id,
            verified=verified,
            rollback=f"restore AWSCURRENT to {previous_version_id or 'previous version'} before revocation",
        )
