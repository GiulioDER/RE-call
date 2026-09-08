"""Runtime secret resolution and rotation receipt primitives."""

from __future__ import annotations

import json
import os
from urllib.request import urlopen
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol

from recall_mcp.settings import SECRET_DESTINATIONS


class SecretProvider(Protocol):
    def get(self, name: str, *, version_id: str | None = None) -> "SecretValue": ...

    def resolve_env(self, mapping: Mapping[str, str]) -> dict[str, "SecretValue"]: ...


@dataclass(frozen=True)
class RuntimeSecretState:
    """Nonsecret version identifiers retained for diagnostics and rotation verification."""

    versions: dict[str, str]


@dataclass(frozen=True)
class SecretValue:
    name: str
    value: str
    version_id: str
    arn: str | None = None


@dataclass(frozen=True)
class RotationReceipt:
    secret_name: str
    new_version_id: str
    previous_version_id: str | None
    rotated_at: str
    verified: bool
    rollback: str

    def to_dict(self) -> dict[str, object]:
        return {
            "secret_name": self.secret_name,
            "new_version_id": self.new_version_id,
            "previous_version_id": self.previous_version_id,
            "rotated_at": self.rotated_at,
            "verified": self.verified,
            "rollback": self.rollback,
        }


class AwsSecretsManagerProvider:
    """Fetch a JSON or plain text secret lazily, never at module import."""

    def __init__(self, *, region_name: str | None = None, client: Any | None = None) -> None:
        self._region_name = region_name or os.environ.get("RECALL_AWS_REGION") or os.environ.get("AWS_REGION")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover, optional AWS extra
                raise RuntimeError("boto3 is required for AWS Secrets Manager") from exc
            self._client = boto3.client("secretsmanager", region_name=self._region_name)
        return self._client

    def get(self, name: str, *, version_id: str | None = None) -> SecretValue:
        params: dict[str, str] = {"SecretId": name}
        if version_id:
            params["VersionId"] = version_id
        response = self._get_client().get_secret_value(**params)
        value = response.get("SecretString")
        if value is None:
            import base64

            value = base64.b64decode(response["SecretBinary"]).decode("utf-8")
        return SecretValue(
            name=name,
            value=str(value),
            version_id=str(response.get("VersionId", "unknown")),
            arn=response.get("ARN"),
        )

    def resolve_env(self, mapping: Mapping[str, str]) -> dict[str, SecretValue]:
        """Resolve secret names to values for a process bootstrapper.

        The returned values are intentionally kept in memory only. Callers must not log this
        mapping or write it to task definitions, receipts, or configuration files.
        """
        return {env_name: self.get(secret_name) for env_name, secret_name in mapping.items()}


def secret_mapping_from_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    raw = source.get("RECALL_AWS_SECRET_MAPPING", "")
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RECALL_AWS_SECRET_MAPPING must be a JSON object") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in decoded.items()
    ):
        raise ValueError("RECALL_AWS_SECRET_MAPPING must map environment names to secret names")
    unknown = sorted(set(decoded) - SECRET_DESTINATIONS)
    if unknown:
        raise ValueError(
            "RECALL_AWS_SECRET_MAPPING contains forbidden destination(s): "
            f"{', '.join(unknown)}"
        )
    return decoded


def rotation_receipt(
    secret_name: str,
    new_version_id: str,
    previous_version_id: str | None,
    *,
    verified: bool,
    rollback: str,
) -> RotationReceipt:
    return RotationReceipt(
        secret_name=secret_name,
        new_version_id=new_version_id,
        previous_version_id=previous_version_id,
        rotated_at=datetime.now(UTC).isoformat(),
        verified=verified,
        rollback=rollback,
    )


def apply_aws_secret_mapping(
    env: dict[str, str] | None = None, *, provider: SecretProvider | None = None
) -> dict[str, str]:
    """Apply configured secret values to the process environment once, without persisting them."""
    target = os.environ if env is None else env
    mapping = secret_mapping_from_env(target)
    if not mapping:
        return {}
    resolved_by = provider or AwsSecretsManagerProvider(
        region_name=target.get("RECALL_AWS_REGION") or target.get("AWS_REGION")
    )
    values = resolved_by.resolve_env(mapping)
    for env_name, secret in values.items():
        target[env_name] = secret.value
    return {env_name: secret.version_id for env_name, secret in values.items()}


def secret_version_mapping_from_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the nonsecret environment to Secrets Manager ARN mapping for task tagging."""
    source = os.environ if env is None else env
    raw = source.get("RECALL_SECRET_VERSION_SECRETS", "").strip()
    if not raw:
        return {}
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RECALL_SECRET_VERSION_SECRETS must be a JSON object") from exc
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in mapping.items()
    ):
        raise ValueError("RECALL_SECRET_VERSION_SECRETS must map names to secret ARNs")
    return {key: value for key, value in mapping.items()}


def tag_ecs_task_secret_versions(
    *, region_name: str | None = None, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Tag the current ECS task with AWSCURRENT version ids, never with secret values.

    The tags are the verification receipt consumed by the rotation runbook. Outside ECS, or
    when no mapping is configured, this is a no-op so local and stdio startup remain unchanged.
    Production ECS startup fails if a configured tag cannot be published, because an unverified
    replacement task must not be mistaken for a healthy rotation.
    """
    source = os.environ if env is None else env
    mapping = secret_version_mapping_from_env(source)
    metadata_url = source.get("ECS_CONTAINER_METADATA_URI_V4", "").rstrip("/")
    if not mapping or not metadata_url:
        return {}
    try:
        with urlopen(f"{metadata_url}/task", timeout=2) as response:  # nosec B310, ECS metadata
            task = json.load(response)
        task_arn = str(task["TaskARN"])
        import boto3

        session = boto3.session.Session(
            region_name=region_name
            or source.get("RECALL_AWS_REGION")
            or source.get("AWS_REGION")
        )
        secrets = session.client("secretsmanager")
        ecs = session.client("ecs")
        versions: dict[str, str] = {}
        tags: list[dict[str, str]] = []
        for name, secret_arn in mapping.items():
            description = secrets.describe_secret(SecretId=secret_arn)
            current = next(
                (
                    version_id
                    for version_id, stages in description.get("VersionIdsToStages", {}).items()
                    if "AWSCURRENT" in stages
                ),
                None,
            )
            if not current:
                raise RuntimeError(f"secret {name!r} has no AWSCURRENT version")
            versions[name] = str(current)
            tags.append({"key": f"recall:secret-version:{name}", "value": str(current)})
        ecs.tag_resource(resourceArn=task_arn, tags=tags)
        return versions
    except Exception:  # BROAD-CATCH: fail-closed
        if source.get("RECALL_ENV", "development").strip().lower() == "production":
            raise
        return {}
