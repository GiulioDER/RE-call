"""Runtime secret resolution and rotation receipt primitives."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol


class SecretProvider(Protocol):
    def get(self, name: str, *, version_id: str | None = None) -> "SecretValue": ...


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
        self._region_name = region_name or os.environ.get("AWS_REGION")
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


def secret_mapping_from_env() -> dict[str, str]:
    raw = os.environ.get("RECALL_AWS_SECRET_MAPPING", "")
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RECALL_AWS_SECRET_MAPPING must be a JSON object") from exc
    if not isinstance(decoded, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in decoded.items()):
        raise ValueError("RECALL_AWS_SECRET_MAPPING must map environment names to secret names")
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


def apply_aws_secret_mapping() -> dict[str, str]:
    """Apply configured secret values to the process environment once, without persisting them."""
    mapping = secret_mapping_from_env()
    if not mapping:
        return {}
    provider = AwsSecretsManagerProvider()
    values = provider.resolve_env(mapping)
    for env_name, secret in values.items():
        os.environ[env_name] = secret.value
    return {env_name: secret.version_id for env_name, secret in values.items()}
