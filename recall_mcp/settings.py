"""Typed configuration and the single runtime bootstrap boundary.

The MCP process receives configuration as strings, but the rest of the server should receive one
validated, immutable snapshot.  Secret resolution is deliberately owned by ``bootstrap_settings``
and never happens while this module is imported.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from recall.trust_policy import TrustPolicy


DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"
DEFAULT_TABLE = "chunks"
DEFAULT_TENANT = "default"
DEFAULT_POOL_SIZE = 8
DEFAULT_MAX_TENANTS = 1000
DEFAULT_READINESS_TENANT_PROBES = 3
MAX_READINESS_TENANT_PROBES = 10
DEFAULT_STATEMENT_TIMEOUT_MS = 15_000

# These are the only process environment destinations that may receive a value fetched from AWS.
# In particular, deployment mode, trust policy, authentication selectors and OIDC policy are not
# secret destinations and cannot be overridden by a secret payload.
SECRET_DESTINATIONS = frozenset(
    {
        "RECALL_SERVING_DSN",
        "RECALL_MIGRATION_DSN",
        "RECALL_REDIS_URL",
        "VOYAGE_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "RECALL_REASONING_EXPANSION_API_KEY",
        "RECALL_REASONING_ANSWER_API_KEY",
    }
)


@dataclass(frozen=True)
class EnvironmentSpec:
    """One documented configuration key from the canonical schema."""

    name: str
    section: str
    description: str
    default: str = ""
    secret: bool = False


# The deployable MCP surface is kept here so generated documentation and runtime names have one
# owner.  The remaining values in ``env`` are intentionally preserved for library and CLI options
# that are outside the MCP bootstrap boundary.
ENVIRONMENT_SCHEMA: tuple[EnvironmentSpec, ...] = (
    EnvironmentSpec("RECALL_ENV", "Core runtime", "deployment environment", "development"),
    EnvironmentSpec("RECALL_SERVING_DSN", "Core runtime", "serving database URL", secret=True),
    EnvironmentSpec("RECALL_MIGRATION_DSN", "Core runtime", "migration database URL", secret=True),
    EnvironmentSpec("RECALL_EMBEDDER", "Core runtime", "embedding backend", "fastembed"),
    EnvironmentSpec("RECALL_TABLE", "Core runtime", "legacy table name", DEFAULT_TABLE),
    EnvironmentSpec("RECALL_TENANT", "Core runtime", "stdio tenant", DEFAULT_TENANT),
    EnvironmentSpec("RECALL_TRANSPORT", "MCP", "stdio, sse, or streamable-http", "stdio"),
    EnvironmentSpec("RECALL_HOST", "MCP", "HTTP bind host", "127.0.0.1"),
    EnvironmentSpec("RECALL_PORT", "MCP", "HTTP bind port", "8000"),
    EnvironmentSpec("RECALL_MCP_STATELESS", "MCP", "stateless HTTP sessions", "auto"),
    EnvironmentSpec("RECALL_AUTH_MODE", "Authentication", "static or oidc"),
    EnvironmentSpec("RECALL_AUTH_TOKENS_FILE", "Authentication", "development token file"),
    EnvironmentSpec("RECALL_AUTH_ISSUER_URL", "Authentication", "protected resource issuer URL"),
    EnvironmentSpec("RECALL_AUTH_RESOURCE_URL", "Authentication", "protected resource URL"),
    EnvironmentSpec("RECALL_OIDC_ISSUER", "Authentication", "OIDC issuer URL"),
    EnvironmentSpec("RECALL_OIDC_AUDIENCE", "Authentication", "OIDC audience"),
    EnvironmentSpec("RECALL_OIDC_TENANTS", "Authentication", "OIDC tenant allowlist"),
    EnvironmentSpec("RECALL_OIDC_ALGORITHMS", "Authentication", "OIDC algorithm allowlist"),
    EnvironmentSpec("RECALL_OIDC_SUBJECT_TENANTS", "Authentication", "OIDC subject bindings"),
    EnvironmentSpec("RECALL_OIDC_TRUST_TENANT_CLAIM", "Authentication", "trust IdP tenant claim"),
    EnvironmentSpec("RECALL_POOL_SIZE", "Resources", "database pool size", str(DEFAULT_POOL_SIZE)),
    EnvironmentSpec("RECALL_CONNECTION_BUDGET", "Resources", "database connection budget"),
    EnvironmentSpec("RECALL_MAX_TENANTS", "Resources", "maximum provisioned tenants", str(DEFAULT_MAX_TENANTS)),
    EnvironmentSpec("RECALL_READINESS_TENANT_PROBES", "Resources", "startup tenant probes", str(DEFAULT_READINESS_TENANT_PROBES)),
    EnvironmentSpec("RECALL_STATEMENT_TIMEOUT_MS", "Resources", "database statement timeout", str(DEFAULT_STATEMENT_TIMEOUT_MS)),
    EnvironmentSpec("RECALL_RATE_LIMIT_BACKEND", "Rate limits", "local, redis, or off", "local"),
    EnvironmentSpec("RECALL_REDIS_URL", "Rate limits", "Redis URL", secret=True),
    EnvironmentSpec("RECALL_REDIS_TIMEOUT_SECONDS", "Rate limits", "Redis operation timeout", "0.25"),
    EnvironmentSpec("RECALL_REDIS_MAX_CONNECTIONS", "Rate limits", "Redis connection cap", "32"),
    EnvironmentSpec("RECALL_AWS_REGION", "AWS", "AWS Secrets Manager region"),
    EnvironmentSpec("RECALL_AWS_SECRET_MAPPING", "AWS", "JSON map of approved secret destinations"),
    EnvironmentSpec("RECALL_SECRET_VERSION_SECRETS", "AWS", "JSON map of task tag secret ARNs"),
    EnvironmentSpec("OPENROUTER_API_KEY", "Provider credentials", "OpenRouter API key", secret=True),
    EnvironmentSpec("VOYAGE_API_KEY", "Provider credentials", "Voyage API key", secret=True),
    EnvironmentSpec("OPENAI_API_KEY", "Provider credentials", "OpenAI API key", secret=True),
    EnvironmentSpec("RECALL_TRANSLATION_ENABLED", "Translation", "enable translation", "0"),
    EnvironmentSpec("RECALL_TRANSLATION_ENDPOINT", "Translation", "translation endpoint"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_ENABLED", "Answer provider", "enable answer provider", "0"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_PROVIDER", "Answer provider", "answer provider", "ollama"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_MODEL", "Answer provider", "answer model"),
)


def _int(source: Mapping[str, str], name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = str(source.get(name, default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name}={value} is out of range; expected {bound}")
    return value


def _bool(source: Mapping[str, str], name: str, default: bool) -> bool:
    raw = str(source.get(name, "1" if default else "0")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean; expected true or false")


def _secret_mapping(source: Mapping[str, str]) -> dict[str, str]:
    raw = str(source.get("RECALL_AWS_SECRET_MAPPING", "")).strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RECALL_AWS_SECRET_MAPPING must be a JSON object") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in decoded.items()
    ):
        raise ValueError("RECALL_AWS_SECRET_MAPPING must map environment names to secret names")
    unknown = sorted(set(decoded) - SECRET_DESTINATIONS)
    if unknown:
        allowed = ", ".join(sorted(SECRET_DESTINATIONS))
        raise ValueError(
            "RECALL_AWS_SECRET_MAPPING contains forbidden destination(s): "
            f"{', '.join(unknown)}; allowed destinations are {allowed}"
        )
    return {str(key): str(value) for key, value in decoded.items()}


class _SecretResolver(Protocol):
    def resolve_env(self, mapping: Mapping[str, str]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Settings:
    """The immutable, validated configuration snapshot used by one server instance."""

    env: Mapping[str, str] = field(repr=False, compare=False)
    transport: str
    host: str
    port: int
    serving_dsn: str
    embedder: str
    table: str
    tenant: str
    pool_size: int
    connection_budget: int
    max_tenants: int
    readiness_tenant_probes: int
    statement_timeout_ms: int
    mcp_stateless: bool
    enterprise_control_plane: bool
    trust_policy: TrustPolicy
    aws_region: str | None
    secret_mapping: Mapping[str, str] = field(repr=False, compare=False)
    secret_versions: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        object.__setattr__(self, "secret_mapping", MappingProxyType(dict(self.secret_mapping)))
        object.__setattr__(self, "secret_versions", MappingProxyType(dict(self.secret_versions)))

    @property
    def values(self) -> Mapping[str, str]:
        """Compatibility name for consumers that call the snapshot a values mapping."""
        return self.env

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        secret_versions: Mapping[str, str] | None = None,
    ) -> "Settings":
        source = dict(os.environ if env is None else env)
        transport = source.get("RECALL_TRANSPORT", "stdio")
        if transport not in {"stdio", "sse", "streamable-http"}:
            raise ValueError(
                f"RECALL_TRANSPORT={transport!r} is not a valid transport; expected stdio, sse, streamable-http"
            )
        table = source.get("RECALL_TABLE", "").strip() or DEFAULT_TABLE
        if not table.isidentifier():
            raise ValueError(f"RECALL_TABLE={table!r} is not a valid SQL identifier")
        pool_size = _int(source, "RECALL_POOL_SIZE", DEFAULT_POOL_SIZE, minimum=1)
        connection_budget = _int(source, "RECALL_CONNECTION_BUDGET", pool_size, minimum=1)
        if pool_size > connection_budget:
            raise ValueError(
                f"RECALL_POOL_SIZE={pool_size} exceeds RECALL_CONNECTION_BUDGET={connection_budget}"
            )
        secret_mapping = _secret_mapping(source)
        source["RECALL_AWS_SECRET_MAPPING"] = json.dumps(secret_mapping, sort_keys=True) if secret_mapping else source.get("RECALL_AWS_SECRET_MAPPING", "")
        source["RECALL_SERVING_DSN"] = source.get("RECALL_SERVING_DSN") or source.get("RECALL_DSN", DEFAULT_DSN)
        return cls(
            env=source,
            transport=transport,
            host=source.get("RECALL_HOST", "127.0.0.1"),
            port=_int(source, "RECALL_PORT", 8000, minimum=1, maximum=65535),
            serving_dsn=source["RECALL_SERVING_DSN"],
            embedder=source.get("RECALL_EMBEDDER", "fastembed"),
            table=table,
            tenant=source.get("RECALL_TENANT", DEFAULT_TENANT),
            pool_size=pool_size,
            connection_budget=connection_budget,
            max_tenants=_int(source, "RECALL_MAX_TENANTS", DEFAULT_MAX_TENANTS, minimum=1),
            readiness_tenant_probes=_int(
                source,
                "RECALL_READINESS_TENANT_PROBES",
                DEFAULT_READINESS_TENANT_PROBES,
                minimum=1,
                maximum=MAX_READINESS_TENANT_PROBES,
            ),
            statement_timeout_ms=_int(
                source, "RECALL_STATEMENT_TIMEOUT_MS", DEFAULT_STATEMENT_TIMEOUT_MS, minimum=1
            ),
            mcp_stateless=_bool(source, "RECALL_MCP_STATELESS", transport in {"sse", "streamable-http"}),
            enterprise_control_plane=_bool(source, "RECALL_ENTERPRISE_CONTROL_PLANE", False),
            trust_policy=TrustPolicy.from_env(source),
            aws_region=source.get("RECALL_AWS_REGION") or source.get("AWS_REGION"),
            secret_mapping=secret_mapping,
            secret_versions=secret_versions or {},
        )


def bootstrap_settings(
    env: Mapping[str, str] | None = None,
    *,
    secret_provider: _SecretResolver | None = None,
) -> Settings:
    """Resolve environment and AWS secrets exactly once for a server instance."""
    source = dict(os.environ if env is None else env)
    mapping = _secret_mapping(source)
    versions: dict[str, str] = {}
    if mapping:
        if secret_provider is None:
            from recall.ops.secrets import AwsSecretsManagerProvider

            secret_provider = AwsSecretsManagerProvider(
                region_name=source.get("RECALL_AWS_REGION") or source.get("AWS_REGION")
            )
        resolved = secret_provider.resolve_env(mapping)
        for destination, secret in resolved.items():
            source[destination] = secret.value
            versions[destination] = secret.version_id
    # ECS tagging is a bootstrap receipt, not configuration resolution. It is still performed once,
    # after values are resolved and before the server is assembled.
    if source.get("RECALL_SECRET_VERSION_SECRETS", "").strip():
        from recall.ops.secrets import tag_ecs_task_secret_versions

        versions.update(
            tag_ecs_task_secret_versions(
                env=source,
                region_name=source.get("RECALL_AWS_REGION") or source.get("AWS_REGION"),
            )
        )
    return Settings.from_env(source, secret_versions=versions)


def render_env_example() -> str:
    """Render the generated configuration reference from ``ENVIRONMENT_SCHEMA``."""
    lines = [
        "# BEGIN GENERATED SETTINGS, do not edit by hand",
        "# Generated from recall_mcp.settings.ENVIRONMENT_SCHEMA.",
    ]
    section = ""
    for spec in ENVIRONMENT_SCHEMA:
        if spec.section != section:
            section = spec.section
            lines.extend(["", f"# {section}"])
        value = spec.default if spec.default else ""
        lines.append(f"# {spec.name}={value:<12} # {spec.description}")
    lines.extend(["", "# END GENERATED SETTINGS"])
    return "\n".join(lines)


__all__ = [
    "ENVIRONMENT_SCHEMA",
    "SECRET_DESTINATIONS",
    "EnvironmentSpec",
    "Settings",
    "bootstrap_settings",
    "render_env_example",
]
