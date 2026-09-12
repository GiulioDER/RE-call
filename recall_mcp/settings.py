"""Typed configuration and the single runtime bootstrap boundary.

The MCP process receives configuration as strings, but the rest of the server should receive one
validated, immutable snapshot.  Secret resolution is deliberately owned by ``bootstrap_settings``
and never happens while this module is imported.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from contextvars import ContextVar, Token
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
        "RECALL_FACT_WRITE_DSN",
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
# owner. Library callers may still pass an environment mapping explicitly, but every setting
# consumed by the MCP process is listed here.
ENVIRONMENT_SCHEMA: tuple[EnvironmentSpec, ...] = (
    EnvironmentSpec("RECALL_ENV", "Core runtime", "deployment environment", "development"),
    EnvironmentSpec("RECALL_DEPLOYMENT", "Core runtime", "deployment identity"),
    EnvironmentSpec("RECALL_SERVING_DSN", "Core runtime", "serving database URL", secret=True),
    EnvironmentSpec("RECALL_MIGRATION_DSN", "Core runtime", "migration database URL", secret=True),
    EnvironmentSpec("RECALL_FACT_WRITE_DSN", "Core runtime", "isolated fact write database URL", secret=True),
    EnvironmentSpec("RECALL_DSN", "Core runtime", "legacy database URL alias"),
    EnvironmentSpec("RECALL_EMBEDDER", "Core runtime", "embedding backend", "fastembed"),
    EnvironmentSpec("RECALL_EMBED_PROFILE", "Core runtime", "registered embedding profile"),
    EnvironmentSpec("RECALL_MODEL_CACHE", "Core runtime", "local model cache"),
    EnvironmentSpec("RECALL_MODEL_SHA256", "Core runtime", "local model tree digest"),
    EnvironmentSpec("RECALL_QWEN_MODEL_PATH", "Core runtime", "Qwen model path"),
    EnvironmentSpec("RECALL_TABLE", "Core runtime", "legacy table name", DEFAULT_TABLE),
    EnvironmentSpec("RECALL_TENANT", "Core runtime", "stdio tenant", DEFAULT_TENANT),
    EnvironmentSpec("RECALL_TRUST_MODE", "Core runtime", "strict or development trust gate", "strict"),
    EnvironmentSpec("RECALL_INDEX_MODE", "Core runtime", "legacy or generation", "legacy"),
    EnvironmentSpec("RECALL_INDEX_ROOT", "Core runtime", "filesystem indexing root", "."),
    EnvironmentSpec("RECALL_INDEX_BATCH_CHUNKS", "Core runtime", "embedding batch size", "64"),
    EnvironmentSpec("RECALL_FASTEMBED_BATCH", "Core runtime", "fastembed batch size"),
    EnvironmentSpec("RECALL_EMBED_THREADS", "Core runtime", "local embedder thread cap"),
    EnvironmentSpec("RECALL_MAX_PRUNE_FRACTION", "Core runtime", "maximum source prune fraction", "0.5"),
    EnvironmentSpec("RECALL_INDEX_ALLOW_CONCURRENT", "Core runtime", "disable the single writer lock"),
    EnvironmentSpec("RECALL_DEPENDENCY_INVALIDATION", "Core runtime", "dependency invalidation mode", "off"),
    EnvironmentSpec("RECALL_INDEX_MAX_FILES", "Core runtime", "maximum files per indexing request", "2000"),
    EnvironmentSpec("RECALL_INDEX_MAX_BYTES", "Core runtime", "maximum bytes per indexing request", "20000000"),
    EnvironmentSpec("RECALL_EMBED_DIMENSIONS", "Core runtime", "hosted embedding dimensions"),
    EnvironmentSpec("RECALL_ACCEPT_RESEARCH_MODEL_LICENSE", "Core runtime", "accept research model license", "0"),
    EnvironmentSpec("RECALL_ACCEPT_REMOTE_MODEL_CODE", "Core runtime", "accept remote model code", "0"),
    EnvironmentSpec("RECALL_SOURCE_POLICY_FILE", "Core runtime", "source authorization policy file"),
    EnvironmentSpec("RECALL_PRINCIPAL", "Core runtime", "local source authorization principal", "cli"),
    EnvironmentSpec("RECALL_CLEARANCE", "Core runtime", "local source authorization clearance", "internal"),
    EnvironmentSpec("RECALL_EGRESS_ALLOWED", "Core runtime", "allow local source egress", "0"),
    EnvironmentSpec("RECALL_TRANSPORT", "MCP", "stdio, sse, or streamable-http", "stdio"),
    EnvironmentSpec("RECALL_HOST", "MCP", "HTTP bind host", "127.0.0.1"),
    EnvironmentSpec("RECALL_PORT", "MCP", "HTTP bind port", "8000"),
    EnvironmentSpec("RECALL_MCP_STATELESS", "MCP", "stateless HTTP sessions; boolean", "1"),
    EnvironmentSpec("RECALL_MCP_TOOLS", "MCP", "comma or space separated tool names or presets"),
    EnvironmentSpec("RECALL_LOG_LEVEL", "MCP", "log level", "INFO"),
    EnvironmentSpec("RECALL_LOG_FORMAT", "MCP", "text or json log format", "text"),
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
    EnvironmentSpec("RECALL_OIDC_JWKS_REFRESH_SECONDS", "Authentication", "OIDC JWKS refresh interval", "300"),
    EnvironmentSpec("RECALL_OIDC_MAX_TOKEN_LIFETIME_SECONDS", "Authentication", "OIDC maximum token lifetime", "900"),
    EnvironmentSpec("RECALL_POOL_SIZE", "Resources", "database pool size", str(DEFAULT_POOL_SIZE)),
    EnvironmentSpec("RECALL_CONNECTION_BUDGET", "Resources", "database connection budget"),
    EnvironmentSpec("RECALL_MAX_TENANTS", "Resources", "maximum provisioned tenants", str(DEFAULT_MAX_TENANTS)),
    EnvironmentSpec("RECALL_READINESS_TENANT_PROBES", "Resources", "startup tenant probes", str(DEFAULT_READINESS_TENANT_PROBES)),
    EnvironmentSpec("RECALL_STATEMENT_TIMEOUT_MS", "Resources", "database statement timeout", str(DEFAULT_STATEMENT_TIMEOUT_MS)),
    EnvironmentSpec("RECALL_RATE_LIMIT_BACKEND", "Rate limits", "local, redis, or off", "local"),
    EnvironmentSpec("RECALL_REDIS_URL", "Rate limits", "Redis URL", secret=True),
    EnvironmentSpec("RECALL_REDIS_TIMEOUT_SECONDS", "Rate limits", "Redis operation timeout", "0.25"),
    EnvironmentSpec("RECALL_REDIS_MAX_CONNECTIONS", "Rate limits", "Redis connection cap", "32"),
    EnvironmentSpec("RECALL_RATE_LIMIT_KEY_PREFIX", "Rate limits", "Redis key prefix", "recall:rate"),
    EnvironmentSpec("RECALL_RATE_READ_FALLBACK_BUDGET", "Rate limits", "read fallback budget", "3"),
    EnvironmentSpec("RECALL_RATE_READ_PER_MIN", "Rate limits", "read calls per minute", "120"),
    EnvironmentSpec("RECALL_RATE_WRITE_PER_MIN", "Rate limits", "write calls per minute", "20"),
    EnvironmentSpec("RECALL_RATE_FORGET_PER_MIN", "Rate limits", "forget calls per minute", "10"),
    EnvironmentSpec("RECALL_RATE_ADMIN_PER_MIN", "Rate limits", "admin calls per minute", "10"),
    EnvironmentSpec("RECALL_RATE_AUTH_FAILURES_PER_MIN", "Rate limits", "authentication failures per minute", "60"),
    EnvironmentSpec("RECALL_INDEX_BYTES_PER_HOUR", "Rate limits", "indexed bytes per hour", "209715200"),
    EnvironmentSpec("RECALL_AWS_REGION", "AWS", "AWS Secrets Manager region"),
    EnvironmentSpec("RECALL_AWS_SECRET_MAPPING", "AWS", "JSON map of approved secret destinations"),
    EnvironmentSpec("RECALL_SECRET_VERSION_SECRETS", "AWS", "JSON map of task tag secret ARNs"),
    EnvironmentSpec("OPENROUTER_API_KEY", "Provider credentials", "OpenRouter API key", secret=True),
    EnvironmentSpec("VOYAGE_API_KEY", "Provider credentials", "Voyage API key", secret=True),
    EnvironmentSpec("OPENAI_API_KEY", "Provider credentials", "OpenAI API key", secret=True),
    EnvironmentSpec("RECALL_REASONING_EXPANSION", "Reasoning", "enable reasoning expansion", "0"),
    EnvironmentSpec("RECALL_REASONING_EXPANSION_MODEL", "Reasoning", "reasoning expansion model"),
    EnvironmentSpec("RECALL_REASONING_EXPANSION_BASE_URL", "Reasoning", "reasoning expansion base URL"),
    EnvironmentSpec("RECALL_REASONING_EXPANSION_API_KEY", "Reasoning", "reasoning expansion API key", secret=True),
    EnvironmentSpec("RECALL_REASONING_EXPANSION_TIMEOUT", "Reasoning", "reasoning expansion timeout", "30"),
    EnvironmentSpec("RECALL_REASONING_EXPANSION_EFFORT", "Reasoning", "reasoning expansion effort", "minimal"),
    EnvironmentSpec("RECALL_REASONING_EXPANSION_REVISION", "Reasoning", "reasoning expansion revision", "unpinned"),
    EnvironmentSpec("RECALL_REASONING_EXPANSION_COST_PER_1K_TOKENS", "Reasoning", "reasoning cost metadata"),
    EnvironmentSpec("RECALL_REASONING_MODEL", "Reasoning", "legacy reasoning model"),
    EnvironmentSpec("RECALL_REASONING_BASE_URL", "Reasoning", "legacy reasoning base URL"),
    EnvironmentSpec("RECALL_REASONING_API_KEY", "Reasoning", "legacy reasoning API key"),
    EnvironmentSpec("RECALL_REASONING_TIMEOUT", "Reasoning", "legacy reasoning timeout", "30"),
    EnvironmentSpec("RECALL_TRANSLATION_ENABLED", "Translation", "enable translation", "0"),
    EnvironmentSpec("RECALL_TRANSLATION_ENDPOINT", "Translation", "translation endpoint"),
    EnvironmentSpec("RECALL_TRANSLATION_TIMEOUT_SECONDS", "Translation", "translation timeout", "5"),
    EnvironmentSpec("RECALL_TRANSLATION_MAX_BATCH", "Translation", "translation batch size", "32"),
    EnvironmentSpec("RECALL_TRANSLATION_MAX_TEXT_CHARS", "Translation", "translation text limit", "20000"),
    EnvironmentSpec("RECALL_TRANSLATION_MAX_RESPONSE_BYTES", "Translation", "translation response limit", "2000000"),
    EnvironmentSpec("RECALL_TRANSLATION_ALLOW_HTTP", "Translation", "allow non HTTPS translation endpoint", "0"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_ENABLED", "Answer provider", "enable answer provider", "0"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_PROVIDER", "Answer provider", "answer provider", "ollama"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_MODEL", "Answer provider", "answer model"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_API_KEY", "Answer provider", "answer provider API key", secret=True),
    EnvironmentSpec("RECALL_REASONING_ANSWER_BASE_URL", "Answer provider", "answer provider base URL"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_TIMEOUT", "Answer provider", "answer provider timeout", "60"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_MAX_TOKENS", "Answer provider", "answer provider token cap", "512"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_REASONING_EFFORT", "Answer provider", "OpenRouter reasoning effort", "none"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_THINKING", "Answer provider", "answer provider thinking mode", "0"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_REVISION", "Answer provider", "answer provider revision", "unpinned"),
    EnvironmentSpec("RECALL_ROUTING_MODE", "Retrieval", "shadow or active routing", "shadow"),
    EnvironmentSpec("RECALL_RETRIEVAL_PROFILE", "Retrieval", "legacy, fast, quality, or code"),
    EnvironmentSpec("RECALL_SEARCH_CONCURRENCY", "Retrieval", "retrieval concurrency"),
    EnvironmentSpec("RECALL_SEARCH_QUEUE", "Retrieval", "retrieval queue capacity"),
    EnvironmentSpec("RECALL_RERANK", "Retrieval", "enable legacy reranking"),
    EnvironmentSpec("RECALL_RERANK_MODEL", "Retrieval", "reranker model"),
    EnvironmentSpec("RECALL_RERANK_REVISION", "Retrieval", "reranker revision"),
    EnvironmentSpec("RECALL_RERANK_BATCH_SIZE", "Retrieval", "reranker batch size", "4"),
    EnvironmentSpec("RECALL_RERANK_THREADS", "Retrieval", "reranker threads", "1"),
    EnvironmentSpec("RECALL_RERANK_PATH", "Retrieval", "local reranker path"),
    EnvironmentSpec("RECALL_RERANK_SHA256", "Retrieval", "local reranker digest"),
    EnvironmentSpec("RECALL_ENTAILMENT", "Retrieval", "enable entailment judge"),
    EnvironmentSpec("RECALL_ENTAILMENT_MODEL", "Retrieval", "entailment model"),
    EnvironmentSpec("RECALL_ENTAILMENT_REVISION", "Retrieval", "entailment revision"),
    EnvironmentSpec("RECALL_HNSW_EF_SEARCH_MULTIPLIER", "Retrieval", "HNSW candidate widening"),
    EnvironmentSpec("RECALL_HNSW_EF_SEARCH_FILTERED", "Retrieval", "filtered HNSW search width"),
    EnvironmentSpec("RECALL_HNSW_ITERATIVE_SCAN_FILTERED", "Retrieval", "filtered HNSW iterative scan"),
    EnvironmentSpec("RECALL_GRAPH_TAIL_REPLACEMENT_MARGIN", "Retrieval", "opt in calibrated graph tail replacement"),
    EnvironmentSpec("RECALL_BENCHMARK_PIN", "Retrieval", "allow pinned benchmark generation", "0"),
    EnvironmentSpec("RECALL_PINNED_GENERATION_ID", "Retrieval", "pinned benchmark generation"),
    EnvironmentSpec("RECALL_ENTERPRISE_CONTROL_PLANE", "Enterprise", "enable enterprise routing", "0"),
    EnvironmentSpec("RECALL_ALLOW_INSECURE_DSN", "Security", "allow insecure database DSN", "0"),
    EnvironmentSpec("RECALL_SERVING_ENV", "Security", "serving environment identity"),
    EnvironmentSpec("RECALL_SCHEMA_LOCK_TIMEOUT_MS", "Operations", "schema lock timeout", "5000"),
    EnvironmentSpec("RECALL_DECISION_LEDGER", "Operations", "decision ledger audit", "0"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS", "Answer provider", "answer cost metadata"),
    EnvironmentSpec("RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN", "Answer provider", "answer call budget", "30"),
    EnvironmentSpec("RECALL_SHADOW_MODEL_CACHE", "Retrieval", "shadow model cache"),
    EnvironmentSpec("RECALL_SHADOW_MODEL_SHA256", "Retrieval", "shadow model digest"),
    EnvironmentSpec("RECALL_SHADOW_QWEN_MODEL_PATH", "Retrieval", "shadow Qwen model path"),
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


def _number(source: Mapping[str, str], name: str, default: float, *, minimum: float) -> float:
    raw = str(source.get(name, default)).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name}={value} is out of range; expected a finite number >= {minimum}")
    return value


def _number_or_off(source: Mapping[str, str], name: str, default: float) -> None:
    raw = str(source.get(name, default)).strip().lower()
    if not raw or raw in {"off", "none", "disabled"}:
        return
    _number({name: raw}, name, default, minimum=0.0)


def _validate_runtime_options(source: Mapping[str, str]) -> None:
    """Validate scalar MCP options before any stores, providers, or listeners are created."""
    for name in (
        "RECALL_TRANSLATION_ENABLED",
        "RECALL_TRANSLATION_ALLOW_HTTP",
        "RECALL_REASONING_ANSWER_ENABLED",
        "RECALL_ENTERPRISE_CONTROL_PLANE",
        "RECALL_BENCHMARK_PIN",
    ):
        _bool(source, name, False)
    if source.get("RECALL_RATE_LIMIT_BACKEND", "local").strip().lower() not in {
        "local",
        "memory",
        "in-memory",
        "redis",
        "off",
        "none",
    }:
        raise ValueError("RECALL_RATE_LIMIT_BACKEND must be local, redis, or off")
    if source.get("RECALL_ROUTING_MODE", "shadow").strip().lower() not in {"shadow", "active"}:
        raise ValueError("RECALL_ROUTING_MODE must be shadow or active")
    if source.get("RECALL_INDEX_MODE", "legacy").strip().lower() not in {"legacy", "generation"}:
        raise ValueError("RECALL_INDEX_MODE must be legacy or generation")
    for name, default_int in (
        ("RECALL_INDEX_MAX_FILES", 2000),
        ("RECALL_INDEX_MAX_BYTES", 20_000_000),
        ("RECALL_TRANSLATION_MAX_BATCH", 32),
        ("RECALL_TRANSLATION_MAX_TEXT_CHARS", 20_000),
        ("RECALL_TRANSLATION_MAX_RESPONSE_BYTES", 2_000_000),
        ("RECALL_REDIS_MAX_CONNECTIONS", 32),
        ("RECALL_RERANK_BATCH_SIZE", 4),
        ("RECALL_RERANK_THREADS", 1),
    ):
        _int(source, name, default_int, minimum=1)
    for name, default_float in (
        ("RECALL_REDIS_TIMEOUT_SECONDS", 0.25),
        ("RECALL_RATE_READ_FALLBACK_BUDGET", 3.0),
        ("RECALL_REASONING_ANSWER_TIMEOUT", 60.0),
    ):
        _number(source, name, default_float, minimum=0.0)
    for name, off_default in (
        ("RECALL_RATE_READ_PER_MIN", 120.0),
        ("RECALL_RATE_WRITE_PER_MIN", 20.0),
        ("RECALL_RATE_FORGET_PER_MIN", 10.0),
        ("RECALL_RATE_ADMIN_PER_MIN", 10.0),
        ("RECALL_RATE_AUTH_FAILURES_PER_MIN", 60.0),
        ("RECALL_INDEX_BYTES_PER_HOUR", 209_715_200.0),
    ):
        _number_or_off(source, name, off_default)


_RUNTIME_SETTINGS: ContextVar["Settings | None"] = ContextVar(
    "recall_runtime_settings", default=None
)


def secret_mapping_from_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
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
        secret_mapping = secret_mapping_from_env(source)
        source["RECALL_AWS_SECRET_MAPPING"] = json.dumps(secret_mapping, sort_keys=True) if secret_mapping else source.get("RECALL_AWS_SECRET_MAPPING", "")
        source["RECALL_SERVING_DSN"] = source.get("RECALL_SERVING_DSN") or source.get("RECALL_DSN", DEFAULT_DSN)
        _validate_runtime_options(source)
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


def activate_runtime_settings(settings: Settings) -> Token[Settings | None]:
    """Make one immutable settings snapshot visible to nested MCP request code."""
    return _RUNTIME_SETTINGS.set(settings)


def reset_runtime_settings(token: Token[Settings | None]) -> None:
    """Restore the previous settings context after a server lifespan ends."""
    _RUNTIME_SETTINGS.reset(token)


def runtime_environment() -> Mapping[str, str]:
    """Return the active server environment, falling back to the process environment for libraries."""
    settings = _RUNTIME_SETTINGS.get()
    return settings.env if settings is not None else os.environ


def bootstrap_settings(
    env: Mapping[str, str] | None = None,
    *,
    secret_provider: _SecretResolver | None = None,
) -> Settings:
    """Resolve environment and AWS secrets exactly once for a server instance."""
    source = dict(os.environ if env is None else env)
    mapping = secret_mapping_from_env(source)
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
    "activate_runtime_settings",
    "bootstrap_settings",
    "reset_runtime_settings",
    "render_env_example",
    "secret_mapping_from_env",
    "runtime_environment",
]
