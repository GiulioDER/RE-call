"""Immutable runtime configuration for RE-call Hosted."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from recall_aml.variants import DEFAULT_VARIANT, variant


PRODUCT_NAME = "RE-call Hosted 1.0"
#: ``RECALL_AML_AUTHORIZED_USER_ID`` value that scopes the API key to an evaluation platform, which
#: sends a different ``user_id`` per sample; every other value binds the key to that one user.
PLATFORM_SCOPE = "*"
PRODUCT_VERSION = "1.0.0"
SCHEMA_VERSION = "0024"
EMBEDDING_PROFILE = "voyage-context-4-v1"
RETRIEVAL_PROFILE = "hosted-quality"
GENERATION_PROVIDER = "openrouter"
GENERATION_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RERANK_MODEL = "voyage:rerank-2.5"
RERANK_PRICE_USD_PER_MILLION_TOKENS = 0.05
RERANK_PRICE_SOURCE_DATE = "2026-09-18"
RERANK_PRICE_SOURCE_URL = "https://docs.voyageai.com/docs/pricing"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
SPARSE_REVISION = "762be6a7206e2f299182705972a65e5c46e62be2"


@dataclass(frozen=True)
class HostedSettings:
    database_url: str
    api_key: str
    git_commit: str
    table: str = "recall_chunks"
    generation_id: str = "aml-hosted-v1"
    host: str = "127.0.0.1"
    port: int = 18_004
    add_concurrency: int = 16
    search_concurrency: int = 16
    context_chars: int = 7_000
    variant_name: str = DEFAULT_VARIANT
    openrouter_api_key: str | None = None
    voyage_api_key: str | None = None
    splade_device: str = "cpu"
    splade_threads: int = 4
    embedding_lock_path: Path | None = None
    embedding_cache_path: Path | None = None
    authorized_user_id: str | None = None

    def __post_init__(self) -> None:
        if not self.database_url or not self.api_key or not self.git_commit:
            raise ValueError("database_url, api_key, and git_commit must be non-empty")
        if self.database_url != "postgresql://unused" and not (self.authorized_user_id or "").strip():
            raise ValueError(
                "authorized_user_id is required for a non-test hosted deployment"
            )
        if not self.table.isidentifier():
            raise ValueError("table must be a valid SQL identifier")
        for name in (
            "port",
            "add_concurrency",
            "search_concurrency",
            "context_chars",
            "splade_threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        variant(self.variant_name)

    @classmethod
    def from_env(cls) -> "HostedSettings":
        required = {
            "database_url": os.environ.get("RECALL_AML_DATABASE_URL", ""),
            "api_key": os.environ.get("RECALL_AML_API_KEY", ""),
            "git_commit": os.environ.get("RECALL_AML_GIT_COMMIT", ""),
        }
        embedding_lock_path = os.environ.get("RECALL_AML_EMBED_LOCK_PATH", "")
        placeholders = {"CHANGE_ME", "CHANGEME", "REPLACE_ME", "REPLACE-ME"}
        placeholder_names = [
            name for name, value in required.items() if value.strip().upper() in placeholders
        ]
        if embedding_lock_path.strip().upper() in placeholders:
            placeholder_names.append("RECALL_AML_EMBED_LOCK_PATH")
        if placeholder_names:
            raise RuntimeError(
                "required hosted settings still contain placeholders: "
                + ", ".join(placeholder_names)
            )
        missing = [
            *[name for name, value in required.items() if not value],
            *([] if embedding_lock_path else ["RECALL_AML_EMBED_LOCK_PATH"]),
        ]
        if missing:
            raise RuntimeError("missing required hosted settings: " + ", ".join(missing))
        authorized_user_id = os.environ.get("RECALL_AML_AUTHORIZED_USER_ID", "")
        if not authorized_user_id.strip():
            raise RuntimeError(
                "missing required hosted setting: RECALL_AML_AUTHORIZED_USER_ID; "
                "the shared API key must be bound to one user, or to '*' for an evaluation "
                "platform that sends a different user_id per sample"
            )
        return cls(
            **required,
            authorized_user_id=authorized_user_id,
            table=os.environ.get("RECALL_AML_TABLE", "recall_chunks"),
            generation_id=os.environ.get("RECALL_AML_GENERATION", "aml-hosted-v1"),
            host=os.environ.get("RECALL_AML_HOST", "127.0.0.1"),
            port=int(os.environ.get("RECALL_AML_PORT", "18004")),
            add_concurrency=int(os.environ.get("RECALL_AML_ADD_CONCURRENCY", "16")),
            search_concurrency=int(os.environ.get("RECALL_AML_SEARCH_CONCURRENCY", "16")),
            context_chars=int(os.environ.get("RECALL_AML_CONTEXT_CHARS", "7000")),
            variant_name=os.environ.get("RECALL_AML_VARIANT", DEFAULT_VARIANT),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
            splade_device=os.environ.get("RECALL_AML_SPLADE_DEVICE", "cpu"),
            splade_threads=int(os.environ.get("RECALL_AML_SPLADE_THREADS", "4")),
            embedding_lock_path=Path(embedding_lock_path),
            embedding_cache_path=(
                Path(value)
                if (value := os.environ.get("RECALL_AML_EMBED_CACHE_PATH"))
                else None
            ),
        )
