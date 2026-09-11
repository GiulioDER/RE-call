"""Immutable runtime configuration for RE-call Hosted."""

from __future__ import annotations

from dataclasses import dataclass
import os

from recall_aml.variants import DEFAULT_VARIANT, variant


PRODUCT_NAME = "RE-call Hosted 1.0"
PRODUCT_VERSION = "1.0.0"
SCHEMA_VERSION = "0024"
EMBEDDING_PROFILE = "voyage-4"
RETRIEVAL_PROFILE = "hosted-quality"
GENERATION_PROVIDER = "openrouter"
GENERATION_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RERANK_MODEL = "voyage:rerank-2.5"


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

    def __post_init__(self) -> None:
        if not self.database_url or not self.api_key or not self.git_commit:
            raise ValueError("database_url, api_key, and git_commit must be non-empty")
        if not self.table.isidentifier():
            raise ValueError("table must be a valid SQL identifier")
        for name in ("port", "add_concurrency", "search_concurrency", "context_chars"):
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
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("missing required hosted settings: " + ", ".join(missing))
        return cls(
            **required,
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
        )
