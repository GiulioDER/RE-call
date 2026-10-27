"""Index a memory directory into the legacy store: what the session-end hook runs.

Moved out of `recall.setup` so `recall_hooks` reaches it without importing the installer,
which imported the hooks back. `recall.setup` re-exports every name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from recall._env import env_is_production
from recall.embeddings import resolve_embedder
from recall.store import scrub_dsn_secrets

DEFAULT_MEMORY_DIR = Path("memory")


def _safe_error(exc: Exception, dsn: str) -> str:
    return scrub_dsn_secrets(f"{type(exc).__name__}: {exc}", dsn)


def index_memory_directory(
    *,
    dsn: str,
    embedder_name: str,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    tenant: str | None = None,
    table: str | None = None,
    env: dict[str, str] | None = None,
    print_fn: Callable[..., None] = print,
) -> None:
    if env_is_production(env):
        print_fn(
            f"Skipping auto-index: RECALL_ENV is production. Index {memory_dir} via your "
            "production build pipeline instead."
        )
        return
    try:
        from recall.context import context_policy_for_profile
        from recall.cache import default_cache
        from recall.embeddings import embedding_profile_id
        from recall.index import Indexer, chunk_text
        from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, PgVectorStore

        embedder = resolve_embedder(embedder_name, env=env)
        with PgVectorStore(
            dsn,
            dim=embedder.dim,
            table=table or DEFAULT_TABLE,
            tenant=tenant or DEFAULT_TENANT,
        ) as store:
            store.check_schema()
            with default_cache() as cache:
                indexer = Indexer(
                    store,
                    embedder,
                    chunker=chunk_text,
                    cache=cache,
                    context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
                )
                stats = indexer.index_path(memory_dir, glob="**/*.md")
    except Exception as exc:  # best effort: scaffolded files must survive even if this fails  # BROAD-CATCH: fail-open
        print_fn(
            f"Could not auto-index {memory_dir}: {_safe_error(exc, dsn)} — run "
            f"'python -m recall.cli index {memory_dir}' once the schema is applied for this "
            "embedder's dimension."
        )
        return
    print_fn(f"Indexed {stats.chunks} chunks from {stats.files} files in {memory_dir}")
