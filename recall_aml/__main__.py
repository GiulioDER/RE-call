"""Production entry point for RE-call Hosted 1.0."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TextIO, cast

from recall.embeddings import Embedder, resolve_registered_embedder
from recall.pool import SharedPool
from recall.rerank import VoyageReranker
from recall.sparse import SpladeEncoder
from recall.store import PgVectorStore
from recall_aml.app import create_app
from recall_aml.compiler import OpenAICompiler
from recall_aml.config import (
    HostedSettings,
    OPENROUTER_BASE_URL,
    SPARSE_MODEL,
    SPARSE_REVISION,
)
from recall_aml.embedding_lock import (
    CachedEmbedder,
    CachedMultimodalEmbedder,
    LockedEmbedder,
    LockedMultimodalEmbedder,
    embedding_call_lock,
)
from recall_aml.retrieval import HostedRetriever
from recall_aml.readiness import verify_model_readiness
from recall_aml.service import HostedService
from recall_aml.storage import PgHostedRepository
from recall_aml.multimodal import MultimodalEmbedder, VoyageMultimodalEmbedder
from recall_aml.variants import HostedVariant, variant


def build_openrouter_client(api_key: str, *, factory: Any = None) -> Any:
    if factory is None:
        from openai import OpenAI

        factory = OpenAI
    return factory(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=20.0,
        max_retries=0,
    )


def _resolve_hosted_embedders(
    settings: HostedSettings, behavior: HostedVariant
) -> tuple[Embedder, dict[str, Embedder]]:
    """Construct provider backed embedders while covering their live SDK probes."""
    assert settings.voyage_api_key is not None
    provider_env = {"VOYAGE_API_KEY": settings.voyage_api_key}
    if (timeout := os.environ.get("RECALL_VOYAGE_TIMEOUT_SECONDS")) is not None:
        provider_env["RECALL_VOYAGE_TIMEOUT_SECONDS"] = timeout
    with embedding_call_lock(settings.embedding_lock_path):
        embedder = resolve_registered_embedder(
            behavior.embedding_profile, provider_env
        )
        specialist_embedders: dict[str, Embedder] = {}
        if behavior.context_specialist:
            context_embedder = resolve_registered_embedder(
                behavior.context_embedding_profile,
                provider_env,
            )
            specialist_embedders[behavior.context_embedding_profile] = context_embedder
    if settings.embedding_lock_path is not None:
        embedder = LockedEmbedder(embedder, settings.embedding_lock_path)
        specialist_embedders = {
            profile: LockedEmbedder(specialist, settings.embedding_lock_path)
            for profile, specialist in specialist_embedders.items()
        }
    if settings.embedding_cache_path is not None:
        # Keep cache hits outside the provider lock. A miss still delegates through the locked
        # inner embedder, so concurrent processes serialize provider calls without serializing
        # local SQLite reads.
        embedder = CachedEmbedder(embedder, settings.embedding_cache_path)
        specialist_embedders = {
            profile: CachedEmbedder(specialist, settings.embedding_cache_path)
            for profile, specialist in specialist_embedders.items()
        }
    return embedder, specialist_embedders


def build_app(settings: HostedSettings | None = None) -> Any:
    settings = settings or HostedSettings.from_env()
    behavior = variant(settings.variant_name)
    if not settings.voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is required")
    if settings.embedding_lock_path is None:
        raise RuntimeError("RECALL_AML_EMBED_LOCK_PATH is required for hosted production")
    if (behavior.compiler or behavior.facets) and not settings.openrouter_api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY is required for {behavior.name}")
    embedder, specialist_embedders = _resolve_hosted_embedders(settings, behavior)
    sparse_encoder = None
    if behavior.learned_sparse:
        import torch

        torch.set_num_threads(settings.splade_threads)
        sparse_encoder = SpladeEncoder.from_pretrained(
            SPARSE_MODEL,
            revision=SPARSE_REVISION,
            device=settings.splade_device,
        )
    pool = SharedPool(
        settings.database_url,
        min_size=1,
        max_size=max(36, settings.add_concurrency + settings.search_concurrency + 2),
        statement_timeout_ms=25_000,
    )
    try:
        store = PgVectorStore(
            settings.database_url,
            embedder.dim,
            table=settings.table,
            tenant="aml_service_readiness",
            generation_id=settings.generation_id,
            shared_pool=pool,
        )
        store.check_schema()
    except BaseException:
        pool.close()
        raise
    repository = PgHostedRepository(
        store,
        embedder,
        sparse_encoder,
        specialist_embedders=specialist_embedders,
    )
    compiler = (
        OpenAICompiler(build_openrouter_client(settings.openrouter_api_key))
        if settings.openrouter_api_key
        else None
    )
    reranker = VoyageReranker(model="rerank-2.5", api_key=settings.voyage_api_key)
    multimodal_embedder: MultimodalEmbedder | None = (
        VoyageMultimodalEmbedder(settings.voyage_api_key)
        if behavior.multimodal_native
        else None
    )
    if multimodal_embedder is not None and settings.embedding_lock_path is not None:
        multimodal_embedder = cast(
            MultimodalEmbedder,
            LockedMultimodalEmbedder(multimodal_embedder, settings.embedding_lock_path),
        )
    if multimodal_embedder is not None and settings.embedding_cache_path is not None:
        multimodal_embedder = cast(
            MultimodalEmbedder,
            CachedMultimodalEmbedder(multimodal_embedder, settings.embedding_cache_path),
        )
    try:
        readiness = verify_model_readiness(
            embedder=embedder,
            compiler=compiler,
            reranker=reranker,
            sparse_encoder=sparse_encoder,
            multimodal_embedder=multimodal_embedder,
            specialist_embedders=specialist_embedders,
            behavior=behavior,
        )
    except BaseException:
        pool.close()
        raise
    retriever = HostedRetriever(embedder, reranker, sparse_encoder=sparse_encoder)
    specialist_retrievers = {
        profile: HostedRetriever(specialist, reranker)
        for profile, specialist in specialist_embedders.items()
    }
    service = HostedService(
        repository,
        compiler,
        retriever,
        context_chars=settings.context_chars,
        behavior=behavior,
        multimodal_embedder=multimodal_embedder,
        specialist_retrievers=specialist_retrievers,
        model_clients_ready=all(
            readiness[name]
            for name in ("embedder_ready",)
            + (("compiler_ready",) if behavior.compiler or behavior.facets else ())
            + (("reranker_ready",) if behavior.reranker else ())
            + (("sparse_ready",) if behavior.learned_sparse else ())
            + (("multimodal_ready",) if behavior.multimodal_native else ())
            + (
                (f"specialist:{behavior.context_embedding_profile}",)
                if behavior.context_specialist
                else ()
            )
        ),
    )
    return create_app(settings, service, shutdown=pool.close)


_RECORD_ATTRIBUTES = frozenset(
    logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__
) | {"message", "asctime"}


class ExtraFieldsFormatter(logging.Formatter):
    """Default text format plus each record's ``extra`` fields as one JSON object.

    ``logging.basicConfig`` renders only ``levelname:name:message``, so every field passed through
    ``extra`` (latency, route, fallback flags, error class) was dropped from the journal for the
    whole official AML Multimodal run ``teval_dcc1109c4331c3e3``. The prefix is unchanged, so
    existing journal searches keep matching. A message that already ends with the identical
    serialization (``recall_aml.compiler._log_diagnostics``) is not repeated.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        line = super().formatMessage(record)
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RECORD_ATTRIBUTES and not key.startswith("_")
        }
        if not fields:
            return line
        rendered = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
        if record.message.endswith(rendered):
            return line
        return f"{line} {rendered}"


def configure_logging(stream: TextIO | None = None) -> None:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ExtraFieldsFormatter(logging.BASIC_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def main() -> None:
    import uvicorn

    configure_logging()
    settings = HostedSettings.from_env()
    uvicorn.run(build_app(settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
