"""Production entry point for RE-call Hosted 1.0."""

from __future__ import annotations

import logging

from recall.embeddings import VoyageEmbedder
from recall.pool import SharedPool
from recall.rerank import VoyageReranker
from recall.store import PgVectorStore
from recall_aml.app import create_app
from recall_aml.compiler import OpenAICompiler
from recall_aml.config import HostedSettings, OPENROUTER_BASE_URL
from recall_aml.retrieval import HostedRetriever
from recall_aml.readiness import verify_model_readiness
from recall_aml.service import HostedService
from recall_aml.storage import PgHostedRepository
from recall_aml.variants import variant


def build_openrouter_client(api_key: str, *, factory=None):
    if factory is None:
        from openai import OpenAI

        factory = OpenAI
    return factory(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=20.0,
        max_retries=0,
    )


def build_app(settings: HostedSettings | None = None):
    settings = settings or HostedSettings.from_env()
    behavior = variant(settings.variant_name)
    if not settings.voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is required")
    if (behavior.compiler or behavior.facets) and not settings.openrouter_api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY is required for {behavior.name}")
    embedder = VoyageEmbedder(model="voyage-4", api_key=settings.voyage_api_key)
    pool = SharedPool(
        settings.database_url,
        min_size=1,
        max_size=max(36, settings.add_concurrency + settings.search_concurrency + 2),
        statement_timeout_ms=25_000,
    )
    store = PgVectorStore(
        settings.database_url,
        embedder.dim,
        table=settings.table,
        tenant="aml_service_readiness",
        generation_id=settings.generation_id,
        shared_pool=pool,
    )
    store.check_schema()
    repository = PgHostedRepository(store, embedder)
    compiler = (
        OpenAICompiler(build_openrouter_client(settings.openrouter_api_key))
        if settings.openrouter_api_key
        else None
    )
    reranker = VoyageReranker(model="rerank-2.5", api_key=settings.voyage_api_key)
    readiness = verify_model_readiness(
        embedder=embedder,
        compiler=compiler,
        reranker=reranker,
        behavior=behavior,
    )
    retriever = HostedRetriever(embedder, reranker)
    service = HostedService(
        repository,
        compiler,
        retriever,
        context_chars=settings.context_chars,
        behavior=behavior,
        model_clients_ready=all(
            readiness[name]
            for name in ("embedder_ready",)
            + (("compiler_ready",) if behavior.compiler or behavior.facets else ())
            + (("reranker_ready",) if behavior.reranker else ())
        ),
    )
    return create_app(settings, service)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    settings = HostedSettings.from_env()
    uvicorn.run(build_app(settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
