"""Production entry point for RE-call Hosted 1.0."""

from __future__ import annotations

import logging

from recall.embeddings import VoyageEmbedder
from recall.pool import SharedPool
from recall.rerank import VoyageReranker
from recall.store import PgVectorStore
from recall_aml.app import create_app
from recall_aml.compiler import OpenAICompiler
from recall_aml.config import HostedSettings
from recall_aml.retrieval import HostedRetriever
from recall_aml.service import HostedService
from recall_aml.storage import PgHostedRepository


def build_app(settings: HostedSettings | None = None):
    settings = settings or HostedSettings.from_env()
    if not settings.openai_api_key or not settings.voyage_api_key:
        raise RuntimeError("OPENAI_API_KEY and VOYAGE_API_KEY are required")
    from openai import OpenAI

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
    compiler = OpenAICompiler(OpenAI(api_key=settings.openai_api_key, timeout=20.0, max_retries=0))
    retriever = HostedRetriever(
        embedder,
        VoyageReranker(model="rerank-2.5", api_key=settings.voyage_api_key),
    )
    service = HostedService(repository, compiler, retriever, context_chars=settings.context_chars)
    return create_app(settings, service)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    settings = HostedSettings.from_env()
    uvicorn.run(build_app(settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
