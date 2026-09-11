"""Starlette application exposing the AML hosted memory contract."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from typing import Any, Awaitable, Callable

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from recall.errors import IdempotencyConflict
from recall_aml.compiler import prompt_digest
from recall_aml.config import (
    EMBEDDING_PROFILE,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    PRODUCT_NAME,
    PRODUCT_VERSION,
    RERANK_MODEL,
    RETRIEVAL_PROFILE,
    SCHEMA_VERSION,
    HostedSettings,
)
from recall_aml.models import AddRequest, DeleteRequest, SearchRequest
from recall_aml.service import HostedService


log = logging.getLogger("recall_aml")
MAX_BODY_BYTES = 2_000_000


def _authenticated(request: Request, expected: str) -> bool:
    authorization = request.headers.get("authorization", "")
    bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    api_key = request.headers.get("x-api-key", "")
    return any(
        value and hmac.compare_digest(value.encode(), expected.encode())
        for value in (bearer, api_key)
    )


async def _payload(request: Request) -> Any:
    length = request.headers.get("content-length")
    if length is not None and int(length) > MAX_BODY_BYTES:
        raise ValueError("request body is too large")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("request body is too large")
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc


def create_app(settings: HostedSettings, service: HostedService) -> Starlette:
    add_slots = asyncio.Semaphore(settings.add_concurrency)
    search_slots = asyncio.Semaphore(settings.search_concurrency)

    async def protected(
        request: Request,
        operation: Callable[[], Awaitable[Response]],
    ) -> Response:
        if not _authenticated(request, settings.api_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            return await operation()
        except IdempotencyConflict:
            return JSONResponse({"error": "request_id conflict"}, status_code=409)
        except (ValidationError, ValueError) as exc:
            return JSONResponse({"error": "invalid_request", "detail": str(exc)}, status_code=422)
        except Exception as exc:  # BROAD-CATCH: public error translation without content leakage
            log.error("hosted_request_failed", extra={"error_class": type(exc).__name__})
            return JSONResponse({"error": "service_unavailable"}, status_code=503)

    async def add(request: Request) -> Response:
        async def run() -> Response:
            model = AddRequest.model_validate_json(json.dumps(await _payload(request)))
            async with add_slots:
                result = await service.add(model)
            return JSONResponse(result.model_dump(mode="json"), status_code=200)

        return await protected(request, run)

    async def search(request: Request) -> Response:
        async def run() -> Response:
            model = SearchRequest.model_validate_json(json.dumps(await _payload(request)))
            async with search_slots:
                result = await service.search(model)
            return JSONResponse(
                result.model_dump(mode="json"),
                status_code=200,
                headers={
                    "X-Recall-Facet-Fallback": str(int(result.facet_fallback)),
                    "X-Recall-Reranker-Fallback": str(int(result.reranker_fallback)),
                },
            )

        return await protected(request, run)

    async def delete(request: Request) -> Response:
        async def run() -> Response:
            model = DeleteRequest.model_validate_json(json.dumps(await _payload(request)))
            deleted = await service.delete_user(model.user_id)
            return JSONResponse({"status": "deleted", "deleted_count": deleted})

        return await protected(request, run)

    async def health(_: Request) -> Response:
        try:
            detail = await service.health()
        except Exception as exc:  # BROAD-CATCH: dependency readiness is the health contract
            return JSONResponse(
                {"status": "unready", "error_class": type(exc).__name__}, status_code=503
            )
        return JSONResponse({"status": "ready", **detail})

    async def version(_: Request) -> Response:
        return JSONResponse(
            {
                "product": PRODUCT_NAME,
                "version": PRODUCT_VERSION,
                "git_commit": settings.git_commit,
                "schema_version": SCHEMA_VERSION,
                "embedding_profile": EMBEDDING_PROFILE,
                "retrieval_profile": RETRIEVAL_PROFILE,
                "generation_provider": GENERATION_PROVIDER,
                "generation_model": GENERATION_MODEL,
                "reranker": RERANK_MODEL,
                "compiler_prompt_digest": prompt_digest(),
                "variant": service.variant_name,
            }
        )

    return Starlette(
        routes=[
            Route("/v1/add", add, methods=["POST"]),
            Route("/v1/search", search, methods=["POST"]),
            Route("/v1/delete", delete, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/version", version, methods=["GET"]),
        ]
    )
