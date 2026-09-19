"""Starlette application exposing the AML hosted memory contract."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from typing import Any, Awaitable, Callable

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from recall.errors import IdempotencyConflict
from recall_aml.compiler import anchor_prompt_digest, facet_prompt_digest, prompt_digest
from recall_aml.config import (
    EMBEDDING_PROFILE,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    PRODUCT_NAME,
    PRODUCT_VERSION,
    RERANK_MODEL,
    RERANK_PRICE_SOURCE_DATE,
    RERANK_PRICE_SOURCE_URL,
    RERANK_PRICE_USD_PER_MILLION_TOKENS,
    RETRIEVAL_PROFILE,
    SCHEMA_VERSION,
    SPARSE_MODEL,
    SPARSE_REVISION,
    HostedSettings,
)
from recall_aml.models import AddRequest, DeleteRequest, SearchRequest
from recall_aml.retrieval import (
    CANDIDATE_WIDTH,
    CODE_NEIGHBOUR_PREDECESSOR_RADIUS,
    CODE_NEIGHBOUR_SEED_LIMIT,
    CODE_NEIGHBOUR_SUCCESSOR_RADIUS,
    CODE_PROFILE,
    CODE_RRF_WEIGHT,
    RRF_CONSTANT,
)
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
            started = time.perf_counter()
            model = SearchRequest.model_validate_json(json.dumps(await _payload(request)))
            async with search_slots:
                result = await service.search(model)
            search_ms = (time.perf_counter() - started) * 1_000
            return JSONResponse(
                result.model_dump(mode="json"),
                status_code=200,
                headers={
                    "X-Recall-Facet-Fallback": str(int(result.facet_fallback)),
                    "X-Recall-Reranker-Fallback": str(int(result.reranker_fallback)),
                    "X-Recall-Task-Type": result.task_type,
                    "X-Recall-Reranker-Attempted": str(int(result.reranker_attempted)),
                    "X-Recall-Reranker-Completed": str(int(result.reranker_completed)),
                    "X-Recall-Reranker-Provider": result.reranker_provider,
                    "X-Recall-Reranker-Model": result.reranker_model,
                    "X-Recall-Reranker-Input-Count": str(result.candidate_input_count),
                    "X-Recall-Reranker-Output-Count": str(result.candidate_output_count),
                    "X-Recall-Reranker-Permutation-Valid": str(
                        int(result.candidate_permutation_valid)
                    ),
                    "X-Recall-Reranker-Top10-Order-Changed": str(int(result.top_10_order_changed)),
                    "X-Recall-Reranker-Top10-Membership-Changed": str(
                        int(result.top_10_membership_changed)
                    ),
                    "X-Recall-Reranker-Top100-Order-Changed": str(
                        int(result.top_100_order_changed)
                    ),
                    "X-Recall-Reranker-Top100-Membership-Changed": str(
                        int(result.top_100_membership_changed)
                    ),
                    "X-Recall-Reranker-Ms": f"{result.rerank_ms:.3f}",
                    "X-Recall-Search-Ms": f"{search_ms:.3f}",
                    "X-Recall-Reranker-Candidate-Chars": str(result.candidate_character_count),
                    "X-Recall-Reranker-Estimated-Cost-USD": (
                        f"{result.estimated_reranker_cost_usd:.12f}"
                    ),
                    "X-Recall-Served-Commit": settings.git_commit,
                    "X-Recall-Generation": result.generation_id,
                    "X-Recall-Corpus-SHA256": result.corpus_sha256,
                    "X-Recall-Variant": service.variant_name,
                    "X-Recall-Code-Aware-Attempted": str(int(result.code_aware_attempted)),
                    "X-Recall-Code-Aware-Fallback": str(int(result.code_aware_fallback)),
                    "X-Recall-Code-Profile": result.code_profile,
                    "X-Recall-Code-RRF-Weight": f"{result.code_rrf_weight:.6f}",
                    "X-Recall-Code-Query-Tokens": str(result.code_query_token_count),
                    "X-Recall-Code-Match-Candidates": str(result.code_match_candidate_count),
                    "X-Recall-Code-Top10-Order-Changed": str(int(result.code_top_10_order_changed)),
                    "X-Recall-Code-Top10-Membership-Changed": str(
                        int(result.code_top_10_membership_changed)
                    ),
                    "X-Recall-Code-Top100-Order-Changed": str(
                        int(result.code_top_100_order_changed)
                    ),
                    "X-Recall-Code-Top100-Membership-Changed": str(
                        int(result.code_top_100_membership_changed)
                    ),
                    "X-Recall-Neighbour-Seed-Limit": str(result.neighbour_seed_limit),
                    "X-Recall-Neighbour-Seeds": str(result.neighbour_seed_count),
                    "X-Recall-Neighbour-Activated-Seeds": str(
                        result.neighbour_activated_seed_count
                    ),
                    "X-Recall-Neighbour-Ineligible-Seeds": str(
                        result.neighbour_ineligible_seed_count
                    ),
                    "X-Recall-Neighbour-Restored": str(result.neighbour_restored_count),
                    "X-Recall-Neighbour-Invalid": str(result.neighbour_invalid_count),
                    "X-Recall-Code-Duplicate-Outputs": str(result.code_duplicate_output_count),
                },
            )

        return await protected(request, run)

    async def delete(request: Request) -> Response:
        async def run() -> Response:
            model = DeleteRequest.model_validate_json(json.dumps(await _payload(request)))
            deleted = await service.delete_user(model.user_id)
            return JSONResponse({"status": "deleted", "deleted_count": deleted})

        return await protected(request, run)

    async def sparse_backfill(request: Request) -> Response:
        async def run() -> Response:
            model = DeleteRequest.model_validate_json(json.dumps(await _payload(request)))
            detail = await service.prepare_sparse_user(model.user_id)
            return JSONResponse({"status": "ready", **detail})

        return await protected(request, run)

    async def corpus_status(request: Request) -> Response:
        async def run() -> Response:
            model = DeleteRequest.model_validate_json(json.dumps(await _payload(request)))
            detail = await service.corpus_status(model.user_id)
            return JSONResponse(
                {
                    "status": "ready",
                    "variant": service.variant_name,
                    "served_commit": settings.git_commit,
                    **detail,
                }
            )

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
                "generation_id": settings.generation_id,
                "embedding_profile": EMBEDDING_PROFILE,
                "retrieval_profile": RETRIEVAL_PROFILE,
                "generation_provider": GENERATION_PROVIDER,
                "generation_model": GENERATION_MODEL,
                "reranker": RERANK_MODEL,
                "reranker_provider": RERANK_MODEL.split(":", 1)[0],
                "reranker_model": RERANK_MODEL.split(":", 1)[1],
                "reranker_price_usd_per_million_tokens": (RERANK_PRICE_USD_PER_MILLION_TOKENS),
                "reranker_price_source_date": RERANK_PRICE_SOURCE_DATE,
                "reranker_price_source_url": RERANK_PRICE_SOURCE_URL,
                "candidate_width": CANDIDATE_WIDTH,
                "rrf_constant": RRF_CONSTANT,
                "code_profile": CODE_PROFILE,
                "code_rrf_weight": CODE_RRF_WEIGHT,
                "code_neighbour_seed_limit": CODE_NEIGHBOUR_SEED_LIMIT,
                "code_neighbour_predecessor_radius": CODE_NEIGHBOUR_PREDECESSOR_RADIUS,
                "code_neighbour_successor_radius": CODE_NEIGHBOUR_SUCCESSOR_RADIUS,
                "sparse_model": SPARSE_MODEL,
                "sparse_revision": SPARSE_REVISION,
                "compiler_prompt_digest": prompt_digest(),
                "anchor_compiler_prompt_digest": anchor_prompt_digest(),
                "facet_prompt_digest": facet_prompt_digest(),
                "variant": service.variant_name,
                "compiled_kinds": service.compiled_kinds,
                "drop_compiler_fallback": service.drops_compiler_fallback,
            }
        )

    return Starlette(
        routes=[
            Route("/v1/add", add, methods=["POST"]),
            Route("/v1/search", search, methods=["POST"]),
            Route("/v1/delete", delete, methods=["POST"]),
            Route("/v1/sparse/backfill", sparse_backfill, methods=["POST"]),
            Route("/v1/corpus/status", corpus_status, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/version", version, methods=["GET"]),
        ]
    )
