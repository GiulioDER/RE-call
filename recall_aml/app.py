"""Starlette application exposing the AML hosted memory contract."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from recall.errors import IdempotencyConflict
from recall_aml.compiler import anchor_prompt_digest, facet_prompt_digest, prompt_digest
from recall_aml.config import (
    PLATFORM_SCOPE,
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
from recall_aml.graph import (
    GRAPH_MAX_PROMOTIONS,
    GRAPH_PROFILE,
    GRAPH_PROTECTED_PREFIX,
    GRAPH_SEED_K,
)
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
# 30 MiB decoded media expands to about 40 MiB as Base64. Leave bounded room for JSON and text.
MAX_BODY_BYTES = 44 * 1024 * 1024
ModelT = TypeVar("ModelT", bound=BaseModel)


def _authenticated(request: Request, expected: str) -> bool:
    authorization = request.headers.get("authorization", "")
    bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    api_key = request.headers.get("x-api-key", "")
    return any(
        value and hmac.compare_digest(value.encode(), expected.encode())
        for value in (bearer, api_key)
    )


def _authorized_user(request: Request, configured: str | None, requested: str) -> bool:
    """Bind the shared hosted credential to its configured principal.

    Directly constructed settings remain usable by unit tests, while production settings loaded
    from the environment fail closed in ``HostedSettings.from_env`` and always provide this
    binding.  No caller-controlled header is accepted as an identity substitute.

    ``PLATFORM_SCOPE`` is the one explicit exception: the key belongs to an evaluation platform
    that legitimately writes and searches under a different ``user_id`` per sample (the official
    AML run does), so it may act for any user. Isolation between those users is still enforced
    below this check, by the per-user tenant. It must be chosen on purpose; an unset binding
    still refuses to start.
    """
    if configured == PLATFORM_SCOPE:
        return True
    return configured is None or hmac.compare_digest(configured, requested)


def authorized_user_scope(configured: str | None) -> str:
    """How far the key reaches, for ``/version``; never the configured user id itself."""
    if configured is None:
        return "unbound"
    return "platform" if configured == PLATFORM_SCOPE else "single-user"


async def _payload(request: Request) -> Any:
    length = request.headers.get("content-length")
    if length is not None and int(length) > MAX_BODY_BYTES:
        raise ValueError("request body is too large")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("request body is too large")
    try:
        return _scrub_surrogates(json.loads(body))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _scrub_surrogates(value: Any) -> Any:
    """Replace every unpaired UTF-16 surrogate with U+FFFD.

    JSON may escape a lone surrogate (`"\\ud800"`) and `json.loads` keeps it, but pydantic's
    JSON parser refuses it and PostgreSQL cannot store it, so one stray code unit in a log line
    used to refuse the whole request. A string without one is returned as the same object.
    """
    if isinstance(value, str):
        if _LONE_SURROGATE.search(value) is None:
            return value
        return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if isinstance(value, list):
        return [_scrub_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {_scrub_surrogates(key): _scrub_surrogates(item) for key, item in value.items()}
    return value


class InvalidRequest(Exception):
    """The request itself cannot be read or validated: the only failure answered with 422.

    AML treats 422 as permanent, so it must mean "this payload will never be accepted". An
    exception raised later, inside the service, is a fault of ours or of a provider; it used to
    share the 422 branch whenever it happened to be a `ValueError`, which made it permanent and
    left no log line (the Code4 ordering error of C8 reached clients exactly that way).
    """


async def _parse(request: Request, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(json.dumps(await _payload(request)))
    except (ValidationError, ValueError) as exc:
        raise InvalidRequest(str(exc)) from exc


def _blank_query(query: object) -> bool:
    return not query.strip() if isinstance(query, str) else not query


def create_app(
    settings: HostedSettings,
    service: HostedService,
    *,
    shutdown: Callable[[], None] | None = None,
) -> Starlette:
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
        except InvalidRequest as exc:
            return JSONResponse({"error": "invalid_request", "detail": str(exc)}, status_code=422)
        except Exception as exc:  # BROAD-CATCH: public error translation without content leakage
            log.error("hosted_request_failed", extra={"error_class": type(exc).__name__})
            return JSONResponse({"error": "service_unavailable"}, status_code=503)

    async def add(request: Request) -> Response:
        async def run() -> Response:
            async with add_slots:
                model = await _parse(request, AddRequest)
                if not _authorized_user(request, settings.authorized_user_id, model.user_id):
                    return JSONResponse({"error": "forbidden"}, status_code=403)
                result = await service.add(model)
            return JSONResponse(result.model_dump(mode="json"), status_code=200)

        return await protected(request, run)

    async def search(request: Request) -> Response:
        async def run() -> Response:
            started = time.perf_counter()
            async with search_slots:
                model = await _parse(request, SearchRequest)
                if not _authorized_user(request, settings.authorized_user_id, model.user_id):
                    return JSONResponse({"error": "forbidden"}, status_code=403)
                if model.top_k == 0 or _blank_query(model.query):
                    # Nothing can be asked of retrieval, and an empty list is inside the
                    # contract; refusing it would fail the sample permanently instead.
                    return JSONResponse({"data": []}, status_code=200)
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
                    "X-Recall-Specialist-Route": result.specialist_route,
                    "X-Recall-Specialist-Embedding-Profile": (
                        result.specialist_embedding_profile
                    ),
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
                    "X-Recall-Graph-Attempted": str(int(result.graph_attempted)),
                    "X-Recall-Graph-Fallback": str(int(result.graph_fallback)),
                    "X-Recall-Graph-Profile": result.graph_profile,
                    "X-Recall-Graph-Relation-Hits": str(result.graph_relation_hits),
                    "X-Recall-Graph-Candidates": str(result.graph_candidate_count),
                    "X-Recall-Graph-Promoted": str(result.graph_promoted_count),
                    "X-Recall-Graph-Invalid-Relations": str(
                        result.graph_invalid_relation_count
                    ),
                    "X-Recall-Graph-Top10-Order-Changed": str(
                        int(result.graph_top_10_order_changed)
                    ),
                    "X-Recall-Graph-Top100-Membership-Changed": str(
                        int(result.graph_top_100_membership_changed)
                    ),
                    "X-Recall-Atomic-Rescue-Attempted": str(
                        int(result.atomic_rescue_attempted)
                    ),
                    "X-Recall-Atomic-Rescue-Active": str(
                        int(result.atomic_rescue_active)
                    ),
                    "X-Recall-Atomic-Rescue-Fallback": str(
                        int(result.atomic_rescue_fallback)
                    ),
                    "X-Recall-Atomic-Rescue-Candidate-Available": str(
                        int(result.atomic_rescue_candidate_available)
                    ),
                },
            )

        return await protected(request, run)

    async def delete(request: Request) -> Response:
        async def run() -> Response:
            model = await _parse(request, DeleteRequest)
            if not _authorized_user(request, settings.authorized_user_id, model.user_id):
                return JSONResponse({"error": "forbidden"}, status_code=403)
            deleted = await service.delete_user(model.user_id)
            return JSONResponse({"status": "deleted", "deleted_count": deleted})

        return await protected(request, run)

    async def sparse_backfill(request: Request) -> Response:
        async def run() -> Response:
            model = await _parse(request, DeleteRequest)
            if not _authorized_user(request, settings.authorized_user_id, model.user_id):
                return JSONResponse({"error": "forbidden"}, status_code=403)
            detail = await service.prepare_sparse_user(model.user_id)
            return JSONResponse({"status": "ready", **detail})

        return await protected(request, run)

    async def corpus_status(request: Request) -> Response:
        async def run() -> Response:
            model = await _parse(request, DeleteRequest)
            if not _authorized_user(request, settings.authorized_user_id, model.user_id):
                return JSONResponse({"error": "forbidden"}, status_code=403)
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
                "embedding_profile": service.embedding_profile,
                "retrieval_profile": RETRIEVAL_PROFILE,
                "lexical_profile": service.lexical_profile,
                "word_window_size": service.word_window_size,
                "word_window_stride": service.word_window_stride,
                "exact_dense": service.exact_dense,
                "ordering_profile": service.ordering_profile,
                "window_renderer_profile": service.window_renderer_profile,
                "search_content_profile": service.search_content_profile,
                "anchor_prior_records": service.anchor_prior_records,
                "active_components": service.active_components,
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
                "graph_profile": GRAPH_PROFILE,
                "graph_seed_k": GRAPH_SEED_K,
                "graph_protected_prefix": GRAPH_PROTECTED_PREFIX,
                "graph_max_promotions": GRAPH_MAX_PROMOTIONS,
                "sparse_model": SPARSE_MODEL,
                "sparse_revision": SPARSE_REVISION,
                "compiler_prompt_digest": prompt_digest(),
                "anchor_compiler_prompt_digest": anchor_prompt_digest(),
                "facet_prompt_digest": facet_prompt_digest(),
                "variant": service.variant_name,
                "compiled_kinds": service.compiled_kinds,
                "drop_compiler_fallback": service.drops_compiler_fallback,
                "multimodal_preserve": service.multimodal_preserve,
                "multimodal_native": service.multimodal_native,
                "multimodal_embedding_profile": service.multimodal_embedding_profile,
                "multimodal_embedding_model": service.multimodal_embedding_model,
                "context_specialist": service.context_specialist,
                "context_embedding_profile": service.context_embedding_profile,
                "specialist_router_profile": service.specialist_router_profile,
                "specialist_fusion_profile": service.specialist_fusion_profile,
                "embedding_call_lock": settings.embedding_lock_path is not None,
                "embedding_cache": settings.embedding_cache_path is not None,
                "graph_sidecar": service.graph_sidecar,
                "authorized_user_scope": authorized_user_scope(settings.authorized_user_id),
                "atomic_rescue": service.atomic_rescue_profile,
            }
        )

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if shutdown is not None:
                shutdown()

    return Starlette(
        routes=[
            Route("/v1/add", add, methods=["POST"]),
            Route("/v1/search", search, methods=["POST"]),
            Route("/v1/delete", delete, methods=["POST"]),
            Route("/v1/sparse/backfill", sparse_backfill, methods=["POST"]),
            Route("/v1/corpus/status", corpus_status, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/version", version, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
