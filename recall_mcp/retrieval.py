"""Retrieval application boundary for MCP and in process clients.

Profile startup is owned here.  Search and evidence remain forwarding façades until their
dependencies are moved in a later retrieval slice.  The legacy service import path is preserved
for callers during the migration.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from recall.profiles import FAST_PROFILE, RetrievalProfile, resolve_retrieval_profile
from recall.query_class import routing_mode
from recall.rerank import COREB_CODE_RERANKER_MODEL
from recall_mcp.factories import (
    _positive_env,
    _require_remote_model_code_enabled,
    _validate_quality_reranker_config,
    resolve_reranker,
)

if TYPE_CHECKING:
    from recall.calibration import Calibration
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore
    from recall.trust_policy import TrustPolicy
    from recall_mcp.service import EvidenceResult, SearchResult


def search_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    include_related: bool = False,
    related_relation: str = "source",
    related_max_items: int = 3,
    reasoning_available: bool = False,
) -> SearchResult:
    """Run retrieval through the legacy service implementation during extraction."""
    from recall_mcp import service

    return service.search_memory(
        store,
        embedder,
        query,
        source,
        k,
        calibration,
        policy,
        explain,
        include_related,
        related_relation,
        related_max_items,
        reasoning_available,
    )


def evidence_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
    max_items: int | None = None,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    include_related: bool = False,
    related_relation: str = "source",
    related_max_items: int = 3,
) -> EvidenceResult:
    """Build generator neutral evidence through the legacy service implementation."""
    from recall_mcp import service

    return service.evidence_memory(
        store,
        embedder,
        query,
        source,
        k,
        max_items,
        calibration,
        policy,
        explain,
        include_related,
        related_relation,
        related_max_items,
    )


def startup_retrieval_profile(env: dict[str, str] | None = None) -> RetrievalProfile:
    """Resolve and fully validate the process profile. Called once, at server startup.

    Resolution alone used to happen on the first search, which meant a contradictory
    ``RECALL_RETRIEVAL_PROFILE`` / ``RECALL_RERANK`` pair, or a quality profile with no pinned
    reranker artifact, produced a server that started clean and failed on its first client
    request. Startup validation keeps that failure at startup.

    This deliberately does not import torch or load the model. It runs before the store is opened,
    and a configuration error should be reported in milliseconds. The artifact itself is verified
    when the reranker is built.
    """
    values = dict(os.environ) if env is None else env
    selected_routing_mode = routing_mode(values.get("RECALL_ROUTING_MODE", "shadow"))
    profile = resolve_retrieval_profile(values)
    if selected_routing_mode == "active" and profile.name == "legacy":
        # Active routing may select QUALITY_PROFILE on temporal and status queries even when no
        # process profile was configured. Validate that artifact at startup and size the worker
        # pool for FAST_PROFILE, the larger of the two active admission pools.
        _validate_quality_reranker_config(values)
        return FAST_PROFILE
    if profile.name == "quality":
        _validate_quality_reranker_config(values)
    elif profile.name == "code":
        rerank_values = dict(values)
        rerank_values.setdefault("RECALL_RERANK", "1")
        rerank_values.setdefault("RECALL_RERANK_MODEL", "coreb-code")
        spec = resolve_reranker(rerank_values)
        assert spec is not None
        if spec[0] != COREB_CODE_RERANKER_MODEL:
            raise ValueError("the code retrieval profile requires RECALL_RERANK_MODEL=coreb-code")
        _require_remote_model_code_enabled(values, "coreb-code")
        _positive_env(values, "RECALL_RERANK_BATCH_SIZE", 4)
    return profile


__all__ = ["evidence_memory", "search_memory", "startup_retrieval_profile"]
