"""Retrieval application boundary for MCP and in process clients.

The implementation still lives in :mod:`recall_mcp.service` during this migration.  These
functions are the new owner for retrieval imports, while the forwarding calls preserve the old
``recall_mcp.service`` import path for clients and tests.  Keeping the delegation lazy avoids a
cycle while the remaining service operations are extracted in later slices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recall.calibration import Calibration
    from recall.embeddings import Embedder
    from recall.profiles import RetrievalProfile
    from recall.store import PgVectorStore
    from recall.trust_policy import TrustPolicy
    from recall.security_policy import AccessContext, SourceSecurityPolicy
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
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> SearchResult:
    """Run retrieval through the legacy service implementation during extraction."""
    from recall_mcp import service

    args = (
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
    if security_policy is None and access_context is None:
        return service.search_memory(*args)
    return service.search_memory(*args, security_policy, access_context)


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
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
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
        security_policy,
        access_context,
    )


def startup_retrieval_profile(env: dict[str, str] | None = None) -> RetrievalProfile:
    """Resolve the serving profile through the compatibility implementation."""
    from recall_mcp import service

    return service.startup_retrieval_profile(env)


__all__ = ["evidence_memory", "search_memory", "startup_retrieval_profile"]
