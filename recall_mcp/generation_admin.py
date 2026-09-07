"""Generation administration boundary for MCP and desktop clients.

This is the first compatibility seam for generation lifecycle operations.  The service module
continues to provide the old names, while new callers depend on this narrower domain boundary.
The lazy forwarding keeps import order stable until the implementation is moved here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore
    from recall.security_policy import AccessContext, SourceSecurityPolicy
    from recall_mcp.service import IndexResult


def generation_ingest(
    store: PgVectorStore,
    embedder: Embedder,
    staged_root: str,
    category: str,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
) -> IndexResult:
    """Build, validate, and activate a staged generation."""
    from recall_mcp import service

    args = (store, embedder, staged_root, category)
    if security_policy is None and security_context is None:
        return service.generation_ingest(*args)
    return service.generation_ingest(
        *args, security_policy=security_policy, security_context=security_context
    )


def run_calibration(
    store: PgVectorStore,
    embedder: Embedder,
    generation_id: str | None = None,
    queries: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Measure a calibration artifact for a generation."""
    from recall_mcp import service

    return service.run_calibration(store, embedder, generation_id, queries)


def publish_calibration(store: PgVectorStore, calibration_id: str) -> dict[str, object]:
    """Publish a previously measured calibration artifact."""
    from recall_mcp import service

    return service.publish_calibration(store, calibration_id)


__all__ = ["generation_ingest", "publish_calibration", "run_calibration"]
