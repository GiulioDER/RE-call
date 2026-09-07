"""Generation administration boundary for MCP and desktop clients.

Calibration operations are owned here. Generation ingest remains a compatibility forwarder until
its carry forward, cleanup, certification, and promotion helpers are moved as one safe slice.
"""

from __future__ import annotations

import psycopg
from collections.abc import Sequence
from typing import TYPE_CHECKING

from recall.calibration_v2 import CalibrationRepository

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore
    from recall_mcp.service import IndexResult


def generation_ingest(
    store: PgVectorStore,
    embedder: Embedder,
    staged_root: str,
    category: str,
) -> IndexResult:
    """Build, validate, and activate a staged generation."""
    from recall_mcp import service

    return service.generation_ingest(store, embedder, staged_root, category)


def _generated_calibration_queries(store: PgVectorStore, generation_id: str) -> list[dict[str, object]]:
    """Build a deterministic draft query set from the active corpus.

    This is intentionally a prototype helper. The generated labels are useful for checking the
    complete workflow, but a production deployment should replace them with reviewed labels.
    """
    with psycopg.connect(store._dsn, autocommit=True, connect_timeout=10) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,))
        rows = conn.execute(
            "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
            "ORDER BY chunk_id LIMIT 20",
            (store.tenant, generation_id),
        ).fetchall()
    answerable: list[str] = []
    for row in rows:
        value = str(row[0]).strip()
        if value and value not in answerable:
            answerable.append(value[:500])
    if len(answerable) < 2:
        raise ValueError("at least two distinct corpus chunks are required to generate calibration labels")
    return [
        *({"query": query, "answerable": True} for query in answerable),
        *(
            {
                "query": f"Prototype calibration negative sample {index}: {nonce}",
                "answerable": False,
            }
            for index, nonce in enumerate(
                (
                    "the unrecorded weather on Europa",
                    "the private password for a fictional account",
                    "the exact weight of an imaginary blue comet",
                    "the inventory of a library that does not exist",
                    "the recipe for a machine never described here",
                    "the birthplace of a person absent from this corpus",
                    "the result of a future election",
                    "the serial number of a nonexistent device",
                    "the internal schedule of an unrelated company",
                    "the answer to an invented mathematical riddle",
                    "the color of a silent radio signal",
                    "the number of doors in an imaginary building",
                    "the owner of a fictional island",
                    "the temperature inside an empty thought",
                    "the name of a removed document",
                    "the location of a lost moon",
                    "the version of an unreleased program",
                    "the price of an unnamed object",
                    "the title of a nonexistent chapter",
                    "the identity of an imaginary maintainer",
                ),
                start=1,
            )
        ),
    ]


def run_calibration(
    store: PgVectorStore,
    embedder: Embedder,
    generation_id: str | None = None,
    queries: Sequence[dict[str, object]] | None = None,
    *,
    _generated_calibration_queries_fn=_generated_calibration_queries,
) -> dict[str, object]:
    """Measure a draft artifact, generating prototype labels when none were supplied."""
    from recall.generation_store import GenerationStore

    generation_store = GenerationStore(store._dsn, embedder.dim, tenant=store.tenant)
    try:
        selected_generation = generation_id or generation_store.active_generation_id()
    finally:
        generation_store.close()
    labels = list(queries) if queries is not None else _generated_calibration_queries_fn(store, selected_generation)
    artifact = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp").calibrate(
        selected_generation,
        labels,
        embedder,
    )
    return artifact.to_dict()


def publish_calibration(store: PgVectorStore, calibration_id: str) -> dict[str, object]:
    """Publish a certified artifact after the user explicitly confirms the action."""
    artifact = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp").publish(calibration_id)
    return artifact.to_dict()


__all__ = ["generation_ingest", "publish_calibration", "run_calibration"]
