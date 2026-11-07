"""The rule that tells a memo the indexer pruned from one a forget erased, by chunk id."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def latest_chunk_records(rows: Iterable[Sequence[Any]]) -> dict[str, tuple[str, str]]:
    """`{chunk_id: (event_type, source)}` from `(chunk_id, event_type, source)` rows of
    `prune_source_chunks` and `forget_source_chunks` records, in the order they were appended:
    the last record of a chunk decides whether its memo was forgotten or only pruned.

    The one rule both stores apply (`LiteStore.latest_chunk_records` and
    `recall.store_receipts.ReceiptsAuditMixin.latest_chunk_records`, CCA CODE-001 of run
    20261106T212835Z); each orders its rows by append, never by a clock an eval run sets.
    Its own module, importing nothing of recall: the stores import it, and the AML service,
    which imports a store, must not reach `recall.eval` through `recall.fact_retraction`.
    """
    latest: dict[str, tuple[str, str]] = {}
    for chunk_id, kind, source in rows:
        latest[str(chunk_id)] = (str(kind), str(source or ""))
    return latest
