"""The rule that tells a memo the indexer pruned from one a forget erased, by chunk id."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

#: The audit record the indexer's prune leaves: the chunk ids of a memo gone from disk.
PRUNE_SOURCE_CHUNKS = "prune_source_chunks"
#: The audit record a forget leaves: the chunk ids of a memo it erased.
FORGET_SOURCE_CHUNKS = "forget_source_chunks"
#: The audit record a forget leaves: the card names that stood for a memo it erased.
FORGET_SOURCE_ALIASES = "forget_source_aliases"


class ChunkRecord(NamedTuple):
    """The latest prune or forget record of one chunk: which it was, and the absolute source."""

    kind: str
    source: str


def latest_chunk_records(rows: Iterable[Sequence[Any]]) -> dict[str, ChunkRecord]:
    """`{chunk_id: ChunkRecord}` from `(chunk_id, event_type, source)` rows of
    `prune_source_chunks` and `forget_source_chunks` records, in the order they were appended:
    the last record of a chunk decides whether its memo was forgotten or only pruned.

    The one rule both stores apply (`LiteStore.latest_chunk_records` and
    `recall.store_receipts.ReceiptsAuditMixin.latest_chunk_records`, CCA CODE-001 of run
    20261106T212835Z); each orders its rows by append, never by a clock an eval run sets.
    Its own module, importing nothing of recall: the stores import it, and the AML service,
    which imports a store, must not reach `recall.eval` through `recall.fact_retraction`.
    """
    latest: dict[str, ChunkRecord] = {}
    for chunk_id, kind, source in rows:
        latest[str(chunk_id)] = ChunkRecord(str(kind), str(source or ""))
    return latest


def sources_by_name(rows: Iterable[Sequence[Any]]) -> dict[str, set[str]]:
    """`{name: {source, ...}}` from `(name, source)` rows of `forget_source_aliases` records,
    skipping a row with no source: the fold both stores' `sources_forgotten_as` apply (CCA
    CODE-002 of run 20261107T124651Z)."""
    found: dict[str, set[str]] = {}
    for name, source in rows:
        if source:
            found.setdefault(str(name), set()).add(str(source))
    return found
