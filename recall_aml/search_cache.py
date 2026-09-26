"""Per-tenant Search caches that answer exactly what a fresh full-tenant read would.

Two things a hosted Search used to recompute from the whole tenant on every request:

* the Code4 BM25 ranking (``recall_aml.code4.rank_bm25_chunks`` over ``store.iter_chunks()``),
  which tokenised every chunk, rebuilt document frequencies and sorted every positive hit; on a
  13,097 memory tenant that was seconds of GIL-bound Python per Search;
* ``PgVectorStore.explicit_superseded_chunk_ids``, an unindexed JSONB scan of every row. Only the
  graph sidecar's scan goes through this cache: over compiled records, whose metadata is large, it
  measured 58 to 72 ms uncached against 3.5 to 4.7 ms cached (1,682 records). Over raw windows the
  fingerprint check costs as much as the scan it would save, so the raw scan stays direct.

Both are pure functions of the tenant's rows, so both are cached here per physical tenant and
served only while the rows provably have not changed.

**What "provably" means.** Every cached value carries the fingerprint of the rows it was read
from, computed from the SAME statement that read them, so a value and its fingerprint always
describe one snapshot. Every use first asks the database for the current fingerprint and serves
the cached value only on an exact match; anything else re-reads. The fingerprint is
``(row count, max(indexed_at), sum(xmin))`` over the tenant. An INSERT or DELETE moves the count;
an UPDATE (the upsert's ``ON CONFLICT DO UPDATE``) writes a new row version with a different
``xmin``, which almost always moves the sum. Almost, not always: ``xmin`` is not monotonic across
concurrent transactions or across wraparound, so a writer in ANOTHER process could in principle
leave count, latest time and sum all unchanged by exact coincidence, and the stale value would be
served until the next write to that tenant. The fingerprint is asked of the database on every
Search, so that coincidence is the whole residual risk from other processes. The in-process write
paths close it for this process: they mark a tenant suspect, which forces the per-row version
comparison below even when the aggregate happens to match.

**Why the BM25 ranking is bit-identical, not merely equal.** The snapshot keeps each document's
term counts (as postings) and token length, and ``Bm25Snapshot.rank`` evaluates the reference's
own expression, term by term in query order, with the same Python ints and floats. Documents that
contain no query term score exactly 0.0 in the reference and are dropped by its ``score > 0.0``
test, so scoring only the postings of query terms changes nothing. The final order uses the
reference's key, which ends in the chunk id and is therefore total, so ``heapq.nsmallest`` returns
exactly ``sorted(...)[:k]``. ``tests/test_aml_search_cache.py`` holds the parity proofs.

Memory and timings are measured by ``scripts/bench_aml_search_cache.py``, with the results
appended to ``docs/preregistrations/2026-09-26-c9-search-bm25-cache.md``.
"""

from __future__ import annotations

from array import array
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime
import heapq
import math
import threading
from typing import Any
from uuid import uuid4

from recall.store import PgVectorStore
from recall.types import Chunk, ScoredChunk
from recall_aml.code4 import BM25_B, BM25_K1, rank_bm25_chunks, stable_window_key, tokenize


#: How many tenants' BM25 snapshots one retriever keeps (least recently used beyond that). Measured
#: 2026-09-26 by ``scripts/bench_aml_search_cache.py`` on 13,097 Code4 windows: 48.3 MB of rows
#: (chunks, text, metadata) plus 20.5 MB of index, about 69 MB of Python heap per tenant. The
#: hosted process holds two retrievers (Code4 and Context), so four bounds it near 550 MB for
#: tenants of that size, while the official run searches one user at a time.
MAX_BM25_TENANTS = 4
#: Supersession results are a frozenset of ids, a few kilobytes at most per tenant.
MAX_SUPERSESSION_TENANTS = 1_024

Fingerprint = tuple[int, datetime | None, int]

_FINGERPRINT_SQL = (
    "SELECT count(*), max(indexed_at), COALESCE(sum(xmin::text::bigint), 0) "
    "FROM {table} WHERE tenant_id = %s"
)
# The inner statement is ``PgVectorStore.explicit_superseded_chunk_ids`` verbatim; wrapping it
# beside the fingerprint makes the ids and their fingerprint one statement, hence one snapshot.
_SUPERSESSION_SQL = """
    SELECT f.n, f.latest, f.xsum, s.ids
    FROM (
        SELECT count(*) AS n, max(indexed_at) AS latest,
               COALESCE(sum(xmin::text::bigint), 0) AS xsum
        FROM {table} WHERE tenant_id = %(tenant)s
    ) f
    CROSS JOIN (
        SELECT array_agg(d.e) AS ids FROM (
            SELECT DISTINCT jsonb_array_elements_text(metadata->'supersedes') AS e
            FROM {table} WHERE tenant_id = %(tenant)s
            AND jsonb_typeof(metadata->'supersedes') = 'array'
        ) d
    ) s
"""


# Every statement here interpolates only ``store._table``, which ``PgVectorStore.__init__`` has
# already refused unless it is a plain SQL identifier; every value travels as a bound parameter.


def cacheable(store: object) -> bool:
    """Whether ``store`` reads rows exactly as ``PgVectorStore`` does, so SQL here can stand in.

    A test double, or a subclass that overrides either read, keeps the reference path: the cache
    must never substitute its own SQL for a read somebody deliberately changed.
    """
    kind = type(store)
    return (
        isinstance(store, PgVectorStore)
        and getattr(kind, "iter_chunks", None) is PgVectorStore.iter_chunks
        and getattr(kind, "explicit_superseded_chunk_ids", None)
        is PgVectorStore.explicit_superseded_chunk_ids
    )


def _key(store: PgVectorStore) -> tuple[str, str, str]:
    return (store._dsn, store._table, store._tenant)


def _fingerprint_of(rows: Iterable[tuple[int, datetime | None]]) -> Fingerprint:
    """The fingerprint SQL, evaluated over rows read by one statement."""
    count = 0
    latest: datetime | None = None
    total = 0
    for xmin, indexed_at in rows:
        count += 1
        total += int(xmin)
        if indexed_at is not None and (latest is None or indexed_at > latest):
            latest = indexed_at
    return (count, latest, total)


def _normalise(row: Sequence[Any]) -> Fingerprint:
    return (int(row[0]), row[1], int(row[2]))


def current_fingerprint(store: PgVectorStore) -> Fingerprint:
    """One aggregate row: far cheaper than reading, decoding and tokenising every chunk."""
    sql = _FINGERPRINT_SQL.format(table=store._table)
    row = store._with_retry(lambda conn: conn.execute(sql, (store._tenant,)).fetchone())
    return _normalise(row)


def _read_all(store: PgVectorStore) -> tuple[list[tuple[Chunk, int]], Fingerprint]:
    """Every row with its version, in one statement, as ``iter_chunks`` streams it."""
    rows: list[tuple[Chunk, int]] = []
    versions: list[tuple[int, datetime | None]] = []
    with store._borrowed() as conn:
        with conn.transaction():
            with conn.cursor(name=f"recall_bm25_{uuid4().hex[:12]}") as cur:
                cur.itersize = 1000
                cur.execute(
                    f"SELECT id, source, text, metadata, xmin::text::bigint, indexed_at "  # noqa: S608
                    f"FROM {store._table} WHERE tenant_id = %s ORDER BY id",
                    (store._tenant,),
                )
                for cid, source, text, metadata, xmin, indexed_at in cur:
                    rows.append(
                        (Chunk(id=cid, source=source, text=text, metadata=metadata or {}), int(xmin))
                    )
                    versions.append((int(xmin), indexed_at))
    return rows, _fingerprint_of(versions)


def _read_versions(store: PgVectorStore) -> tuple[dict[str, int], Fingerprint]:
    """Every row's id and version, and their fingerprint, in one statement."""
    fetched = store._with_retry(
        lambda conn: conn.execute(
            f"SELECT id, xmin::text::bigint, indexed_at FROM {store._table} "  # noqa: S608
            "WHERE tenant_id = %s",
            (store._tenant,),
        ).fetchall()
    )
    versions = {str(row[0]): int(row[1]) for row in fetched}
    return versions, _fingerprint_of((int(row[1]), row[2]) for row in fetched)


def _read_rows(store: PgVectorStore, ids: Sequence[str]) -> dict[str, tuple[Chunk, int]]:
    fetched = store._with_retry(
        lambda conn: conn.execute(
            f"SELECT id, source, text, metadata, xmin::text::bigint FROM {store._table} "  # noqa: S608
            "WHERE tenant_id = %s AND id = ANY(%s)",
            (store._tenant, list(ids)),
        ).fetchall()
    )
    return {
        str(row[0]): (
            Chunk(id=row[0], source=row[1], text=row[2], metadata=row[3] or {}),
            int(row[4]),
        )
        for row in fetched
    }


def _stable_key_or_none(chunk: Chunk) -> tuple[bytes, int, str] | None:
    try:
        return stable_window_key(chunk)
    except Exception:  # BROAD-CATCH: the reference path re-raises exactly this, at rank time
        return None


def _fresh(chunk: Chunk) -> Chunk:
    """A private copy per Search, as a fresh database read would give; the cache stays pristine."""
    return Chunk(
        id=chunk.id, source=chunk.source, text=chunk.text, metadata=copy.deepcopy(chunk.metadata)
    )


@dataclass(frozen=True, eq=False)
class Bm25Snapshot:
    """An immutable BM25 view of one tenant at one fingerprint.

    Immutable so that a Search can rank from it while another thread builds its successor:
    ``extend`` copies what it changes and shares the rest.
    """

    fingerprint: Fingerprint
    chunks: tuple[Chunk, ...] = ()
    index_of: dict[str, int] = field(default_factory=dict)
    xmins: array = field(default_factory=lambda: array("q"))
    lengths: array = field(default_factory=lambda: array("q"))
    total_length: int = 0
    stable_keys: tuple[tuple[bytes, int, str] | None, ...] = ()
    postings: dict[str, tuple[array, array]] = field(default_factory=dict)

    @classmethod
    def build(cls, rows: Iterable[tuple[Chunk, int]], fingerprint: Fingerprint) -> Bm25Snapshot:
        return cls(fingerprint=fingerprint).extend(rows, fingerprint)

    def extend(self, rows: Iterable[tuple[Chunk, int]], fingerprint: Fingerprint) -> Bm25Snapshot:
        """A new snapshot holding this one's documents plus ``rows``; this one is unchanged."""
        chunks = list(self.chunks)
        index_of = dict(self.index_of)
        xmins = array("q", self.xmins)
        lengths = array("q", self.lengths)
        total_length = self.total_length
        stable_keys = list(self.stable_keys)
        added: dict[str, tuple[list[int], list[int]]] = {}
        for chunk, xmin in rows:
            if chunk.id in index_of:
                raise ValueError(f"chunk {chunk.id!r} is already in the snapshot")
            doc = len(chunks)
            counts = Counter(tokenize(chunk.text))
            length = sum(counts.values())
            chunks.append(chunk)
            index_of[chunk.id] = doc
            xmins.append(xmin)
            lengths.append(length)
            total_length += length
            stable_keys.append(_stable_key_or_none(chunk))
            for term, frequency in counts.items():
                pair = added.get(term)
                if pair is None:
                    added[term] = ([doc], [frequency])
                else:
                    pair[0].append(doc)
                    pair[1].append(frequency)
        postings = dict(self.postings)
        for term, (docs, frequencies) in added.items():
            previous = postings.get(term)
            if previous is None:
                postings[term] = (array("i", docs), array("i", frequencies))
            else:
                merged_docs = array("i", previous[0])
                merged_docs.extend(docs)
                merged_frequencies = array("i", previous[1])
                merged_frequencies.extend(frequencies)
                postings[term] = (merged_docs, merged_frequencies)
        return Bm25Snapshot(
            fingerprint=fingerprint,
            chunks=tuple(chunks),
            index_of=index_of,
            xmins=xmins,
            lengths=lengths,
            total_length=total_length,
            stable_keys=tuple(stable_keys),
            postings=postings,
        )

    def version_of(self, chunk_id: str) -> int | None:
        doc = self.index_of.get(chunk_id)
        return None if doc is None else self.xmins[doc]

    def rank(self, query: str, *, k: int, stable_ties: bool) -> list[ScoredChunk] | None:
        """``rank_bm25_chunks(self.chunks, query, ...)``, or None where that would raise.

        None means a positive candidate has no Code4 window identity; the caller then runs the
        reference itself, so the error raised is exactly the one it always was.
        """
        if k < 1:
            raise ValueError("k must be positive")
        count = len(self.chunks)
        average_length = self.total_length / count if count else 0.0
        query_terms = tokenize(query)
        inverse_document_frequency: dict[str, float] = {}
        per_doc: dict[int, dict[str, int]] = {}
        for term in dict.fromkeys(query_terms):
            posting = self.postings.get(term)
            if posting is None:
                continue
            docs, frequencies = posting
            frequency = len(docs)
            inverse_document_frequency[term] = math.log(
                1.0 + (count - frequency + 0.5) / (frequency + 0.5)
            )
            for doc, term_frequency in zip(docs, frequencies):
                counts = per_doc.get(doc)
                if counts is None:
                    per_doc[doc] = {term: term_frequency}
                else:
                    counts[term] = term_frequency
        ranked: list[tuple[int, float]] = []
        lengths = self.lengths
        for doc, counts in per_doc.items():
            length = lengths[doc]
            score = 0.0
            # The reference's loop and expression, operand for operand: float addition is not
            # associative, so the order of the query terms is part of the score.
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                normalizer = 1.0 - BM25_B + BM25_B * length / (average_length or 1.0)
                denominator = frequency + BM25_K1 * normalizer
                score += (
                    inverse_document_frequency.get(term, 0.0)
                    * frequency
                    * (BM25_K1 + 1.0)
                    / denominator
                )
            if score > 0.0:
                ranked.append((doc, score))
        chunks = self.chunks
        if stable_ties:
            keys = self.stable_keys
            if any(keys[doc] is None for doc, _ in ranked):
                return None
            top = heapq.nsmallest(k, ranked, key=lambda item: (-item[1], keys[item[0]]))
        else:
            top = heapq.nsmallest(
                k, ranked, key=lambda item: (-item[1], (b"", 0, chunks[item[0]].id))
            )
        return [ScoredChunk(_fresh(chunks[doc]), score) for doc, score in top]


@dataclass
class _Bm25Entry:
    build_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Bumped by every in-process write to the tenant; a snapshot is served only while the epoch
    #: it was built under is still current.
    epoch: int = 0
    #: ``(snapshot, epoch it was built under)``, replaced as one reference so that a reader can
    #: never pair a new snapshot with an old epoch or the reverse.
    state: tuple[Bm25Snapshot, int] | None = None


@dataclass
class _SupersessionEntry:
    epoch: int = 0
    value: tuple[Fingerprint, int, frozenset[str]] | None = None


class TenantSearchCache:
    """The BM25 and supersession caches of one retriever, keyed by physical tenant."""

    def __init__(
        self,
        *,
        max_bm25_tenants: int = MAX_BM25_TENANTS,
        max_supersession_tenants: int = MAX_SUPERSESSION_TENANTS,
    ) -> None:
        if max_bm25_tenants < 1 or max_supersession_tenants < 1:
            raise ValueError("cache bounds must be positive")
        self._guard = threading.Lock()
        self._bm25: OrderedDict[tuple[str, str, str], _Bm25Entry] = OrderedDict()
        self._supersession: OrderedDict[tuple[str, str, str], _SupersessionEntry] = OrderedDict()
        self._max_bm25 = max_bm25_tenants
        self._max_supersession = max_supersession_tenants
        #: Counters a test or an operator can read; they never affect a result.
        self.stats: Counter[str] = Counter()

    def _bm25_entry(self, key: tuple[str, str, str]) -> _Bm25Entry:
        with self._guard:
            entry = self._bm25.get(key)
            if entry is None:
                entry = _Bm25Entry()
                self._bm25[key] = entry
                while len(self._bm25) > self._max_bm25:
                    self._bm25.popitem(last=False)
            else:
                self._bm25.move_to_end(key)
            return entry

    def _supersession_entry(self, key: tuple[str, str, str]) -> _SupersessionEntry:
        with self._guard:
            entry = self._supersession.get(key)
            if entry is None:
                entry = _SupersessionEntry()
                self._supersession[key] = entry
                while len(self._supersession) > self._max_supersession:
                    self._supersession.popitem(last=False)
            else:
                self._supersession.move_to_end(key)
            return entry

    def invalidate(self, tenants: Iterable[str], *, drop: bool = False) -> None:
        """Mark every cached value of ``tenants`` suspect, or forget it outright with ``drop``."""
        wanted = set(tenants)
        with self._guard:
            for cache in (self._bm25, self._supersession):
                for key in [key for key in cache if key[2] in wanted]:
                    if drop:
                        del cache[key]
                    else:
                        cache[key].epoch += 1

    def rank_bm25(
        self, store: object, query: str, *, k: int, stable_ties: bool
    ) -> tuple[list[ScoredChunk], str]:
        """``rank_bm25_chunks(list(store.iter_chunks()), query, ...)`` and how it was served."""
        if not cacheable(store):
            self.stats["bm25_bypass"] += 1
            return (
                rank_bm25_chunks(
                    list(store.iter_chunks()),  # type: ignore[attr-defined]
                    query,
                    k=k,
                    stable_ties=stable_ties,
                ),
                "bypass",
            )
        if k < 1:
            raise ValueError("k must be positive")
        assert isinstance(store, PgVectorStore)
        entry = self._bm25_entry(_key(store))
        fingerprint = current_fingerprint(store)
        state = entry.state
        status = "hit"
        if state is not None and state[1] == entry.epoch and state[0].fingerprint == fingerprint:
            snapshot = state[0]
        else:
            with entry.build_lock:
                snapshot, status = self._refresh(entry, store)
        self.stats[f"bm25_{status}"] += 1
        ranked = snapshot.rank(query, k=k, stable_ties=stable_ties)
        if ranked is None:
            self.stats["bm25_reference"] += 1
            return (
                rank_bm25_chunks(
                    list(store.iter_chunks()), query, k=k, stable_ties=stable_ties
                ),
                "reference",
            )
        return ranked, status

    def _refresh(self, entry: _Bm25Entry, store: PgVectorStore) -> tuple[Bm25Snapshot, str]:
        """Bring ``entry`` to the tenant's current rows; the caller holds ``entry.build_lock``."""
        epoch = entry.epoch
        state = entry.state
        snapshot = state[0] if state is not None else None
        if state is not None and state[1] == epoch:
            # Another Search may have refreshed while this one waited for the lock.
            if state[0].fingerprint == current_fingerprint(store):
                return state[0], "hit"
        fresh: Bm25Snapshot | None = None
        status = "rebuild"
        if snapshot is not None:
            versions, fingerprint = _read_versions(store)
            unchanged = len(versions) >= len(snapshot.index_of) and all(
                versions.get(chunk_id) == snapshot.version_of(chunk_id)
                for chunk_id in snapshot.index_of
            )
            if unchanged:
                new_ids = sorted(chunk_id for chunk_id in versions if chunk_id not in snapshot.index_of)
                fetched = _read_rows(store, new_ids) if new_ids else {}
                # Rows re-read by a second statement must be the versions the first one saw;
                # anything else means a write landed in between, and only a one-statement read
                # can give a consistent snapshot then.
                if len(fetched) == len(new_ids) and all(
                    fetched[chunk_id][1] == versions[chunk_id] for chunk_id in new_ids
                ):
                    fresh = snapshot.extend((fetched[chunk_id] for chunk_id in new_ids), fingerprint)
                    status = "incremental" if new_ids else "revalidated"
        if fresh is None:
            rows, fingerprint = _read_all(store)
            fresh = Bm25Snapshot.build(rows, fingerprint)
        entry.state = (fresh, epoch)
        return fresh, status

    def superseded_ids(self, store: object) -> frozenset[str]:
        """``store.explicit_superseded_chunk_ids()``, re-read only when the tenant has changed."""
        if not cacheable(store):
            self.stats["supersession_bypass"] += 1
            return store.explicit_superseded_chunk_ids()  # type: ignore[attr-defined,no-any-return]
        assert isinstance(store, PgVectorStore)
        entry = self._supersession_entry(_key(store))
        epoch = entry.epoch
        fingerprint = current_fingerprint(store)
        cached = entry.value
        if cached is not None and cached[0] == fingerprint and cached[1] == epoch:
            self.stats["supersession_hit"] += 1
            return cached[2]
        sql = _SUPERSESSION_SQL.format(table=store._table)
        row = store._with_retry(
            lambda conn: conn.execute(sql, {"tenant": store._tenant}).fetchone()
        )
        ids = frozenset(str(value) for value in (row[3] or []) if value)
        entry.value = (_normalise(row), epoch, ids)
        self.stats["supersession_miss"] += 1
        return ids
