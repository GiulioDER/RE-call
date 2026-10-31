"""`PgVectorStore`'s supersession reads: the authored graph, its cache and the dependency
projection.

A mixin of `recall.store.PgVectorStore` (architecture step S7d); its caches live on the store.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

import psycopg

from recall.lineage import canonical_sha256
from recall.observability import METRICS, get_logger
from recall.supersession import (
    EdgeCandidates,
    SupersessionGraph,
    resolve_supersession_graph,
    supersession_rows_sql,
)
from recall.supersession_records import (
    PG_RECORD_REVISION_SQL,
    pg_records,
)
from recall.types import Chunk

_log = get_logger("store")  # the store's own logger name, which tests filter on

_T = TypeVar("_T")


class SupersessionMixin:
    if TYPE_CHECKING:  # what this mixin uses of the store it is mixed into
        _tenant: str
        _table: str
        _supersession_cache: tuple | None
        _supersession_scans: int
        _dependency_projection_cache: tuple | None

        def _with_retry(self, op: Callable[["psycopg.Connection"], _T]) -> _T: ...

        def _borrowed(self) -> "AbstractContextManager[psycopg.Connection]": ...

        def _upsert_in(
            self, conn: "psycopg.Connection", chunks: list[Chunk], embeddings: list[list[float]]
        ) -> None: ...

        def _scrub_sparse_rows(
            self, conn: "psycopg.Connection", chunk_table: str, chunk_ids: list[str]
        ) -> int: ...

        @property
        def record_scope(self) -> str: ...

        @property
        def generation_id(self) -> str: ...

        # `dependency_projection` hands the store itself to `project_current_state`, whose
        # `StateStore` protocol also needs these two.
        @property
        def tenant(self) -> str: ...

        def iter_chunks(self, batch_size: int = 1000) -> "Iterator[Chunk]": ...

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        """The supersession relation: ``(edges, unresolved)``.

        `edges` maps superseded file -> superseding file. Both are root-relative, except a path
        that two roots hold, which is keyed by its absolute source so the two stay two documents
        (`SupersessionGraph.collided`). `unresolved` names the files an edge pointed at but could
        not identify, so the read path can fail closed on them rather than serve them as healthy.
        These edges carry no validity windows; the window rule needs `supersession_graph()`.

        A ``supersedes:`` reference is matched by the rule `recall.supersession._resolve_rows`
        states (changed 2026-10-21): a reference naming a folder by its folded path, any
        reference by its folded stem, the claimant never its own candidate. An AMBIGUOUS target
        (several candidates) is skipped rather than mis-mapped, so a stray sibling can never be
        silently marked superseded; `recall lint`'s ``ambiguous-supersedes-target`` still flags
        by stem alone, so it can flag a reference this resolves by path. Only the ambiguous case
        is skipped. A DANGLING target (no such file in the corpus) is instead kept, as an edge
        keyed on the raw name it was written as: harmless, because that key matches no indexed
        file, and dropping it would silently discard a valid supersession claim (e.g. a memo
        superseding a doc since removed).

        The result is cached, but the cache is VALIDATED against the table on every call rather
        than trusted. It previously was not: it was invalidated only by this instance's own
        writes, so a long-lived reader (an MCP server holds one store for its lifetime) never saw
        an edge written by a separate `recall index` run. It kept serving the superseded memory as
        `ok` until someone restarted the process — the trust layer returning exactly the wrong
        answer, silently, which is the failure it exists to prevent.

        Freshness is established by a cheap fingerprint — `(max(indexed_at), count(*))` for this
        tenant — and the expensive grouped scan runs only when that fingerprint moves. Both
        halves are needed: `max(indexed_at)` alone cannot see a DELETE, and deleting a superseding
        document must stop its edge from applying, or the reader keeps demoting a memory that is
        current again.

        Measured on a 50k-row table: fingerprint ~12 ms, full scan ~80 ms. So this is cheaper than
        rescanning on every call and, unlike caching indefinitely, it is correct.

        Fingerprint and scan share ONE connection, so they cannot straddle a concurrent write and
        cache a result under a fingerprint that never described it.
        """

        edges, unresolved, _dates = self.supersession_all()  # already copies
        return edges, unresolved

    def supersession_all(
        self,
    ) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
        """``(edges, unresolved, candidates)`` behind one validated cache and one scan.

        `candidates` maps a superseded file to EVERY document claiming to supersede it, each with
        when that claim was first written: `min(indexed_at)` over the claiming chunks, since a
        chunk existing implies its `supersedes:` frontmatter existed and the earliest is the
        conservative reading.

        **A list, not a winner, and that is the point.** `edges` keeps one superseder per target,
        chosen by scan order, which is a time-independent rule and therefore answers a different
        question. Where `b1.md` (Monday) and `b2.md` (Wednesday) both supersede `a.md`, only
        `b2.md` survived, so a replay of Tuesday saw a single edge dated Wednesday, dropped it,
        and answered `ok` when `a.md` was in fact superseded by `b1.md`. Renaming the two files
        flipped the answer, which is what showed the axis was wrong. `resolve_successor` now picks
        the claim that was live at the instant. `edges` is still last-row-wins over the scan, so
        a caller who never replays sees the scan-order winner; which file a reference matches is
        `_resolve_rows`' rule, changed 2026-10-21 (folded paths, a self-claim dropped, roots kept
        apart).

        This is the only accessor. A second one returning just the dates existed briefly and had
        to warn, in its own docstring, that pairing it with `supersession()` reintroduced the
        two-validation race this exists to close.

        `ORDER BY` is load-bearing, not tidiness. `edges` resolves fan-in by
        last-row-wins, so an unordered scan lets the
        winner change between runs. The predecessor query was `SELECT DISTINCT`, and swapping it
        for `GROUP BY` preserves the row SET but not the row ORDER, which would have re-rolled
        that winner for existing `supersession()` callers who never asked for any of this.
        The ordering is the DATABASE's text collation, so the winner is stable within a
        deployment but not guaranteed identical across one with a different `lc_collate`. That
        residue is deliberately left rather than pinned with `COLLATE "C"`: `ORDER BY 1 COLLATE
        "C"` does not mean what it looks like. An ORDER BY ordinal is only read as an output
        column when it is a bare integer constant, so adding `COLLATE` turns it into the integer
        literal 1, and integers have no collation. It was written that way here and broke 51 tests
        on the first CI run that touched a real database, with nothing catching it locally because
        no test executes this SQL without Postgres.

        The winner is also stable rather than necessarily IDENTICAL to what `SELECT DISTINCT`
        produced: under a Sort+Unique plan it is the same one, under HashAggregate the old one was
        bucket order.

        Dates come from `first_indexed_at` rather than `indexed_at`. The distinction is the
        whole reason the column exists: `indexed_at` records the LAST write, so `replace_sources`
        re-inserting an edited document moved its edge forward and a replay before the edit
        dropped a claim that had been continuously true. `first_indexed_at` is preserved on
        conflict with `LEAST`, and captured and restored across `replace_sources`' delete, which
        `ON CONFLICT` alone cannot cover because the row is gone before the insert runs.
        """

        return self.supersession_graph().parts()

    def supersession_graph(self) -> SupersessionGraph:
        """The resolved graph behind `supersession_all()`, with each claimant's validity window.

        The same validated cache and the same single scan, so the windows and the edges always
        describe one corpus state; `recall.trust` reads this, and the three-part accessors are
        projections of it. A copy, for the reason `supersession_all` gives.
        """

        def _op(conn: "psycopg.Connection") -> SupersessionGraph:
            # The record revision rides in the same statement: a withdrawal of a memo already gone
            # writes a record and deletes no row, so the table's own fingerprint would not move.
            fingerprint = conn.execute(
                f"SELECT max(indexed_at), count(*), ({PG_RECORD_REVISION_SQL}) "
                f"FROM {self._table} WHERE tenant_id = %s",
                (self._tenant, self._tenant),
            ).fetchone()
            cached = self._supersession_cache
            if cached is not None and cached[0] == fingerprint:
                graph: SupersessionGraph = cached[1]
                return graph
            rows = conn.execute(
                supersession_rows_sql(self._table, source_column="source", scope="c.tenant_id = %s"),
                (self._tenant,),
            ).fetchall()
            self._supersession_scans += 1
            METRICS.increment("recall_supersession_scans_total")
            graph = resolve_supersession_graph(
                rows, pg_records(conn, self._tenant, self.record_scope), self._tenant
            )
            self._supersession_cache = (fingerprint, graph)
            return graph

        # COPIES. This is public, and the cache is process-wide and validated by fingerprint
        # rather than rebuilt, so handing out the live structures would let one caller's mutation
        # survive every later cache hit and redirect other callers' supersession verdicts. The
        # candidate LISTS are copied too, because a shallow `dict()` would still share them.
        return self._with_retry(_op).copy()

    def dependency_projection(
        self,
        as_of: datetime | None = None,
        known_as_of: datetime | None = None,
    ) -> Any:
        """Return the ready generation's dependency invalidation projection.

        The persisted marker is checked before rebuilding the time-sensitive closure. This keeps
        enforce mode generation bound: a legacy generation or a generation whose transactional
        projection failed validation cannot silently become enforceable merely because its chunks
        happen to contain dependency metadata.
        """
        from recall.current_state import project_current_state

        generation_reader = getattr(self, "_generation_id", None)
        generation_id = (
            str(generation_reader()) if callable(generation_reader) else self.generation_id
        )
        instant = as_of.isoformat() if as_of is not None else None
        known_instant = known_as_of.isoformat() if known_as_of is not None else None
        cached = getattr(self, "_dependency_projection_cache", None)
        if cached is not None:
            cached_key = cached[0]
            if (
                cached_key[0] == generation_id
                and cached_key[2] == instant
                and cached_key[3] == known_instant
            ):
                return cached[-1]

        def _load_persisted(
            conn: "psycopg.Connection",
        ) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
            row = conn.execute(
                "SELECT validation_summary FROM recall_generations "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchone()
            summary = row[0] if row and isinstance(row[0], Mapping) else {}
            edge_rows = conn.execute(
                "SELECT edge_id FROM recall_dependency_edges_v1 "
                "WHERE tenant_id = %s AND generation_id = %s ORDER BY edge_id",
                (self._tenant, generation_id),
            ).fetchall()
            diagnostic_rows = conn.execute(
                "SELECT diagnostic_id FROM recall_dependency_diagnostics_v1 "
                "WHERE tenant_id = %s AND generation_id = %s ORDER BY diagnostic_id",
                (self._tenant, generation_id),
            ).fetchall()
            return (
                summary,
                tuple(str(item[0]) for item in edge_rows),
                tuple(str(item[0]) for item in diagnostic_rows),
            )

        summary, persisted_edge_ids, persisted_diagnostic_ids = self._with_retry(_load_persisted)
        marker = summary.get("dependency_projection")
        if not isinstance(marker, Mapping) or marker.get("ready") is not True:
            return None
        cache_key = (
            generation_id,
            str(marker.get("projection_id", "")),
            instant,
            known_instant,
            persisted_edge_ids,
            persisted_diagnostic_ids,
        )
        projection = project_current_state(
            self, as_of=as_of, known_as_of=known_as_of
        ).dependency_projection
        if projection is None:
            return None
        computed_edge_ids = tuple(edge.id for edge in projection.edges)
        computed_diagnostic_ids = tuple(
            sorted(
                "depdiag_"
                + canonical_sha256(
                    {
                        "kind": diagnostic.kind,
                        "source": diagnostic.source,
                        "dependency": diagnostic.dependency,
                    }
                )[:24]
                for diagnostic in projection.diagnostics
            )
        )
        if (
            computed_edge_ids != persisted_edge_ids
            or computed_diagnostic_ids != persisted_diagnostic_ids
        ):
            _log.error(
                "dependency projection rows do not match generation marker for %s",
                generation_id,
            )
            return None
        self._dependency_projection_cache = (cache_key, projection)
        return projection

    def supersession_map(self) -> dict[str, str]:
        """Resolvable supersession edges only — convenience view of `supersession`."""
        return self.supersession()[0]
