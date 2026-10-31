"""`PgVectorStore`'s source mutation: replace, delete and touch the rows of a source.

A mixin of `recall.store.PgVectorStore` (architecture step S7d); it owns no state of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

import psycopg

from recall.observability import get_logger
from recall.supersession_records import (
    pg_delete_removals,
)
from recall.types import Chunk

_log = get_logger("store")  # the store's own logger name, which tests filter on

_T = TypeVar("_T")


def _chunk_identity(chunks: list[Chunk]) -> tuple[str | None, list[str]]:
    """The project and root-relative file names a batch of chunks belongs to.

    `(None, [])` when the chunks declare no project, which is what confines the identity delete to a
    corpus whose owner declared one: two roots with the same relative layout and no project may be
    different corpora, and merging them is not a guess this layer may make.

    Also `(None, [])` if a batch ever mixed two projects, rather than guessing which one owns the
    delete.
    """
    projects = {
        value
        for chunk in chunks
        if isinstance(value := chunk.metadata.get("project"), str) and value
    }
    if len(projects) != 1:
        return None, []
    names: dict[str, None] = {}
    for chunk in chunks:
        name = chunk.metadata.get("file")
        if isinstance(name, str) and name:
            names.setdefault(name, None)
    return projects.pop(), list(names)


class SourceMutationMixin:
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

    def replace_sources(
        self, sources: list[str], chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        """Atomically replace every row of `sources` with the given chunks.

        ⛔ **A second delete, by `(project, root-relative file)`, reaches what `sources` cannot.**
        Deleting by absolute path alone left the previous rows in place whenever the same file was
        re-indexed from a different root, so the new chunks landed BESIDE the old ones rather than
        over them. Measured on a live corpus: 452 duplicate rows in one project and 615 in another.
        The two keys are ORed, never substituted, so this can only remove more, never less.

        The identity is derived from `chunks` rather than passed in. Two reasons, and the second is
        the one that changed the design: the rows deleted and the rows inserted then cannot describe
        different files, because they come from one list; and a keyword argument here broke five
        test doubles in three files, which is what an interface implemented in many places does when
        it grows a parameter the callee could work out for itself.

        Delete + insert run in ONE transaction: a failure (or a concurrent reader) never
        observes the sources deleted without their replacement rows. Callers must compute
        `embeddings` BEFORE calling — an embedding failure then leaves the old rows intact.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None

        def _op(conn: "psycopg.Connection") -> int:
            with conn.transaction():
                # `first_indexed_at` has to survive the DELETE. `ON CONFLICT ... LEAST` in
                # `_upsert_in` cannot help here: the row is gone by the time the insert runs, so
                # there is no conflict and the re-inserted chunk would claim it was first written
                # now. That is the whole re-index defect, and this is the path that causes it.
                preserved: dict[str, datetime] = {}
                removed_ids: set[str] = set()
                if sources:
                    # COALESCE to the row's OWN `indexed_at`, not `now()`. Two bugs deep here.
                    # First: a NULL key was still PRESENT, so the restore wrote that NULL back
                    # over the timestamp `_upsert_in` had just written, and the row could never
                    # acquire a first write at all. Filtering the NULL fixed that but stamped
                    # `now()` — so a memo written in January, migrated, then re-indexed once in
                    # August claimed August as its first write, and a replay of March said it had
                    # never existed. Its real `indexed_at` was sitting on the same row, and it is
                    # exactly what BOTH read paths COALESCE to. The write path now agrees with
                    # them: migrated rows keep their last-known write as their first, and freeze.
                    preserved = {
                        cid: first
                        for cid, first in conn.execute(
                            f"SELECT id, COALESCE(first_indexed_at, indexed_at) "
                            f"FROM {self._table} "
                            f"WHERE tenant_id = %s AND source = ANY(%s)",
                            (self._tenant, sources),
                        ).fetchall()
                        if first is not None
                    }
                    removed_ids.update(
                        row[0]
                        for row in conn.execute(
                            f"DELETE FROM {self._table} "
                            f"WHERE tenant_id = %s AND source = ANY(%s) RETURNING id",
                            (self._tenant, sources),
                        ).fetchall()
                    )
                identity_project, identity_files = _chunk_identity(chunks)
                if identity_project and identity_files:
                    # Same transaction as the insert below, so a reader never sees the old rows
                    # gone without their replacement. The project is a scalar because one index run
                    # carries one project; PostgreSQL also refuses an anonymous composite as a
                    # bound parameter, so the pair was never an option.
                    removed_ids.update(
                        row[0]
                        for row in conn.execute(
                            f"DELETE FROM {self._table} "
                            f"WHERE tenant_id = %s AND metadata->>'project' = %s "
                            f"AND metadata->>'file' = ANY(%s) RETURNING id",
                            (self._tenant, identity_project, identity_files),
                        ).fetchall()
                    )
                # Before the upsert: a re-inserted chunk id's old sparse vector is stale for the
                # new text anyway, and the indexer's sparse hook re-encodes in the same run when
                # an encoder is configured. Re-indexing WITHOUT an encoder therefore shrinks
                # sidecar coverage, and `assert_sparse_coverage` refuses loudly (undercount)
                # instead of silently serving stale vectors for changed text — the intended
                # direction.
                self._scrub_sparse_rows(conn, self._table, sorted(removed_ids))
                if chunks:
                    self._upsert_in(conn, chunks, embeddings)  # savepoint, same commit
                    restore = [
                        (preserved[c.id], self._tenant, c.id) for c in chunks if c.id in preserved
                    ]
                    if restore:
                        # executemany, matching `_upsert_in`: psycopg3 pipelines it. A chunk id
                        # absent from `preserved` is genuinely new and keeps its now().
                        with conn.cursor() as cur:
                            cur.executemany(
                                # LEAST(..., now()) mirrors `_upsert_in`. Without it the two
                                # write paths disagreed on exactly the input the upsert comment
                                # names: a stored date in the FUTURE (clock skew, a restore) was
                                # clamped by upsert and preserved verbatim here — on the ONLY
                                # path `recall index` takes, leaving the row permanently
                                # invisible to every as-of replay.
                                f"UPDATE {self._table} SET first_indexed_at = LEAST(%s, now()) "
                                f"WHERE tenant_id = %s AND id = %s",
                                restore,
                            )
            return len(chunks)

        self._with_retry(_op)
        return len(chunks)

    def delete_sources(self, sources: list[str], *, withdraw: bool = False) -> int:
        """Delete every chunk belonging to the given `source` values; returns rows removed.

        Standalone removal API (the Indexer uses the atomic `replace_sources` instead).

        A memo that declared `supersedes:` leaves a `SupersessionRecord` in the same transaction
        (`recall.supersession_records`): a forget keeps retired what it retired, and `withdraw`
        gives that back instead and splices its chain. `withdraw` also covers a source with no
        rows left that a record stands in for.

        ⚠️ The returned count can UNDERCOUNT after a reconnect-retry. The delete and its sidecar
        scrub now share a transaction whose COMMIT runs inside the retried op, so a connection
        lost at commit time has an indeterminate outcome: the retry re-runs, finds the rows
        already gone, and returns 0 for an erasure that in fact happened. On the right-to-erasure
        path the deletion is still real; only the receipt may say fewer rows than it removed.
        """
        if not sources:
            return 0
        self._supersession_cache = None

        # An explicit transaction, where a single statement used to suffice: the chunk delete
        # and the sidecar scrub MUST share one commit, or a failure between them leaves term
        # weights for text the caller was told is gone. The row count is computed inside the
        # borrow for the reason the old comment gave: in pooled mode `_with_retry` returns from
        # within `with self._pool.connection()`, so nothing from the cursor may escape it.
        def _op(conn: "psycopg.Connection") -> int:
            with conn.transaction():
                ids = pg_delete_removals(conn, self._tenant, self._table, sources, withdraw=withdraw)
                self._scrub_sparse_rows(conn, self._table, ids)
                return len(ids)

        return self._with_retry(_op)

    def delete_sources_across(
        self, tables: list[str], sources: list[str], *, withdraw: bool = False
    ) -> int:
        """Atomically erase tenant sources from active and shadow generation tables.

        Each table keeps its own `SupersessionRecord`s, as `delete_sources` does.
        """
        if not sources:
            return 0
        from recall.sql_identifiers import validate_table_name

        unique_tables = list(dict.fromkeys(tables))
        if not unique_tables:
            raise ValueError("at least one generation table is required")
        # The SAME allowlist the control plane validates registry rows with, not a second,
        # weaker one. This method interpolates every name into a DELETE, and it used to accept
        # anything `str.isidentifier()` liked: non-ASCII, uppercase that PostgreSQL folds to
        # something else, and names past the truncation point. It does NOT refuse unquoted SQL
        # keywords such as `select`; those are a syntax error at query time, not an injection.
        for table in unique_tables:
            validate_table_name(table)
        self._supersession_cache = None

        def _op(conn: "psycopg.Connection") -> int:
            removed = 0
            with conn.transaction():
                for table in unique_tables:
                    ids = pg_delete_removals(conn, self._tenant, table, sources, withdraw=withdraw)
                    # Under THAT table's name: the sidecar keys shadow-table rows by the
                    # chunk_table column, so each table scrubs its own.
                    self._scrub_sparse_rows(conn, table, ids)
                    removed += len(ids)
            return removed

        return self._with_retry(_op)

    def touch_files(self, files: list[str]) -> int:
        """Reset ``indexed_at`` to now(), carrying ``first_indexed_at`` across, for every chunk
        whose metadata file name matches.

        A timestamp-only touch — text and embeddings are untouched by construction, so it is
        the honest way to simulate a re-sync (the eval's recency arm uses it; re-indexing
        identical text would also re-embed, which is only a no-op for deterministic
        embedders). Matches on the ``file`` metadata key (basename), so it works for nested
        corpora too. Returns rows updated.

        **`first_indexed_at` is carried across the touch.** A row predating that column holds NULL
        there, and both read paths COALESCE the NULL to `indexed_at` — so `indexed_at` IS that
        row's only evidence of age. Overwriting it alone destroyed the evidence, and the next
        re-index then COALESCEd the TOUCH instant into `first_indexed_at` and froze it, so a memo
        written in January permanently claimed it was first seen the moment someone simulated a
        re-sync. A timestamp-only touch is supposed to change what a staleness check sees, not
        rewrite when the corpus was acquired.

        Clamped with `LEAST(..., now())` like the other two writers. This is the THIRD path that
        writes `first_indexed_at`, and it was the only one that did not clamp: a migrated row
        whose `indexed_at` sat in the future (clock skew, a restore) went from visible to
        permanently `not_yet_known` after a touch, and no further touch repaired it. A rule
        enforced on two paths out of three is not a rule.

        Captured in the SAME statement, because Postgres evaluates every `SET` expression against
        the OLD tuple: the COALESCE reads the pre-touch `indexed_at`, not `now()`. Two statements
        would race, and the ordering that writes `indexed_at` first would capture the value it had
        just destroyed. A row that already has a first write IN THE PAST is unaffected: COALESCE returns
        it and LEAST keeps it. One in the FUTURE is clamped to now() like everywhere else, so
        "unaffected" was too strong once the clamp landed.

        ⚠️ `LEAST` does not propagate NULL, so a row with BOTH time columns NULL is stamped with
        the touch instant rather than staying unknown. Unreachable through a store-created table,
        where `indexed_at` is NOT NULL; reachable only by pointing `ensure_schema` at a
        hand-rolled table whose `indexed_at` is nullable, which the shape check does not forbid.
        """
        if not files:
            return 0
        # rowcount read inside the borrow — see `delete_sources` for why.
        return self._with_retry(
            lambda conn: (
                conn.execute(
                    f"UPDATE {self._table} SET "
                    f"first_indexed_at = LEAST(COALESCE(first_indexed_at, indexed_at), now()), "
                    f"indexed_at = now() "
                    f"WHERE tenant_id = %s AND metadata->>'file' = ANY(%s)",
                    (self._tenant, files),
                ).rowcount
                or 0
            )
        )
