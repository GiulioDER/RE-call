"""Read only store view pinned to one tenant's active immutable generation."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import psycopg
from pgvector import Vector

from recall.generations import NoActiveGeneration
from recall.semantic_graph import (
    GraphReadiness,
    SemanticGraphProjection,
    delete_semantic_graph,
    load_semantic_graph,
    write_semantic_graph,
)
from recall.store import DEFAULT_TABLE
from recall.store import (
    EdgeCandidates,
    PgVectorStore,
    _EXACT_SCAN_GUARDS,
    resolve_supersession_candidates,
)
from recall.types import Chunk, ScoredChunk
from recall.errors import RecallError

if TYPE_CHECKING:
    from recall.calibration_v2 import CalibrationResolution
    from recall.pool import SharedPool


class ImmutableGenerationError(RuntimeError, RecallError):
    pass


#: The generation states whose manifests describe the CORPUS, for erasure purposes.
#:
#: One list, read by every query that answers "does the tenant hold this source". It was
#: written twice once, and the second copy was unguarded: the test that pins the `failed`
#: exclusion drives the CLI, so adding `failed` to the other copy left the whole forget
#: suite green while an MCP erasure could tombstone a URI only a failed build ever named.
#: A tombstone is permanent and bars that URI from every future build, so admitting a URI
#: that was never in the corpus is irreversible. Dropping a state is the mirror harm:
#: a genuine right-to-erasure request answered "check for typos".
#:
#: `failed` is the only exclusion. `building` MUST be included: it is the state a
#: generation occupies for the whole of its ingest, and `build()` re-checks `_is_tombstoned`
#: per object exactly so an erasure issued mid-build lands.
LIVE_MANIFEST_STATES = ("building", "validating", "ready", "active", "retired")


class GenerationStore(PgVectorStore):
    """The retrieval surface for v1, scoped to a request-consistent active generation."""

    def __init__(
        self,
        dsn: str,
        dim: int,
        *,
        tenant: str,
        migration_target: str = DEFAULT_TABLE,
        pool_size: int | None = None,
        statement_timeout_ms: int | None = None,
        shared_pool: "SharedPool | None" = None,
    ) -> None:
        super().__init__(
            dsn,
            dim,
            table="recall_chunks_v1",
            tenant=tenant,
            pool_size=pool_size,
            statement_timeout_ms=statement_timeout_ms,
            shared_pool=shared_pool,
        )
        if not migration_target.isidentifier():
            raise ValueError("migration_target must be a valid SQL identifier")
        self._migration_target = migration_target
        self._pinned_generation: ContextVar[str | None] = ContextVar(
            f"recall_generation_{uuid.uuid4().hex}", default=None
        )
        self._fixed_generation: str | None = None

    def _reset_tenant_state(self) -> None:
        """Also rebuild the pinned-generation ContextVar, which is tenant-derived.

        `for_tenant` copies `__dict__` by reference, so without this a view shares the SOURCE
        store's ContextVar object: a generation pinned while serving tenant A silently governs
        queries issued through a view bound to tenant B. RLS and the explicit `tenant_id`
        predicate still hold, so it is not disclosure — it is worse-shaped than that, an empty or
        wrong-generation result that reads like an honest answer.
        """
        super()._reset_tenant_state()
        self._pinned_generation = ContextVar(
            f"recall_generation_{uuid.uuid4().hex}", default=None
        )
        self._fixed_generation = None

    def check_schema(self) -> None:
        from recall.schema import check_schema

        self._with_retry(
            lambda conn: check_schema(conn, table=self._migration_target, dim=self._dim)
        )

    def ensure_schema(self) -> None:
        raise ImmutableGenerationError(
            "GenerationStore never migrates; run `recall schema apply` with the migration role"
        )

    def drop_table(self) -> None:
        raise ImmutableGenerationError("the shared generation table cannot be dropped by a store")

    def active_generation_id(self) -> str:
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT s.active_generation_id FROM recall_tenant_state s "
                "JOIN recall_generations g ON g.tenant_id = s.tenant_id "
                "AND g.generation_id = s.active_generation_id "
                "WHERE s.tenant_id = %s AND g.state = 'active'",
                (self._tenant,),
            ).fetchone()
        )
        if not row or not row[0]:
            raise NoActiveGeneration(f"tenant {self._tenant!r} has no active generation")
        return str(row[0])

    @contextmanager
    def snapshot(self) -> Iterator[str]:
        """Pin all reads in this context to one atomically observed active pointer."""
        existing = self._pinned_generation.get()
        if existing is not None:
            yield existing
            return
        # A benchmark server may deliberately read a retired, immutable snapshot.  The fixed
        # process pin must win here as well as in `_generation_id`; otherwise `trusted_search`
        # enters this context manager and silently replaces the pin with the active generation.
        generation_id = self._fixed_generation or self.active_generation_id()
        token = self._pinned_generation.set(generation_id)
        try:
            yield generation_id
        finally:
            self._pinned_generation.reset(token)

    def _generation_id(self) -> str:
        return self._pinned_generation.get() or self._fixed_generation or self.active_generation_id()

    def set_fixed_generation(self, generation_id: str) -> None:
        """Pin this read-only store to one immutable generation for its whole process.

        This is intentionally separate from ``pin_generation``: that context manager is for a
        short administrative operation, while a benchmark server needs every request task to
        see the same retired snapshot. Callers must opt into this explicitly and the generation
        is validated before it becomes process state.
        """

        generation_id = generation_id.strip()
        if not generation_id:
            raise ValueError("generation_id must be non-empty")
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT 1 FROM recall_generations WHERE tenant_id = %s "
                "AND generation_id = %s AND state IN ('ready', 'active', 'retired')",
                (self._tenant, generation_id),
            ).fetchone()
        )
        if row is None:
            raise NoActiveGeneration(
                f"tenant {self._tenant!r} has no fixed readable generation {generation_id!r}"
            )
        self._fixed_generation = generation_id

    @contextmanager
    def pin_generation(self, generation_id: str) -> Iterator[str]:
        """Administrative read view for calibrating one explicit immutable generation."""
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT 1 FROM recall_generations WHERE tenant_id = %s "
                "AND generation_id = %s AND state IN ('ready', 'active', 'retired')",
                (self._tenant, generation_id),
            ).fetchone()
        )
        if row is None:
            raise NoActiveGeneration(
                f"tenant {self._tenant!r} has no calibratable generation {generation_id!r}"
            )
        token = self._pinned_generation.set(generation_id)
        try:
            yield generation_id
        finally:
            self._pinned_generation.reset(token)

    def generation_binding(self) -> dict[str, str]:
        """Identity of the generation this store reads, including WHICH MODEL wrote its vectors.

        `embedder_model` and `embedder_dimension` are read here rather than left to a readiness
        check, because this is the only place the serving path looks the generation up.
        `check_enterprise_readiness` does
        compare the runtime embedder against the generation's, but only `if control_plane is not
        None`, and a stdio server has no control plane; its sibling check on the calibration's
        identity is documented at `recall_mcp/server.py` as unreachable from startup. So on the
        stdio path nothing compared them at all, and a mismatch is invisible rather than loud.
        """
        generation_id = self._generation_id()
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT pipeline_fingerprint, corpus_fingerprint, pipeline_identity "
                "FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchone()
        )
        if row is None:
            raise NoActiveGeneration(generation_id)
        binding = {
            "tenant_id": self._tenant,
            "generation_id": generation_id,
            "pipeline_fingerprint": str(row[0]),
            "corpus_fingerprint": str(row[1]),
        }
        identity = row[2] if isinstance(row[2], Mapping) else {}
        embedder = identity.get("embedder")
        if isinstance(embedder, Mapping):
            # `model` and `dimension`, NOT a `provider:model` string compared against
            # `embedding_profile_id`. That function returns a registered `profile_id` when there
            # is one and the bare `embedder.name` otherwise, so it does not spell
            # `provider:model`, and comparing the two would have refused the CORRECT embedder.
            # This pair is the comparison `CalibrationRepository.calibrate` already makes, and it
            # is known to accept the matching case: the 2026-08-20 carry-forward passed it.
            model = str(embedder.get("model", "")).strip()
            if model:
                binding["embedder_model"] = model
            dimension = embedder.get("dimension")
            if isinstance(dimension, int):
                binding["embedder_dimension"] = str(dimension)
        return binding

    def load_semantic_graph(self, generation_id: str | None = None) -> SemanticGraphProjection | None:
        """Load the semantic graph for the pinned or active generation."""
        target = generation_id or self._generation_id()
        return self._with_retry(lambda conn: load_semantic_graph(conn, self._tenant, target))

    def load_generation_graph(
        self, tenant_id: str, generation_id: str
    ) -> SemanticGraphProjection | None:
        """Graph-store contract adapter with explicit tenant and generation binding."""
        if tenant_id != self._tenant:
            raise ValueError("graph store tenant does not match this GenerationStore")
        return self.load_semantic_graph(generation_id)

    def write_generation_graph(self, graph: SemanticGraphProjection) -> None:
        """Persist one complete graph atomically without changing generation metadata."""
        if graph.tenant_id != self._tenant:
            raise ValueError("graph tenant does not match this GenerationStore")

        def _op(conn: psycopg.Connection) -> None:
            with conn.transaction():
                write_semantic_graph(conn, graph)

        self._with_retry(_op)

    def graph_readiness(self, generation_id: str | None = None) -> GraphReadiness:
        """Return graph readiness without changing retrieval behavior."""
        target = generation_id or self._generation_id()

        def _op(conn: psycopg.Connection) -> GraphReadiness:
            row = conn.execute(
                "SELECT validation_summary FROM recall_generations "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, target),
            ).fetchone()
            marker = row[0].get("semantic_graph") if row and isinstance(row[0], dict) else None
            graph = load_semantic_graph(conn, self._tenant, target)
            if graph is None or not isinstance(marker, dict):
                return GraphReadiness(
                    ready=False,
                    tenant_id=self._tenant,
                    generation_id=target,
                    graph_id=None,
                    graph_fingerprint=None,
                    entity_count=0,
                    mention_count=0,
                    relation_count=0,
                    diagnostic_count=0,
                    reason="GRAPH_NOT_READY",
                )
            readiness = graph.readiness()
            if marker.get("graph_id") != readiness.graph_id or marker.get("graph_fingerprint") != readiness.graph_fingerprint:
                return replace(readiness, ready=False, reason="GRAPH_FINGERPRINT_MISMATCH")
            return readiness

        return self._with_retry(_op)

    def delete_generation_graph(self, generation_id: str | None = None) -> int:
        """Delete all derived graph rows for one generation."""
        target = generation_id or self._generation_id()

        def _op(conn: psycopg.Connection) -> int:
            return delete_semantic_graph(conn, self._tenant, target)

        return self._with_retry(_op)

    def resolve_calibration(self) -> CalibrationResolution:
        from recall.calibration_v2 import CalibrationRepository

        return CalibrationRepository(self._dsn, self._tenant, actor="generation-search").resolve(
            self._generation_id()
        )

    @staticmethod
    def _generation_rows(rows: list[tuple[Any, ...]]) -> list[ScoredChunk]:
        hits: list[ScoredChunk] = []
        for chunk_id, source_uri, text, metadata, indexed_at, score in rows:
            value = metadata if isinstance(metadata, dict) else json.loads(metadata)
            hits.append(
                ScoredChunk(
                    chunk=Chunk(
                        id=str(chunk_id),
                        source=str(source_uri),
                        text=str(text),
                        metadata=value,
                    ),
                    score=float(score),
                    indexed_at=indexed_at,
                    first_indexed_at=indexed_at,
                )
            )
        return hits

    def _query_dense(
        self, vector: list[float], k: int, source: str | None = None
    ) -> list[ScoredChunk]:
        """Generation-scoped dense search. PRIVATE on purpose: the timed public `query_dense` on
        `PgVectorStore` delegates here, so this subclass inherits the instrumentation and the
        `k <= 0` check rather than re-stating them.

        Overriding the PUBLIC method (as this did) silently drops the timing, and
        `RECALL_ENV=production` selects exactly this class — so the metric fired on a laptop and
        recorded nothing in production, with an empty series and a free store reading the same.
        """
        generation_id = self._generation_id()
        source_filter = (
            "AND (metadata->>'file' = %(source)s OR source_uri = %(source)s)" if source else ""
        )
        sql = f"""
            SELECT chunk_id, source_uri, text, metadata, indexed_at,
                   1 - (embedding <=> %(vec)s) AS score
            FROM recall_chunks_v1
            WHERE tenant_id = %(tenant)s AND generation_id = %(generation)s {source_filter}
            ORDER BY embedding <=> %(vec)s
            LIMIT %(k)s
        """
        params: dict[str, Any] = {
            "vec": Vector(vector),
            "k": k,
            "tenant": self._tenant,
            "generation": generation_id,
        }
        if source:
            params["source"] = source
        ef_search, iterative_scan = self._hnsw_filtered_tuning()

        def _op(conn: psycopg.Connection) -> list[tuple[Any, ...]]:
            with conn.transaction():
                conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                conn.execute(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}")
                return conn.execute(sql, params).fetchall()

        rows = self._with_retry(_op)
        if not rows:
            # The filter-blind walk can exhaust its candidates on other tenants' / generations'
            # rows and report an occupied generation as empty; see `_dense_exact_fallback`.
            rows = self._dense_exact_fallback(sql, params)
        return self._generation_rows(rows)

    def top_cosine(self, vector: list[float]) -> float:
        """Exact best cosine within the PINNED generation. See `PgVectorStore.top_cosine`.

        Overridden because the base implementation scopes by tenant alone, and a tenant here holds
        every generation it has ever built. Carry-forward re-scores a parent's query set against
        the CHILD generation specifically; a tenant-wide maximum would silently read the parent's
        vectors too and report that the corpus had not moved.

        `1 - min(distance)` rather than `max(1 - distance)` for the NaN reason the base method
        documents; the two forms disagree whenever one row of the scope is a zero-norm vector.

        Runs under `_EXACT_SCAN_GUARDS` for the reason the base method documents: Postgres's
        min/max optimization can otherwise hand this aggregate to the ordering index and make
        the measurement approximate.
        """
        generation_id = self._generation_id()

        def _op(conn: psycopg.Connection) -> tuple[Any, ...] | None:
            with conn.transaction():
                for guard in _EXACT_SCAN_GUARDS:
                    conn.execute(guard)
                return conn.execute(
                    "SELECT 1 - min(embedding <=> %(vec)s) FROM recall_chunks_v1 "
                    "WHERE tenant_id = %(tenant)s AND generation_id = %(generation)s",
                    {"vec": Vector(vector), "tenant": self._tenant, "generation": generation_id},
                ).fetchone()

        row = self._with_retry(_op)
        return 0.0 if row is None or row[0] is None else float(row[0])

    def _query_sparse(
        self,
        text: str,
        k: int,
        source: str | None = None,
        vec: list[float] | None = None,
    ) -> list[ScoredChunk]:
        """Generation-scoped sparse search. PRIVATE for the same reason as `_query_dense`."""
        generation_id = self._generation_id()
        source_filter = (
            "AND (c.metadata->>'file' = %(source)s OR c.source_uri = %(source)s)" if source else ""
        )
        score = "1 - (embedding <=> %(vec)s)" if vec is not None else "rank"
        sql = f"""
            WITH config AS (
                SELECT pipeline_identity #>> '{{fts_configuration,language}}' AS language
                FROM recall_generations
                WHERE tenant_id = %(tenant)s AND generation_id = %(generation)s
            ), q AS (
                SELECT (
                    SELECT string_agg(quote_literal(lexeme), ' | ')
                    FROM config,
                         unnest(to_tsvector(config.language::regconfig, %(q)s))
                )::tsquery AS tsq
            ), top_k AS (
                SELECT c.chunk_id, c.source_uri, c.text, c.metadata, c.indexed_at,
                       c.embedding, ts_rank(c.tsv, q.tsq) AS rank
                FROM recall_chunks_v1 c, q
                WHERE c.tenant_id = %(tenant)s AND c.generation_id = %(generation)s
                  AND c.tsv @@ q.tsq {source_filter}
                ORDER BY rank DESC
                LIMIT %(k)s
            )
            SELECT chunk_id, source_uri, text, metadata, indexed_at, {score} AS score
            FROM top_k ORDER BY rank DESC
        """
        params: dict[str, Any] = {
            "q": text,
            "k": k,
            "tenant": self._tenant,
            "generation": generation_id,
        }
        if vec is not None:
            params["vec"] = Vector(vec)
        if source:
            params["source"] = source
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        return self._generation_rows(rows)

    def _newest_indexed_at(self) -> datetime | None:
        """Generation-scoped freshness. PRIVATE so the timed public wrapper is inherited —
        see `store.TIMED_PUBLIC_METHODS`."""
        generation_id = self._generation_id()
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT max(indexed_at) FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchone()
        )
        return row[0] if row else None

    def _cosines_for(self, ids: Sequence[str], vec: list[float]) -> dict[str, float]:
        """Generation-scoped rescore. PRIVATE for the same reason as `_newest_indexed_at`.

        The base implementation selects `id` from `self._table`, but `recall_chunks_v1`'s
        identifier column is `chunk_id`, not `id`, and it holds rows from every generation the
        tenant has in `LIVE_MANIFEST_STATES` at once, so an unscoped query would raise
        `UndefinedColumn` and, once the column name were fixed, could still return a cosine for a
        chunk from a retired or not yet active generation, not the one being searched.
        """
        wanted = list(dict.fromkeys(str(i) for i in ids))
        generation_id = self._generation_id()

        def _op(conn: psycopg.Connection) -> list[tuple[Any, ...]]:
            return conn.execute(
                "SELECT chunk_id, 1 - (embedding <=> %s) FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s AND chunk_id = ANY(%s)",
                (Vector(vec), self._tenant, generation_id, wanted),
            ).fetchall()

        rows = self._with_retry(_op)
        return {str(row[0]): float(row[1]) for row in rows}

    def count(self) -> int:
        generation_id = self._generation_id()
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchone()
        )
        return int(row[0]) if row else 0

    def source_content_hashes(self) -> dict[str, str]:
        generation_id = self._generation_id()
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT DISTINCT source_uri, source_sha256 FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchall()
        )
        return {str(source): str(digest) for source, digest in rows}

    def sources_in_any_generation(self) -> frozenset[str]:
        """Every source this tenant has indexed, across ALL generations.

        `source_content_hashes()` is scoped to one generation, so it answers "can I read this
        now", not "does this exist". Erasure needs the second question: a source that dropped
        out of the active generation must still be erasable, while one that was never indexed
        must still be refused as a typo, because forgetting it writes a permanent tombstone
        that bars that URI from every future build.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT DISTINCT source_uri FROM recall_chunks_v1 WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchall()
        )
        return frozenset(str(row[0]) for row in rows)

    def sources_in_legacy_table(self) -> frozenset[str]:
        """Every source still held in the adopted v0.8 table for this tenant.

        Migration 0008 adopts a v0.8 install's rows IN PLACE: they never enter
        `recall_chunks_v1`, and its `legacy_unverified` generation carries a
        `{"legacy_table": ...}` manifest with no `objects`, so neither of the other two probes
        can see them. Without this an erasure request for a legacy source was answered with
        "check for typos" about data the tenant demonstrably holds.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL", (self._migration_target,)
            ).fetchone()
        )
        if not rows or not rows[0]:
            return frozenset()
        found = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT DISTINCT source FROM {self._migration_target} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchall()
        )
        return frozenset(str(row[0]) for row in found)

    def _legacy_rows_for_identifiers(self, identifiers: list[str]) -> list[tuple[Any, ...]]:
        """(source, file) pairs for the adopted v0.8 rows matching any requested identifier."""
        exists = self._with_retry(
            lambda conn: conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL", (self._migration_target,)
            ).fetchone()
        )
        if not exists or not exists[0]:
            return []
        return list(
            self._with_retry(
                lambda conn: conn.execute(
                    f"SELECT DISTINCT source, metadata->>'file' FROM {self._migration_target} "
                    "WHERE tenant_id = %s "
                    "AND (metadata->>'file' = ANY(%s) OR source = ANY(%s))",
                    (self._tenant, identifiers, identifiers),
                ).fetchall()
            )
        )

    def manifest_uris_matching(self, identifiers: list[str]) -> frozenset[str]:
        """Which of `identifiers` any live generation's manifest names.

        The one question BOTH erasure surfaces ask, so the live-state list has a single
        reader per query and the `failed` exclusion is pinned wherever an erasure resolves.
        Asking it wholesale and intersecting in Python is what let the two surfaces drift
        onto separate copies of that list, with only one of them under a test.

        The membership test runs in SQL rather than materialising every manifest: pulling the
        tenant's whole manifest set into Python to answer "is this one URI in it" costs
        seconds on a tenant with many generations, and is paid by any erasure request
        containing a single unresolved identifier, a typo included.

        Two guards that look redundant and are not. `jsonb_typeof(g.manifest->'objects')`
        sits in WHERE while `jsonb_array_elements` sits in FROM, which reads like the trap
        where the guard cannot fire; it references only `g`, so it is a baserel restriction
        applied at the scan node BEFORE the lateral function scan. It is load-bearing for a
        JSON `null` or a JSON scalar `objects`, which make `jsonb_array_elements` raise
        `cannot extract elements from a scalar` and would abort an erasure request rather
        than answer it. A MISSING `objects` key does not need it: that yields SQL NULL and
        zero rows. `jsonb_typeof(entry->'uri') = 'string'` keeps this identical to the Python
        path it replaced: `->>` casts a JSON number to text, so without it a manifest
        carrying `{"uri": 123}` made the identifier `123` resolve, and a resolved identifier
        is what `forget()` turns into a permanent tombstone.
        """
        if not identifiers:
            return frozenset()
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT DISTINCT entry->>'uri' FROM recall_generations g, "
                "jsonb_array_elements(g.manifest->'objects') entry "
                "WHERE g.tenant_id = %s "
                "AND g.state = ANY(%s) "
                "AND jsonb_typeof(g.manifest->'objects') = 'array' "
                "AND jsonb_typeof(entry->'uri') = 'string' "
                "AND entry->>'uri' = ANY(%s)",
                (self._tenant, list(LIVE_MANIFEST_STATES), identifiers),
            ).fetchall()
        )
        return frozenset(str(row[0]) for row in rows)

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        edges, unresolved, _candidates = self.supersession_all()
        return edges, unresolved

    def supersession_all(
        self,
    ) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
        generation_id = self._generation_id()
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT metadata->>'file', metadata->>'supersedes', min(indexed_at) "
                "FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
                "AND metadata ? 'file' GROUP BY 1, 2 ORDER BY 1, 2",
                (self._tenant, generation_id),
            ).fetchall()
        )
        edges, unresolved, candidates = resolve_supersession_candidates(rows)
        return (
            dict(edges),
            unresolved,
            {target: list(claims) for target, claims in candidates.items()},
        )

    def sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
        """Resolve erasure identifiers to source URIs, across everything the tenant holds.

        Scoped to the ACTIVE generation this used to disagree with the CLI about what could be
        erased, and to raise `NoActiveGeneration` on a tenant whose rows were adopted in place
        by migration 0008 and never entered `recall_chunks_v1` — so the MCP `recall_forget`
        left that data on disk while `recall forget` erased it. Its one consumer,
        `forget_memory`, deletes through `GenerationManager.forget`, which sweeps every
        generation anyway, so resolving narrower than that only ever lost erasures.
        """
        if not identifiers:
            return {}
        rows = list(
            self._with_retry(
                lambda conn: conn.execute(
                    "SELECT DISTINCT source_uri, metadata->>'file' FROM recall_chunks_v1 "
                    "WHERE tenant_id = %s "
                    "AND (metadata->>'file' = ANY(%s) OR source_uri = ANY(%s))",
                    (self._tenant, identifiers, identifiers),
                ).fetchall()
            )
        )
        rows += self._legacy_rows_for_identifiers(identifiers)
        requested = set(identifiers)
        resolved: dict[str, list[str]] = {}
        for source, file in rows:
            for identifier in {file, source} & requested:
                # De-duplicate, as the base implementation does. Now that this spans every
                # generation and the legacy table, one source legitimately appears more than
                # once: under different `metadata->>'file'` values in different generations, or
                # in both recall_chunks_v1 and the adopted v0.8 table.
                bucket = resolved.setdefault(str(identifier), [])
                if str(source) not in bucket:
                    bucket.append(str(source))
        unresolved = requested - set(resolved)
        if unresolved:
            # Chunk rows are not the corpus. An object that chunks to nothing has no rows, and
            # during a build NOTHING has rows yet for any object not already committed --
            # `build()` opens a transaction per manifest entry. Resolving on rows alone meant
            # an erasure issued through MCP mid-build was answered "not found", wrote no
            # tombstone, and the build then indexed the content the user asked to erase, while
            # the CLI (which consults the manifest) erased it. The two surfaces must agree.
            for identifier in self.manifest_uris_matching(sorted(unresolved)):
                resolved[identifier] = [identifier]
        return resolved

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        generation_id = self._generation_id()
        # The explicit transaction is NOT optional, and its absence here made five MCP tools
        # unusable under `RECALL_ENV=production` — the only mode that selects this class. A
        # server-side cursor is transaction-scoped, and these connections are autocommit, so
        # `DECLARE CURSOR` fails outright. `PgVectorStore.iter_chunks` has carried the same
        # `conn.transaction()` and the same reasoning all along; this override inherited the
        # cursor and dropped the transaction around it.
        with (
            self._borrowed() as conn,
            conn.transaction(),
            conn.cursor(name=f"recall_gen_{uuid.uuid4().hex[:12]}") as cur,
        ):
            with conn.transaction():
                cur.execute(
                    "SELECT chunk_id, source_uri, text, metadata FROM recall_chunks_v1 "
                    "WHERE tenant_id = %s AND generation_id = %s ORDER BY chunk_id",
                    (self._tenant, generation_id),
                )
                while rows := cur.fetchmany(batch_size):
                    for chunk_id, source, text, metadata in rows:
                        value = metadata if isinstance(metadata, dict) else json.loads(metadata)
                        yield Chunk(str(chunk_id), str(source), str(text), value)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        raise ImmutableGenerationError("active generations are read only")

    def replace_sources(
        self, sources: list[str], chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        raise ImmutableGenerationError("active generations are read only")

    def delete_sources(self, sources: list[str]) -> int:
        from recall.generations import GenerationManager

        manager = GenerationManager(
            self._dsn,
            self._tenant,
            actor="generation-store-forget",
        )
        return sum(
            manager.forget(source, legacy_table=self._migration_target).chunks_removed
            for source in sources
        )

    def touch_files(self, files: list[str]) -> int:
        raise ImmutableGenerationError("immutable generations cannot be touched")
