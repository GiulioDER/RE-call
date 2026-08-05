"""Versioned generation routing and durable shadow migration coordination."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Literal
from typing import TYPE_CHECKING
import threading

import psycopg

if TYPE_CHECKING:
    from recall.store import PgVectorStore

GenerationState = Literal["building", "ready", "active", "retired", "failed"]
EventKind = Literal["index", "forget"]


def validate_table_name(table: str) -> str:
    if not table.isidentifier():
        raise ValueError("physical table must be a valid SQL identifier")
    return table


@dataclass(frozen=True)
class IndexGeneration:
    generation_id: str
    physical_table: str
    embedding_profile: str
    dimension: int
    state: GenerationState
    chunk_count: int
    source_count: int
    created_at: datetime
    ready_at: datetime | None


@dataclass(frozen=True)
class TenantRoute:
    tenant_id: str
    active: IndexGeneration
    shadow: IndexGeneration | None
    updated_at: datetime


@dataclass(frozen=True)
class MigrationEvent:
    sequence_id: int
    tenant_id: str
    operation_id: str
    operation_kind: EventKind
    status: str
    payload: dict[str, object] | None
    active_count: int
    shadow_count: int


class ControlPlane:
    """Small PostgreSQL control plane. Runtime calls are always tenant scoped."""

    def __init__(self, dsn: str, *, connect_timeout_s: int = 10) -> None:
        self._dsn = dsn
        self._connect_timeout_s = connect_timeout_s

    def _connect(self) -> "psycopg.Connection":
        return psycopg.connect(
            self._dsn, autocommit=True, connect_timeout=self._connect_timeout_s
        )

    def watch_routes(
        self, callback: Callable[[str], None], stop: threading.Event
    ) -> None:
        """Deliver route invalidations through LISTEN, reconnecting after transient failures."""
        while not stop.is_set():
            try:
                with self._connect() as conn:
                    conn.execute("LISTEN recall_route_changed")
                    while not stop.is_set():
                        for notification in conn.notifies(timeout=5.0, stop_after=1):
                            callback(notification.payload)
            except psycopg.Error:
                stop.wait(1.0)

    @staticmethod
    def _set_tenant(conn: "psycopg.Connection", tenant: str) -> None:
        if not tenant:
            raise ValueError("tenant must be non-empty")
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))

    def apply_migrations(self) -> None:
        """Apply checked, immutable SQL migrations using the caller's migration role."""
        directory = Path(__file__).with_name("sql")
        migrations = sorted(directory.glob("[0-9][0-9][0-9]_*.sql"))
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS recall_schema_versions ("
                "version integer PRIMARY KEY, checksum text NOT NULL, "
                "applied_at timestamptz NOT NULL DEFAULT now())"
            )
            for path in migrations:
                version = int(path.name.split("_", 1)[0])
                sql = path.read_text(encoding="utf-8")
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                row = conn.execute(
                    "SELECT checksum FROM recall_schema_versions WHERE version = %s",
                    (version,),
                ).fetchone()
                if row is not None:
                    if row[0] != checksum:
                        raise RuntimeError(f"migration {version} checksum does not match")
                    continue
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute(
                        "INSERT INTO recall_schema_versions(version, checksum) VALUES (%s, %s)",
                        (version, checksum),
                    )

    def register_generation(
        self,
        generation_id: str,
        physical_table: str,
        embedding_profile: str,
        dimension: int,
        state: GenerationState = "building",
    ) -> None:
        validate_table_name(physical_table)
        if not generation_id or not embedding_profile or dimension < 1:
            raise ValueError("generation identity, profile, and dimension are required")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO recall_index_generations "
                "(generation_id, physical_table, embedding_profile, dimension, state) "
                "VALUES (%s, %s, %s, %s, %s)",
                (generation_id, physical_table, embedding_profile, dimension, state),
            )

    def set_generation_state(
        self,
        generation_id: str,
        state: GenerationState,
        *,
        chunk_count: int | None = None,
        source_count: int | None = None,
    ) -> None:
        if ((chunk_count is not None and chunk_count < 0)
                or (source_count is not None and source_count < 0)):
            raise ValueError("generation counts cannot be negative")
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE recall_index_generations SET state = %s, "
                "chunk_count = COALESCE(%s, chunk_count), source_count = COALESCE(%s, source_count), "
                "ready_at = CASE WHEN %s = 'ready' THEN now() ELSE ready_at END, "
                "retired_at = CASE WHEN %s = 'retired' THEN now() ELSE retired_at END "
                "WHERE generation_id = %s RETURNING generation_id",
                (state, chunk_count, source_count, state, state, generation_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown generation: {generation_id}")

    @staticmethod
    def _generation(row: tuple) -> IndexGeneration:
        return IndexGeneration(
            generation_id=row[0], physical_table=validate_table_name(row[1]),
            embedding_profile=row[2], dimension=row[3], state=row[4],
            chunk_count=row[5], source_count=row[6], created_at=row[7], ready_at=row[8],
        )

    def route(self, tenant: str) -> TenantRoute | None:
        with self._connect() as conn:
            self._set_tenant(conn, tenant)
            row = conn.execute(
                "SELECT r.tenant_id, r.updated_at, "
                "a.generation_id, a.physical_table, a.embedding_profile, a.dimension, a.state, "
                "a.chunk_count, a.source_count, a.created_at, a.ready_at, "
                "s.generation_id, s.physical_table, s.embedding_profile, s.dimension, s.state, "
                "s.chunk_count, s.source_count, s.created_at, s.ready_at "
                "FROM recall_tenant_routes r "
                "JOIN recall_index_generations a ON a.generation_id = r.active_generation "
                "LEFT JOIN recall_index_generations s ON s.generation_id = r.shadow_generation "
                "WHERE r.tenant_id = %s",
                (tenant,),
            ).fetchone()
        if row is None:
            return None
        active = self._generation(tuple(row[2:11]))
        shadow = self._generation(tuple(row[11:20])) if row[11] is not None else None
        return TenantRoute(row[0], active, shadow, row[1])

    def set_route(
        self, tenant: str, active_generation: str, shadow_generation: str | None = None
    ) -> None:
        """Migration-role route update followed by a content-free notification."""
        with self._connect() as conn, conn.transaction():
            # Routes are FORCE RLS protected, including from their table owner.  The
            # dedicated migration role is intentionally neither superuser nor BYPASSRLS,
            # so it must establish the same tenant GUC as runtime readers before writing.
            self._set_tenant(conn, tenant)
            state_rows = conn.execute(
                    "SELECT generation_id, state FROM recall_index_generations "
                    "WHERE generation_id = ANY(%s)",
                    ([value for value in (active_generation, shadow_generation) if value],),
                ).fetchall()
            states: dict[str, str] = {str(row[0]): str(row[1]) for row in state_rows}
            if states.get(active_generation) not in {"ready", "active"}:
                raise ValueError("active generation must be ready or active")
            if shadow_generation and states.get(shadow_generation) not in {"building", "ready"}:
                raise ValueError("shadow generation must be building or ready")
            conn.execute(
                "INSERT INTO recall_tenant_routes "
                "(tenant_id, active_generation, shadow_generation) VALUES (%s, %s, %s) "
                "ON CONFLICT (tenant_id) DO UPDATE SET active_generation = EXCLUDED.active_generation, "
                "shadow_generation = EXCLUDED.shadow_generation, updated_at = now()",
                (tenant, active_generation, shadow_generation),
            )
            conn.execute("SELECT pg_notify('recall_route_changed', %s)", (tenant,))

    def append_event(
        self,
        tenant: str,
        operation_id: str,
        operation_kind: EventKind,
        payload: dict[str, object],
        *,
        active_count: int = 0,
    ) -> int:
        """Append once by operation ID. Retries return the original ordered sequence."""
        with self._connect() as conn:
            self._set_tenant(conn, tenant)
            row = conn.execute(
                "INSERT INTO recall_migration_events "
                "(tenant_id, operation_id, operation_kind, payload, active_count) "
                "VALUES (%s, %s, %s, %s::jsonb, %s) "
                "ON CONFLICT (tenant_id, operation_id) DO UPDATE SET operation_id = EXCLUDED.operation_id "
                "RETURNING sequence_id",
                (tenant, operation_id, operation_kind, json.dumps(payload), active_count),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def pending_events(self, tenant: str) -> list[MigrationEvent]:
        with self._connect() as conn:
            self._set_tenant(conn, tenant)
            rows = conn.execute(
                "SELECT sequence_id, tenant_id, operation_id, operation_kind, status, payload, "
                "active_count, shadow_count FROM recall_migration_events "
                "WHERE tenant_id = %s AND status = 'pending' ORDER BY sequence_id",
                (tenant,),
            ).fetchall()
        return [MigrationEvent(*row) for row in rows]

    def complete_event(self, tenant: str, operation_id: str, shadow_count: int) -> None:
        """Mark replay complete and erase the potentially sensitive replay payload."""
        with self._connect() as conn:
            self._set_tenant(conn, tenant)
            row = conn.execute(
                "UPDATE recall_migration_events SET status = 'complete', payload = NULL, "
                "shadow_count = %s, completed_at = now() "
                "WHERE tenant_id = %s AND operation_id = %s AND status = 'pending' "
                "RETURNING sequence_id",
                (shadow_count, tenant, operation_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"pending migration event not found: {operation_id}")

    def _require_parity(
        self, conn: psycopg.Connection, tenant: str, active: str, shadow: str
    ) -> None:
        """Refuse a cutover whose shadow does not match the generation it replaces.

        `recall/migration.py` has carried `validate_generation_parity` since the enterprise
        program landed, with a docstring saying it runs "before an enterprise cutover", and no
        caller anywhere in the package. This is that caller.
        """
        from recall.migration import validate_generation_parity
        from recall.store import PgVectorStore

        rows = {
            str(r[0]): (str(r[1]), int(r[2]), str(r[3]), int(r[4]))
            for r in conn.execute(
                "SELECT generation_id, physical_table, dimension, embedding_profile, "
                "chunk_count FROM recall_index_generations WHERE generation_id = ANY(%s)",
                ([active, shadow],),
            ).fetchall()
        }
        if active not in rows or shadow not in rows:
            raise RuntimeError("cutover requires both generations to be registered")
        active_table, active_dim, _active_profile, _active_chunks = rows[active]
        shadow_table, shadow_dim, _shadow_profile, declared_chunks = rows[shadow]
        # Deliberately NOT comparing embedding_profile or dimension. Re-indexing onto a new
        # embedder is the main reason to build a shadow generation at all, and
        # `validate_generation_parity` is built for exactly that: it compares source sets and
        # raw content hashes "while allowing embeddings and metadata to differ". Requiring
        # equality here would refuse the workflow this machinery exists to serve.
        validate_table_name(active_table)
        validate_table_name(shadow_table)
        with (
            PgVectorStore(self._dsn, dim=active_dim, table=active_table, tenant=tenant) as before,
            PgVectorStore(self._dsn, dim=shadow_dim, table=shadow_table, tenant=tenant) as after,
        ):
            parity = validate_generation_parity(before, after)
        if parity.shadow_chunks == 0:
            # Parity alone passes vacuously here: an empty shadow over an empty active is
            # 0 == 0, so the gate would report "verified" having verified nothing, and promote
            # a generation whose `mark-ready` counts were pure assertion. An empty shadow is
            # never a corpus worth serving, whatever the active holds.
            raise RuntimeError(
                "cutover refused: the shadow generation holds no rows "
                f"(recall_index_generations declares chunk_count={declared_chunks})"
            )
        if not parity.valid:
            raise RuntimeError(
                "cutover refused: " + "; ".join(parity.failures)
                + " (re-run with --allow-divergent-corpus only if the corpus change is"
                " intended)"
            )

    def cutover(self, tenant: str, *, allow_divergent_corpus: bool = False) -> None:
        """Promote the ready shadow only when its ordered outbox has no lag.

        `state = 'ready'` is an operator ASSERTION, not a measurement: `mark-ready` stores its
        `--chunks`/`--sources` argparse ints verbatim and compares them to nothing. On its own
        it says nothing about what the shadow table holds, so an empty generation could be
        marked ready and cut over, sending every read for the tenant to an empty index. This
        therefore compares the two generations before swapping.
        """
        with self._connect() as conn, conn.transaction():
            self._set_tenant(conn, tenant)
            pending = conn.execute(
                "SELECT count(*) FROM recall_migration_events "
                "WHERE tenant_id = %s AND status = 'pending'", (tenant,)
            ).fetchone()
            if pending and pending[0]:
                raise RuntimeError("cutover refused while migration events remain pending")
            row = conn.execute(
                "SELECT active_generation, shadow_generation FROM recall_tenant_routes "
                "WHERE tenant_id = %s FOR UPDATE",
                (tenant,),
            ).fetchone()
            if row is None or row[1] is None:
                raise RuntimeError("cutover requires a configured shadow generation")
            state = conn.execute(
                "SELECT state FROM recall_index_generations WHERE generation_id = %s", (row[1],)
            ).fetchone()
            if state is None or state[0] != "ready":
                raise RuntimeError("cutover requires a ready shadow generation")
            if not allow_divergent_corpus:
                self._require_parity(conn, tenant, str(row[0]), str(row[1]))
            conn.execute(
                "UPDATE recall_tenant_routes SET active_generation = shadow_generation, "
                "shadow_generation = active_generation, updated_at = now() WHERE tenant_id = %s",
                (tenant,),
            )
            conn.execute("SELECT pg_notify('recall_route_changed', %s)", (tenant,))

    def replay_pending(
        self, tenant: str, stores: dict[str, "PgVectorStore"]
    ) -> int:
        """Replay ordered, idempotent shadow writes and clear completed payloads."""
        from recall.types import Chunk

        completed = 0
        for event in self.pending_events(tenant):
            payload = event.payload or {}
            raw_sources = payload.get("sources", [])
            if not isinstance(raw_sources, list):
                raise ValueError("migration event sources must be a list")
            sources = [str(value) for value in raw_sources]
            if event.operation_kind == "index":
                active_generation = str(payload.get("active_generation", ""))
                shadow_generation = str(payload.get("shadow_generation", ""))
                active_store = stores.get(active_generation)
                shadow_store = stores.get(shadow_generation)
                if active_store is None or shadow_store is None:
                    raise KeyError("replay stores for active and shadow generations are required")

                def _decode(key: str) -> tuple[list[Chunk], list[list[float]]]:
                    records = payload.get(key, [])
                    if not isinstance(records, list):
                        raise ValueError(f"migration event {key} must be a list")
                    decoded_chunks: list[Chunk] = []
                    decoded_vectors: list[list[float]] = []
                    for record in records:
                        if not isinstance(record, dict):
                            raise ValueError("migration event chunk must be an object")
                        metadata = record.get("metadata")
                        embedding = record.get("embedding")
                        if not isinstance(metadata, dict) or not isinstance(embedding, list):
                            raise ValueError(
                                "migration event chunk metadata or embedding is invalid"
                            )
                        decoded_chunks.append(
                            Chunk(
                                id=str(record["id"]), source=str(record["source"]),
                                text=str(record["text"]), metadata=dict(metadata),
                            )
                        )
                        decoded_vectors.append([float(value) for value in embedding])
                    return decoded_chunks, decoded_vectors

                active_chunks, active_vectors = _decode("active_chunks")
                chunks, vectors = _decode("chunks")
                active_store.replace_sources(sources, active_chunks, active_vectors)
                count = shadow_store.replace_sources(sources, chunks, vectors)
            else:
                generation = str(payload.get("active_generation", ""))
                store = stores.get(generation)
                if store is None:
                    raise KeyError(f"no replay store for active generation {generation!r}")
                count = store.delete_sources(sources)
            self.complete_event(tenant, event.operation_id, count)
            completed += 1
        return completed
