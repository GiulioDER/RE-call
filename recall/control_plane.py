"""Versioned generation routing and durable shadow migration coordination."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
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

#: States a generation may be in and still back a served request. `retired` and `failed` are
#: excluded deliberately: `docs/ENTERPRISE_RETRIEVAL.md` requires that no request ever name a
#: retired table, and a `failed` generation is one whose DDL did not finish.
SERVABLE_STATES: frozenset[str] = frozenset({"building", "ready", "active"})

#: The ACTIVE slot is narrower, and it is derived from `set_route`'s own gate rather than
#: restated: a route may only be pointed at a `ready` or `active` generation, so serving one
#: that is still `building` would be wider than the write path that created the route.
#: `building` remains servable for the SHADOW slot, where a dual write opens a generation that
#: is by definition still being built.
SERVABLE_ACTIVE_STATES: frozenset[str] = frozenset({"ready", "active"})

#: Advisory lock name for the control-plane ledger. Deliberately NOT `recall/schema.py`'s
#: `MIGRATION_LOCK_NAME`: the two ledgers are separate on purpose (see `docs/MIGRATIONS.md`), and
#: sharing one lock would make a chunk-table migration and a control-plane bootstrap block each
#: other for no reason. Separate ledger, separate lock, both locked.
CONTROL_PLANE_LOCK_NAME = "recall-control-plane-migrations-v1"

#: Advisory lock guarding one tenant's outbox. Both the replay drain and the erasure scrub take
#: it, because they are the two writers of `recall_migration_events.payload` and the ONLY thing
#: that can order them is a lock they share. `SELECT ... FOR UPDATE` cannot: `pending_events` is a
#: plain non-locking read, and under READ COMMITTED a plain SELECT is never blocked by a row lock,
#: so the drain would read the pre-scrub snapshot and replay erased text back into both
#: generations. A comment claiming FOR UPDATE prevented that was wrong, which is the failure mode
#: this project keeps hitting: a guard that reads as protection and cannot fire.
OUTBOX_LOCK_PREFIX = "recall-outbox-v1"

#: Physical table identifiers are interpolated into SQL, so this is an allowlist, not a filter.
#:
#: `str.isidentifier()` was the previous gate and is too weak in three separate ways, each a live
#: defect rather than a hypothetical. It accepts non-ASCII, because `café` is a valid Python
#: identifier. It accepts uppercase, and PostgreSQL folds an unquoted `Chunks_G1` to `chunks_g1`,
#: so the registry row and the physical table silently disagree. It accepts any length, and
#: PostgreSQL truncates identifiers at NAMEDATALEN-1 = 63 bytes, so two registry rows differing
#: only after byte 63 map to ONE table. The allowlist below refuses all three.
#:
#: The bound is 46, not 63, because nothing is named by the table name alone. Every derived
#: object suffixes it, and the longest suffix shipped is `_tenant_isolation` at 17 bytes
#: (`recall/migrations/sql/0001_v08_baseline.sql`). A 63-byte table therefore yields an 80-byte
#: policy or index name that PostgreSQL silently truncates, and `readiness_facts` compares index
#: names by EXACT string, so readiness would report a missing index for one that exists and is
#: valid. A bound is named by what it bounds: this one bounds the derived identifier.
_TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,45}$")


def validate_table_name(table: str) -> str:
    """Return `table` if it matches the physical-identifier allowlist, else raise.

    This is the chokepoint for every table name that reaches an f-string THROUGH THE CONTROL
    PLANE: it runs on the way IN (`register_generation`) and again on the way OUT (`_generation`),
    so a row written by another client, or by a direct `INSERT`, cannot smuggle an identifier into
    a query at read time.

    It is not yet the only gate in the codebase. `PgVectorStore.__init__` and
    `recall.schema._validate_target` still use `str.isidentifier()`, so a store constructed
    directly, or `recall schema apply --table`, can still carry a name this allowlist would
    refuse. Those two are a deliberate follow-up rather than an oversight: tightening them changes
    behaviour for tables that already exist, which needs a compatibility decision this change does
    not make. Do not describe this function as the single chokepoint until they are converted.
    """
    if not isinstance(table, str) or not _TABLE_NAME.fullmatch(table):
        raise ValueError(
            "physical table must match ^[a-z_][a-z0-9_]{0,45}$ (lowercase ASCII, at most 46 "
            f"bytes, leaving room for the 17-byte `_tenant_isolation` suffix); got {table!r}"
        )
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


class ConcurrentControlPlaneMigrator(RuntimeError):
    """Another process holds the control-plane migration advisory lock."""


class ConcurrentOutboxDrain(RuntimeError):
    """Another process is replaying or scrubbing this tenant's migration outbox."""


@dataclass(frozen=True)
class ControlPlaneLedgerState:
    """What `recall_schema_versions` says, against what this package ships."""

    ledger_present: bool
    missing: tuple[int, ...]
    unknown: tuple[int, ...]
    checksum_mismatches: tuple[int, ...]

    @property
    def current(self) -> bool:
        """Whether this database's control plane is safe to serve against.

        `unknown` is deliberately NOT a failure. A database carrying migrations this package does
        not ship means the DATABASE IS AHEAD, which is the normal and intended state during a
        staged rollout and after an application rollback: migrate first, then roll the code.
        Treating it as fatal would mean a forward migration bricks every replica still running the
        previous release, turning a routine rollback into an outage. `missing` and
        `checksum_mismatches` stay fatal, because those mean the code expects schema the database
        does not have, or that committed bytes changed under it.
        """
        return self.ledger_present and not self.missing and not self.checksum_mismatches

    @property
    def ahead(self) -> bool:
        """The database has migrations this package does not ship. Degraded, not fatal."""
        return bool(self.unknown)

    def describe(self) -> str:
        if not self.ledger_present:
            return "control plane ledger recall_schema_versions is absent"
        parts = []
        if self.missing:
            parts.append(f"missing migrations {list(self.missing)}")
        if self.unknown:
            parts.append(f"unknown migrations {list(self.unknown)}")
        if self.checksum_mismatches:
            parts.append(f"checksum mismatch on {list(self.checksum_mismatches)}")
        return "control plane ledger is current" if not parts else "; ".join(parts)


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

    @staticmethod
    def _outbox_lock_key(tenant: str) -> str:
        return f"{OUTBOX_LOCK_PREFIX}:{tenant}"

    @staticmethod
    @lru_cache(maxsize=1)
    def bundled_migrations() -> tuple[tuple[int, str, str], ...]:
        """`(version, sql, sha256)` for every control-plane migration shipped in this package."""
        directory = Path(__file__).with_name("sql")
        bundled: list[tuple[int, str, str]] = []
        for path in sorted(directory.glob("[0-9][0-9][0-9]_*.sql")):
            sql = path.read_text(encoding="utf-8")
            bundled.append(
                (int(path.name.split("_", 1)[0]), sql, hashlib.sha256(sql.encode("utf-8")).hexdigest())
            )
        return tuple(bundled)

    def apply_migrations(self) -> None:
        """Apply checked, immutable SQL migrations using the caller's migration role.

        Serialised by a PostgreSQL advisory lock, for the same reason `recall schema apply` takes
        one: two `recall-enterprise migrate` jobs against one database otherwise interleave, and
        `CREATE TABLE IF NOT EXISTS` plus a ledger `INSERT` is not atomic across sessions. The
        loser raises rather than waiting, because the winner is doing the identical work and the
        caller wants to know a second migrator existed.
        """
        with self._connect() as conn:
            lock = conn.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (CONTROL_PLANE_LOCK_NAME,)
            ).fetchone()
            if not lock or not lock[0]:
                raise ConcurrentControlPlaneMigrator(
                    "another recall-enterprise migrate is already running against this database"
                )
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS recall_schema_versions ("
                    "version integer PRIMARY KEY, checksum text NOT NULL, "
                    "applied_at timestamptz NOT NULL DEFAULT now())"
                )
                for version, sql, checksum in self.bundled_migrations():
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
                            "INSERT INTO recall_schema_versions(version, checksum) "
                            "VALUES (%s, %s)",
                            (version, checksum),
                        )
            finally:
                conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (CONTROL_PLANE_LOCK_NAME,),
                )

    def ledger_state(self) -> "ControlPlaneLedgerState":
        """Compare the shipped control-plane migrations against what this database recorded.

        Read-only, and deliberately separate from `recall.schema.check_schema`: that one covers
        `recall_schema_migrations` (per chunk table), this one covers `recall_schema_versions`
        (database-global). Enterprise readiness checks BOTH, because a process whose control plane
        is behind routes requests using a schema it has not verified.
        """
        bundled = {version: checksum for version, _sql, checksum in self.bundled_migrations()}
        with self._connect() as conn:
            present = conn.execute(
                "SELECT to_regclass('recall_schema_versions')"
            ).fetchone()
            if not present or present[0] is None:
                return ControlPlaneLedgerState(
                    ledger_present=False,
                    missing=tuple(sorted(bundled)),
                    unknown=(),
                    checksum_mismatches=(),
                )
            recorded = {
                int(version): str(checksum)
                for version, checksum in conn.execute(
                    "SELECT version, checksum FROM recall_schema_versions"
                ).fetchall()
            }
        return ControlPlaneLedgerState(
            ledger_present=True,
            missing=tuple(sorted(set(bundled) - set(recorded))),
            unknown=tuple(sorted(set(recorded) - set(bundled))),
            checksum_mismatches=tuple(
                sorted(v for v in set(bundled) & set(recorded) if bundled[v] != recorded[v])
            ),
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

    _GENERATION_COLUMNS = (
        "generation_id, physical_table, embedding_profile, dimension, state, "
        "chunk_count, source_count, created_at, ready_at"
    )

    def generation(self, generation_id: str) -> IndexGeneration | None:
        """One validated registry row, or None. The ONLY sanctioned source of a physical table."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._GENERATION_COLUMNS} FROM recall_index_generations "
                "WHERE generation_id = %s",
                (generation_id,),
            ).fetchone()
        return None if row is None else self._generation(tuple(row))

    def generations(self) -> list[IndexGeneration]:
        """Every VALID registry row, oldest first. `recall_index_generations` is not tenant scoped.

        Invalid rows are skipped here and reported by `invalid_generations`, deliberately. The
        identifier allowlist validates on the way OUT, so a row written under the older, weaker
        `str.isidentifier()` rule raises when it is read. Letting that escape from here made the
        one command an operator would reach for to FIND the bad row (`recall-enterprise status`)
        the command that crashes on it. A diagnostic that dies on the thing it diagnoses is not a
        diagnostic. Serving still refuses: `route()` validates the routed rows and is unchanged.
        """
        valid, _invalid = self._read_generations()
        return valid

    def invalid_generations(self) -> list[tuple[str, str, str]]:
        """`(generation_id, physical_table, reason)` for rows the allowlist would reject.

        The upgrade preflight. A deployment carrying such a row will fail to serve that tenant
        after this change, and the operator needs to see WHICH row before that happens.
        """
        _valid, invalid = self._read_generations()
        return invalid

    def _read_generations(self) -> tuple[list[IndexGeneration], list[tuple[str, str, str]]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self._GENERATION_COLUMNS} FROM recall_index_generations "
                "ORDER BY created_at, generation_id"
            ).fetchall()
        valid: list[IndexGeneration] = []
        invalid: list[tuple[str, str, str]] = []
        for row in rows:
            try:
                valid.append(self._generation(tuple(row)))
            except ValueError as exc:
                invalid.append((str(row[0]), str(row[1]), str(exc)))
        return valid, invalid

    def retire_generation(self, generation_id: str, tenant: str) -> None:
        """Retire a generation, refusing while `tenant`'s route still points at it.

        Scoped to ONE tenant on purpose. `recall_tenant_routes` carries FORCE row level security
        and the migration role is neither superuser nor `BYPASSRLS`, so no caller can enumerate
        every tenant's routes to prove a generation is globally unrouted. Rather than weaken the
        isolation model to make a convenient check possible, the operator names the tenant they
        are retiring for, and the *serving* path refuses a retired generation independently
        (`SERVABLE_STATES`, enforced in `recall_mcp.stores.StoreRegistry`). That second guard is
        the one that actually protects a request; this one is here to stop the obvious mistake.
        """
        with self._connect() as conn, conn.transaction():
            self._set_tenant(conn, tenant)
            row = conn.execute(
                "SELECT active_generation, shadow_generation FROM recall_tenant_routes "
                "WHERE tenant_id = %s FOR UPDATE",
                (tenant,),
            ).fetchone()
            if row is None:
                # The refusal is the whole point of naming a tenant, so an absent route must not
                # be the quiet path through it. A mistyped `--tenant` otherwise retired the
                # generation unconditionally and still printed success.
                raise KeyError(
                    f"tenant {tenant!r} has no route in this control plane; refusing to confirm "
                    f"retirement of {generation_id!r} against a tenant it cannot check"
                )
            if generation_id in {row[0], row[1]}:
                role = "active" if generation_id == row[0] else "shadow"
                raise RuntimeError(
                    f"refusing to retire {generation_id!r}: it is tenant {tenant!r}'s {role} "
                    f"generation. Route the tenant elsewhere first."
                )
            updated = conn.execute(
                "UPDATE recall_index_generations SET state = 'retired', retired_at = now() "
                "WHERE generation_id = %s RETURNING generation_id",
                (generation_id,),
            ).fetchone()
            if updated is None:
                raise KeyError(f"unknown generation: {generation_id}")
            # Every other route-affecting write notifies. Without this the serving refusal does
            # not bind until each process's route cache expires, so a retired generation keeps
            # answering requests for up to `route_poll_seconds`.
            conn.execute("SELECT pg_notify('recall_route_changed', %s)", (tenant,))

    def erase_sources_from_pending(self, tenant: str, sources: list[str]) -> int:
        """Scrub erased sources out of pending outbox payloads. Returns events changed.

        Erasure previously stopped at the chunk tables, and a pending migration event's payload
        holds the full text and the vectors of every chunk in the batch
        (`recall/index.py::Indexer._flush`). A tenant who invoked their right to erasure while an
        index event was pending kept their text in `recall_migration_events.payload` until an
        unrelated replay happened to complete it, and a replay that DID run would have written the
        erased text back into both generations.

        Scrubbing rather than discarding, because one event covers a batch of sources and only
        some of them are being erased. When nothing is left to replay the event is completed with
        a zero shadow count: its work is genuinely void, and leaving it pending would block
        `cutover` forever for an operation that must never be replayed.
        """
        if not sources:
            return 0
        erased = set(sources)
        changed = 0
        # The lock, not `FOR UPDATE`, is what excludes a concurrent drain. `pending_events` is a
        # plain non-locking read, and under READ COMMITTED a plain SELECT is never blocked by a
        # row lock, so a replay would happily read the pre-scrub payload and write the erased text
        # back into both generations. `replay_pending` takes this same lock; `FOR UPDATE` below
        # only orders this scrub against another scrub.
        #
        # BLOCKING here, unlike the drain's try-lock: erasure is irreversible and legally loaded,
        # so waiting for a drain to finish is right and skipping the scrub is not. `lock_timeout`
        # bounds the wait so a stuck drain surfaces as an error rather than a hang.
        with self._connect() as conn:
            conn.execute("SET lock_timeout = '30s'")
            conn.execute(
                "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                (self._outbox_lock_key(tenant),),
            )
            try:
                changed = self._scrub_locked(conn, tenant, erased)
            finally:
                conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (self._outbox_lock_key(tenant),),
                )
        return changed

    def _scrub_locked(
        self, conn: "psycopg.Connection", tenant: str, erased: set[str]
    ) -> int:
        changed = 0
        with conn.transaction():
            self._set_tenant(conn, tenant)
            # Filtered SERVER SIDE. The payload of one event carries every chunk's text AND its
            # embedding for a batch of up to DEFAULT_BATCH_CHUNKS, which is megabytes; pulling
            # every pending payload back to filter in Python put that on the forget request path
            # even when nothing matched. `jsonb_exists_any` is the function spelling of `?|`,
            # used to keep the `?` out of a psycopg query string.
            rows = conn.execute(
                "SELECT operation_id, payload FROM recall_migration_events "
                "WHERE tenant_id = %s AND status = 'pending' "
                "AND (jsonb_exists_any(payload -> 'sources', %s) "
                # `payload -> 'sources' IS NULL` is NOT enough: jsonb null is not SQL
                # NULL, so `{"sources": null}` and a non-array `sources` were filtered out
                # server side and never reached the chunk-derived fallback below. Selecting
                # on the TYPE keeps the optimisation for well-formed arrays while letting
                # every malformed shape through to be scrubbed.
                "     OR jsonb_typeof(payload -> 'sources') IS DISTINCT FROM 'array') "
                "ORDER BY sequence_id FOR UPDATE",
                (tenant, list(erased)),
            ).fetchall()
            for operation_id, payload in rows:
                if not isinstance(payload, dict):
                    continue
                # Derived, not declared. A payload whose `sources` is absent or JSON null still
                # carries the text in its chunk records, and trusting the declared list meant a
                # malformed event kept the erased text (or raised TypeError mid-scrub, after the
                # irreversible chunk deletes had already committed).
                declared = payload.get("sources")
                declared_list = [str(s) for s in declared] if isinstance(declared, list) else []
                present = set(declared_list)
                for key in ("active_chunks", "chunks"):
                    records = payload.get(key)
                    if isinstance(records, list):
                        present |= {
                            str(r["source"]) for r in records
                            if isinstance(r, dict) and "source" in r
                        }
                if not present & erased:
                    continue
                remaining = [s for s in declared_list if s not in erased]
                if not declared_list:
                    remaining = sorted(present - erased)
                if not remaining:
                    # active_count goes to 0 with the payload. Leaving it at N would make a
                    # voided event byte-identical, in the retained audit record, to a shadow
                    # write that lost all N chunks.
                    conn.execute(
                        "UPDATE recall_migration_events SET status = 'complete', payload = NULL, "
                        "active_count = 0, shadow_count = 0, completed_at = now() "
                        "WHERE tenant_id = %s AND operation_id = %s",
                        (tenant, operation_id),
                    )
                    changed += 1
                    continue
                scrubbed = dict(payload)
                scrubbed["sources"] = remaining
                for key in ("active_chunks", "chunks"):
                    records = scrubbed.get(key)
                    if isinstance(records, list):
                        scrubbed[key] = [
                            r for r in records
                            if not (isinstance(r, dict) and r.get("source") in erased)
                        ]
                surviving = scrubbed.get("active_chunks")
                if isinstance(surviving, list):
                    # Only when the payload HAS active chunks. Zeroing the count for a
                    # payload that never described any would be a value change dressed as
                    # a consistency fix.
                    conn.execute(
                        "UPDATE recall_migration_events SET payload = %s::jsonb, "
                        "active_count = %s "
                        "WHERE tenant_id = %s AND operation_id = %s",
                        (json.dumps(scrubbed), len(surviving), tenant, operation_id),
                    )
                else:
                    conn.execute(
                        "UPDATE recall_migration_events SET payload = %s::jsonb "
                        "WHERE tenant_id = %s AND operation_id = %s",
                        (json.dumps(scrubbed), tenant, operation_id),
                    )
                changed += 1
        return changed

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

    def pending_event_summaries(self, tenant: str) -> list[dict[str, object]]:
        """Pending events WITHOUT their payloads, for operator surfaces and for counting.

        `pending_events` selects `payload`, which carries every chunk's text and embedding for a
        batch. An operator status command that only prints ids and counts must not move that over
        the wire, and it must not be a retrieval path by accident.
        """
        with self._connect() as conn:
            self._set_tenant(conn, tenant)
            rows = conn.execute(
                "SELECT sequence_id, operation_id, operation_kind, active_count, shadow_count, "
                "payload ->> 'active_generation', payload ->> 'shadow_generation' "
                "FROM recall_migration_events "
                "WHERE tenant_id = %s AND status = 'pending' ORDER BY sequence_id",
                (tenant,),
            ).fetchall()
        return [
            {
                "sequence_id": row[0], "operation_id": row[1], "operation_kind": row[2],
                "active_count": row[3], "shadow_count": row[4],
                "active_generation": row[5], "shadow_generation": row[6],
            }
            for row in rows
        ]

    def pending_count(self, tenant: str) -> int:
        with self._connect() as conn:
            self._set_tenant(conn, tenant)
            row = conn.execute(
                "SELECT count(*) FROM recall_migration_events "
                "WHERE tenant_id = %s AND status = 'pending'",
                (tenant,),
            ).fetchone()
        return int(row[0]) if row else 0

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

    def cutover(self, tenant: str) -> None:
        """Promote the ready shadow only when its ordered outbox has no lag."""
        with self._connect() as conn, conn.transaction():
            self._set_tenant(conn, tenant)
            pending = conn.execute(
                "SELECT count(*) FROM recall_migration_events "
                "WHERE tenant_id = %s AND status = 'pending'", (tenant,)
            ).fetchone()
            if pending and pending[0]:
                raise RuntimeError("cutover refused while migration events remain pending")
            row = conn.execute(
                "SELECT shadow_generation FROM recall_tenant_routes WHERE tenant_id = %s FOR UPDATE",
                (tenant,),
            ).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError("cutover requires a configured shadow generation")
            state = conn.execute(
                "SELECT state FROM recall_index_generations WHERE generation_id = %s", (row[0],)
            ).fetchone()
            if state is None or state[0] != "ready":
                raise RuntimeError("cutover requires a ready shadow generation")
            conn.execute(
                "UPDATE recall_tenant_routes SET active_generation = shadow_generation, "
                "shadow_generation = active_generation, updated_at = now() WHERE tenant_id = %s",
                (tenant,),
            )
            conn.execute("SELECT pg_notify('recall_route_changed', %s)", (tenant,))

    def replay_pending(
        self, tenant: str, stores: dict[str, "PgVectorStore"]
    ) -> int:
        """Replay ordered, idempotent shadow writes and clear completed payloads.

        Serialised against the erasure scrub AND against a second drain by a per-tenant advisory
        lock. Both are needed. Two concurrent drains both perform every write and the loser then
        raises `KeyError` from `complete_event`, and without the lock a drain reads the pre-scrub
        payload and writes erased text back into both generations, which is the exact outcome
        `erase_sources_from_pending` exists to prevent. `FOR UPDATE` cannot supply that ordering:
        this read is a plain SELECT, and under READ COMMITTED a plain SELECT is never blocked by
        a row lock.

        Try-lock rather than blocking, unlike the scrub: an operator can re-run a drain, and
        waiting behind an erasure that is itself waiting is worse than being told to retry.
        """
        with self._connect() as lock_conn:
            got = lock_conn.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
                (self._outbox_lock_key(tenant),),
            ).fetchone()
            if not got or not got[0]:
                raise ConcurrentOutboxDrain(
                    f"another replay or erasure is working tenant {tenant!r}'s outbox"
                )
            try:
                return self._replay_locked(tenant, stores)
            finally:
                lock_conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (self._outbox_lock_key(tenant),),
                )

    def _replay_locked(self, tenant: str, stores: dict[str, "PgVectorStore"]) -> int:
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
