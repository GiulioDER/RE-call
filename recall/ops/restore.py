"""Restore validation primitives and cutover safety state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


_TENANT_TABLES = """
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p')
      AND EXISTS (
          SELECT 1
          FROM pg_attribute a
          WHERE a.attrelid = c.oid
            AND a.attname = 'tenant_id'
            AND NOT a.attisdropped
      )
"""


@dataclass(frozen=True)
class RestoreValidation:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "checks": self.checks, "failures": list(self.failures)}


def validate_restored_database(
    connection: Any,
    *,
    expected_schema_version: str,
    expected_tenant: str | None = None,
    expected_generation: str | None = None,
    expected_role: str | None = None,
    representative_chunk_id: str | None = None,
    expected_checksums: dict[str, str] | None = None,
    checksum_provider: Callable[[Any], dict[str, str]] | None = None,
    calibration_check: Callable[[Any], bool] | None = None,
    representative_search: Callable[[Any], bool] | None = None,
) -> RestoreValidation:
    """Validate structural and direct database serving invariants without returning corpus text.

    The representative search checks below execute SQL directly on the restored database. They do
    not start the MCP application and must not be described as authenticated HTTP validation.

    ``expected_tenant`` is required by the restore drill. Keeping it optional preserves the
    lower level validator's compatibility with callers that only need structural checks, while
    the generation, calibration, and search checks are enabled whenever a tenant is supplied.
    """
    checks: dict[str, bool] = {}

    def scalar(sql: str, params: tuple[Any, ...] = ()) -> Any:
        with connection.cursor() as cursor:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            row = cursor.fetchone()
            return row[0] if row else None

    tenant = expected_tenant.strip() if expected_tenant else None
    tenant_context = True
    tenant_params: tuple[Any, ...]
    if tenant:
        try:
            scalar("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        except Exception:  # BROAD-CATCH: fail-closed
            tenant_context = False

    checks["role"] = False
    try:
        role_is_serving = bool(
            scalar(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_roles "
                "WHERE rolname = current_user AND NOT rolsuper AND NOT rolbypassrls"
                ")"
            )
        )
        current_role = scalar("SELECT current_user")
        checks["role"] = role_is_serving and (
            expected_role is None or str(current_role) == expected_role
        )
    except Exception:  # BROAD-CATCH: fail-closed
        checks["role"] = False

    checks["grants"] = False
    try:
        checks["grants"] = bool(
            scalar(
                "SELECT has_schema_privilege(current_user, 'public', 'USAGE') "
                "AND count(*) > 0 "
                "AND bool_and(has_table_privilege(current_user, c.oid, 'SELECT')) "
                + _TENANT_TABLES
            )
        )
    except Exception:  # BROAD-CATCH: fail-closed
        checks["grants"] = False

    checks["schema"] = (
        str(scalar("SELECT COALESCE(max(version), '') FROM recall_schema_migrations"))
        == expected_schema_version
    )
    checks["pgvector"] = bool(
        scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
    )
    checks["rls"] = False
    try:
        checks["rls"] = bool(
            scalar(
                # `_TENANT_TABLES` is a fixed, source controlled catalog predicate, not input.
                "SELECT count(*) > 0 "  # noqa: S608
                "AND bool_and(c.relrowsecurity AND c.relforcerowsecurity "
                "AND EXISTS ("
                "SELECT 1 FROM pg_policy p "
                "WHERE p.polrelid = c.oid "
                "AND pg_get_expr(p.polqual, p.polrelid) LIKE '%current_setting%' "
                "AND pg_get_expr(p.polwithcheck, p.polrelid) LIKE '%current_setting%'"
                ")) "
                + _TENANT_TABLES
            )
        )
    except Exception:  # BROAD-CATCH: fail-closed
        checks["rls"] = False
    checks["indexes"] = bool(
        scalar(
            "SELECT count(*) = 2 FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = 'recall_chunks_v1' "
            "AND indexname IN ('recall_chunks_v1_tsv_idx', 'recall_chunks_v1_embedding_idx')"
        )
    )

    if tenant:
        active_generation_sql = (
            "SELECT s.active_generation_id FROM recall_tenant_state s "
            "JOIN recall_generations g ON g.tenant_id = s.tenant_id "
            "AND g.generation_id = s.active_generation_id "
            "WHERE s.tenant_id = %s AND g.state = 'active' LIMIT 1"
        )
        tenant_params = (tenant,)
    else:
        active_generation_sql = (
            "SELECT s.active_generation_id FROM recall_tenant_state s "
            "JOIN recall_generations g ON g.tenant_id = s.tenant_id "
            "AND g.generation_id = s.active_generation_id "
            "WHERE s.tenant_id = current_setting('recall.tenant_id', true) "
            "AND g.state = 'active' LIMIT 1"
        )
        tenant_params = ()
    active_generation = scalar(active_generation_sql, tenant_params)
    if tenant:
        checks["tenant"] = tenant_context
        checks["active_generation"] = tenant_context and active_generation is not None
        if expected_generation is not None:
            checks["active_generation"] = (
                checks["active_generation"] and str(active_generation) == expected_generation
            )
    elif expected_generation is not None:
        checks["active_generation"] = active_generation == expected_generation

    if tenant:
        checks["calibration"] = False
        if tenant_context and active_generation is not None:
            try:
                if calibration_check is not None:
                    checks["calibration"] = bool(calibration_check(connection))
                else:
                    checks["calibration"] = bool(
                        scalar(
                            "SELECT EXISTS ("
                            "SELECT 1 FROM recall_calibrations c "
                            "JOIN recall_generations g ON g.tenant_id = c.tenant_id "
                            "AND g.generation_id = c.generation_id "
                            "JOIN recall_calibration_query_sets q ON q.tenant_id = c.tenant_id "
                            "AND q.query_set_digest = c.query_set_digest "
                            "WHERE c.tenant_id = %s AND c.generation_id = %s "
                            "AND c.lifecycle_state = 'published' AND c.certified "
                            "AND c.pipeline_fingerprint = g.pipeline_fingerprint "
                            "AND c.corpus_fingerprint = g.corpus_fingerprint "
                            "AND q.sample_count > 0"
                            ")",
                            (tenant, str(active_generation)),
                        )
                    )
            except Exception:  # BROAD-CATCH: fail-closed
                checks["calibration"] = False

        if representative_chunk_id:
            search_params = (tenant, str(active_generation), tenant, representative_chunk_id)
            search_sql = (
                "SELECT EXISTS (SELECT 1 FROM ("
                "SELECT c.chunk_id FROM recall_chunks_v1 c "
                "JOIN recall_tenant_state s ON s.tenant_id = c.tenant_id "
                "AND s.active_generation_id = c.generation_id "
                "JOIN recall_chunks_v1 seed ON seed.tenant_id = c.tenant_id "
                "AND seed.generation_id = c.generation_id "
                "WHERE c.tenant_id = %s AND c.generation_id = %s "
                "AND seed.tenant_id = %s AND seed.chunk_id = %s "
                "ORDER BY c.embedding <=> seed.embedding LIMIT 1"
                ") hit)"
            )
            checks["direct_database_search"] = False
            checks["representative_retrieval"] = False
            if tenant_context and active_generation is not None:
                try:
                    checks["direct_database_search"] = bool(scalar(search_sql, search_params))
                    checks["representative_retrieval"] = bool(
                        scalar(
                            search_sql[:-1] + " WHERE hit.chunk_id = %s)",
                            (*search_params, representative_chunk_id),
                        )
                    )
                except Exception:  # BROAD-CATCH: fail-closed
                    checks["direct_database_search"] = False
                    checks["representative_retrieval"] = False

    if expected_checksums is not None:
        if checksum_provider is None:
            checks["checksums"] = False
        else:
            try:
                checks["checksums"] = checksum_provider(connection) == expected_checksums
            except Exception:  # BROAD-CATCH: fail-closed
                checks["checksums"] = False
    if representative_search is not None:
        checks["representative_search"] = bool(representative_search(connection))
    failures = tuple(name for name, passed in checks.items() if not passed)
    return RestoreValidation(passed=not failures, checks=checks, failures=failures)


@dataclass
class CutoverGuard:
    """Require an explicit operator confirmation and retain the previous target for rollback."""

    write_frozen: bool = False
    previous_target: str | None = None
    active_target: str | None = None
    events: list[str] = field(default_factory=list)

    def freeze_writes(self) -> None:
        self.write_frozen = True
        self.events.append("writes_frozen")

    def cutover(self, restored_target: str, *, confirmation: str) -> None:
        if not self.write_frozen:
            raise RuntimeError("writes must be frozen before restore cutover")
        if confirmation != "CUTOVER_RESTORED_CLUSTER":
            raise ValueError("restore cutover requires confirmation=CUTOVER_RESTORED_CLUSTER")
        self.previous_target = self.active_target
        self.active_target = restored_target
        self.events.append("cutover")

    def rollback(self, *, confirmation: str) -> None:
        if not self.write_frozen:
            raise RuntimeError("writes must be frozen before restore rollback")
        if confirmation != "ROLLBACK_RESTORE":
            raise ValueError("restore rollback requires confirmation=ROLLBACK_RESTORE")
        if self.previous_target is None:
            raise RuntimeError("no previous production target is retained")
        self.active_target, self.previous_target = self.previous_target, self.active_target
        self.events.append("rollback")
