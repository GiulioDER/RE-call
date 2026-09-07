"""Restore validation primitives and cutover safety state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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
    expected_generation: str | None = None,
    expected_checksums: dict[str, str] | None = None,
    checksum_provider: Callable[[Any], dict[str, str]] | None = None,
    representative_search: Callable[[Any], bool] | None = None,
) -> RestoreValidation:
    """Validate structural and serving invariants without returning corpus text."""
    checks: dict[str, bool] = {}

    def scalar(sql: str) -> Any:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
            return row[0] if row else None

    checks["schema"] = str(scalar("SELECT COALESCE(max(version), '') FROM recall_schema_migrations")) == expected_schema_version
    checks["pgvector"] = bool(scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
    checks["rls"] = bool(scalar("SELECT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'recall_chunks_v1' AND relrowsecurity)"))
    checks["indexes"] = bool(scalar("SELECT count(*) > 0 FROM pg_indexes WHERE tablename = 'recall_chunks_v1'"))
    if expected_generation is not None:
        checks["active_generation"] = scalar("SELECT generation_id FROM recall_generations WHERE state = 'active' LIMIT 1") == expected_generation
    if expected_checksums is not None:
        if checksum_provider is None:
            checks["checksums"] = False
        else:
            try:
                checks["checksums"] = checksum_provider(connection) == expected_checksums
            except Exception:
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
