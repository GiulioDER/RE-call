"""Resolve the one indexing and serving route used by a process.

The legacy table and immutable generation paths are intentionally still supported, but they must
never be selected independently by individual call sites.  A process resolves this object once
and passes it to both its read and write paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping

from recall.db_constants import DEFAULT_TABLE, DEFAULT_TENANT
from recall.errors import RecallError

RouteMode = Literal["legacy", "generation"]


class RouteConfigurationError(ValueError, RecallError):
    """The configured indexing and serving route is invalid or contradictory."""


@dataclass(frozen=True)
class RuntimeRoute:
    """The immutable route decision shared by indexing and serving."""

    mode: RouteMode
    environment: str
    source: str
    table: str
    explicit: bool
    enterprise: bool = False

    @property
    def uses_generation(self) -> bool:
        return self.mode == "generation"

    @property
    def generation_table(self) -> str:
        return "recall_chunks_v1"

    def identity(self) -> dict[str, object]:
        """Return only operator safe route metadata."""
        return {
            "mode": self.mode,
            "environment": self.environment,
            "source": self.source,
            "table": self.generation_table if self.uses_generation else self.table,
            "explicit": self.explicit,
            "enterprise": self.enterprise,
        }

    def table_refusal(self, table: str, command: str) -> str | None:
        """Why `--table` cannot be honoured on this route, or None when it can.

        Every generation lives in `recall_chunks_v1`, so a store opened on the generation route
        reads and erases that table whatever `--table` names. Dropping the flag sent
        `recall --table scratch forget --yes` to the real store, so a table other than the
        default is refused before anything is opened, as the MCP server refuses `RECALL_TABLE`
        (`recall_mcp.server_lifespan.table_override_refusal`). Compared case-folded, because
        PostgreSQL folds an unquoted `Chunks` to the default table.
        """
        if not self.uses_generation or table.casefold() == DEFAULT_TABLE:
            return None
        return (
            f"{command}: --table {table!r} cannot be honoured on the generation route "
            f"({self.source}): every generation is kept in {self.generation_table!r}, so this "
            "would act on that table instead. Nothing was read or changed. Drop --table, or use "
            "that table on the legacy route (RECALL_INDEX_MODE=legacy, development only)."
        )

    def describe(self) -> str:
        """Render the route in a stable, human readable form."""
        identity = self.identity()
        return (
            f"mode={identity['mode']} table={identity['table']} "
            f"environment={identity['environment']} source={identity['source']} "
            f"explicit={str(identity['explicit']).lower()}"
        )


def resolve_runtime_route(
    env: Mapping[str, str] | None = None,
    *,
    enterprise: bool = False,
    requested_mode: RouteMode | None = None,
) -> RuntimeRoute:
    """Resolve a route once, rejecting combinations that could split read and write paths.

    ``RECALL_INDEX_MODE`` is the explicit operator setting.  Its absence retains the historical
    development default for compatibility, but the returned object marks that decision as
    implicit.  Production and enterprise deployments
    are always generation based and refuse an explicit legacy request.
    """

    source = os.environ if env is None else env
    raw_environment = source.get("RECALL_ENV", "development")
    environment = raw_environment.strip().lower()
    if environment not in {"development", "production"}:
        raise RouteConfigurationError(
            f"RECALL_ENV={raw_environment!r} is invalid; expected development or production"
        )

    raw_mode = requested_mode or source.get("RECALL_INDEX_MODE", "").strip().lower()
    explicit = bool(raw_mode)
    if raw_mode and raw_mode not in {"legacy", "generation"}:
        raise RouteConfigurationError(
            f"RECALL_INDEX_MODE={raw_mode!r} is invalid; expected legacy or generation"
        )

    if raw_mode:
        mode: RouteMode = raw_mode  # type: ignore[assignment]
    else:
        mode = "generation" if environment == "production" else "legacy"

    if enterprise and mode != "generation":
        raise RouteConfigurationError(
            "enterprise control plane requires the generation route; "
            "set RECALL_INDEX_MODE=generation"
        )
    if environment == "production" and mode != "generation":
        raise RouteConfigurationError(
            "production requires the generation route; legacy indexing and serving are "
            "development only"
        )

    return RuntimeRoute(
        mode=mode,
        environment=environment,
        source="RECALL_INDEX_MODE" if explicit else "RECALL_ENV compatibility default",
        table=source.get("RECALL_TABLE", "chunks"),
        explicit=explicit,
        enterprise=enterprise,
    )


def generations_table_refusal(table: str | None, command: str) -> str | None:
    """Why `--table` cannot be honoured by a command that only works on generations, or None.

    `recall generation`, `recall graph` and `recall calibration` on PostgreSQL act on the tenant's
    generations and calibrations on either route, and those name no chunk table, so a scratch
    `--table` would leave them acting on the real store (#1421). Compared case-folded, as in
    `RuntimeRoute.table_refusal`. None is a caller that named no table, as when a handler is
    called with a namespace the CLI parser did not build.
    """
    if table is None or table.casefold() == DEFAULT_TABLE:
        return None
    return (
        f"{command}: --table {table!r} cannot be honoured: this command works on the tenant's "
        "generations and calibrations, which no --table selects, so it would act on the real "
        "store. Nothing was read or changed. Drop --table, or use another --tenant to keep an "
        "experiment apart."
    )


def throwaway_table_refusal(table: str | None, tenant: str | None, command: str) -> str | None:
    """Why `--table` or `--tenant` would be ignored by a command that embeds into its own table, or None.

    `recall calibrate` and `recall lint --semantic` create, use and drop a throwaway table
    (`cal_<uuid>`, `semlint_<uuid>`) and read neither flag (#1441). The table is compared
    case-folded, as in `generations_table_refusal`. None is a caller that named no value, as when a
    handler is called with a namespace the CLI parser did not build.
    """
    for flag, value, ignored in (
        ("--table", table, table is not None and table.casefold() != DEFAULT_TABLE),
        ("--tenant", tenant, tenant is not None and tenant != DEFAULT_TENANT),
    ):
        if ignored:
            return (
                f"{command}: {flag} {value!r} would be ignored: "
                "this command embeds into a throwaway table of its own and reads neither --table "
                f"nor --tenant. Nothing was read or changed. Drop {flag}."
            )
    return None


__all__ = [
    "RouteConfigurationError",
    "RouteMode",
    "RuntimeRoute",
    "generations_table_refusal",
    "resolve_runtime_route",
    "throwaway_table_refusal",
]
