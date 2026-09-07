"""Resolve the one indexing and serving route used by a process.

The legacy table and immutable generation paths are intentionally still supported, but they must
never be selected independently by individual call sites.  A process resolves this object once
and passes it to both its read and write paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping

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
    implicit so ``recall route status`` can surface it.  Production and enterprise deployments
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


__all__ = ["RouteConfigurationError", "RouteMode", "RuntimeRoute", "resolve_runtime_route"]
