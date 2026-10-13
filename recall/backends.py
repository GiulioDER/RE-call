"""Which store a DSN opens: PostgreSQL, or one SQLite file (`sqlite:///<path>`).

The lite store implements the calls the legacy (non-generation) route makes: indexing, dense and
keyword retrieval, supersession, calibration. It is not a `PgVectorStore` subclass, and it is
returned typed as one on purpose, so the CLI and the server keep one code path; anything outside
that subset raises `recall.lite.LiteUnsupported`, which names the call and says it needs the full
install, and reads as absent to `getattr(store, name, None)` probes.

Generations, manifests, the enterprise control plane and learned sparse retrieval exist only on
Postgres, so a lite DSN on the generation route is refused before anything opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from recall.lite import LiteStore, is_lite_dsn
from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, PgVectorStore

if TYPE_CHECKING:
    from argparse import Namespace

    from recall.runtime_route import RuntimeRoute

__all__ = [
    "LITE_REFUSED",
    "LITE_SERVED",
    "is_lite_dsn",
    "lite_command_refusal",
    "lite_route_refusal",
    "open_legacy_store",
]

#: Every top-level `recall` command that works with a `sqlite:///` DSN, or never opens a store.
#: `tests/test_lite_cli_refusals.py` fails when a command registered in `recall.cli` is in neither
#: this set nor `LITE_REFUSED`, so a new command must be classified the day it is added: serve the
#: lite store (and add it here), or refuse it by name (and add it there). Never neither.
#: `forget`, `provenance`, `scopes` and `calibration` serve lite through their own pull requests
#: (#914, #915, #917, #918); until those merge they still crash on a lite DSN.
LITE_SERVED: frozenset[str] = frozenset(
    {
        "index", "search", "forget", "scopes", "doctor", "dashboard", "calibration", "provenance",
        # Never open the store a DSN names: files, manifests, backups, secrets, client config.
        "check", "extract", "rewrite", "manifest", "backup", "secret", "uninstall",
        # Takes its database from its own config file, not from the global DSN.
        "wizard",
        # Partly served, partly refused: the refused forms are decided in `lite_command_refusal`.
        "lint", "schema", "setup", "quickstart",
    }
)

#: Commands a lite store cannot serve, with what to do instead. Refused by name before anything
#: is opened, rather than reaching PostgreSQL with a `sqlite:///` string.
LITE_REFUSED: dict[str, str] = {
    "generation": "Generations, manifests and blue-green promotion need the full (PostgreSQL) install.",
    "graph": "The semantic graph is built per generation and needs the full (PostgreSQL) install.",
    "reasoning": "The reasoning projection and proposals need the full (PostgreSQL) install.",
    "idempotency": "Reconciliation repairs the PostgreSQL receipt and Redis replay pair, which a lite store does not have; it needs the full (PostgreSQL) install.",
    "calibrate": "A lite store calibrates itself when it is indexed (`recall index`).",
    "demo": "The demo indexes into its own PostgreSQL table; try `recall setup --lite` instead.",
    "code": "Code memory lives in its own table, which a lite file does not have; it needs the full (PostgreSQL) install.",
}


def lite_command_refusal(args: Namespace) -> str | None:
    """Why the parsed command cannot run on its `sqlite:///` DSN, or None when it can."""
    command = getattr(args, "cmd", None)
    # Judge the DSN the command will open, not merely the serving DSN.
    if command == "quickstart":
        dsn = getattr(args, "existing_dsn", None)
    elif command == "schema":
        # `apply` opens only the migration DSN; `status` and `plan` open it when given, else the
        # serving DSN (`recall.cli_commands.schema_cmd`).
        migration = getattr(args, "migration_dsn", None)
        serving = getattr(args, "dsn", None)
        dsn = migration if getattr(args, "schema_cmd", None) == "apply" else (migration or serving)
    else:
        dsn = getattr(args, "dsn", None)
    if not is_lite_dsn(dsn):
        return None
    if command == "schema":
        if getattr(args, "schema_cmd", None) == "grants":
            return None  # prints SQL for an operator to run; opens nothing
        reason = "A lite file has no migrations; `recall doctor` reports its state."
    elif command == "lint":
        if not getattr(args, "semantic", False):
            return None  # the file checks open no store
        reason = "Semantic lint compares stored vectors and needs the full (PostgreSQL) install."
    elif command == "setup":
        if getattr(args, "lite", False):
            return None
        reason = "Run `recall setup --lite`, which builds, calibrates and registers a lite store."
    elif command == "quickstart":
        reason = "Quickstart provisions PostgreSQL; `recall setup --lite` is the zero-setup path."
    elif command in LITE_REFUSED:
        reason = LITE_REFUSED[command]
    else:
        return None
    return f"recall {command}: not available on a lite store ({dsn}). {reason}"


def lite_route_refusal(dsn: str | None, route: RuntimeRoute) -> str | None:
    """Why `route` cannot run on `dsn`, or None when it can."""
    if not is_lite_dsn(dsn) or not route.uses_generation:
        return None
    return (
        f"{route.describe()}: the lite store ({dsn}) has no generations. Use it with "
        "RECALL_ENV=development and RECALL_INDEX_MODE=legacy (the defaults), or point "
        "RECALL_DSN at PostgreSQL for the generation route"
    )


def open_legacy_store(
    dsn: str,
    *,
    dim: int,
    table: str = DEFAULT_TABLE,
    tenant: str = DEFAULT_TENANT,
    pool_size: int | None = None,
    statement_timeout_ms: int | None = None,
) -> PgVectorStore:
    """The legacy-route store for `dsn`: a `LiteStore` for `sqlite:///`, else a `PgVectorStore`.

    The pool size and statement timeout apply to Postgres only; a SQLite file has neither.
    """
    if is_lite_dsn(dsn):
        return cast("PgVectorStore", LiteStore.from_dsn(dsn, dim=dim, tenant=tenant, table=table))
    return PgVectorStore(
        dsn, dim=dim, table=table, tenant=tenant, pool_size=pool_size, statement_timeout_ms=statement_timeout_ms
    )
