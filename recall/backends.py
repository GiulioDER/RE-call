"""Which store a DSN opens: PostgreSQL, or one SQLite file (`sqlite:///<path>`).

The lite store implements the calls the legacy (non-generation) route makes: indexing, dense and
keyword retrieval, supersession, calibration. It is not a `PgVectorStore` subclass, and it is
returned typed as one on purpose, so the CLI and the server keep one code path; anything outside
that subset raises `recall.lite.LiteUnsupported`, which names the call and says it needs the full
install, and reads as absent to `getattr(store, name, None)` probes.

Generations, manifests, the enterprise control plane and learned sparse retrieval exist only on
Postgres, so a lite DSN on the generation route is refused before anything opens. Deletion by
source (`recall forget`, `recall_forget`) works on the legacy route. A lite file holds one table,
so a table other than the default is refused rather than ignored.

Every top-level `recall` command is classified here for a lite DSN: `LITE_SERVED` or
`LITE_REFUSED`. `lite_command_refusal` is what `recall.cli` asks before it opens anything, so a
command a lite file cannot serve is refused by name instead of reaching PostgreSQL with a
`sqlite:///` string.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, cast

from recall.lite import LiteStore, LiteStoreError, is_lite_dsn
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
    "write_session",
]

#: Every top-level `recall` command that works with a `sqlite:///` DSN, or never opens a store.
#: `tests/test_lite_cli_refusals.py` fails when a command registered in `recall.cli` is in neither
#: this set nor `LITE_REFUSED`, so a new command must be classified the day it is added: serve the
#: lite store (and add it here), or refuse it by name (and add it there). Never neither.
#: `forget`, `provenance`, `scopes` and `calibration` serve lite through #914, #915, #917 and #918.
#: `calibration` is served in part: #918 serves `list`, `show` and `export` and refuses the other
#: subcommands by name in its own handler, not in `lite_command_refusal`. `lite` (#916) is the
#: lite store's own refresh command.
LITE_SERVED: frozenset[str] = frozenset(
    {
        "index", "search", "lite", "forget", "scopes", "doctor", "dashboard", "calibration", "provenance",
        # Never open the store a DSN names: files, manifests, backups, secrets, client config.
        "check", "extract", "rewrite", "manifest", "backup", "secret", "uninstall",
        # Takes its database from its own config file, not from the global DSN.
        "wizard",
        # Partly served, partly refused: the refused forms are decided in `lite_command_refusal`.
        "lint", "schema", "setup", "quickstart",
    }
)

_FULL_INSTALL = "the full (PostgreSQL) install"
_SETUP_LITE = "`recall setup --lite`"

#: Commands a lite store cannot serve, with what to do instead. Refused by name before anything
#: is opened, rather than reaching PostgreSQL with a `sqlite:///` string.
LITE_REFUSED: dict[str, str] = {
    "generation": f"Generations, manifests and blue-green promotion need {_FULL_INSTALL}.",
    "graph": f"The semantic graph is built per generation and needs {_FULL_INSTALL}.",
    "reasoning": f"The reasoning projection and proposals need {_FULL_INSTALL}.",
    "idempotency": "Reconciliation repairs the PostgreSQL receipt and Redis replay pair, which a lite store does not have; "
    f"it needs {_FULL_INSTALL}.",
    "calibrate": "A lite store calibrates itself when it is indexed (`recall index`).",
    "demo": f"The demo indexes into its own PostgreSQL table; try {_SETUP_LITE} instead.",
    "code": f"Code memory lives in its own table, which a lite file does not have; it needs {_FULL_INSTALL}.",
}


def lite_command_refusal(args: Namespace) -> str | None:
    """Why the parsed command cannot run on its `sqlite:///` DSN, or None when it can."""
    command = getattr(args, "cmd", None)
    # Judge the DSN the command will open, not merely the serving DSN, and name where it came from.
    source = "--serving-dsn names, else RECALL_SERVING_DSN or RECALL_DSN"
    if command == "quickstart":
        dsn = getattr(args, "existing_dsn", None)
        source = "--existing-dsn names"
    elif command == "schema":
        # `apply` opens only the migration DSN; `status` and `plan` open it when given, else the
        # serving DSN (`recall.cli_commands.schema_cmd`).
        migration = getattr(args, "migration_dsn", None)
        serving = getattr(args, "dsn", None)
        dsn = migration if getattr(args, "schema_cmd", None) == "apply" else (migration or serving)
        if migration:
            source = "--migration-dsn names, else RECALL_MIGRATION_DSN"
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
        reason = f"Semantic lint compares stored vectors and needs {_FULL_INSTALL}."
    elif command == "setup":
        if getattr(args, "lite", False):
            return None
        reason = f"Run {_SETUP_LITE}, which builds, calibrates and registers a lite store."
    elif command == "quickstart":
        reason = f"Quickstart provisions PostgreSQL; {_SETUP_LITE} is the zero-setup path."
    elif command in LITE_REFUSED:
        reason = LITE_REFUSED[command]
    else:
        return None
    return (
        f"recall {command}: not available on a lite store ({dsn}). {reason} The store is the one "
        f"{source}; point that at PostgreSQL to use it instead."
    )


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

    A non-default `table` is refused for a lite DSN rather than ignored: a lite file has one table,
    and ignoring the name ran `recall --table scratch ...`, meant to keep an experiment away from
    real memory, against the real memory.
    """
    if is_lite_dsn(dsn):
        if table != DEFAULT_TABLE:
            raise LiteStoreError(
                f"a lite store ({dsn}) has one table, '{DEFAULT_TABLE}'; table {table!r} (--table or "
                f"RECALL_TABLE) cannot be "
                "honoured. Use another sqlite:/// file to keep an experiment apart from this memory"
            )
        return cast("PgVectorStore", LiteStore.from_dsn(dsn, dim=dim, tenant=tenant, table=table))
    return PgVectorStore(
        dsn, dim=dim, table=table, tenant=tenant, pool_size=pool_size, statement_timeout_ms=statement_timeout_ms
    )


def write_session(store: object, purpose: str, **kwargs: Any) -> AbstractContextManager[Any]:
    """Hold the lite store's write lane for a write operation; a no-op for any other store.

    Postgres has its own guard on the same operations (`recall.index_lock.single_writer`), so the
    lane exists only where that one cannot reach. kwargs go to `recall.lite.writer_lock.lite_lane`
    (lane, wait_s, coalesce, on_busy).
    """
    if isinstance(store, LiteStore):
        from recall.lite.writer_lock import lite_lane

        return lite_lane(store.path, purpose=purpose, **kwargs)
    return nullcontext(None)
