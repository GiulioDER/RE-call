from __future__ import annotations

import argparse
import importlib
import os
import sys

from recall._env import load_dotenv
from recall.observability import configure_logging
from recall.schema import (
    ConcurrentMigrator,
    InterruptedConcurrentIndex,
    MigrationChecksumMismatch,
    SchemaError,
    SchemaIncompatible,
)
from recall.store import (
    DEFAULT_TENANT,
    _env_opt_out,
    require_secure_dsn,
    warn_if_insecure_dsn,
)

# `recall setup` writes its answers to .env, so the file has to be read BEFORE the DSN
# defaults below are computed from os.environ. Without this the wizard appears to succeed
# and the very next command silently ignores every setting it just captured.
#
# The failure is RECORDED here rather than acted on: SystemExit is not safe at import time
# (it would kill `import recall.cli` for library consumers), so refusing a command over a
# broken .env has to happen inside `main()`, where it is. Printing a warning here and moving
# on was tried and an audit caught what it misses: warn-and-continue still lets the exact
# hazard through, a request that carries the wrong DSN, it just prints a line first.
_DOTENV_ERROR: Exception | None = None
try:
    load_dotenv()
except Exception as _dotenv_exc:  # noqa: BLE001 - see below  # BROAD-CATCH: fail-open
    # Deliberately broad: this runs at IMPORT time, so anything escaping here kills
    # `recall --help`, every command, and `import recall.cli` for library consumers and test
    # collection. Enumerating types was tried twice and was wrong twice — (OSError,
    # UnicodeDecodeError) missed the ValueError that a NUL byte produces, and a NUL is valid
    # UTF-8 so the read itself succeeds.
    _DOTENV_ERROR = _dotenv_exc
    try:
        print(
            f"warning: .env could not be applied — {type(_dotenv_exc).__name__}: {_dotenv_exc}",
            file=sys.stderr,
        )
    except Exception:  # noqa: BLE001 - this handler must not be able to fail either  # BROAD-CATCH: fail-open
        # A write to a closed or broken stderr (a daemonised or service-wrapped host) must not
        # take an import down. The refusal in `main()` below does not depend on this line
        # having printed; it depends only on `_DOTENV_ERROR` being set.
        pass

DEFAULT_DSN = os.environ.get(
    "RECALL_SERVING_DSN",
    os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"),
)
DEFAULT_MIGRATION_DSN = os.environ.get("RECALL_MIGRATION_DSN")


def _require_secure(dsn: str) -> None:
    """Indirection so ONE call site decides which DSNs are guarded; see `main`.

    A bug audit proposed converting the PermissionError this raises into a SystemExit, on the
    grounds that every other operator-facing refusal in this file is a SystemExit and this one
    arrives as a traceback. That was REJECTED: `test_cli_db_commands_fail_closed_on_insecure_
    default_dsn` asserts the PermissionError propagates, and it is a security test pinning
    fail-closed behaviour. Rewriting a security assertion to accommodate a cosmetic improvement
    is the wrong trade. The exception type is deliberate; do not "tidy" it.

    Resolving `require_secure_dsn` through the module global at call time is also deliberate:
    that test monkeypatches it, and a `from`-bound local would make the patch inert.
    """
    require_secure_dsn(dsn)


_COMMAND_REGISTRATIONS: tuple[tuple[frozenset[str], str, str], ...] = (
    (frozenset({"doctor"}), "recall.cli_commands.doctor_cmd", "register"),
    (
        frozenset({"setup", "wizard", "uninstall"}),
        "recall.cli_commands.setup_wizard",
        "register",
    ),
    (frozenset({"quickstart"}), "recall.cli_commands.setup_wizard", "register_quickstart"),
    (frozenset({"schema"}), "recall.cli_commands.schema_cmd", "register"),
    (frozenset({"manifest"}), "recall.cli_commands.manifest_cmd", "register"),
    (frozenset({"generation"}), "recall.cli_commands.generation_cmd", "register"),
    (frozenset({"graph"}), "recall.cli_commands.graph_cmd", "register"),
    (
        frozenset({"index", "forget", "search", "scopes"}),
        "recall.cli_commands.index_search",
        "register",
    ),
    (frozenset({"reasoning"}), "recall.cli_commands.reasoning_cmd", "register"),
    (frozenset({"extract", "rewrite"}), "recall.cli_commands.extract_rewrite", "register"),
    (frozenset({"demo", "code"}), "recall.cli_commands.index_search", "register_demo_code"),
    (frozenset({"lint", "check"}), "recall.cli_commands.lint_check", "register"),
    (frozenset({"calibration"}), "recall.cli_commands.calibration_cmd", "register"),
    (frozenset({"provenance"}), "recall.cli_commands.provenance_cmd", "register"),
    (frozenset({"backup"}), "recall.cli_commands.backup_cmd", "register"),
    (frozenset({"secret"}), "recall.cli_commands.secret_cmd", "register"),
    (frozenset({"idempotency"}), "recall.cli_commands.idempotency_cmd", "register"),
    (frozenset({"dashboard"}), "recall.cli_commands.dashboard_cmd", "register"),
)


def build_parser(command: str | None = None) -> argparse.ArgumentParser:
    """Build the command tree without opening a database or resolving providers.

    Passing a command limits imports to the module that registers that command. The default
    builds the complete tree for API introspection and top level help.
    """

    parser = argparse.ArgumentParser(
        prog="recall",
        description="Retrieval-augmented memory for long-running agents.",
        epilog=(
            "Starting out? `recall quickstart` demonstrates the system; `recall setup` is THE "
            "install; `recall wizard` is the saved-config workflow; `recall doctor` diagnoses "
            "an existing install."
        ),
    )
    parser.add_argument(
        "--serving-dsn",
        "--dsn",
        dest="dsn",
        default=DEFAULT_DSN,
        help="unprivileged application DSN (env: RECALL_SERVING_DSN; --dsn is deprecated)",
    )
    parser.add_argument(
        "--migration-dsn",
        default=DEFAULT_MIGRATION_DSN,
        help="DDL-owner DSN used only by `schema apply` (env: RECALL_MIGRATION_DSN)",
    )
    parser.add_argument(
        "--embedder",
        default=os.environ.get("RECALL_EMBEDDER", "fastembed"),
        help=(
            "hashing, fastembed[:model], st:<model>, voyage[:model], openai[:model]. "
            "Set RECALL_EMBED_PROFILE for a registered profile such as "
            "bge-small-context-section-v1 or bge-large-context-section-v1."
        ),
    )
    parser.add_argument(
        "--table",
        default="chunks",
        help="table to read/write (default: chunks). Use a throwaway name to keep an experiment out of your real memory index.",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help=f"tenant namespace to operate on (default: {DEFAULT_TENANT}).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # These modules own the complete argument declarations and handlers. Keeping registration in
    # one place prevents the executable parser and the API introspection parser from drifting.
    # The command index is only an import optimization. An unknown command falls back to the full
    # registry so adding a command cannot silently make the executable undiscoverable.
    registrations = (
        _COMMAND_REGISTRATIONS
        if command is None or not any(command in commands for commands, _, _ in _COMMAND_REGISTRATIONS)
        else tuple(item for item in _COMMAND_REGISTRATIONS if command in item[0])
    )
    for _, module_name, function_name in registrations:
        module = importlib.import_module(module_name)
        getattr(module, function_name)(sub)
    return parser


_GLOBAL_OPTIONS_WITH_VALUES = frozenset(
    {"--serving-dsn", "--dsn", "--migration-dsn", "--embedder", "--table", "--tenant"}
)


def _command_from_argv(argv: list[str]) -> str | None:
    """Find the top level command without importing command modules."""

    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in _GLOBAL_OPTIONS_WITH_VALUES:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return None


_SCHEMA_REMEDY: dict[type[SchemaError], str] = {
    SchemaIncompatible: (
        "This table was created for a different embedder or RE-call version. Pass the matching "
        "--embedder or use a different --table."
    ),
    MigrationChecksumMismatch: (
        "A migration file no longer matches the bytes recorded as applied. Restore the committed "
        "file or use a reviewed upgrade; do not edit applied migration history."
    ),
    ConcurrentMigrator: "Another migrator holds the lock. Wait for it to finish, then retry.",
    InterruptedConcurrentIndex: (
        "A concurrently built index was left invalid. Drop the named index and re-run "
        "`recall schema apply`."
    ),
}


def schema_error_message(exc: SchemaError) -> str:
    remedy = _SCHEMA_REMEDY.get(type(exc))
    return str(exc) if remedy is None else f"{exc}\n\n{remedy}"


def _main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # clean UTF-8 output on Windows consoles
        # `errors=` as well as `encoding=`, because reconfiguring the encoding RESETS errors to
        # strict. The inherited handler is surrogateescape, and dropping it made every `print`
        # of a filename raise for a name that is not valid UTF-8: `recall extract run` over a
        # corpus holding one such file exited 1 with EMPTY stdout, throwing away a completed
        # extraction at the REPORT step. That is the same "one bad memo kills the run" failure
        # the extractor guards against everywhere else, arriving at the last possible moment.
        # Showing a mangled name beats showing nothing.
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        # For the ENCODING, not the error handler, and the first version of this comment had it
        # wrong: CPython already defaults stderr to `backslashreplace`, and keeps it there even
        # under `PYTHONIOENCODING=utf-8:strict`, which sets stdout to strict alone. So deleting
        # this line would not turn a refusal into a traceback. What it does is give stderr the
        # same UTF-8 encoding stdout gets on a Windows console, and hold the handler if a
        # caller has replaced stderr with a strict wrapper of its own.
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    # Without this the library's loggers have no handler, so every _log.info is discarded — which
    # is how `index` came to prune rows while printing nothing about it.
    configure_logging()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser(_command_from_argv(raw_argv))
    args = parser.parse_args(raw_argv)
    # Commands that will actually open a connection FAIL CLOSED on the insecure default DSN;
    # everything else only warns.
    #
    # Every command that will open a connection FAILS CLOSED on the insecure default DSN;
    # the rest only warn.
    #
    # An earlier version of this set listed six commands by hand and missed four that connect
    # (generation, calibration, schema, and lint --semantic), so the guard read as coverage and
    # was not. The set is derived from the parsers now: a subcommand declares `_opens_db=True`
    # beside its own definition, so a new one cannot be added without answering the question.
    opens_db = bool(getattr(args, "_opens_db", False))
    if args.cmd == "provenance" and getattr(args, "sqlite_path", None):
        # The local provenance adapter is deliberately independent of PostgreSQL and the
        # embedder. It still revalidates source bytes below, so this is not a trust bypass.
        opens_db = False
    if args.cmd == "lint":  # only the --semantic path reaches a database
        opens_db = bool(getattr(args, "semantic", False))
    if args.cmd == "schema" and getattr(args, "schema_cmd", None) == "grants":
        opens_db = False  # prints SQL for an operator to run; opens nothing

    if (
        opens_db
        and args.cmd not in {"setup", "wizard"}  # see the setup-specific carve-out for _require_secure below —
        # `recall setup` is the command you run to REPAIR a broken .env, so blocking it on a
        # broken .env is the same dead end that carve-out exists to avoid, one guard down. A
        # round-6 audit caught this: it fired unconditionally and refused `setup` even when the
        # operator had already passed an explicit --dsn that resolved the ambiguity.
        #
        # `setup` is not left silent: the note comes from the import-time stderr print above
        # (near `_DOTENV_ERROR = _dotenv_exc`), which runs for every command before args.cmd is
        # even known — NOT from run_setup_wizard, which has no .env-specific messaging of its
        # own. A round-7 audit caught an earlier version of this comment misattributing it,
        # which is worth naming: believing the notice were conditional on reaching the wizard
        # could lead a later change to gate or remove the import-time print, leaving `setup`
        # with zero indication anything was wrong.
        and _DOTENV_ERROR is not None
        and not _env_opt_out("RECALL_IGNORE_BROKEN_DOTENV")
    ):
        # `.env` exists but could not be applied, so any variable it would have set — most
        # dangerously RECALL_SERVING_DSN — is silently absent from this process, and args.dsn
        # below is the LOCAL fallback rather than whatever was configured. Warning about that
        # at import time and proceeding anyway was tried; it still lets a request reach the
        # wrong database, which is the exact hazard this whole guard exists to prevent, so a
        # DB-opening command refuses instead. Reading it, fixing it, or deleting it are all
        # legitimate; running against a database neither the operator nor the file chose is not.
        raise SystemExit(
            f".env exists but could not be applied "
            f"({type(_DOTENV_ERROR).__name__}: {_DOTENV_ERROR}), and this command connects to a "
            f"database. Fix the file, or set RECALL_IGNORE_BROKEN_DOTENV=1 to proceed anyway — "
            f"variables the file would have set (including RECALL_SERVING_DSN) are absent, so "
            f"the DSN in effect may not be the one you intended."
        )

    if opens_db:
        if args.cmd == "setup":
            # The wizard is the command you run to REPAIR a bad configuration, so a bare
            # refusal is a dead end: it takes `dsn=args.dsn` verbatim and never prompts for
            # one. Still guarded, because it does connect when the operator accepts the
            # calibrate prompt, and also when the operator accepts the CLAUDE.md/memory
            # scaffold prompt (which defaults to yes and auto-indexes memory/) — but the
            # refusal has to name the way out.
            try:
                _require_secure(args.dsn)
            except PermissionError as exc:
                raise SystemExit(
                    f"{exc}\n\n"
                    "This is `recall setup`, which cannot prompt its way out of this: it uses "
                    "the DSN it was given and never asks for another. Passing that same value "
                    "again with `--dsn` or `--serving-dsn` does not help, because the refusal "
                    "is about the credentials inside the DSN, not about how it reached the "
                    "command. Re-run with a DSN carrying a real password, or set "
                    "RECALL_ALLOW_INSECURE_DSN=1 to accept the risk deliberately."
                ) from exc
        elif args.cmd != "wizard":
            _require_secure(args.dsn)
    else:
        warn_if_insecure_dsn(args.dsn)  # loud stderr note if default creds target a remote host

    # The DDL-owner credential was never checked or even warned about on any path, which is the
    # wrong way round: it is the most privileged DSN this CLI accepts.
    migration_dsn = getattr(args, "migration_dsn", None)
    if migration_dsn and opens_db:  # grants stays exempt because it does not open a database
        _require_secure(migration_dsn)

    handler = getattr(args, "func", None)
    if handler is None:
        raise SystemExit(f"no handler registered for command: {args.cmd}")
    handler(args)


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except SchemaError as exc:
        raise SystemExit(schema_error_message(exc)) from exc


if __name__ == "__main__":
    main()
