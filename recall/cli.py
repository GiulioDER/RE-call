from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

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
    DEFAULT_TABLE,
    DEFAULT_TENANT,
    _env_opt_out,
    require_secure_dsn,
    warn_if_insecure_dsn,
)

from recall.cli_commands import (
    calibration_cmd,
    doctor_cmd,
    extract_rewrite,
    generation_cmd,
    graph_cmd,
    index_search,
    lint_check,
    manifest_cmd,
    reasoning_cmd,
    schema_cmd,
    setup_wizard,
)

_DOTENV_ERROR: Exception | None = None
try:
    load_dotenv()
except Exception as _dotenv_exc:  # noqa: BLE001
    _DOTENV_ERROR = _dotenv_exc
    try:
        print(
            f"warning: .env could not be applied — {type(_dotenv_exc).__name__}: {_dotenv_exc}",
            file=sys.stderr,
        )
    except Exception:  # noqa: BLE001
        pass

DEFAULT_DSN = os.environ.get(
    "RECALL_SERVING_DSN",
    os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"),
)
DEFAULT_MIGRATION_DSN = os.environ.get("RECALL_MIGRATION_DSN")


def _require_secure(dsn: str) -> None:
    """Keep the security check patchable for callers and tests."""
    require_secure_dsn(dsn)


#: Printed under the subcommand list. `recall --help` names twenty commands in alphabetical-ish
#: order with one line each, which answers "what exists" and not "which one do I want" — and a
#: newcomer only ever has the second question. Four of these commands can each reasonably be read
#: as "the install" (`quickstart`, `setup`, `wizard`, and the `recall-install` window), so the
#: first job of this text is to say which one, and what the other three are for instead.
#:
#: Kept to what a person needs before their first successful query. Everything past that point they
#: will reach through the docs, and a help epilogue long enough to scroll is one that gets skipped
#: whole.
_EPILOGUE = """\
Starting out? In order:

  recall quickstart     see it work in one command, on a sample corpus, in its own throwaway
                        database. Calibrates nothing and wires nothing up. `--remove` undoes it.
  recall setup          THE install. Asks what you need, points recall at your own notes, fits an
                        abstention threshold to them, and registers the MCP server with Claude Code.
  recall doctor         when something is wrong and the symptom does not say what. Reads only.

  recall wizard         `setup` as a scriptable pipeline (`--headless --config`), for CI and for
                        rebuilding an install from a file. Same engine, no questions.

Everything else operates an install that already exists: index, search, schema, calibration,
generation, forget, lint, check, reasoning, graph, extract, rewrite.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recall",
        epilog=_EPILOGUE,
        # Without this argparse rewraps the epilogue into a single paragraph and the alignment that
        # makes it a table becomes noise.
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="hashing, fastembed[:model], st:<model>, voyage[:model], openai[:model]; "
        "RECALL_EMBED_PROFILE selects a registered profile such as "
        "bge-small-context-section-v1 or bge-large-context-section-v1",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help="table to read/write (default: chunks). Use a throwaway name to isolate experiments.",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help=f"tenant namespace to operate on (default: {DEFAULT_TENANT})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    setup_wizard.register(sub)
    schema_cmd.register(sub)
    manifest_cmd.register(sub)
    generation_cmd.register(sub)
    index_search.register(sub)
    reasoning_cmd.register(sub)
    graph_cmd.register(sub)
    extract_rewrite.register(sub)
    setup_wizard.register_quickstart(sub)
    index_search.register_demo_code(sub)
    lint_check.register(sub)
    calibration_cmd.register(sub)
    doctor_cmd.register(sub)
    return parser


#: What to do about each schema fault, for the ones whose message does not already say.
#:
#: Deliberately NOT a remedy for every class. `SchemaTooOld` already ends "run `recall schema
#: apply`" and `SchemaTooNew` ends "upgrade RE-call", so adding a second sentence would restate
#: the first in different words -- and a remedy that disagrees with the exception's own is worse
#: than none, because the reader cannot tell which is current. Entries exist only where the
#: message stops at the diagnosis.
_SCHEMA_REMEDY: dict[type[SchemaError], str] = {
    SchemaIncompatible: (
        "This table was created for a different embedder or a different RE-call version. An "
        "embedding dimension belongs to one model, so the fix is to pass the embedder this table "
        "was built with (--embedder), or to point at a different table (--table). Re-indexing "
        "into a table whose width does not match cannot work, and must not be forced."
    ),
    MigrationChecksumMismatch: (
        "A migration file on disk no longer matches the one recorded as applied. That is a "
        "question about which of the two is right, so it is not fixed by re-running anything: "
        "an applied migration is history, and editing one after the fact is what produces this."
    ),
    ConcurrentMigrator: (
        "Another migrator holds the lock. Wait for it and retry; two migrators against one "
        "database is the case this refuses on purpose."
    ),
    InterruptedConcurrentIndex: (
        "A concurrently-built index was left invalid, which happens when a migration is "
        "interrupted partway. Drop the named index and re-run `recall schema apply`."
    ),
}


def schema_error_message(exc: SchemaError) -> str:
    """Render a schema fault as the CLI error it is, rather than a traceback.

    Every member of this family is raised deliberately, with a message written for a person, and
    every one of them used to reach the operator as a Python traceback -- so a database that
    simply needed `recall schema apply` looked identical to a crash in the tool. The distinction
    matters more here than almost anywhere else in the CLI, because the whole family is
    *operator-fixable by construction*: none of them means recall is broken.

    Rendered rather than re-raised bare so the remedy can be attached where the exception itself
    stops at the diagnosis. `SchemaIncompatible` is the case that motivated this: its message,
    `table 'chunks' uses vector(384), requested dimension is 64`, is a precise statement of fact
    that tells a reader nothing about what to do, and it is the one people actually hit, because
    it fires whenever a table meets an embedder it was not built for.

    Looked up by exact type, not by `isinstance`, so a subclass added later gets no remedy rather
    than silently inheriting a parent's -- wrong advice presented confidently is the failure this
    whole audit kept finding, and an absent line is recoverable where a misleading one is not.
    """
    remedy = _SCHEMA_REMEDY.get(type(exc))
    if remedy is None:
        return str(exc)
    return f"{exc}\n\n{remedy}"


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    opens_db = bool(getattr(args, "_opens_db", False))
    if args.cmd == "lint":
        opens_db = bool(getattr(args, "semantic", False))
    if args.cmd == "schema" and getattr(args, "schema_cmd", None) == "grants":
        opens_db = False

    if (
        opens_db
        and args.cmd != "setup"
        and _DOTENV_ERROR is not None
        and not _env_opt_out("RECALL_IGNORE_BROKEN_DOTENV")
    ):
        raise SystemExit(
            f".env exists but could not be applied ({type(_DOTENV_ERROR).__name__}: "
            f"{_DOTENV_ERROR}), and this command connects to a database. Fix the file, or set "
            "RECALL_IGNORE_BROKEN_DOTENV=1 to proceed."
        )

    if opens_db:
        if args.cmd == "setup":
            try:
                _require_secure(args.dsn)
            except PermissionError as exc:
                raise SystemExit(
                    f"{exc}\n\nPass a DSN carrying a real password, or set "
                    "RECALL_ALLOW_INSECURE_DSN=1 to accept the risk deliberately."
                ) from exc
        elif args.cmd != "wizard":
            _require_secure(args.dsn)
    else:
        warn_if_insecure_dsn(args.dsn)

    migration_dsn = getattr(args, "migration_dsn", None)
    if migration_dsn and opens_db:
        _require_secure(migration_dsn)

    handler: Callable[[argparse.Namespace], None] = args.func
    try:
        handler(args)
    except SchemaError as exc:
        raise SystemExit(schema_error_message(exc)) from exc


if __name__ == "__main__":
    main()
