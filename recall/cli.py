from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from recall._env import load_dotenv
from recall.observability import configure_logging
from recall.store import (
    DEFAULT_TABLE,
    DEFAULT_TENANT,
    _env_opt_out,
    require_secure_dsn,
    warn_if_insecure_dsn,
)

from recall.cli_commands import (
    calibration_cmd,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recall")
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
        "bge-small-context-section-v1",
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
    return parser


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
    handler(args)


if __name__ == "__main__":
    main()
