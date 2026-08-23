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
    index_search,
    lint_check,
    manifest_cmd,
    reasoning_cmd,
    schema_cmd,
    setup_wizard,
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
except Exception as _dotenv_exc:  # noqa: BLE001 - see below
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
    except Exception:  # noqa: BLE001 - this handler must not be able to fail either
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
    # No `choices=`: the accepted set is whatever `resolve_embedder` supports
    # (hashing, fastembed[:model], st:<model>, voyage[:model], openai[:model]), and
    # duplicating it here is how it drifted out of step with the setup wizard.
    parser.add_argument(
        "--embedder",
        default=os.environ.get("RECALL_EMBEDDER", "fastembed"),
        help="hashing, fastembed[:model], st:<model>, voyage[:model], openai[:model]",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,  # imported, not retyped: the wizard compares against it
        help="table to read/write (default: chunks). Use a throwaway name to keep an "
        "experiment out of your real memory index.",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help=f"tenant namespace to operate on (default: {DEFAULT_TENANT}). Every command is "
        f"scoped to one tenant; `forget` in particular deletes nothing outside it, so an "
        f"erasure request against another tenant needs this flag.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Registration order is the `--help` listing order, so families whose commands are not
    # adjacent in that listing register in more than one step.
    setup_wizard.register(sub)
    schema_cmd.register(sub)
    manifest_cmd.register(sub)
    generation_cmd.register(sub)
    index_search.register(sub)
    reasoning_cmd.register(sub)
    extract_rewrite.register(sub)
    setup_wizard.register_quickstart(sub)
    index_search.register_demo_code(sub)
    lint_check.register(sub)
    calibration_cmd.register(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
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
    parser = build_parser()

    args = parser.parse_args(argv)
    # Commands that will actually open a connection FAIL CLOSED on the insecure default DSN;
    # everything else only warns.
    #
    # This is grafted FORWARD from the merge, not restored from before it: the pre-merge CLI had
    # no such check at all, only the warning. An earlier version of this comment claimed the
    # merge had dropped the wiring, which was simply false, and a bug audit caught it.
    #
    # ⚠️ This set is INCOMPLETE and known to be: `generation`, `calibration`, `schema` and
    # `lint --semantic` all open connections and are not in it, and `--migration-dsn` (the
    # DDL-owner credential) is never checked on any path. Tracked as follow-up; do not read the
    # presence of this guard as coverage.
    # Every command that will open a connection FAILS CLOSED on the insecure default DSN;
    # the rest only warn.
    #
    # An earlier version of this set listed six commands by hand and missed four that connect
    # (generation, calibration, schema, and lint --semantic), so the guard read as coverage and
    # was not. The set is derived from the parsers now: a subcommand declares `_opens_db=True`
    # beside its own definition, so a new one cannot be added without answering the question.
    opens_db = bool(getattr(args, "_opens_db", False))
    if args.cmd == "lint":  # only the --semantic path reaches a database
        opens_db = bool(getattr(args, "semantic", False))
    if args.cmd == "schema" and getattr(args, "schema_cmd", None) == "grants":
        opens_db = False  # prints SQL for an operator to run; opens nothing

    if (
        opens_db
        and args.cmd != "setup"  # see the setup-specific carve-out for _require_secure below —
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
        elif args.cmd == "wizard":
            # `recall wizard` never contacts `args.dsn`. Every DSN it uses comes from --config, so
            # checking the global one is wrong in BOTH directions and both were reproduced: a
            # config naming `recall:recall` against a remote host sailed through unchecked, and a
            # `--serving-dsn`/RECALL_SERVING_DSN pointing at a remote host refused the command
            # outright, as a raw PermissionError traceback, before the config was even read. The
            # real DSNs are checked in the wizard handler, once they exist.
            pass
        else:
            _require_secure(args.dsn)
    else:
        warn_if_insecure_dsn(args.dsn)  # loud stderr note if default creds target a remote host

    # The DDL-owner credential was never checked or even warned about on any path, which is the
    # wrong way round: it is the most privileged DSN this CLI accepts.
    #
    # No longer scoped to `schema`. That scoping meant the credential was guarded on exactly one of
    # the commands that use it, and `generation`, `calibration` and the wizard's own DDL all ran
    # unchecked. `opens_db` still keeps `schema grants` exempt, since it only prints SQL.
    migration_dsn = getattr(args, "migration_dsn", None)
    if migration_dsn and opens_db:
        _require_secure(migration_dsn)

    # `warn_return_any` is on, so the Any-typed attribute is bound to a typed local rather than
    # returned directly.
    handler: Callable[[argparse.Namespace], None] = args.func
    handler(args)


if __name__ == "__main__":
    main()
