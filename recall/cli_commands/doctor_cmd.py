"""`recall doctor`: report what is wrong with this install, and write nothing.

⚠️ **This deliberately does NOT set `_opens_db`, and that is the whole design of the command.**
`_opens_db` makes `recall.cli` run `_require_secure` on the DSN and refuse a broken `.env` before
the handler is reached. Both are right for a command that is about to index or search, and both are
wrong here: a diagnostic that refuses to start because the thing it diagnoses is misconfigured has
nothing left to diagnose. `recall.doctor` opens its own connection and turns a failure into a
reported check with a repair line, which is what the person running it needs.

The insecure-DSN warning still prints, because `recall.cli` warns on the `not opens_db` branch.
That is correct: it is advice, not a refusal.
"""

from __future__ import annotations

import argparse
import json


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_doctor = sub.add_parser(
        "doctor",
        help="check this install end to end and say what is wrong, changing nothing",
        description=(
            "Check the install end to end: interpreter, package, console scripts on PATH, "
            "embedder backend, Docker, database, pgvector, schema, whether the table and tenant "
            "you are configured for actually hold anything, calibration, and the Claude Code "
            "wiring. Writes nothing and repairs nothing; every problem is printed with the "
            "command that fixes it. Exits non-zero only when something is BLOCKED, so a missing "
            "calibration does not fail a script."
        ),
    )
    p_doctor.set_defaults(func=_cmd_doctor)
    p_doctor.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the checks as JSON instead of a readable report",
    )
    p_doctor.add_argument(
        "--project-root",
        default=None,
        help="the directory the MCP registration is keyed by (default: the current directory). A "
        "local-scope server lives under projects/<dir>, so checking the wrong one reports a "
        "healthy registration as absent.",
    )


def _cmd_doctor(args: argparse.Namespace) -> None:
    from pathlib import Path

    from recall.doctor import run_checks

    report = run_checks(
        dsn=args.dsn,
        embedder=args.embedder,
        table=args.table,
        tenant=args.tenant,
        project_root=Path(args.project_root) if args.project_root else None,
    )
    if args.as_json:
        print(json.dumps({"checks": [c.as_dict() for c in report.checks]}, indent=2))
    else:
        print(report.render())
    # SystemExit rather than sys.exit, to match every other refusal in this CLI, and only on a
    # blocking failure. See `Report.exit_code`.
    if report.exit_code():
        raise SystemExit(report.exit_code())
