"""What RE-call's server would run with, reported without a single secret value.

`python -m recall_mcp.config_report --json` prints every key of `ENVIRONMENT_SCHEMA` with its
state (set, unset), its default, and its value where showing it is safe, plus the first problem
the server's own startup validation would refuse. `--env-from-stdin` reports a given mapping
(a registered MCP server's `env`) instead of this process's environment.

It runs as its own process on purpose, and the dashboard calls it that way rather than importing
it: the redaction happens where the values live, so on a remote host (`ssh host ...`) a secret
never crosses the network, and the `recall` library keeps its import contract with `recall_mcp`.

A value is withheld when:

1. the schema flags its key as secret;
2. the key's NAME ends like a credential (`_KEY`, `_TOKEN`, `_SECRET`, `_PASSWORD`, ...), so a key
   someone forgot to flag is still withheld;
3. and in every value that IS shown, a password inside a URL or a libpq keyword string is
   replaced, because a "URL" setting can carry one.

Only the schema's keys are reported, plus the NAMES of other `RECALL_*` variables that are set
(never their values), so a typo such as `RECALL_TRUST_MOD` is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

from recall_mcp.settings import ENVIRONMENT_SCHEMA, Settings

__all__ = ["REPORT_VERSION", "build_report", "main", "redact_argv", "redact_value", "withheld"]

REPORT_VERSION = 1
WITHHELD = "(set, withheld)"

#: Names that end like a credential. Anchored at the end so `RECALL_OIDC_MAX_TOKEN_LIFETIME_SECONDS`
#: or `RECALL_RATE_LIMIT_KEY_PREFIX` stay visible, while `RECALL_REASONING_API_KEY` does not.
_CREDENTIAL_NAME = re.compile(r"(?:_|^)(?:API_KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|ACCESS_KEY_ID)$", re.I)
#: `scheme://user:password@host` and `scheme://:password@host`.
_URL_PASSWORD = re.compile(r"(?P<head>[a-z][a-z0-9+.-]*://[^/@\s:]*:)(?P<pw>[^/@\s]+)(?P<at>@)", re.I)
#: libpq keyword form, `host=x password=y`, and query parameters such as `?password=y`.
_KEYWORD_PASSWORD = re.compile(r"(?P<head>(?:^|[\s?&;])(?:password|passwd|sslpassword|api_key|apikey|token|secret)=)(?P<pw>[^\s&;]+)", re.I)

_SCHEMA = {spec.name: spec for spec in ENVIRONMENT_SCHEMA}


def withheld(name: str) -> bool:
    """Whether `name`'s value is never shown: flagged secret, or named like a credential."""
    spec = _SCHEMA.get(name)
    return bool(spec and spec.secret) or bool(_CREDENTIAL_NAME.search(name))


def redact_value(value: str) -> str:
    """`value` with any password inside a URL or a keyword string replaced by `***`."""
    value = _URL_PASSWORD.sub(lambda m: f"{m.group('head')}***{m.group('at')}", value)
    return _KEYWORD_PASSWORD.sub(lambda m: f"{m.group('head')}***", value)


#: `NAME=value` inside an argument, as an ssh command string or `env NAME=value` carries it.
_ASSIGNMENT = re.compile(r"(?P<name>\b[A-Za-z_][A-Za-z0-9_]*)=(?P<value>'[^']*'|\"[^\"]*\"|[^\s;&|]+)")
#: A flag whose NEXT argument is a credential.
_CREDENTIAL_FLAG = re.compile(r"^--?(?:api[-_]?key|token|secret|password|passwd|key)$", re.I)
#: The same flag with its value inside one argument: `--api-key=x`, or `--api-key x` in an ssh
#: command string.
_INLINE_FLAG = re.compile(r"(?P<flag>(?:^|\s)--?(?:api[-_]?key|token|secret|password|passwd)(?:=|\s+))(?P<value>[^\s;&|]+)", re.I)


def redact_argv(argv: list[str]) -> list[str]:
    """A server's launch command with every credential in it replaced by `***`.

    Covers the three shapes a command carries one in: `NAME=value` with a withheld NAME (also
    inside a single ssh command string), the argument after a flag such as `--api-key`, and a
    password inside a URL or keyword string.
    """
    out: list[str] = []
    hide_next = False
    for arg in argv:
        if hide_next:
            out.append("***")
            hide_next = False
            continue
        hide_next = bool(_CREDENTIAL_FLAG.match(arg))
        arg = _ASSIGNMENT.sub(lambda m: f"{m.group('name')}=***" if withheld(m.group("name")) else m.group(0), arg)
        arg = _INLINE_FLAG.sub(lambda m: f"{m.group('flag')}***", arg)
        out.append(redact_value(arg))
    return out


def _scrub_message(message: str, env: Mapping[str, str]) -> str:
    """A validation message with every withheld value that appears in it replaced."""
    for name, value in env.items():
        if value and len(value) >= 4 and withheld(name):
            message = message.replace(value, "***")
    return redact_value(message)


def build_report(env: Mapping[str, str], *, source: str, argv: list[str] | None = None) -> dict[str, Any]:
    """The report for `env` (and a server's launch `argv`, redacted). Pure: reads and writes nothing else."""
    entries = []
    for spec in ENVIRONMENT_SCHEMA:
        raw = env.get(spec.name)
        state = "set" if raw not in (None, "") else "unset"
        if state == "unset":
            value = None
        elif withheld(spec.name):
            value = WITHHELD
        else:
            value = redact_value(str(raw))
        entries.append(
            {
                "name": spec.name,
                "section": spec.section,
                "description": spec.description,
                "default": spec.default,
                "state": state,
                "value": value,
                "withheld": withheld(spec.name),
            }
        )
    unknown = sorted(name for name in env if name.startswith("RECALL_") and name not in _SCHEMA and env[name] != "")
    problem = None
    try:
        Settings.from_env(dict(env))
    except Exception as exc:  # BROAD-CATCH: fail-open  # any refusal is reported, scrubbed, not raised
        problem = _scrub_message(f"{type(exc).__name__}: {exc}", env)
    return {
        "version": REPORT_VERSION,
        "source": source,
        "entries": entries,
        "unknown_recall_variables": unknown,
        "problem": problem,
        "argv": redact_argv(list(argv)) if argv is not None else None,
    }


def _text(report: Mapping[str, Any]) -> str:
    lines = [f"configuration ({report['source']})"]
    section = ""
    for entry in report["entries"]:
        if entry["state"] != "set":
            continue
        if entry["section"] != section:
            section = entry["section"]
            lines.append(f"\n[{section}]")
        lines.append(f"  {entry['name']} = {entry['value']}")
    if report["unknown_recall_variables"]:
        lines.append("\nset but not in the schema (a typo?): " + ", ".join(report["unknown_recall_variables"]))
    lines.append("\n" + (f"the server would refuse to start: {report['problem']}" if report["problem"] else "valid: the server's startup checks pass"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recall_mcp.config_report",
        description="Report RE-call's configuration with every secret value withheld.",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--env-from-stdin", action="store_true",
        help="report the JSON on stdin instead of this process's environment: a server's env as an "
        'object, or {"env": {...}, "argv": [...]} to report its launch command as well',
    )
    args = parser.parse_args(argv)
    launch: list[str] | None = None  # a registered server's command, when one is given on stdin
    if args.env_from_stdin:
        given = json.load(sys.stdin)
        if not isinstance(given, dict):
            parser.error("stdin must hold a JSON object")
        if isinstance(given.get("env"), dict) or isinstance(given.get("argv"), list):
            raw_launch = given.get("argv")
            launch = [str(a) for a in raw_launch] if isinstance(raw_launch, list) else None
            given = given.get("env") if isinstance(given.get("env"), dict) else {}
        env = {str(k): str(v) for k, v in given.items()}
        source = "given"
    else:
        env = dict(os.environ)
        source = "environment"
    report = build_report(env, source=source, argv=launch)
    print(json.dumps(report, indent=1) if args.json else _text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
