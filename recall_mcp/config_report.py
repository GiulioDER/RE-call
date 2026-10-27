"""What RE-call's server would run with, with credential values withheld by name and by shape.

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
3. and in every value that IS shown, the credential shapes below are replaced: URL userinfo
   (`user:password@`, or a bare token before `@`), query parameters named like a credential
   (`?key=`, `?access_token=`, `X-Amz-Signature=`, ...), libpq keyword passwords (quoted or not)
   and `Bearer` or `Basic` authorization values.

A credential in any other shape, inside a setting whose name and flag do not mark it, IS shown:
the rules are a list, not a proof. Only the schema's keys are reported, plus the NAMES of other
`RECALL_*` variables that are set (never their values), so a typo such as `RECALL_TRUST_MOD` is
visible.
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

#: Names that end like a credential, with or without a separator before the word, so `PGPASSWORD`,
#: `MYSQL_PWD` and `X_APIKEY` are caught as well as `RECALL_REASONING_API_KEY`. Anchored at the end
#: so `RECALL_OIDC_MAX_TOKEN_LIFETIME_SECONDS`, `RECALL_RATE_LIMIT_KEY_PREFIX` and the token-count
#: settings (`..._MAX_TOKENS`) stay visible. A `_DSN` carries a host and often a password.
_CREDENTIAL_NAME = re.compile(
    r"(?:API_?KEY|KEYS?|TOKEN|SECRETS?|PASSWORD|PASSWD|PASS|PWD|PAT|CREDENTIALS?|ACCESS_KEY_ID|_DSN)$", re.I
)
#: Named like a credential and not one: a boolean switch.
_NOT_A_CREDENTIAL = frozenset({"RECALL_ALLOW_INSECURE_DSN"})
#: A flag that names a credential: `--key`, `--api-key`, `--auth-token`, `--access-token`,
#: `--bearer`, `--db-password`. The credential word is whole, so `--no-auth`, `--sort-key`,
#: `--bypass` and `--pass-through` stay ordinary flags and keep the argument after them visible.
_FLAG_WORD = (
    r"(?:key|(?:[\w-]*[-_])?(?:api[-_]?key|access[-_]?key|secret[-_]?key|private[-_]?key|token|secret"
    r"|password|passwd|pass|pwd|bearer))"
)
#: Everything between `scheme://` and the `@` before the host: `user:password@`, `token@`, and a
#: password holding an unencoded `@` (the last `@` before the path ends the userinfo).
_URL_USERINFO = re.compile(r"(?P<head>[a-z][a-z0-9+.-]*://)(?P<userinfo>[^\s/?#]*)@", re.I)
#: A query parameter whose NAME ends in a whole credential word: `?key=`, `?apiKey=`,
#: `?access_token=`, `&sig=`, `X-Amz-Signature=`, `X-Amz-Credential=`, `?password=`, libpq's
#: `?sslpassword=`. `?author=`,
#: `&design=` and `&keyword=` stay readable.
_QUERY_CREDENTIAL = re.compile(
    r"(?P<head>[?&;](?:[\w.-]*[_.-])?(?:api_?key|key|token|secret|(?:ssl)?passw(?:or)?d|pwd|sig|signature|auth"
    r"|credentials?)=)(?P<value>[^&\s#;]+)",
    re.I,
)
#: libpq keyword form, `host=x password=y` or `password = 'a b'`, as `recall.store` parses it.
_KEYWORD_PASSWORD = re.compile(
    r"(?P<head>(?:^|\s)(?:password|passwd|sslpassword|api_key|apikey|token|secret)\s*=\s*)"
    r"(?P<pw>'(?:[^'\\]|\\.)*'|[^\s&;]+)",
    re.I,
)
#: An HTTP authorization value: `Authorization: Bearer x`, `Basic x`.
_AUTHORIZATION = re.compile(r"(?P<head>\b(?:Bearer|Basic)\s+)[A-Za-z0-9._~+/=-]+", re.I)

_SCHEMA = {spec.name: spec for spec in ENVIRONMENT_SCHEMA}


def withheld(name: str) -> bool:
    """Whether `name`'s value is never shown: flagged secret, or named like a credential."""
    spec = _SCHEMA.get(name)
    if spec and spec.secret:
        return True
    return name not in _NOT_A_CREDENTIAL and bool(_CREDENTIAL_NAME.search(name))


def redact_value(value: str) -> str:
    """`value` with URL userinfo, credential query parameters, libpq keyword passwords and
    authorization values replaced by `***`."""
    value = _URL_USERINFO.sub(lambda m: f"{m.group('head')}***@", value)
    value = _QUERY_CREDENTIAL.sub(lambda m: f"{m.group('head')}***", value)
    value = _KEYWORD_PASSWORD.sub(lambda m: f"{m.group('head')}***", value)
    return _AUTHORIZATION.sub(lambda m: f"{m.group('head')}***", value)


#: `NAME=value` inside an argument, as an ssh command string or `env NAME=value` carries it.
_ASSIGNMENT = re.compile(r"(?P<name>\b[A-Za-z_][A-Za-z0-9_]*)=(?P<value>'[^']*'|\"[^\"]*\"|[^\s;&|]+)")
#: A flag whose NEXT argument is a credential: `--api-key`, `--auth-token`, `--access-token`, ...
_CREDENTIAL_FLAG = re.compile(rf"^--?{_FLAG_WORD}$", re.I)
#: The same flag with its value inside one argument: `--api-key=x`, or `--api-key x` in an ssh
#: command string.
_INLINE_FLAG = re.compile(rf"(?P<flag>(?:^|\s)--?{_FLAG_WORD}(?:=|\s+))(?P<value>'[^']*'|\"[^\"]*\"|[^\s;&|]+)", re.I)


def redact_argv(argv: list[str]) -> list[str]:
    """A server's launch command with the credentials it recognises replaced by `***`.

    The shapes: `NAME=value` with a withheld NAME (also inside a single ssh command string), the
    argument after a credential flag such as `--api-key` or `--auth-token`, and every shape
    `redact_value` replaces (URL userinfo, credential query parameters, keyword passwords,
    `Bearer` values).
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
        lines.append(
            "\nset but not in the MCP server's settings schema (a typo, or a variable another part of "
            "RE-call reads): " + ", ".join(report["unknown_recall_variables"])
        )
    lines.append("\n" + (f"the server would refuse to start: {report['problem']}" if report["problem"] else "valid: the server's settings validation passes"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recall_mcp.config_report",
        description="Report RE-call's configuration, with credential values withheld by name and by shape.",
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
