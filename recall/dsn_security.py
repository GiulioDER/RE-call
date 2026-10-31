"""DSN safety: refuse or warn about connection strings that would leak credentials.

Moved out of `recall.store` (architecture step S7d). `recall.store` re-exports every name, so
existing imports and patches of `recall.store.<name>` keep working.
"""

from __future__ import annotations

import os
import re
from ipaddress import ip_address
from urllib.parse import unquote, urlsplit

from recall.observability import get_logger

_log = get_logger("store")  # the store's own logger name, which tests filter on

#: The built-in dev credentials shipped in the default DSN — safe only against a local database.
_DEFAULT_CREDS = ("recall", "recall")

#: "" covers a hostless/unix-socket DSN. Bracketed IPv6 is absent on purpose: urlsplit strips
#: the brackets. All of 127.0.0.0/8 is handled numerically by `_is_local_host`.
#: `0.0.0.0` is here as a CONNECT target, not a bind address: as a destination it resolves to
#: the local host, so a DSN naming it carries the built-in credentials no further than
#: `localhost` would. S104 is about binding a LISTENER to every interface, which this is not;
#: the listener default lives in `recall_mcp/server.py` and is `127.0.0.1`.
_LOCAL_HOSTS = ("", "localhost", "::1", "0.0.0.0")  # noqa: S104

#: Hosts that get a WARNING but not a refusal. `host.docker.internal` used to sit in
#: `_LOCAL_HOSTS`, which was wrong in one direction: from inside a container it reaches the
#: container HOST, which can be a shared, network-reachable machine. It stays out of the
#: refusal so the compose quickstart keeps working, and the warning says what it reaches.
_WARN_ONLY_HOSTS = ("host.docker.internal",)

#: Values that count as "yes" for an opt-OUT of a safety guard. Deliberately an allowlist rather
#: than a truthiness test: `0`/`false`/`no` must read as "keep the guard", and anything
#: unrecognised must too, because a typo in a security switch may not grant permission.
_ENV_TRUE = frozenset({"1", "true", "yes", "on"})


#: Opt-out for `require_secure_dsn`. Named so it cannot be set by accident, and so its presence
#: in a deploy is a visible, greppable decision rather than an oversight.
INSECURE_DSN_OPT_OUT = "RECALL_ALLOW_INSECURE_DSN"


def _is_local_host(host: str) -> bool:
    """True when `host` cannot reach a shared database (loopback, unix socket, or unset)."""
    if host in _LOCAL_HOSTS or host.startswith(("/", "%2f")):  # %2f: percent-encoded socket dir
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _dsn_passwords(dsn: str) -> tuple[str, ...]:
    """Every password this DSN could carry, in either libpq form. Never raises.

    The URI form (`postgresql://u:pw@host/db`) and the keyword form (`host=x password=pw`) are both
    accepted by libpq, so both are scrubbed. A DSN that parses as neither yields nothing to scrub,
    which is the honest answer: this is a redaction helper, not a validator, and a malformed DSN is
    exactly when the caller most needs it to keep working.
    """
    found: list[str] = []
    try:
        uri_password = urlsplit(dsn).password
    except ValueError:
        uri_password = None
    if uri_password:
        found.append(uri_password)
        # ⛔ **The DECODED form too.** `urlsplit` does not percent-decode and libpq does, so a
        # password containing a reserved character (which MUST be encoded in a URI DSN, making this
        # the ordinary case rather than an edge one) was scrubbed in a spelling that never appears
        # in an error message. Measured: `postgresql://u:p%40ss@h/db` scrubbed `p%40ss` while the
        # text carried `p@ss`. Three auditors found this independently.
        decoded = unquote(uri_password)
        if decoded != uri_password:
            found.append(decoded)

    # The keyword form. libpq allows single-quoted values with backslash escapes, so a password
    # containing a space or a quote is reachable and must still be matched in full.
    for match in re.finditer(r"password\s*=\s*(?:'((?:[^'\\]|\\.)*)'|(\S+))", dsn):
        quoted, bare = match.group(1), match.group(2)
        value = quoted.replace("\\'", "'").replace("\\\\", "\\") if quoted is not None else bare
        if value:
            found.append(value)
    return tuple(found)


def scrub_dsn_secrets(text: str, *dsns: str) -> str:
    """Remove any of these DSNs' passwords from `text`.

    `redacted_dsn` is not enough on its own, because the password can be inside the EXCEPTION rather
    than inside a DSN we format. Two measurements, years apart in spirit and days apart in fact:

    * a password containing `%` produced
      `psycopg.ProgrammingError: invalid percent-encoded token: "S3cr%tPw"`, so wrapping the error
      with a redacted DSN beside it still printed the secret verbatim;
    * a MALFORMED dsn produced
      `ProgrammingError: missing "=" after "not-a-dsn://user:PASSWORD@x" in connection info string`,
      echoing the entire connection string. That one reached a label in the desktop settings page,
      which is on screen, in screenshots, and in whatever a user pastes into an issue — and a
      mistyped DSN is exactly when a person is looking at that label.

    Anything derived from a connection attempt goes through here. Lives beside `redacted_dsn`
    rather than in one caller, because it was written twice before this: once in
    `recall/wizard/headless.py` and once, nearly, in the preflight.

    ⚠️ **A password shorter than 4 characters is NOT removed.** A blind `str.replace` of a
    one-character password rewrote every occurrence of that letter: measured, a password of `a`
    turned "cannot connect to database at host a" into
    "c***nnot connect to d***t***b***se ***t host ***". An unreadable error is its own kind of
    failure, and the callers put this on screen. So the guarantee below has a floor, and it is
    stated HERE rather than in a comment inside the loop, because a caller who needs an
    unconditional guarantee must be able to see that this is not one.

    ⛔ **Both DSN forms, because libpq accepts both and the installer's field is free text.**
    This handled only the URI form: `urlsplit("host=db password=s3cret dbname=recall").password` is
    `None`, so a keyword/value DSN — which psycopg accepts, and which is what somebody pasting from
    a hosting provider's console often has — passed through with the password intact, while the
    caller's docstring promised it had been removed. The parse is hand-rolled rather than delegated
    to `psycopg.conninfo`, because this function must not fail on a MALFORMED dsn: malformed is the
    case that produced the second measurement above.
    """
    for dsn in dsns:
        for password in _dsn_passwords(dsn):
            # ⚠️ **Short passwords are skipped, deliberately.** A blind `str.replace` of a
            # one-character password rewrote every occurrence of that letter: measured, a password
            # of `a` turned "cannot connect to database at host a" into
            # "c***nnot connect to d***t***b***se ***t host ***". An unreadable error is its own
            # kind of failure, and a password that short is not the secret worth protecting at the
            # cost of every diagnosis the user needs.
            if len(password) < 4:
                continue
            text = text.replace(password, "***")
    return text


def redacted_dsn(dsn: str) -> str:
    """`dsn` with any password removed — safe to print to a log or a systemd journal.

    A connection failure is exactly when an operator wants the DSN in the logs, and exactly when
    printing it verbatim would write the password to disk.
    """
    try:
        parts = urlsplit(dsn)
        if not parts.hostname:
            return "<dsn>"
        userinfo = (
            f"{parts.username}:***@"
            if parts.password
            else (f"{parts.username}@" if parts.username else "")
        )
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{userinfo}{parts.hostname}{port}{parts.path}"
    except ValueError:  # pragma: no cover - malformed URL
        return "<dsn>"


def warn_if_insecure_dsn(dsn: str) -> str | None:
    """Warn (to stderr) when the built-in ``recall:recall`` credentials target a NON-local host.

    A shared, well-known password is fine against localhost but a footgun the moment the DSN
    points at a real remote database. This warns loudly and returns the message; it never blocks
    execution (returns None when there is nothing to warn about).
    """
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return None
    # unquote: urlsplit returns the RAW percent-encoded form, so "recal%6C" is the password
    # "recall" and must not slip past the comparison
    if (unquote(parts.username or ""), unquote(parts.password or "")) != _DEFAULT_CREDS:
        return None
    host = (parts.hostname or "").lower()
    if _is_local_host(host):
        return None
    if host in _WARN_ONLY_HOSTS:
        # Warn without returning a message: `require_secure_dsn` raises on any returned
        # message, and this host must warn rather than block (it is the documented way a
        # containerised quickstart reaches its database).
        _log.warning(
            "recall: WARNING — default 'recall:recall' credentials against %r, which reaches "
            "the container HOST, not this container. Fine for a local quickstart; set a strong "
            "password if that machine is shared.",
            parts.hostname,
        )
        return None
    msg = (
        f"recall: WARNING — using the default 'recall:recall' credentials against non-local host "
        f"{parts.hostname!r}. Set a strong password via RECALL_DSN before using a remote database."
    )
    _log.warning(msg)
    return msg


def _env_opt_out(name: str) -> bool:
    """True only when `name` is set to an explicit affirmative (see `_ENV_TRUE`)."""
    return os.environ.get(name, "").strip().lower() in _ENV_TRUE


def require_secure_dsn(dsn: str) -> None:
    """Raise unless `dsn` is safe to use unattended; the fail-closed form of the warning above.

    `warn_if_insecure_dsn` detects the built-in `recall:recall` credentials against a remote host
    and then RETURNS, so the process carries on talking to a shared database with a password
    published in this repository's README. A warning on stderr is not a control: under systemd it
    lands in a journal nobody reads, and the server comes up looking healthy.

    A server should therefore call this instead. The escape hatch is an explicit environment
    variable, because the legitimate case (a private network where the operator has genuinely
    accepted the risk) must be expressible — just not by default and not silently.
    """
    # Parsed, not merely truthy. A bare `os.environ.get(...)` reads ANY non-empty string as
    # "opt out", so `RECALL_ALLOW_INSECURE_DSN=0` — which an operator writes meaning "keep the
    # guard on" — silently switched the guard OFF, the exact inverse of the intent. Anything
    # unrecognised keeps the guard: this is a security control, so an ambiguous value must not
    # be read as permission.
    if _env_opt_out(INSECURE_DSN_OPT_OUT):
        return
    if warn_if_insecure_dsn(dsn) is None:
        return
    raise PermissionError(
        f"refusing to start against {redacted_dsn(dsn)}: the default 'recall:recall' credentials "
        f"are published in this project's README and this DSN points at a non-local host. Set a "
        f"real password, or set {INSECURE_DSN_OPT_OUT}=1 to accept the risk deliberately."
    )
