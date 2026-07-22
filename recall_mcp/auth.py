"""Bearer-token authentication and principal -> tenant mapping for the HTTP transport.

Why this module exists
----------------------
`PgVectorStore` is bound to ONE tenant for its whole lifetime: the pool's `configure` hook sets
`recall.tenant_id` per connection, and the RLS policy compares against that GUC (`store.py`
`_prepare` / `_enable_rls`). Tenant isolation is therefore a property of *which store you hold*,
not of an argument you pass. So authentication here does not just answer "may this caller in?" —
it answers "which store is this caller allowed to touch?", and the answer has to be decided before
any tool body runs.

Threat model
------------
This is a **static bearer-token** scheme: an operator provisions tokens out of band, a client
presents one, and the server maps it to a principal. It is deliberately NOT an OAuth authorisation
server. That buys simplicity at a real cost, stated plainly so nobody has to discover it in
production:

- **No revocation without a restart.** Tokens live in a file read at startup. Removing one takes
  effect when the process reloads, not when you save the file.
- **No rotation protocol.** Overlapping validity has to be arranged by hand: add the new token,
  restart, migrate clients, remove the old one, restart again.
- **Bearer means bearer.** A leaked token is full access for that principal until it is removed.
  There is no proof-of-possession and no audience binding.

For a deployment that needs any of those, put a real identity provider in front and supply the
SDK's `auth_server_provider` instead of this. What this module *does* give you is the property
that matters most and is most often missing: an unauthenticated network listener is impossible to
create by accident, because `build_auth` fails closed (see `AuthConfigError`).

Tokens come from a FILE, never an environment variable
------------------------------------------------------
`RECALL_AUTH_TOKENS_FILE` is the only source. There is no `RECALL_AUTH_TOKENS=<secret>`, and that
omission is the point: environment variables leak through `/proc/<pid>/environ`, `ps e`, container
inspection APIs, crash dumps and — worst — every child process the server ever spawns. A secrets
file is what Kubernetes and Docker mount anyway, and it can be permission-checked, which is what
`_warn_if_world_readable` does.

The file may hold either plaintext tokens or their SHA-256 digests (`token_sha256`), so an
operator provisioning access never has to write a live credential to disk in recoverable form.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from recall.observability import get_logger
from recall.store import DEFAULT_TENANT

_log = get_logger("mcp.auth")

#: Scope required by each tool. These mirror the tools' own MCP annotations — `recall_search` and
#: `recall_stats` are `readOnlyHint`, `recall_index` is a write, `recall_forget` is
#: `destructiveHint` — so a principal's scopes line up with the risk the tool actually carries.
#: They are separate rather than a single "write" because the two writes fail differently:
#: `recall_index` burns embedding spend, `recall_forget` destroys memory irreversibly.
SCOPE_READ = "recall:read"
SCOPE_WRITE = "recall:write"
SCOPE_FORGET = "recall:forget"
ALL_SCOPES = (SCOPE_READ, SCOPE_WRITE, SCOPE_FORGET)

#: Minimum accepted token length. 32 characters is ~192 bits at base64/hex alphabets — far past
#: anything brute-forceable — and the floor is deliberately NOT configurable: an operator who can
#: lower it will, under deadline, and a 6-character bearer token on a public listener is the
#: failure this whole module exists to prevent. `secrets.token_urlsafe(32)` produces 43 chars.
MIN_TOKEN_LENGTH = 32

_ENV_TOKENS_FILE = "RECALL_AUTH_TOKENS_FILE"


class AuthConfigError(RuntimeError):
    """Raised when the auth configuration is absent, malformed, or unsafe.

    Always fatal at startup and never downgraded to a warning: every path that raises this would
    otherwise produce a listener that is open, or open to the wrong tenant.
    """


@dataclass(frozen=True)
class Principal:
    """An authenticated caller, and the tenant whose memory it may reach.

    `tenant` is the security boundary. It selects the store — and therefore the RLS GUC — that
    every tool call from this principal runs against.
    """

    name: str
    tenant: str
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({SCOPE_READ}))
    expires_at: datetime | None = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def is_expired(self, *, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or datetime.now(timezone.utc)) >= self.expires_at


def _sha256_hex(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


#: Fixed digest-shaped value compared against on the lookup-miss path, so a miss costs the same
#: comparison a hit does. Not a secret and not derived from one.
_MISS_SENTINEL = "0" * 64


class TokenRegistry:
    """Maps a presented bearer token to a `Principal` in constant time w.r.t. the secret.

    Lookup is keyed on the SHA-256 digest of the token rather than the token itself. Two properties
    follow, and both matter:

    1. The registry never holds a plaintext credential in memory, so a heap dump or a stray repr
       does not hand over live tokens.
    2. Recovering a valid token from the keys requires a preimage attack on SHA-256. Dict-lookup
       timing does vary with key content, but here that content is a digest of the secret, not the
       secret — so what leaks is information about a hash the attacker already computed.

    Note what is deliberately NOT claimed: there is no `secrets.compare_digest` on the hit path,
    because there is nothing left to compare. The dict lookup *is* the check, and wrapping it in a
    constant-time comparison of a value against itself would look like a security control while
    being a no-op. The one real timing concern — a dict miss returning faster than a hit — is
    handled in `verify` instead.
    """

    def __init__(self, principals: dict[str, Principal]) -> None:
        #: digest -> principal. Never digest -> token.
        self._by_digest = dict(principals)

    def __len__(self) -> int:
        return len(self._by_digest)

    @property
    def tenants(self) -> frozenset[str]:
        """Every tenant reachable by some configured token.

        The set of tenants is fixed by configuration, not by traffic. `StoreRegistry` relies on
        this: it can only ever open a store for a tenant an operator explicitly provisioned, so
        there is no request-driven growth in pools or connections.
        """
        return frozenset(p.tenant for p in self._by_digest.values())

    def verify(self, token: str, *, now: datetime | None = None) -> Principal | None:
        """Return the principal for `token`, or None if it is unknown or expired."""
        if not token:
            return None
        digest = _sha256_hex(token)
        principal = self._by_digest.get(digest)
        if principal is None:
            # Burn a comparison of equal length on the miss path. Without it, "unknown token"
            # returns measurably sooner than "known token", which is a free oracle: an attacker
            # learns a candidate is valid from the response time alone, without needing the
            # request to succeed. `_MISS_SENTINEL` is a fixed digest-shaped string, so the work
            # done here matches the work done on a hit.
            secrets.compare_digest(digest, _MISS_SENTINEL)
            return None
        if principal.is_expired(now=now):
            _log.warning("rejected expired token for principal %r", principal.name)
            return None
        return principal


def authorize(
    token_scopes: "Sequence[str] | None", claims: dict | None, required: str
) -> str:
    """Decide a single call: return the tenant it may touch, or raise PermissionError.

    Deliberately a free function over plain data rather than a method on a request object. The
    authorisation decision is the highest-consequence branch in this codebase, and as a pure
    function it can be tested exhaustively — every missing-scope, missing-claim and empty-tenant
    case — without a database, a transport, or a live server. A version of this living inside a
    closure in `build_server` was only reachable through a full end-to-end request, which in
    practice means the failure paths never get tested at all.
    """
    if required not in (token_scopes or ()):
        raise PermissionError(f"this token lacks the {required!r} scope required for this tool")
    tenant = (claims or {}).get("tenant")
    if not isinstance(tenant, str) or not tenant:
        # Fail closed rather than defaulting. A token that authenticates but carries no tenant is
        # a provisioning bug, and the "helpful" fallback — DEFAULT_TENANT — would hand it the
        # namespace a single-tenant install keeps everything in.
        raise PermissionError("authenticated token carries no tenant claim")
    return tenant


def _parse_expiry(raw: object, *, who: str) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise AuthConfigError(f"{who}: 'expires_at' must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise AuthConfigError(f"{who}: 'expires_at' is not valid ISO-8601: {raw!r}") from exc
    if parsed.tzinfo is None:
        # A naive timestamp would be compared against an aware `now` and raise at request time —
        # i.e. the server would start fine and then fail on the first authenticated call.
        raise AuthConfigError(
            f"{who}: 'expires_at' must include a timezone offset (e.g. '2027-01-01T00:00:00Z')"
        )
    return parsed


def _parse_scopes(raw: object, *, who: str) -> frozenset[str]:
    if raw is None:
        return frozenset({SCOPE_READ})  # least privilege by default: read, nothing else
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        raise AuthConfigError(f"{who}: 'scopes' must be a list of strings")
    unknown = sorted(set(raw) - set(ALL_SCOPES))
    if unknown:
        # Refuse rather than ignore. A typo'd scope silently dropped means a principal quietly has
        # less access than the operator wrote down — discovered later as a mystery 403.
        raise AuthConfigError(
            f"{who}: unknown scope(s) {unknown}; valid scopes are {list(ALL_SCOPES)}"
        )
    return frozenset(raw)


def _digest_for_entry(entry: dict, *, who: str) -> str:
    plaintext = entry.get("token")
    digest = entry.get("token_sha256")
    if (plaintext is None) == (digest is None):
        raise AuthConfigError(f"{who}: provide exactly one of 'token' or 'token_sha256'")
    if digest is not None:
        if not isinstance(digest, str) or len(digest) != 64:
            raise AuthConfigError(f"{who}: 'token_sha256' must be a 64-char hex SHA-256 digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise AuthConfigError(f"{who}: 'token_sha256' is not hex") from exc
        return digest.lower()
    if not isinstance(plaintext, str):
        raise AuthConfigError(f"{who}: 'token' must be a string")
    if len(plaintext) < MIN_TOKEN_LENGTH:
        # Deliberately does NOT echo the token — not even a prefix. An error message is the most
        # likely thing in this module to reach a log aggregator or a bug report.
        raise AuthConfigError(
            f"{who}: token is shorter than {MIN_TOKEN_LENGTH} characters. Generate one with "
            f"`python -c 'import secrets; print(secrets.token_urlsafe(32))'`."
        )
    return _sha256_hex(plaintext)


def parse_principals(payload: object) -> dict[str, Principal]:
    """Validate a parsed token document and return digest -> Principal.

    Every malformed entry raises. Nothing is skipped: a skipped entry is a principal that either
    cannot authenticate at all, or — if the malformed field was `tenant` — authenticates into the
    wrong isolation boundary. Both are worse than refusing to start.
    """
    if not isinstance(payload, dict) or "principals" not in payload:
        raise AuthConfigError("token file must be a JSON object with a 'principals' array")
    entries = payload["principals"]
    if not isinstance(entries, list) or not entries:
        raise AuthConfigError("'principals' must be a non-empty array")

    by_digest: dict[str, Principal] = {}
    seen_names: set[str] = set()
    for i, entry in enumerate(entries):
        who = f"principals[{i}]"
        if not isinstance(entry, dict):
            raise AuthConfigError(f"{who}: must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AuthConfigError(f"{who}: 'name' is required and must be a non-empty string")
        if name in seen_names:
            raise AuthConfigError(f"{who}: duplicate principal name {name!r}")
        seen_names.add(name)

        tenant = entry.get("tenant", DEFAULT_TENANT)
        if not isinstance(tenant, str) or not tenant.strip():
            raise AuthConfigError(f"{who}: 'tenant' must be a non-empty string")

        digest = _digest_for_entry(entry, who=who)
        if digest in by_digest:
            # Two principals sharing a token makes the audit trail a lie: you could never tell
            # from a log line which of them acted.
            raise AuthConfigError(
                f"{who}: token collides with principal {by_digest[digest].name!r}"
            )
        by_digest[digest] = Principal(
            name=name,
            tenant=tenant,
            scopes=_parse_scopes(entry.get("scopes"), who=who),
            expires_at=_parse_expiry(entry.get("expires_at"), who=who),
        )
    return by_digest


def _warn_if_world_readable(path: Path) -> None:
    """Warn when a token file is readable beyond its owner.

    A warning rather than a hard failure: on Windows and on some bind-mounted container volumes
    the POSIX bits are not meaningful, and refusing to start there would push operators toward
    disabling auth entirely — a strictly worse outcome than a loud log line.
    """
    if os.name != "posix":
        # Windows reports POSIX-looking bits that do not reflect its actual ACLs — a temp file
        # there reads as 0o666 while being perfectly well protected. Warning anyway would make
        # this message fire on every Windows start, and a warning that is always wrong on a
        # platform is one operators learn to ignore everywhere.
        return
    try:
        mode = path.stat().st_mode
    except OSError:  # pragma: no cover - stat failure is reported by the read that follows
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        _log.warning(
            "token file %s is readable by group or other (mode %s) — tighten it with "
            "`chmod 600 %s`", path, oct(stat.S_IMODE(mode)), path,
        )


def load_token_registry(path: str | os.PathLike[str]) -> TokenRegistry:
    """Read and validate the token file at `path`."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise AuthConfigError(f"token file not found: {p}")
    _warn_if_world_readable(p)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthConfigError(f"cannot read token file {p}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # `exc` carries only position/message, never file content — safe to surface.
        raise AuthConfigError(f"token file {p} is not valid JSON: {exc}") from exc
    registry = TokenRegistry(parse_principals(payload))
    _log.info(
        "loaded %d principal(s) across %d tenant(s)", len(registry), len(registry.tenants)
    )
    return registry


def token_registry_from_env(env: dict[str, str] | None = None) -> TokenRegistry | None:
    """Build a registry from `RECALL_AUTH_TOKENS_FILE`, or None when unset.

    None means "no auth configured". It is the caller's job to decide whether that is acceptable
    for the transport it is about to start — `build_auth` in `server.py` refuses it for HTTP and
    permits it for stdio.
    """
    src = (env if env is not None else dict(os.environ)).get(_ENV_TOKENS_FILE)
    if not src:
        return None
    return load_token_registry(src)
