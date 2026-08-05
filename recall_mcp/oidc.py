"""External OIDC identity: discovery, JWKS, and fail-closed JWT validation.

Why this exists alongside `auth.py`
-----------------------------------
`auth.py` holds STATIC bearer tokens: a shared secret with no expiry and no issuer, valid until
somebody notices and edits a file. This module takes identity from an external provider instead,
so revocation, rotation and expiry belong to the IdP.

Rejections are a taxonomy, not a boolean
----------------------------------------
Every refusal carries a stable `reason`, and the reasons are the API: alerting on `bad_signature`
versus `unknown_kid` watches two different events (a forgery attempt versus a rotation this
process has not caught up with).

Two exception types, on purpose. `TokenRejected` states a fact about the TOKEN and maps to 401.
`IdentityProviderUnavailable` states a fact about the IdP and maps to 503. Collapsing them means
answering 401 for a server-side outage, which tells every client to re-authenticate against the
provider that is already down.

The four rejections that carry the most weight
-----------------------------------------------
- **`algorithm_not_allowed`** is checked before any key lookup, and the pinned algorithm — not the
  whole allowlist — is what reaches `jwt.decode`. Membership in the allowlist is not agreement
  with the selected key's family: with a mixed RSA/EC allowlist, `alg: ES256` against an RSA `kid`
  otherwise reaches `ECAlgorithm.prepare_key(RSAPublicKey)` and raises a bare `TypeError` that is
  not a `jwt.InvalidTokenError` and escapes every handler.
- **`missing_kid`** is enforced. Trying every key turns rotation into a signature oracle.
- **`missing_expiry`** because a token with no `exp` never stops being valid.
- **`missing_tenant`** because tenant identity is the isolation boundary, and an empty tenant is
  not "all tenants" — it must never reach the store, where an empty GUC reads as "match nothing".

The JWKS cache
--------------
Refreshed every `jwks_refresh_s`. If the IdP is unreachable the cached keys keep working up to
`max_stale_key_s`, then validation fails closed: an IdP blip must not take the fleet down, and a
withdrawn key must stop working in bounded time.

Two properties that are easy to state and easy to get wrong, so both are pinned by test:

1. **A failed fetch never updates `fetched_at`.** Otherwise each failure renews the stale window
   and a withdrawn key works forever — a validator that never fails closed passes a suite whose
   tests only ever fail once.
2. **A forced refresh is bounded per PROCESS, not per call.** An unknown `kid` triggers an
   out-of-band refresh so rotation is picked up without waiting the interval, but bounding that
   per call bounds nothing an attacker controls: N requests with N random kids would be N fetches.

Network IO happens OUTSIDE the cache lock (single-flight), so a slow IdP cannot serialise every
request in the process behind one HTTP call.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import jwt
from jwt.algorithms import RSAAlgorithm

from recall.observability import get_logger
from recall_mcp.auth import ALL_SCOPES, AuthConfigError, Principal

_log = get_logger("mcp.oidc")

#: Spec defaults. Named so a caller overriding one is visibly overriding a documented value.
DEFAULT_CLOCK_SKEW_S = 60
DEFAULT_JWKS_REFRESH_S = 15 * 60
DEFAULT_MAX_STALE_KEY_S = 60 * 60

#: Asymmetric only. Including any HS* algorithm alongside a published verification key IS the
#: algorithm-confusion vulnerability, not a configuration preference.
DEFAULT_ALGORITHMS = ("RS256", "RS384", "RS512", "ES256", "ES384", "PS256", "PS384", "PS512")

#: Which JWK key types can serve which algorithm family. Selection is checked against this at key
#: load time, so a family mismatch is a clean rejection rather than a library TypeError.
_ALG_KTY = {"RS": "RSA", "PS": "RSA", "ES": "EC"}

_HTTP_TIMEOUT_S = 10
#: Floor between forced (unknown-kid) refreshes, as a fraction of the refresh interval.
_FORCED_REFRESH_DIVISOR = 10


class TokenRejected(Exception):
    """A token was refused. `reason` is a stable slug; the message never contains the token."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


class IdentityProviderUnavailable(TokenRejected):
    """The IdP could not be consulted. A 503, not a 401.

    Subclasses `TokenRejected` so an existing caller that catches only that still fails closed;
    a caller that wants the distinction catches this first.
    """


@dataclass(frozen=True)
class OidcConfig:
    issuer: str
    audience: str
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    clock_skew_s: int = DEFAULT_CLOCK_SKEW_S
    jwks_refresh_s: int = DEFAULT_JWKS_REFRESH_S
    max_stale_key_s: int = DEFAULT_MAX_STALE_KEY_S
    tenant_claim: str = "tenant"
    scope_claim: str = "scope"
    #: Require `azp` to name us when a token carries several audiences (OIDC Core 3.1.3.7).
    require_azp_for_multi_audience: bool = True
    #: Tenants this deployment provisioned, or None to accept whatever the IdP asserts.
    #:
    #: The IdP vouches for identity; it knows nothing about our topology. A token reading
    #: `tenant: initech` is not forged merely because we never provisioned initech — but admitting
    #: it would open a store for a namespace no operator configured, and `StoreRegistry` is built
    #: on the property that its tenant set comes from configuration and never from traffic.
    #:
    #: None is permissive so the validator remains usable on its own terms; `build_auth` is what
    #: refuses to DEPLOY without one, which is where the decision belongs.
    allowed_tenants: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.issuer.strip() or not self.audience.strip():
            raise ValueError("issuer and audience are required")
        object.__setattr__(self, "algorithms", tuple(self.algorithms))
        if not self.algorithms:
            raise ValueError("algorithms must not be empty: an allowlist that permits nothing "
                             "rejects every token, which is an outage rather than a policy")
        weak = [a for a in self.algorithms if a.upper().startswith("HS") or a.lower() == "none"]
        if weak:
            raise ValueError(
                f"refusing symmetric or unsigned algorithms in an OIDC allowlist: {weak}. "
                "With a published verification key an HMAC algorithm IS the algorithm-confusion "
                "attack, because the key an attacker needs to forge with is public."
            )
        if self.jwks_refresh_s <= 0:
            raise ValueError("jwks_refresh_s must be positive; 0 refetches on every request")
        if self.clock_skew_s < 0:
            raise ValueError("clock_skew_s must be >= 0; a negative skew narrows validity")
        if self.max_stale_key_s < self.jwks_refresh_s:
            raise ValueError("max_stale_key_s must be >= jwks_refresh_s")
        if not self.tenant_claim.strip() or not self.scope_claim.strip():
            raise ValueError("tenant_claim and scope_claim must be non-empty")
        if self.allowed_tenants is not None:
            object.__setattr__(self, "allowed_tenants", frozenset(self.allowed_tenants))
            if not self.allowed_tenants:
                raise ValueError(
                    "allowed_tenants must not be empty: an allowlist that permits no tenant "
                    "rejects every token, which is an outage rather than a policy. Leave it None "
                    "to accept any tenant the IdP asserts."
                )
            # `validate` REFUSES a padded tenant claim rather than trimming it, so a padded entry
            # here could never match any token. That tenant would read as provisioned in the
            # configuration and be refused at every single request.
            unmatchable = sorted(t for t in self.allowed_tenants if not t or t != t.strip())
            if unmatchable:
                raise ValueError(
                    f"allowed_tenants entries are empty or padded and could never match a "
                    f"tenant claim: {unmatchable}"
                )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects outright on identity endpoints.

    `urlopen` follows 30x by default and its handler permits an https -> http downgrade, so a
    scheme check on the URL we were handed says nothing about the URL actually fetched. Key
    material is not something to retrieve over a connection we did not verify.
    """

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise IdentityProviderUnavailable(
            "discovery_failed", f"identity endpoint attempted a {code} redirect"
        )


_OPENER = urllib.request.build_opener(_NoRedirect)


def _http_get(url: str) -> bytes:
    if not url.startswith("https://"):
        raise IdentityProviderUnavailable(
            "discovery_failed", "refusing a non-https identity endpoint"
        )
    try:
        with _OPENER.open(url, timeout=_HTTP_TIMEOUT_S) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        # HTTPError IS the response object and owns the socket; the `with` above never ran.
        exc.close()
        raise IdentityProviderUnavailable("jwks_unavailable", f"identity endpoint {exc.code}") from exc


def discover_jwks_uri(issuer: str, *, opener: Callable[[str], bytes] | None = None) -> str:
    """Resolve the JWKS URI from the issuer's OIDC discovery document.

    The returned URI is constrained to the issuer's own origin. Without that, a hostile or
    compromised discovery document points this server's fetch at an internal address (a cloud
    metadata endpoint, an admin port) and thereby chooses the key set that verifies every token —
    which makes it a token-forgery path, not merely an information leak.
    """
    fetch = opener or _http_get
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        doc = json.loads(fetch(url))
    except IdentityProviderUnavailable:
        raise
    except Exception as exc:
        raise IdentityProviderUnavailable("discovery_failed", "discovery document unreadable") from exc
    if not isinstance(doc, dict):
        raise IdentityProviderUnavailable("discovery_failed", "discovery document is not an object")
    if str(doc.get("issuer", "")).rstrip("/") != issuer.rstrip("/"):
        raise IdentityProviderUnavailable(
            "discovery_failed", "discovery issuer does not match configuration"
        )
    uri = doc.get("jwks_uri")
    if not isinstance(uri, str) or not uri.startswith("https://"):
        raise IdentityProviderUnavailable("discovery_failed", "jwks_uri missing or not https")
    if urllib.parse.urlparse(uri).netloc != urllib.parse.urlparse(issuer).netloc:
        raise IdentityProviderUnavailable(
            "discovery_failed", "jwks_uri is not on the issuer's origin"
        )
    return uri


@dataclass
class _CachedKeys:
    keys: dict[str, Any] = field(default_factory=dict)
    fetched_at: float = 0.0
    ever_fetched: bool = False


def _usable_keys(document: Any) -> dict[str, Any]:
    """Parse a JWKS into kid -> key, refusing anything ambiguous.

    Three refusals worth naming, because each one is silent if you skip it:

    - A **duplicate `kid`** is rejected outright rather than last-one-wins. An IdP publishing two
      keys under one id is a rotation bug or an attack; overwriting means the real key stops
      verifying and the other one starts, with nothing logged.
    - A key marked for **encryption** (`use: enc`, or `key_ops` without `verify`) is skipped. It
      was not published to verify signatures, and accepting it widens the trusted set for free.
    - A **family mismatch** is caught here rather than inside PyJWT, where it surfaces as a
      `TypeError` that is not an `InvalidTokenError`.
    """
    if not isinstance(document, dict):
        raise IdentityProviderUnavailable("jwks_unavailable", "JWKS document is not an object")
    entries = document.get("keys")
    if not isinstance(entries, list):
        raise IdentityProviderUnavailable("jwks_unavailable", "JWKS 'keys' is not a list")

    keys: dict[str, Any] = {}
    dropped = 0
    for entry in entries:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        kid = entry.get("kid")
        if not isinstance(kid, str) or not kid:
            dropped += 1  # a key we cannot name cannot be selected by `kid`
            continue
        use = entry.get("use")
        ops = entry.get("key_ops")
        if (use is not None and use != "sig") or (
            isinstance(ops, list) and "verify" not in ops
        ):
            dropped += 1
            continue
        if kid in keys:
            raise IdentityProviderUnavailable(
                "jwks_unavailable", "JWKS publishes two keys under one key id"
            )
        kty = entry.get("kty")
        try:
            if kty == "RSA":
                keys[kid] = RSAAlgorithm.from_jwk(json.dumps(entry))
            elif kty == "EC":
                from jwt.algorithms import ECAlgorithm

                keys[kid] = ECAlgorithm.from_jwk(json.dumps(entry))
            else:
                dropped += 1
        except Exception:
            # Logged rather than silently swallowed: an unparseable published key otherwise
            # reaches the operator as `unknown_kid`, which says "rotation" when the truth is
            # "the IdP served something we cannot read".
            dropped += 1
            _log.warning("dropping unparseable JWKS entry", extra={"kid_len": len(kid)})
    if dropped:
        _log.warning("dropped %d unusable JWKS entries", dropped)
    if entries and not keys:
        raise IdentityProviderUnavailable("jwks_unavailable", "no usable keys in JWKS")
    return keys


def _kty_for(alg: str) -> str | None:
    return _ALG_KTY.get(alg[:2].upper())


class OidcValidator:
    """Validates a bearer JWT and returns a `Principal`. Tenant comes from the token, only.

    `validate` deliberately takes no tenant argument. With one, some call site would eventually
    pass a request parameter and the isolation model would be decided there.
    """

    def __init__(
        self,
        config: OidcConfig,
        *,
        jwks_uri: str | None = None,
        _jwks_fetcher: Callable[[], dict] | None = None,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._jwks_uri = jwks_uri
        self._fetcher = _jwks_fetcher
        # Wall clock, not monotonic: `max_stale_key_s` is a bound on how long a WITHDRAWN key may
        # keep working, which is a fact about the outside world. CLOCK_MONOTONIC does not advance
        # while a machine is suspended, so a snapshot-restored VM would silently extend it.
        self._clock = _clock or time.time
        self._cache = _CachedKeys()
        self._lock = threading.Lock()
        #: Held only while fetching, and never together with `_lock` around IO. Single-flight: one
        #: thread fetches, the rest carry on with what is cached.
        self._fetch_lock = threading.Lock()
        self._last_forced_at = 0.0
        self._missing_kids: set[str] = set()

    @property
    def config(self) -> OidcConfig:
        return self._config

    def _fetch_jwks(self) -> dict:
        if self._fetcher is not None:
            document = self._fetcher()
        else:
            uri = self._jwks_uri or discover_jwks_uri(self._config.issuer)
            self._jwks_uri = uri
            document = json.loads(_http_get(uri))
        if not isinstance(document, dict):
            raise IdentityProviderUnavailable("jwks_unavailable", "JWKS document is not an object")
        return document

    def _snapshot(self) -> _CachedKeys:
        with self._lock:
            return self._cache

    def _refresh(self, cached: _CachedKeys) -> dict[str, Any]:
        """Fetch and install, OUTSIDE the cache lock. Falls back to stale keys inside the window.

        The lock is taken only to read and to publish. Holding it across the HTTP call would make
        one slow IdP serialise every request in the process, which is the same outage the stale
        window exists to prevent, arriving as latency instead of as errors.
        """
        now = self._clock()
        age = now - cached.fetched_at
        if not self._fetch_lock.acquire(blocking=False):
            # Another thread is already fetching. Serve what we have rather than queue behind it.
            return cached.keys
        try:
            try:
                keys = _usable_keys(self._fetch_jwks())
            except Exception as exc:
                # Every fetch-side failure lands here, including the TokenRejected subclasses this
                # module raises for a malformed document. Routing those around the stale window
                # would mean a JSON error page causes a total outage while an HTML one is survived.
                if cached.ever_fetched and age <= self._config.max_stale_key_s:
                    _log.warning("serving stale JWKS: identity provider unreachable")
                    return cached.keys
                raise IdentityProviderUnavailable(
                    "jwks_unavailable", "identity provider unreachable and cached keys are stale"
                ) from exc
            with self._lock:
                # `fetched_at` advances ONLY here, on success. Updating it on failure would renew
                # the stale window at every failed attempt, so a withdrawn key would work forever.
                self._cache = _CachedKeys(keys=keys, fetched_at=now, ever_fetched=True)
                self._missing_kids.clear()
                return keys
        finally:
            self._fetch_lock.release()

    def _keys_for(self, kid: str) -> dict[str, Any]:
        cached = self._snapshot()
        due = not cached.ever_fetched or (
            self._clock() - cached.fetched_at >= self._config.jwks_refresh_s
        )
        keys = self._refresh(cached) if due else cached.keys
        if kid in keys:
            return keys

        # Unknown kid: refresh out of band so a rotation is picked up without waiting the
        # interval. Bounded PER PROCESS, because bounding it per call bounds nothing an attacker
        # controls — N requests with N random kids would be N fetches against the IdP.
        now = self._clock()
        floor = self._config.jwks_refresh_s / _FORCED_REFRESH_DIVISOR
        with self._lock:
            recently_missing = kid in self._missing_kids
            too_soon = now - self._last_forced_at < floor
            if not (recently_missing or too_soon):
                self._last_forced_at = now
                forced = True
            else:
                forced = False
        if not forced:
            return keys
        keys = self._refresh(self._snapshot())
        if kid not in keys:
            with self._lock:
                self._missing_kids.add(kid)
        return keys

    def validate(self, token: str) -> Principal:
        cfg = self._config
        if not isinstance(token, str) or not token.strip():
            raise TokenRejected("malformed", "empty bearer token")
        try:
            header = jwt.get_unverified_header(token)
        except Exception as exc:
            raise TokenRejected("malformed", "token header is not decodable") from exc

        # BEFORE any key lookup: checking after would mean the token's own header had already
        # influenced which key was chosen.
        alg = header.get("alg")
        if not isinstance(alg, str) or alg not in cfg.algorithms:
            shown = alg if isinstance(alg, str) else type(alg).__name__
            raise TokenRejected("algorithm_not_allowed", f"alg {shown[:16]!r} is not allowed")

        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise TokenRejected("missing_kid", "token header carries no key id")

        keys = self._keys_for(kid)
        if kid not in keys:
            raise TokenRejected("unknown_kid", "no published key matches this token's key id")

        key = keys[kid]
        # Membership in the allowlist is not agreement with THIS key's family. Caught here, it is
        # a clean rejection; left to PyJWT it is a TypeError that is not an InvalidTokenError and
        # escapes every handler below.
        expected = _kty_for(alg)
        actual = type(key).__name__
        if expected == "RSA" and "RSA" not in actual:
            raise TokenRejected("algorithm_not_allowed", "algorithm does not match the key type")
        if expected == "EC" and "Elliptic" not in actual and "EC" not in actual:
            raise TokenRejected("algorithm_not_allowed", "algorithm does not match the key type")

        try:
            claims = jwt.decode(
                token,
                key=key,
                # The PINNED algorithm, not the allowlist. The pre-check already reduced it to
                # exactly one, and handing PyJWT the full list lets the header re-select a family.
                algorithms=[alg],
                audience=cfg.audience,
                issuer=cfg.issuer,
                leeway=cfg.clock_skew_s,
                options={"require": ["exp", "iss", "aud"], "verify_signature": True},
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenRejected("expired", "token expiry is in the past") from exc
        except jwt.ImmatureSignatureError as exc:
            raise TokenRejected("not_yet_valid", "token is not valid yet") from exc
        except jwt.InvalidAudienceError as exc:
            raise TokenRejected("bad_audience", "audience does not match") from exc
        except jwt.InvalidIssuerError as exc:
            raise TokenRejected("bad_issuer", "issuer does not match") from exc
        except jwt.MissingRequiredClaimError as exc:
            missing = getattr(exc, "claim", "")
            reason = "missing_expiry" if missing == "exp" else "missing_claim"
            raise TokenRejected(reason, f"required claim {missing!r} is absent") from exc
        except jwt.InvalidSignatureError as exc:
            raise TokenRejected("bad_signature", "signature does not verify") from exc
        except jwt.InvalidTokenError as exc:
            raise TokenRejected("invalid", "token failed validation") from exc
        except Exception as exc:
            # The contract is that EVERY ambiguity resolves to a TokenRejected. `InvalidKeyError`
            # and `PyJWKError` are not `InvalidTokenError` subclasses, and a library that raises
            # something new in a future version must not turn a 401 into a 500.
            raise TokenRejected("invalid", "token failed validation") from exc

        self._check_authorized_party(claims)

        raw_tenant = claims.get(cfg.tenant_claim)
        if not isinstance(raw_tenant, str) or not raw_tenant.strip():
            raise TokenRejected(
                "missing_tenant",
                f"claim {cfg.tenant_claim!r} is absent or empty; an empty tenant is not a tenant",
            )
        if raw_tenant != raw_tenant.strip():
            # Rejected, not normalised. `auth.parse_principals` stores tenants verbatim and the
            # registry does exact string membership, so silently trimming here would make one
            # tenant id mean two different things depending on which factory built the Principal.
            raise TokenRejected(
                "malformed_tenant", "tenant claim has leading or trailing whitespace"
            )
        # AFTER the signature, deliberately. Reached before `jwt.decode`, this check would answer
        # "does tenant X exist here?" for a caller holding no credential at all: a forgery naming
        # a provisioned tenant would come back `bad_signature` while one naming an unprovisioned
        # tenant came back `tenant_not_allowed`, and the difference enumerates the deployment.
        if cfg.allowed_tenants is not None and raw_tenant not in cfg.allowed_tenants:
            raise TokenRejected(
                "tenant_not_allowed",
                "the token is genuine, but names a tenant this deployment did not provision",
            )

        return Principal(
            name=str(claims.get("sub") or raw_tenant),
            tenant=raw_tenant,
            scopes=_parse_scope_claim(claims.get(cfg.scope_claim)),
            # Carried onto the Principal, not merely checked here. The Principal is the object
            # that travels; without this a five-minute JWT becomes a credential downstream code
            # believes never expires.
            expires_at=datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc),
        )

    def _check_authorized_party(self, claims: dict) -> None:
        """OIDC Core 3.1.3.7: a multi-audience token must name us in `azp`.

        Without it, any other client of the same IdP that can obtain a token listing our audience
        is a valid caller here — and the `tenant` claim it carries becomes our isolation boundary.
        """
        if not self._config.require_azp_for_multi_audience:
            return
        aud = claims.get("aud")
        if isinstance(aud, (list, tuple)) and len(aud) > 1:
            if claims.get("azp") != self._config.audience:
                raise TokenRejected(
                    "bad_audience",
                    "multi-audience token is not authorised to this party",
                )


def _parse_scope_claim(raw: object) -> frozenset[str]:
    """Accept the space-delimited string of RFC 8693 or a JSON list; ignore unknown scopes.

    Unknown scopes are dropped rather than rejected, unlike the static-token file. The file is OUR
    configuration, so a typo there is an operator error worth failing on; the token comes from an
    IdP that legitimately issues scopes for other audiences. The filter is an allowlist
    intersection, so the failure direction is always LESS privilege.
    """
    if isinstance(raw, str):
        candidates = raw.split()
    elif isinstance(raw, (list, tuple)):
        candidates = [str(item) for item in raw]
    else:
        candidates = []
    return frozenset(scope for scope in candidates if scope in ALL_SCOPES)


#: Env keys. Named here so `build_auth` and the error messages cannot drift apart.
ENV_ISSUER = "RECALL_OIDC_ISSUER"
ENV_AUDIENCE = "RECALL_OIDC_AUDIENCE"
ENV_TENANTS = "RECALL_OIDC_TENANTS"
ENV_ALGORITHMS = "RECALL_OIDC_ALGORITHMS"


def _csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def oidc_validator_from_env(env: dict[str, str] | None = None) -> OidcValidator | None:
    """Build a validator from `RECALL_OIDC_*`, or None when the issuer is unset.

    None means "OIDC not configured", exactly as `token_registry_from_env` returns None for "no
    static tokens". Deciding whether that is acceptable belongs to the caller, which knows the
    transport.

    Constructing a validator performs NO network IO. Discovery and the first JWKS fetch happen on
    the first token, so an IdP that is briefly unreachable delays a request rather than preventing
    the server from booting — the same reasoning as the stale-key window, applied to startup.
    """
    source = dict(os.environ) if env is None else env
    issuer = source.get(ENV_ISSUER, "").strip()
    if not issuer:
        return None

    audience = source.get(ENV_AUDIENCE, "").strip()
    if not audience:
        raise AuthConfigError(
            f"{ENV_ISSUER} is set, so {ENV_AUDIENCE} is required. It is the audience this server "
            f"accepts; without it any token the IdP issued for any of its clients would verify "
            f"here, and that token's tenant claim would become this server's isolation boundary."
        )

    raw_tenants = source.get(ENV_TENANTS, "").strip()
    if not raw_tenants:
        raise AuthConfigError(
            f"{ENV_TENANTS} is required with {ENV_ISSUER}: a comma-separated list of the tenants "
            f"this deployment serves. Absent is not 'every tenant' — the IdP vouches for identity "
            f"and knows nothing about this deployment's topology, so without the list any tenant "
            f"claim it signs would open a store no operator provisioned."
        )
    tenants = frozenset(_csv(raw_tenants))

    algorithms = tuple(_csv(source.get(ENV_ALGORITHMS, ""))) or DEFAULT_ALGORITHMS
    try:
        config = OidcConfig(
            issuer=issuer,
            audience=audience,
            algorithms=algorithms,
            allowed_tenants=tenants,
        )
    except ValueError as exc:
        # A ValueError out of a config constructor reads as a bug at the call site. Restated as
        # AuthConfigError it joins the class this server already refuses to boot on.
        raise AuthConfigError(f"invalid RECALL_OIDC_* configuration: {exc}") from exc

    _log.info(
        "OIDC identity configured: issuer=%s audience=%s, %d provisioned tenant(s)",
        issuer,
        audience,
        len(tenants),
    )
    return OidcValidator(config)


__all__ = [
    "DEFAULT_ALGORITHMS",
    "DEFAULT_CLOCK_SKEW_S",
    "DEFAULT_JWKS_REFRESH_S",
    "DEFAULT_MAX_STALE_KEY_S",
    "ENV_ALGORITHMS",
    "ENV_AUDIENCE",
    "ENV_ISSUER",
    "ENV_TENANTS",
    "IdentityProviderUnavailable",
    "OidcConfig",
    "OidcValidator",
    "TokenRejected",
    "discover_jwks_uri",
    "oidc_validator_from_env",
]
