"""External OIDC identity: discovery, JWKS, and fail-closed JWT validation.

Why this exists alongside `auth.py`
-----------------------------------
`auth.py` holds STATIC bearer tokens: an operator provisions a file, and a token is a shared
secret with no expiry and no issuer. That is fine for a laptop and unacceptable for a fleet — a
leaked static token is valid until somebody notices and edits a file. This module takes identity
from an external identity provider instead, so revocation, rotation and expiry belong to the IdP.

Static tokens remain, but development-only, and production configuration refuses them.

What "fail closed" means here
-----------------------------
Every ambiguity resolves to rejection, and each rejection carries a STABLE `reason` string. The
reasons are the API: an operator alerting on `bad_signature` versus `unknown_kid` is watching for
two very different events (a forgery attempt versus a rotation this process has not caught up
with), and collapsing them into "invalid token" throws that distinction away.

The rejections that matter most, and why each is separate
---------------------------------------------------------
- **`algorithm_not_allowed`** covers algorithm confusion, which is the attack this class of code
  exists to stop. The classic form re-signs the token with HS256 using the RSA *public* key as the
  HMAC secret: the public key is, by definition, public, so a validator that infers the algorithm
  from the token's own header is asking the attacker which lock to use. The allowlist is
  asymmetric-only by default and is applied BEFORE any key lookup. `alg: none` falls out of the
  same check.
- **`missing_kid`** is enforced rather than worked around. A token without a key id forces the
  validator to try every key, which turns key rotation into a signature-oracle and makes an
  unknown-key rejection indistinguishable from a bad signature.
- **`missing_expiry`** because a token with no `exp` never stops being valid, and "the IdP always
  sets one" is an assumption about someone else's configuration.
- **`missing_tenant`** because tenant identity is the isolation boundary. An empty tenant is not
  "all tenants"; it is a token that cannot be authorised, and it must not reach the store layer
  where an empty GUC reads as "match nothing" and returns a plausible empty answer.

The JWKS cache and the stale window
-----------------------------------
Keys are cached and refreshed every `jwks_refresh_s` (15 minutes). If the IdP is unreachable at
refresh time the cached keys keep working up to `max_stale_key_s` (1 hour), then validation fails
closed. The two bounds encode a real trade-off: an IdP blip must not take the fleet down, and a
key withdrawn by the IdP must stop working within a bounded time. Serving stale keys forever would
mean revocation never propagates; refusing immediately would make the IdP a hard single point of
failure for every request.

An unknown `kid` triggers at most ONE out-of-band refresh, so a rotation is picked up without
waiting out the interval. The bound matters: without it, an attacker presenting random kids has a
free amplification channel against the IdP.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

import jwt
from jwt.algorithms import RSAAlgorithm

from recall_mcp.auth import ALL_SCOPES, Principal

#: Spec defaults. Named so a caller overriding one is visibly overriding a documented value.
DEFAULT_CLOCK_SKEW_S = 60
DEFAULT_JWKS_REFRESH_S = 15 * 60
DEFAULT_MAX_STALE_KEY_S = 60 * 60

#: Asymmetric only, deliberately. Including any HS* algorithm here alongside a public key is the
#: algorithm-confusion vulnerability, not a configuration preference.
DEFAULT_ALGORITHMS = ("RS256", "RS384", "RS512", "ES256", "ES384", "PS256", "PS384", "PS512")

_HTTP_TIMEOUT_S = 10


class TokenRejected(Exception):
    """A token was refused. `reason` is a stable slug; the message never contains the token."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class OidcConfig:
    issuer: str
    audience: str
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    clock_skew_s: int = DEFAULT_CLOCK_SKEW_S
    jwks_refresh_s: int = DEFAULT_JWKS_REFRESH_S
    max_stale_key_s: int = DEFAULT_MAX_STALE_KEY_S
    #: Claim carrying the tenant. Configurable because IdPs differ; never taken from a request.
    tenant_claim: str = "tenant"
    scope_claim: str = "scope"

    def __post_init__(self) -> None:
        if not self.issuer.strip() or not self.audience.strip():
            raise ValueError("issuer and audience are required")
        weak = [a for a in self.algorithms if a.upper().startswith("HS") or a.lower() == "none"]
        if weak:
            raise ValueError(
                f"refusing symmetric or unsigned algorithms in an OIDC allowlist: {weak}. "
                "With a public verification key an HMAC algorithm IS the algorithm-confusion "
                "attack, because the key an attacker needs to forge with is published."
            )
        if self.max_stale_key_s < self.jwks_refresh_s:
            raise ValueError("max_stale_key_s must be >= jwks_refresh_s")


def discover_jwks_uri(issuer: str, *, opener: Callable[[str], bytes] | None = None) -> str:
    """Resolve the JWKS URI from the issuer's OIDC discovery document."""
    fetch = opener or _http_get
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    doc = json.loads(fetch(url))
    if not isinstance(doc, dict):
        raise TokenRejected("discovery_failed", "discovery document is not an object")
    # The discovery document is fetched from the issuer, but a document that DISAGREES with the
    # issuer it was fetched from cannot be trusted to name a key endpoint either.
    if str(doc.get("issuer", "")).rstrip("/") != issuer.rstrip("/"):
        raise TokenRejected("discovery_failed", "discovery issuer does not match configuration")
    uri = doc.get("jwks_uri")
    if not isinstance(uri, str) or not uri.startswith("https://"):
        raise TokenRejected("discovery_failed", "jwks_uri missing or not https")
    return uri


def _http_get(url: str) -> bytes:
    if not url.startswith("https://"):
        raise TokenRejected("discovery_failed", "refusing a non-https identity endpoint")
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as response:  # noqa: S310
        return bytes(response.read())


@dataclass
class _CachedKeys:
    keys: dict[str, Any] = field(default_factory=dict)
    fetched_at: float = 0.0
    ever_fetched: bool = False


class OidcValidator:
    """Validates a bearer JWT and returns a `Principal`. Tenant comes from the token, only.

    `validate` deliberately takes no tenant argument. If it did, some call site would eventually
    pass one from a request parameter and the isolation model would be decided there; not having
    the parameter makes that unrepresentable rather than merely discouraged.
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
        self._clock = _clock or time.monotonic
        self._cache = _CachedKeys()
        self._lock = threading.Lock()

    @property
    def config(self) -> OidcConfig:
        return self._config

    def _fetch_jwks(self) -> dict:
        if self._fetcher is not None:
            return self._fetcher()
        uri = self._jwks_uri or discover_jwks_uri(self._config.issuer)
        self._jwks_uri = uri
        document = json.loads(_http_get(uri))
        # Checked rather than cast. `json.loads` returns Any, so silencing the type error would
        # push a malformed JWKS downstream into an AttributeError on `.get("keys")` — an
        # unreachable-IdP symptom wearing a crash's clothes.
        if not isinstance(document, dict):
            raise TokenRejected("jwks_unavailable", "JWKS document is not a JSON object")
        return document

    def _load_keys(self, *, force: bool) -> dict[str, Any]:
        """Return kid -> key, refreshing when due. Serves stale keys inside the stale window."""
        with self._lock:
            now = self._clock()
            age = now - self._cache.fetched_at
            due = force or not self._cache.ever_fetched or age >= self._config.jwks_refresh_s
            if not due:
                return self._cache.keys
            try:
                raw = self._fetch_jwks()
            except TokenRejected:
                raise
            except Exception as exc:
                # The IdP is unreachable. Keep serving what we have until the stale window closes,
                # then fail closed: a key the IdP has withdrawn must stop working in bounded time.
                if self._cache.ever_fetched and age <= self._config.max_stale_key_s:
                    return self._cache.keys
                raise TokenRejected(
                    "jwks_unavailable", "identity provider unreachable and cached keys are stale"
                ) from exc
            keys: dict[str, Any] = {}
            for entry in (raw or {}).get("keys", []):
                kid = entry.get("kid")
                if not isinstance(kid, str) or not kid:
                    continue  # a key we cannot name cannot be selected by `kid`
                try:
                    keys[kid] = RSAAlgorithm.from_jwk(json.dumps(entry))
                except Exception:
                    try:
                        from jwt.algorithms import ECAlgorithm

                        keys[kid] = ECAlgorithm.from_jwk(json.dumps(entry))
                    except Exception:
                        continue
            self._cache = _CachedKeys(keys=keys, fetched_at=now, ever_fetched=True)
            return keys

    def validate(self, token: str) -> Principal:
        cfg = self._config
        if not isinstance(token, str) or not token.strip():
            raise TokenRejected("malformed", "empty bearer token")
        try:
            header = jwt.get_unverified_header(token)
        except Exception as exc:
            raise TokenRejected("malformed", "token header is not decodable") from exc

        # BEFORE any key lookup. Checking the algorithm after selecting a key would mean the
        # token's own header had already influenced which key was chosen.
        alg = str(header.get("alg", ""))
        if alg not in cfg.algorithms:
            raise TokenRejected("algorithm_not_allowed", f"alg {alg!r} is not in the allowlist")

        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise TokenRejected("missing_kid", "token header carries no key id")

        keys = self._load_keys(force=False)
        if kid not in keys:
            # One out-of-band refresh, so a rotation is picked up without waiting the interval —
            # and exactly one, so arbitrary kids cannot be used to hammer the IdP.
            keys = self._load_keys(force=True)
        if kid not in keys:
            raise TokenRejected("unknown_kid", "no published key matches this token's key id")

        try:
            claims = jwt.decode(
                token,
                key=keys[kid],
                algorithms=list(cfg.algorithms),
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
            # Deliberately last: every specific reason above is more useful than this one, and a
            # broad handler placed first would swallow all of them.
            raise TokenRejected("invalid", "token failed validation") from exc

        tenant = claims.get(cfg.tenant_claim)
        if not isinstance(tenant, str) or not tenant.strip():
            raise TokenRejected(
                "missing_tenant",
                f"claim {cfg.tenant_claim!r} is absent or empty; an empty tenant is not a tenant",
            )

        return Principal(
            name=str(claims.get("sub") or tenant),
            tenant=tenant.strip(),
            scopes=_parse_scope_claim(claims.get(cfg.scope_claim)),
        )


def _parse_scope_claim(raw: object) -> frozenset[str]:
    """Accept the space-delimited string of RFC 8693 or a JSON list; ignore unknown scopes.

    Unknown scopes are dropped rather than rejected, unlike the static-token file. The file is OUR
    configuration, so a typo there is an operator error worth failing on; the token comes from an
    IdP that legitimately issues scopes for other audiences, and refusing those would make this
    service brittle to unrelated changes in someone else's authorisation server.
    """
    if isinstance(raw, str):
        candidates = raw.split()
    elif isinstance(raw, (list, tuple)):
        candidates = [str(item) for item in raw]
    else:
        candidates = []
    return frozenset(scope for scope in candidates if scope in ALL_SCOPES)


__all__ = [
    "DEFAULT_ALGORITHMS",
    "DEFAULT_CLOCK_SKEW_S",
    "DEFAULT_JWKS_REFRESH_S",
    "DEFAULT_MAX_STALE_KEY_S",
    "OidcConfig",
    "OidcValidator",
    "TokenRejected",
    "discover_jwks_uri",
]
