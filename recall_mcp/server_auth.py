"""Authentication and tenancy for the MCP server: token verifiers, the auth builder, the RLS gate.

Moved out of `recall_mcp.server` (architecture review, server split). `recall_mcp.server`
re-imports every name, so `recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl

from recall.observability import get_logger
from recall_mcp.auth import (
    AuthConfigError,
    ENV_TOKENS_FILE,
    TokenRegistry,
    token_registry_from_env,
)
from recall_mcp.limits import (
    FailedAuthThrottle,
    failed_auth_throttle_from_env,
)
from recall_mcp.oidc import (
    ENV_AUDIENCE,
    ENV_ISSUER,
    ENV_TENANTS,
    IdentityProviderUnavailable,
    OidcValidator,
    TokenRejected,
    oidc_env_present,
    oidc_validator_from_env,
)
from recall_mcp.threads import _to_thread

_T = TypeVar("_T")
_log = get_logger("mcp")  # the server's own logger name


#: Transport to serve. `stdio` is a private pipe between one client and this process — there is no
#: network listener and no remote caller to authenticate, so auth is not required there. The HTTP
#: transports open a socket, and `build_auth` refuses to start them unless an authentication
#: mechanism is configured — a static token file, or an OIDC provider.
Transport = Literal["stdio", "sse", "streamable-http"]
TRANSPORTS: tuple[Transport, ...] = ("stdio", "sse", "streamable-http")
HTTP_TRANSPORTS = frozenset({"streamable-http", "sse"})


class TenantProvisioning(Protocol):
    """Whatever decided which tenants exist. `TokenRegistry` and `ProvisionedTenants` both satisfy it.

    The lifespan needs the tenant set and nothing else about how authentication works, so this is
    the whole surface. Keeping it this narrow is what lets a second identity mechanism land
    without the store layer learning that it exists.
    """

    @property
    def tenants(self) -> frozenset[str]: ...


class RecallTokenVerifier:
    """Adapts `TokenRegistry` to the MCP SDK's `TokenVerifier` protocol.

    The tenant travels in `claims`, not in `scopes`. Scopes answer "what may this caller do";
    the tenant answers "whose data", and conflating them into one string list is how you end up
    with a caller whose scope string is parsed as a tenant name.
    """

    def __init__(
        self,
        registry: TokenRegistry,
        throttle: FailedAuthThrottle | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._throttle = (
            throttle if throttle is not None else failed_auth_throttle_from_env(env)
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        # No pre-verification gate here, deliberately. Verifying a static token is a
        # constant-cost digest lookup, so there is no expensive work for a throttle to
        # protect — and gating BEFORE the lookup would refuse VALID tokens whenever a
        # failure storm had drained the shared bucket, since `allow()` cannot tell a valid
        # token from garbage. The throttle exists to cap the OIDC path's JWKS/RSA work; on
        # this cheap path a failed lookup is recorded (so the signal exists) but never gates.
        principal = self._registry.verify(token)
        if principal is None:
            self._throttle.record_failure()
            # No token material in the log line — only the fact of a rejection. A logged prefix
            # is enough to shrink a brute-force search space, and logs travel further than
            # anyone expects.
            _log.warning("rejected an unrecognised bearer token")
            return None
        return AccessToken(
            token=token,
            client_id=principal.name,
            scopes=sorted(principal.scopes),
            expires_at=(int(principal.expires_at.timestamp()) if principal.expires_at else None),
            claims={"tenant": principal.tenant, "principal": principal.name},
        )


class OidcTokenVerifier:
    """Adapts `OidcValidator` to the MCP SDK's `TokenVerifier` protocol.

    The static counterpart above looks a token up in a table. This one verifies a signature
    against an external provider, but the OUTPUT is identical by construction: an `AccessToken`
    whose tenant travels in `claims`. Everything downstream — `StoreRegistry`, the per-tool scope
    check — is written against that shape and cannot tell the two apart, which is the point.

    One thing is lost at this boundary and cannot be recovered: `verify_token` returns an
    `AccessToken` or `None`, so the 401/503 distinction the validator is careful to preserve has
    nowhere to go. It survives into the log instead, because "the IdP is unreachable" and
    "somebody is forging tokens" call for opposite responses and both arrive here as a refusal.
    """

    def __init__(
        self,
        validator: OidcValidator,
        throttle: FailedAuthThrottle | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._validator = validator
        self._throttle = (
            throttle if throttle is not None else failed_auth_throttle_from_env(env)
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        if not self._throttle.reserve():
            # Gated here, unlike the static path, because validation is EXPENSIVE — a JWKS
            # fetch and an RSA verify — and capping that work is what stops a forgery wave
            # from driving the CPU and the provider round-trips. The cost of the gate is that
            # during a sustained storm a VALID OIDC token is refused too: admission cannot
            # tell it from garbage without doing the very work the gate defers. That is a
            # deliberate availability-for-integrity trade on this path, not a property the
            # throttle can avoid; the per-source ASGI middleware named in FailedAuthThrottle
            # is the way to narrow the blast radius, and it is not built yet.
            _log.warning("refusing unverified tokens: failed-authentication throttle is closed")
            return None
        try:
            # OFF THE LOOP (PERF-001). `validate` is synchronous, and on a cache miss it makes
            # two blocking HTTPS calls — discovery, then JWKS — each bounded by a 10s timeout.
            # Called inline from here that stops the entire server for up to 20s, and the
            # single-flight `acquire(blocking=False)` written to prevent exactly that outage is
            # inert against it: it lets other THREADS carry on with cached keys, and an event
            # loop has none, only coroutines that never get scheduled. Even fully warm, RSA
            # verification is ~2ms of uninterruptible loop time on every authenticated request.
            principal = await _to_thread(lambda: self._validator.validate(token))
        except IdentityProviderUnavailable as exc:
            self._throttle.refund()
            # Distinct message, deliberately: this is an outage on our side of the trust
            # relationship, and reading it as a wave of forgeries would send an operator hunting
            # an attacker while the IdP stays down.
            _log.warning(
                "refusing tokens: the identity provider could not be consulted (reason=%s)",
                exc.reason,
            )
            return None
        except TokenRejected as exc:
            # Keep the reservation for rejected tokens. Provider outages return it above:
            # an unreachable provider must not spend the failed-authentication budget.
            # `exc.reason` is a stable slug and never contains token material; the exception's
            # detail is deliberately not logged, and neither is the token.
            _log.warning("rejected a bearer token (reason=%s)", exc.reason)
            return None
        except Exception:  # BROAD-CATCH: fail-closed
            # Defence in depth (NUM-001). The validator's contract is that every ambiguity
            # resolves to a TokenRejected, but this is the boundary where a breach of it turns a
            # 401 into a 500: the SDK does not wrap `verify_token`. A failure to authenticate
            # must fail CLOSED as a refusal, never as a stack trace.
            _log.warning("rejected a bearer token (reason=validator_error)", exc_info=True)
            return None
        self._throttle.refund()
        return AccessToken(
            token=token,
            client_id=principal.name,
            scopes=sorted(principal.scopes),
            expires_at=(int(principal.expires_at.timestamp()) if principal.expires_at else None),
            # `subject` and `iss` are populated (SEC-007) because the SDK's session-principal
            # comparison is built from client_id + subject + claims["iss"], and silently degrades
            # to whichever of those the verifier supplied. Both are known here, verified.
            subject=principal.name,
            claims={
                "tenant": principal.tenant,
                "principal": principal.name,
                "iss": self._validator.config.issuer,
            },
        )


@dataclass(frozen=True)
class ProvisionedTenants:
    """The tenant set a deployment serves, when it did not come from a token file.

    `_make_lifespan` needs exactly one thing from the auth configuration: which tenants may have a
    store. `TokenRegistry` supplies that via `.tenants`, and this supplies the same for OIDC, so
    the lifespan does not branch on which mechanism is in use.
    """

    tenants: frozenset[str]

    def __post_init__(self) -> None:
        # Coerced and checked here rather than trusted from the caller (DAT-002). This type is
        # the carrier of "the tenant set comes from configuration, never from traffic", and an
        # annotation is not an enforcement: a plain `set` passed in stays mutable, and an empty
        # one reaches `min(registry.allowed_tenants)` in the lifespan. Every future provisioning
        # mechanism passes through here, so the check belongs here and not in each of them.
        if isinstance(self.tenants, (str, bytes)):
            # `frozenset("acme")` is {"a","c","e","m"} (BUG-002), and that set is handed straight
            # to StoreRegistry as `allowed_tenants`. The emptiness check below would not catch it:
            # single characters are neither empty nor padded.
            raise AuthConfigError(
                "provisioned tenants must be a collection of tenant ids, not a single string "
                "(a string would be split into its characters)"
            )
        object.__setattr__(self, "tenants", frozenset(self.tenants))
        if not self.tenants:
            raise AuthConfigError(
                "a provisioned tenant set must not be empty: an authenticated server that can "
                "serve no tenant refuses every request it authenticates"
            )


def build_auth(
    transport: str | None = None, env: dict[str, str] | None = None
) -> tuple[
    RecallTokenVerifier | OidcTokenVerifier | None,
    AuthSettings | None,
    TenantProvisioning | None,
]:
    """Resolve the auth configuration for `transport`, failing closed on the HTTP ones.

    This is the function that makes an unauthenticated network listener impossible to create by
    accident. Starting an HTTP transport with NO mechanism configured raises instead of warning,
    because the failure mode of a warning here is a server that comes up looking healthy with
    every memory in it world-readable — and the warning lands in a journal nobody reads until
    afterwards.

    Two mechanisms, exactly one of which may be active:

    - `RECALL_AUTH_TOKENS_FILE` — static bearer tokens. Development only; `load_token_registry`
      refuses to load it under `RECALL_ENV=production`.
    - `RECALL_OIDC_ISSUER` (with `RECALL_OIDC_AUDIENCE` and `RECALL_OIDC_TENANTS`) — identity from
      an external provider. With this set, `RECALL_AUTH_ISSUER_URL` defaults to the provider.

    Both together raises **only when `RECALL_AUTH_MODE` is unset**. They are two trust models, so
    an undeclared precedence means one of them sits in the configuration looking effective;
    declaring it with `RECALL_AUTH_MODE=oidc|static` is a choice, and it is what makes a staged
    cutover possible. A mode naming a mechanism that is not configured also raises, as does a mode
    outside that pair.

    Only the SELECTED mechanism is constructed. The OIDC block is still validated whenever it is
    present, because doing so has no side effect and step 1 of a cutover is supposed to rehearse
    it; the token file is not, because `load_token_registry` refuses under `RECALL_ENV=production`
    and that refusal must not fire for a file that is standing down.
    """
    e = env if env is not None else dict(os.environ)
    transport = transport or e.get("RECALL_TRANSPORT", "stdio")
    # Presence is decided from the ENV KEYS, before either mechanism is built, because building
    # one can raise for its own reasons: `load_token_registry` refuses under RECALL_ENV=production,
    # and that fired before the selector could say "we are not using the token file", which is
    # exactly the state a production cutover passes through.
    # `oidc_env_present` rather than a raw key read: it also refuses a PARTIAL block, and that
    # check must not depend on whether we go on to build a validator (SEC-001).
    has_oidc = oidc_env_present(e)
    has_static = bool(e.get(ENV_TOKENS_FILE, "").strip())

    mode = e.get("RECALL_AUTH_MODE", "").strip().lower()
    if mode and mode not in {"oidc", "static"}:
        raise AuthConfigError(
            f"RECALL_AUTH_MODE={mode!r} is not a valid mode; expected one of oidc, static"
        )
    if mode == "oidc" and not has_oidc:
        raise AuthConfigError(f"RECALL_AUTH_MODE=oidc but {ENV_ISSUER} is not set")
    if mode == "static" and not has_static:
        raise AuthConfigError(f"RECALL_AUTH_MODE=static but {ENV_TOKENS_FILE} is not set")

    if has_oidc and has_static and not mode:
        # The ambiguity guard, narrowed rather than removed. Refusing was right when nobody chose;
        # it also made the cutover atomic, because the intermediate state of a staged rollout is
        # precisely "both configured". Declaring precedence is a choice, so it is allowed; leaving
        # it undeclared still is not.
        raise AuthConfigError(
            f"{ENV_ISSUER} and {ENV_TOKENS_FILE} are both set, and this server will not "
            f"guess between them. They are two trust models: one where the IdP owns revocation, "
            f"expiry and rotation, and one where a static shared secret is valid until somebody "
            f"edits a file. Set RECALL_AUTH_MODE=oidc or =static to declare which is active "
            f"(that is the supported way to stage a cutover), or unset one of them."
        )

    use_oidc = has_oidc and mode != "static"

    # The OIDC block is VALIDATED whenever it is present, even when static is selected, and only
    # the static side is conditionally loaded. The asymmetry is the point (SEC-002):
    #
    # - Loading the token file has a side effect that must not happen when it is standing down:
    #   `load_token_registry` refuses under RECALL_ENV=production, which would abort the very
    #   cutover the selector exists to enable.
    # - Building the OIDC config has NO such effect. It performs no network IO by design, so
    #   skipping it buys nothing and costs the whole point of cutover step 1, which docs/AUTH.md
    #   describes as "add every OIDC variable, change nothing. Verify." Unparsed, a malformed
    #   subject binding or algorithm list would surface only at the step-2 flip, which is the
    #   moment with the least rollback slack.
    oidc = oidc_validator_from_env(e) if has_oidc else None
    validator = oidc if use_oidc else None
    registry = token_registry_from_env(e) if not use_oidc else None

    if has_oidc and has_static:
        # Precedence was declared, so the refusal no longer carries the warning. The log has to.
        inactive = ENV_TOKENS_FILE if use_oidc else ENV_ISSUER
        _log.warning(
            "both authentication mechanisms are configured; RECALL_AUTH_MODE=%s is active, so "
            "%s is set but NOT enforcing anything. Remove it once the cutover has settled.",
            "oidc" if use_oidc else "static",
            inactive,
        )

    configured = validator is not None or registry is not None
    if transport not in HTTP_TRANSPORTS:
        # Configured but inapplicable. Silence here would let an operator believe stdio is
        # access-controlled when the pipe itself is the only boundary. Two messages rather than
        # one generic one: an operator needs to know WHICH knob is inert, and "authentication is
        # unused" leaves them checking both.
        if registry is not None:
            _log.warning(
                "%s is set but transport is %r — stdio has no remote "
                "caller to authenticate, so the tokens are unused and the single tenant "
                "RECALL_TENANT=%r applies. Set RECALL_TRANSPORT=streamable-http to use them.",
                ENV_TOKENS_FILE,
                transport,
                e.get("RECALL_TENANT", "default"),
            )
        if validator is not None:
            _log.warning(
                "%s is set but transport is %r — stdio has no remote caller to authenticate, so "
                "the OIDC configuration is unused and the single tenant RECALL_TENANT=%r "
                "applies. Set RECALL_TRANSPORT=streamable-http to use it.",
                ENV_ISSUER,
                transport,
                e.get("RECALL_TENANT", "default"),
            )
        return None, None, None

    if not configured:
        raise AuthConfigError(
            f"transport {transport!r} opens a network listener, so authentication is required. "
            f"Set {ENV_ISSUER} (with {ENV_AUDIENCE} and {ENV_TENANTS}) to take identity from an "
            f"OIDC provider, or RECALL_AUTH_TOKENS_FILE to a JSON file of principals for "
            f"development (see docs/AUTH.md), or use RECALL_TRANSPORT=stdio for a private "
            f"single-client pipe."
        )

    # With an IdP there is exactly one right answer for the metadata issuer, and requiring an
    # operator to restate it is a chance to state it differently — at which point clients are
    # directed to a provider that did not sign the tokens this server accepts.
    # Defaulted from the OIDC block ONLY when OIDC is the selected mechanism. Defaulting whenever
    # a block was merely present (the first attempt at DEPLOY-004's rollback asymmetry) advertised
    # the IdP as this resource's authorization server while static bearer tokens were the thing
    # actually enforcing — sending clients to a provider whose tokens this server would refuse,
    # which is the exact misdirection the paragraph below exists to prevent. The rollback
    # asymmetry is real and is answered in docs/AUTH.md by keeping RECALL_AUTH_ISSUER_URL set
    # explicitly for the duration of a cutover; a wrong default is worse than a required value.
    issuer = e.get("RECALL_AUTH_ISSUER_URL") or (
        validator.config.issuer if validator is not None else ""
    )
    resource = e.get("RECALL_AUTH_RESOURCE_URL")
    if not issuer or not resource:
        raise AuthConfigError(
            "RECALL_AUTH_ISSUER_URL and RECALL_AUTH_RESOURCE_URL are required for an HTTP "
            "transport. They are published in this server's protected-resource metadata so a "
            "client knows where to get a token and which audience it is for; set both to this "
            "server's own public URL if you are provisioning tokens by hand. With "
            f"{ENV_ISSUER} set, the issuer defaults to the provider and only "
            "RECALL_AUTH_RESOURCE_URL is required."
        )
    settings = AuthSettings(
        issuer_url=AnyHttpUrl(issuer),
        resource_server_url=AnyHttpUrl(resource),
        # Left empty on purpose: a global required_scope would reject a principal provisioned for
        # exactly one capability (a forget-only retention job holds recall:forget and nothing
        # else). Scope is enforced per tool in `_require`, against what that tool actually does.
        required_scopes=[],
    )
    if validator is not None:
        allowed = validator.config.allowed_tenants
        if allowed is None:
            # `oidc_validator_from_env` refuses to build a validator without an allowlist, so
            # reaching this means someone added a second construction path. Raised rather than
            # asserted: `python -O` strips an assert, and this one stands between a token-borne
            # tenant and `StoreRegistry`. A guard that a flag can remove is not a guard.
            raise AuthConfigError(
                "an OIDC validator reached build_auth with no tenant allowlist; refusing to "
                "serve, because every tenant the IdP asserts would otherwise open a store"
            )
        return OidcTokenVerifier(validator, env=e), settings, ProvisionedTenants(allowed)

    if registry is None:  # pragma: no cover - `configured` above already excluded this
        raise AuthConfigError("no authentication mechanism resolved")
    return RecallTokenVerifier(registry, env=e), settings, registry


_RLS_BYPASS = "this database role bypasses row-level security (superuser or BYPASSRLS)"


def require_effective_rls(*, rls_effective: bool, multi_tenant: bool) -> str | None:
    """Refuse ineffective row-level security for authenticated multi-tenant serving."""
    if rls_effective:
        return None
    if multi_tenant:
        raise RuntimeError(
            f"{_RLS_BYPASS}, and this server is configured for authenticated multi-tenant "
            "serving, so tenant isolation would rest on query predicates alone. Connect as an "
            "unprivileged role that owns neither the managed tables nor the cluster."
        )
    return (
        f"{_RLS_BYPASS}, so tenant isolation rests on query predicates alone. Connect as an "
        "unprivileged role for defence in depth."
    )
