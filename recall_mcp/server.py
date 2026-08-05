from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import anyio.to_thread
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl

from recall.control_plane import ControlPlane
from recall.embeddings import embedding_profile_id
from recall.readiness import check_enterprise_readiness
from recall.observability import METRICS, configure_logging, get_logger
from recall.store import DEFAULT_TENANT, PgVectorStore, redacted_dsn
from recall_mcp.auth import (
    SCOPE_FORGET,
    SCOPE_READ,
    SCOPE_WRITE,
    AuthConfigError,
    TokenRegistry,
    authorize,
    token_registry_from_env,
)
from recall_mcp.limits import limiter_from_env
from recall_mcp.oidc import (
    ENV_AUDIENCE,
    ENV_ISSUER,
    ENV_TENANTS,
    IdentityProviderUnavailable,
    OidcValidator,
    TokenRejected,
    oidc_validator_from_env,
)
from recall_mcp.service import (
    forget_memory,
    index_memory,
    make_embedder,
    make_profile_embedder,
    memory_stats,
    search_memory,
)
from recall_mcp.stores import StoreRegistry

#: Which call budget each scope draws on. Keyed by scope rather than by tool name so a new tool
#: is metered the moment it declares a scope — there is no separate table to remember to update,
#: and an unmetered tool would be one that also skipped authorisation.
_SCOPE_BUDGETS = {SCOPE_READ: "read", SCOPE_WRITE: "write", SCOPE_FORGET: "forget"}

DEFAULT_DSN = os.environ.get(
    "RECALL_SERVING_DSN",
    os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"),
)
#: Transport to serve. `stdio` is a private pipe between one client and this process — there is no
#: network listener and no remote caller to authenticate, so auth is not required there. The HTTP
#: transports open a socket, and `build_auth` refuses to start them unless an authentication
#: mechanism is configured — a static token file, or an OIDC provider.
Transport = Literal["stdio", "sse", "streamable-http"]
TRANSPORTS: tuple[Transport, ...] = ("stdio", "sse", "streamable-http")
HTTP_TRANSPORTS = frozenset({"streamable-http", "sse"})


def _read_transport() -> Transport:
    """`RECALL_TRANSPORT`, validated against the three the SDK accepts.

    Unvalidated, a typo reached `mcp.run(transport=...)` as an arbitrary string. `stdo` does not
    fall back to stdio and does not name a listener — it produces whatever the SDK does with an
    unknown transport, at the end of startup, having already opened a store and read the token
    file. Failing here names the bad value and the valid set instead.
    """
    value = os.environ.get("RECALL_TRANSPORT", "stdio")
    if value not in TRANSPORTS:
        raise ValueError(
            f"RECALL_TRANSPORT={value!r} is not a valid transport; "
            f"expected one of {', '.join(TRANSPORTS)}"
        )
    return value  # narrowed to Transport by the membership test above


def _read_int_env(name: str, default: int, *, min_value: int, max_value: int | None = None) -> int:
    """An integer knob from the environment, validated — the numeric analogue of `_read_transport`.

    A bare ``int()`` crashes with ``invalid literal for int()`` that names no variable, and never
    bounds-checks, so an out-of-range value is accepted at import and only surfaces later (a
    negative RECALL_STATEMENT_TIMEOUT_MS reaches ``SET statement_timeout``; 0 silently disables the
    cap). Fail here, naming the variable and the expected range.
    """
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if value < min_value or (max_value is not None and value > max_value):
        bound = f"{min_value}..{max_value}" if max_value is not None else f">= {min_value}"
        raise ValueError(f"{name}={value} is out of range; expected {bound}")
    return value


TRANSPORT: Transport = _read_transport()
#: Bind address for the HTTP transports. Exposed as RECALL_* because the SDK's own FASTMCP_HOST /
#: FASTMCP_PORT are read when the FastMCP object is constructed at import time, which makes them
#: unreliable to set from a wrapper — and every other knob in this server is RECALL_*.
#: Default is loopback, NOT 0.0.0.0: binding every interface should be a decision someone makes,
#: not something they inherit.
HTTP_HOST = os.environ.get("RECALL_HOST", "127.0.0.1")
HTTP_PORT = _read_int_env("RECALL_PORT", 8000, min_value=1, max_value=65535)
EMBEDDER_NAME = os.environ.get("RECALL_EMBEDDER", "fastembed")
#: Connections the server keeps open. This bounds concurrent in-flight tool calls at the database,
#: which is where the real limit is — more worker threads than connections just queue on the pool.
POOL_SIZE = _read_int_env("RECALL_POOL_SIZE", 8, min_value=1)
#: Tenant this server instance serves. One store is bound to one tenant, so a
#: multi-tenant deployment runs a server (or a store) per tenant rather than switching
#: tenants on a shared connection — see PgVectorStore._prepare.
TENANT = os.environ.get("RECALL_TENANT", DEFAULT_TENANT)
#: Server-side cap on any single statement. A runaway query otherwise holds its connection until
#: the process dies, and a few of those exhaust the pool while the server still looks healthy.
# min_value=1: 0 is a valid Postgres statement_timeout meaning "no limit", but here it would
# disable the very cap this exists to enforce — a fail-open we refuse rather than accept silently.
STATEMENT_TIMEOUT_MS = _read_int_env("RECALL_STATEMENT_TIMEOUT_MS", 15000, min_value=1)

_T = TypeVar("_T")

_log = get_logger("mcp")


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

    def __init__(self, registry: TokenRegistry) -> None:
        self._registry = registry

    async def verify_token(self, token: str) -> AccessToken | None:
        principal = self._registry.verify(token)
        if principal is None:
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

    def __init__(self, validator: OidcValidator) -> None:
        self._validator = validator

    async def verify_token(self, token: str) -> AccessToken | None:
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
            # Distinct message, deliberately: this is an outage on our side of the trust
            # relationship, and reading it as a wave of forgeries would send an operator hunting
            # an attacker while the IdP stays down.
            _log.warning(
                "refusing tokens: the identity provider could not be consulted (reason=%s)",
                exc.reason,
            )
            return None
        except TokenRejected as exc:
            # `exc.reason` is a stable slug and never contains token material; the exception's
            # detail is deliberately not logged, and neither is the token.
            _log.warning("rejected a bearer token (reason=%s)", exc.reason)
            return None
        except Exception:
            # Defence in depth (NUM-001). The validator's contract is that every ambiguity
            # resolves to a TokenRejected, but this is the boundary where a breach of it turns a
            # 401 into a 500: the SDK does not wrap `verify_token`. A failure to authenticate
            # must fail CLOSED as a refusal, never as a stack trace.
            _log.warning("rejected a bearer token (reason=validator_error)", exc_info=True)
            return None
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
    transport: str = TRANSPORT, env: dict[str, str] | None = None
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

    Both together raises: they are two trust models, and whichever won silently, the other would
    sit in the configuration looking effective.
    """
    e = env if env is not None else dict(os.environ)
    # Both are read BEFORE any transport branch, so a conflicting pair is refused on stdio too.
    # Deferring the conflict to whenever someone first starts an HTTP listener would let a
    # misconfiguration sit in a chart looking fine until the day it decides which trust model
    # applies.
    validator = oidc_validator_from_env(e)
    registry = token_registry_from_env(e)

    if validator is not None and registry is not None:
        raise AuthConfigError(
            f"{ENV_ISSUER} and RECALL_AUTH_TOKENS_FILE are both set, and this server will not "
            f"choose between them. They are two trust models: one where the IdP owns revocation, "
            f"expiry and rotation, and one where a static shared secret is valid until somebody "
            f"edits a file. Whichever won silently, the other would sit in the configuration "
            f"looking effective. Unset one."
        )

    configured = validator is not None or registry is not None
    if transport not in HTTP_TRANSPORTS:
        # Configured but inapplicable. Silence here would let an operator believe stdio is
        # access-controlled when the pipe itself is the only boundary. Two messages rather than
        # one generic one: an operator needs to know WHICH knob is inert, and "authentication is
        # unused" leaves them checking both.
        if registry is not None:
            _log.warning(
                "RECALL_AUTH_TOKENS_FILE is set but transport is %r — stdio has no remote "
                "caller to authenticate, so the tokens are unused and the single tenant "
                "RECALL_TENANT=%r applies. Set RECALL_TRANSPORT=streamable-http to use them.",
                transport,
                TENANT,
            )
        if validator is not None:
            _log.warning(
                "%s is set but transport is %r — stdio has no remote caller to authenticate, so "
                "the OIDC configuration is unused and the single tenant RECALL_TENANT=%r "
                "applies. Set RECALL_TRANSPORT=streamable-http to use it.",
                ENV_ISSUER,
                transport,
                TENANT,
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
        return OidcTokenVerifier(validator), settings, ProvisionedTenants(allowed)

    if registry is None:  # pragma: no cover - `configured` above already excluded this
        raise AuthConfigError("no authentication mechanism resolved")
    return RecallTokenVerifier(registry), settings, registry


async def _to_thread(fn: Callable[[], _T]) -> _T:
    """Run a blocking tool body off the event loop.

    FastMCP awaits an async tool and calls a sync one INLINE (`func_metadata.py`:
    ``return await fn(...)`` vs ``return fn(...)``) — there is no thread offload. A sync tool that
    embeds a query, makes two database round trips and maybe runs a cross-encoder therefore blocks
    the whole loop for its duration: one request at a time, with no response to anything else —
    not even a ping — until it finishes. `recall_index` blocks it for an entire corpus index.

    `anyio.to_thread` rather than `asyncio.to_thread` because FastMCP runs on AnyIO: this inherits
    its worker-thread limiter and cancellation scope instead of starting a second, unmanaged pool
    beside it.
    """
    return await anyio.to_thread.run_sync(fn)


def _make_lifespan(
    token_registry: TenantProvisioning | None,
) -> Callable[[FastMCP], AbstractAsyncContextManager[dict]]:
    """Build the lifespan.

    Two shapes, decided by whether auth is on:

    - **Unauthenticated (stdio).** One store bound to `RECALL_TENANT`, exactly as before. There is
      one caller on the other end of the pipe and it gets one namespace.
    - **Authenticated (HTTP).** A `StoreRegistry` over the tenants the deployment provisions —
      the token file's principals, or `RECALL_OIDC_TENANTS`. This function does not know which.
      Nothing is opened until a request for that tenant arrives, so a server configured for ten
      tenants that only ever serves one holds one pool, not ten.
    """

    @asynccontextmanager
    async def _lifespan(_server: FastMCP) -> AsyncIterator[dict]:
        from recall.store import require_secure_dsn

        # FAIL CLOSED, unlike the CLI's warning: a server is unattended, so a stderr note about
        # published default credentials pointed at a remote database lands in a journal nobody
        # reads while the process comes up looking healthy. RECALL_ALLOW_INSECURE_DSN=1 opts out.
        require_secure_dsn(DEFAULT_DSN)
        store: PgVectorStore | None = None
        registry: StoreRegistry | None = None
        try:
            embedder = make_embedder(EMBEDDER_NAME)
            generation_mode = os.environ.get("RECALL_ENV", "development").lower() == "production"
            # Inspect migration state before PgVectorStore prepares a pgvector codec. On a fresh
            # database the extension deliberately does not exist yet; reporting "migrations
            # pending" is more useful than leaking the driver's missing-type error. This path is
            # SELECT-only and uses the serving credential.
            from recall.schema import SchemaTooOld, schema_status

            schema = schema_status(DEFAULT_DSN, dim=embedder.dim)
            if not schema.compatible:
                pending = [m.version for m in schema.pending]
                raise SchemaTooOld(
                    f"database migrations pending: {pending}; run `recall schema apply`"
                )
            enterprise = os.environ.get("RECALL_ENTERPRISE_CONTROL_PLANE", "").lower() in {
                "1", "true", "yes", "on"
            }
            if enterprise and token_registry is None:
                raise RuntimeError("enterprise control plane requires authenticated tenant routing")
            if token_registry is None:
                # Pooled + timed out: a server shares this store across concurrent tool calls,
                # and one connection would serialise them however many threads are available.
                if generation_mode:
                    from recall.generation_store import GenerationStore

                    store = GenerationStore(
                        DEFAULT_DSN,
                        dim=embedder.dim,
                        tenant=TENANT,
                        pool_size=POOL_SIZE,
                        statement_timeout_ms=STATEMENT_TIMEOUT_MS,
                    )
                else:
                    store = PgVectorStore(
                        DEFAULT_DSN,
                        dim=embedder.dim,
                        tenant=TENANT,
                        pool_size=POOL_SIZE,
                        statement_timeout_ms=STATEMENT_TIMEOUT_MS,
                    )
            else:
                registry = StoreRegistry(
                    dsn=DEFAULT_DSN,
                    dim=embedder.dim,
                    allowed_tenants=token_registry.tenants,
                    pool_size=POOL_SIZE,
                    statement_timeout_ms=STATEMENT_TIMEOUT_MS,
                    generation_mode=generation_mode and not enterprise,
                    control_plane=ControlPlane(DEFAULT_DSN) if enterprise else None,
                    embedding_profile=embedding_profile_id(embedder),
                )
        except Exception:
            _log.error(
                "startup failed (dsn=%s, embedder=%r)",
                redacted_dsn(DEFAULT_DSN),
                EMBEDDER_NAME,
                exc_info=True,
            )
            raise

        try:
            if store is not None:
                store.check_schema()
                probe = store
            else:
                assert registry is not None
                # Open ONE tenant eagerly. Schema compatibility, a missing pgvector extension
                # and a bad DSN fail identically for every tenant, and finding that out on the
                # first client request — per tenant, at request latency — turns a startup error
                # into an intermittent runtime one.
                probe = registry.get(min(registry.allowed_tenants))
                _log.info(
                    "auth enabled: %d tenant(s), up to %d pooled connections at full spread",
                    len(registry.allowed_tenants),
                    registry.max_connections(),
                )
        except Exception:
            if store is not None:
                store.close()
            if registry is not None:
                registry.close()
            _log.error("schema check failed", exc_info=True)
            raise

        if not probe.check_rls_effective():
            _log.warning(
                "this database role bypasses row-level security (superuser or BYPASSRLS), so "
                "tenant isolation rests on query predicates alone. Connect as an unprivileged "
                "role for defence in depth."
            )
        if enterprise:
            readiness = check_enterprise_readiness(
                probe,
                embedder,
                control_plane=registry.control_plane if registry is not None else None,
            )
            if not readiness.ready:
                raise RuntimeError("enterprise readiness failed: " + "; ".join(readiness.failures))
            if readiness.degraded:
                _log.warning("enterprise readiness degraded: %s", "; ".join(readiness.warnings))
        # Built only for the authenticated shape: buckets are keyed by tenant, and stdio has no
        # principal to attribute a call to. Reported at startup so the effective budget is visible
        # in the journal rather than inferred from which requests started failing.
        limiter = limiter_from_env() if registry is not None else None
        if limiter is not None:
            _log.info(
                "per-tenant budgets: %s",
                ", ".join(f"{k}={v.capacity:,.0f}" for k, v in sorted(limiter.limits().items()))
                or "(all disabled)",
            )

        try:
            yield {
                "store": store,
                "stores": registry,
                "embedder": embedder,
                "limiter": limiter,
                "shadow_embedders": {},
                "shadow_embedder_lock": threading.Lock(),
            }
        finally:
            if store is not None:
                store.close()
            if registry is not None:
                registry.close()

    return _lifespan


def build_server() -> FastMCP:
    """Construct the recall_mcp FastMCP server with its four tools registered."""
    verifier, auth_settings, token_registry = build_auth()
    mcp = FastMCP(
        "recall_mcp",
        lifespan=_make_lifespan(token_registry),
        token_verifier=verifier,
        auth=auth_settings,
        host=HTTP_HOST,
        port=HTTP_PORT,
    )

    def _current_tenant(state: dict) -> str | None:
        """The authenticated caller's tenant, or None when running unauthenticated (stdio).

        Read from the access token rather than threaded down from `_require`, so it cannot go
        stale or be passed the wrong value by a future caller.
        """
        if state.get("stores") is None:
            return None
        token = get_access_token()
        if token is None:  # pragma: no cover - `_require` has already rejected this
            return None
        return (token.claims or {}).get("tenant")

    def _state() -> dict:
        ctx = mcp.get_context().request_context.lifespan_context
        if not isinstance(ctx, dict) or "embedder" not in ctx:
            raise RuntimeError(
                "recall_mcp lifespan context is not initialized — tools must be invoked within "
                "the running server (store/embedder are opened in the lifespan)."
            )
        return ctx

    def _require(scope: str) -> PgVectorStore:
        """Authorise this call and return the store for the caller's OWN tenant.

        Every tool body goes through here. The store it hands back is the only one that tool can
        reach, so a missing scope check cannot leak data across tenants — at worst it lets a
        principal do the wrong thing inside its own namespace.

        This is also where the per-tenant call budget is debited, for the same reason: one choke
        point that a new tool cannot forget to call, because it cannot get a store without it.
        """
        state = _state()
        registry: StoreRegistry | None = state.get("stores")
        if registry is None:
            # Unauthenticated stdio: one caller, one tenant, a private pipe. There is no principal
            # to charge and no one to protect the local user from but themselves, so the budget
            # does not apply — matching how auth itself is scoped.
            store: PgVectorStore = state["store"]
            return store

        token = get_access_token()
        if token is None:
            # The SDK's bearer middleware rejects unauthenticated HTTP requests before a tool
            # runs, so this is unreachable through the normal path. It stays because the
            # alternative — falling through to some default store — would turn any future gap in
            # that middleware into a silent full-corpus read.
            raise PermissionError("this server requires authentication")
        try:
            tenant = authorize(token.scopes, token.claims, scope)
        except PermissionError:
            principal = (token.claims or {}).get("principal", token.client_id)
            _log.warning("principal %r denied for scope %s", principal, scope)
            raise
        # After authorisation, so an unauthorised caller cannot burn the tenant's budget by
        # hammering a scope it does not hold.
        limiter = state.get("limiter")
        if limiter is not None:
            limiter.check(tenant, _SCOPE_BUDGETS[scope])
        return registry.get(tenant)

    @mcp.tool(
        name="recall_search",
        annotations=ToolAnnotations(
            title="Search agent memory",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def recall_search(query: str, source: str | None = None, k: int = 5) -> str:
        """Search the agent's OWN memory before acting, and get actionable guidance.

        Call this before proposing an idea, forming a hypothesis, or repeating past work:
        if a closed decision or falsified hypothesis surfaces, do not re-litigate it. Every hit
        carries a trust verdict (only `ok` hits should be relied on), a calibrated confidence,
        provenance (indexed_at) and validity (superseded_by / valid_until). When `abstained` is
        true, NO valid hit survived — say you don't know instead of answering from the hits.
        `advice` states what to do.

        Args:
            query: what to recall (natural language).
            source: optional source filter (only search one file/source).
            k: max hits to return (default 5).

        Returns:
            JSON with abstention, calibration status and ID, tenant/generation/pipeline/corpus/
            query-set identities, freshness, advice, and hits carrying provenance and verdicts.
        """
        state = _state()
        store = _require(SCOPE_READ)
        with METRICS.timer("recall_tool_latency_ms", tool="search"):
            return await _to_thread(
                lambda: search_memory(
                    store,
                    state["embedder"],
                    query,
                    source=source,
                    k=k,
                ).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_index",
        annotations=ToolAnnotations(
            title="Add to agent memory",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def recall_index(path: str) -> str:
        """Index a markdown file or folder into the agent's memory so it can be recalled later.

        Re-indexing a file REPLACES its chunks completely (safe to re-run after edits; a shrunk
        file leaves no stale chunks behind).
        `path` is confined to RECALL_INDEX_ROOT (default: the server's working directory), and the
        request is refused before anything is embedded if it exceeds RECALL_INDEX_MAX_FILES or
        RECALL_INDEX_MAX_BYTES (see `recall_mcp/service.py`).

        Args:
            path: a file or directory path (``**/*.md`` is indexed for directories).

        Returns:
            JSON of {files, chunks, message}.
        """
        state = _state()
        store = _require(SCOPE_WRITE)
        limiter = state.get("limiter")
        tenant = _current_tenant(state)
        registry: StoreRegistry | None = state.get("stores")
        shadow_store = (
            registry.get_shadow(tenant)
            if registry is not None and tenant is not None else None
        )
        shadow_embedder = None
        if shadow_store is not None:
            assert registry is not None and registry.control_plane is not None and tenant is not None
            route = registry.control_plane.route(tenant)
            if route is None or route.shadow is None:
                raise RuntimeError("shadow store was acquired without shadow generation metadata")
            profile_id = route.shadow.embedding_profile
            lock = state["shadow_embedder_lock"]
            with lock:
                cache = state["shadow_embedders"]
                shadow_embedder = cache.get(profile_id)
                if shadow_embedder is None:
                    shadow_embedder = make_profile_embedder(profile_id, shadow=True)
                    if shadow_embedder.dim != route.shadow.dimension:
                        raise RuntimeError("shadow embedder dimension does not match generation")
                    cache[profile_id] = shadow_embedder

        def _debit(_files: int, total_bytes: int) -> None:
            """Charge the tenant for what is about to be embedded, before it is embedded.

            The call budget alone cannot bound spend: one request may carry 20 MB and the next
            200 bytes, so counting requests prices them identically. This meters the thing that
            actually costs money, and it runs pre-flight — a refusal here has spent nothing.
            """
            if limiter is not None and tenant is not None:
                limiter.check(tenant, "index_bytes", float(total_bytes))

        with METRICS.timer("recall_tool_latency_ms", tool="index"):
            return await _to_thread(
                lambda: index_memory(
                    store, state["embedder"], path, on_measured=_debit,
                    shadow_store=shadow_store,
                    shadow_embedder=shadow_embedder,
                    control_plane=registry.control_plane if registry is not None else None,
                ).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_forget",
        annotations=ToolAnnotations(
            title="Forget agent memory",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def recall_forget(sources: list[str]) -> str:
        """Permanently delete indexed memory for the given source(s). IRREVERSIBLE.

        This is the right-to-erasure path: use it to make the agent forget a memory that should
        no longer be recalled (e.g. it indexed something it should not have retained). Deletion
        is scoped to this server's own tenant and cannot reach another tenant's memory. A source
        that does not exist is reported in `sources_not_found` rather than silently counted as
        "removed" — check that list before assuming a name was actually forgotten.

        Args:
            sources: one or more source values to forget, exactly as they appear in
                `recall_search` hits (the `source` field).

        Returns:
            JSON of {chunks_removed, sources_removed, sources_not_found, message}.
        """
        state = _state()
        store = _require(SCOPE_FORGET)
        registry: StoreRegistry | None = state.get("stores")
        tenant = _current_tenant(state)
        shadow = registry.get_shadow(tenant) if registry is not None and tenant is not None else None
        with METRICS.timer("recall_tool_latency_ms", tool="forget"):
            return await _to_thread(
                lambda: forget_memory(store, sources, shadow).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_stats",
        annotations=ToolAnnotations(
            title="Memory freshness & size",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def recall_stats() -> str:
        """Report how much memory exists and whether it is stale (freshness check).

        `stale` is True when the newest indexed content is older than 2 days.

        Returns:
            JSON of {chunks, newest_indexed_at, stale}.
        """
        store = _require(SCOPE_READ)
        return await _to_thread(lambda: memory_stats(store).model_dump_json(indent=2))

    return mcp


mcp = build_server()


def main() -> None:
    # stderr only, and propagate=False — stdout carries JSON-RPC, so a stray log line there
    # would corrupt the protocol.
    configure_logging()
    if TRANSPORT in HTTP_TRANSPORTS:
        # Tenancy is per-token here, so logging a single RECALL_TENANT would be actively
        # misleading about what this process serves.
        _log.info(
            "starting %s server on %s:%s (authenticated)",
            TRANSPORT,
            mcp.settings.host,
            mcp.settings.port,
        )
    else:
        _log.info("starting stdio server", extra={"tenant": TENANT, "embedder": EMBEDDER_NAME})
    mcp.run(transport=TRANSPORT)


if __name__ == "__main__":
    main()
