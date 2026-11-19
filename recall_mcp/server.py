"""The RE-call MCP server: `build_server`, tool registration, idempotency helpers and `main`.

Authentication and tenancy live in `recall_mcp.server_auth`; their public names are re-imported
here so `recall_mcp.server.<name>` keeps resolving for existing callers. The same holds for the
modules split out of this one: the lifespan (`server_lifespan`), each tool family
(`server_search`, `server_facts`, `server_reasoning`, `server_ingest`, `server_calibration`,
`server_memory_admin`, `server_assistant`) and what they share (`server_common`). A test that
patches a name those modules look up must patch it there: `tests/server_modules.py`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlsplit

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from recall.answer_provider import resolve_answer_provider
from recall.errors import IdempotencyConflict, RecallError
from recall.observability import configure_logging
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.backends import is_lite_dsn
from recall_mcp.assistant import (
    ASSISTANT_TOOLS,
    assistant_memos_from_env,
)
from recall.store import PgVectorStore
from recall_mcp.access_log import caller_identity, record_access
from recall_mcp.auth import (
    CLEARANCE_LEVELS,
    SCOPE_ADMIN,
    SCOPE_FORGET,
    SCOPE_FACT_WRITE,
    SCOPE_READ,
    SCOPE_WRITE,
    AuthConfigError,
    authorize,
)
from recall_mcp.limits import (
    IdempotencyResultMissing,
    IdempotencyReplay,
)
from recall.ops.health import HealthController, route_response
from recall_mcp.settings import (
    Settings,
    bootstrap_settings,
)
from recall_mcp.stores import StoreRegistry
from recall_mcp.tool_surface import LITE_UNSERVED_TOOLS, FilteredToolRegistrar, resolve_tool_surface
from recall_mcp.server_auth import (  # noqa: F401  # re-exported: public names of this module
    Transport,
    TRANSPORTS,
    HTTP_TRANSPORTS,
    TenantProvisioning,
    RecallTokenVerifier,
    OidcTokenVerifier,
    ProvisionedTenants,
    build_auth,
    require_effective_rls,
)
from recall_mcp.threads import _to_thread  # noqa: F401  # re-exported: tests call srv._to_thread
from recall_mcp.server_common import (  # re-exported: moved out of this module
    _mutation_fingerprint as _mutation_fingerprint,
    _principal_bound_fingerprint as _principal_bound_fingerprint,
    _record_mutation_result as _record_mutation_result,
    _release_mutation_reservation as _release_mutation_reservation,
    _META_REQUIRED_SCOPE as _META_REQUIRED_SCOPE,
    _T as _T,
    _log as _log,
    _check_limiter_async as _check_limiter_async,
    _Require as _Require,
    _RequireMutation as _RequireMutation,
    _RecordMutationResult as _RecordMutationResult,
    _ToolDeps as _ToolDeps,
    _parse_as_of as _parse_as_of,
    _trust_policy_for as _trust_policy_for,
    _runtime_settings_for as _runtime_settings_for,
    _runtime_env_for as _runtime_env_for,
    _tool_error_for_trust_refusal as _tool_error_for_trust_refusal,
    _token_principal as _token_principal,
)
from recall_mcp.server_lifespan import (  # re-exported: moved out of this module
    table_override_refusal as table_override_refusal,
    benchmark_generation_setting as benchmark_generation_setting,
    RESERVED_WORKER_THREADS as RESERVED_WORKER_THREADS,
    worker_thread_budget as worker_thread_budget,
    apply_worker_thread_budget as apply_worker_thread_budget,
    bounded_tenant_probe_ids as bounded_tenant_probe_ids,
    _StartupConfig as _StartupConfig,
    _ServingBackend as _ServingBackend,
    _make_lifespan as _make_lifespan,
)
from recall_mcp.server_search import (  # re-exported: moved out of this module
    _serving_json as _serving_json,
    _RetrievalResultT as _RetrievalResultT,
    TRANSLATION_THREAD_LIMITER as TRANSLATION_THREAD_LIMITER,
    _translation_to_thread as _translation_to_thread,
    _federation_config_for as _federation_config_for,
    _tool_error_for_overload as _tool_error_for_overload,
    _record_federation as _record_federation,
    _retrieval_plan_for as _retrieval_plan_for,
    _register_search_tools as _register_search_tools,
)
from recall_mcp.server_facts import (  # re-exported: moved out of this module
    _register_fact_tools as _register_fact_tools,
)
from recall_mcp.server_reasoning import (  # re-exported: moved out of this module
    PROOF_THREAD_LIMITER as PROOF_THREAD_LIMITER,
    _proof_to_thread as _proof_to_thread,
    _proof_payload as _proof_payload,
    _report_snapshot as _report_snapshot,
    _paths_named as _paths_named,
    _register_reasoning_tools as _register_reasoning_tools,
)
from recall_mcp.server_ingest import (  # re-exported: moved out of this module
    _MutationPreflightFailure as _MutationPreflightFailure,
    _ingest_mutation_fingerprint as _ingest_mutation_fingerprint,
    _check_limiter_from_worker as _check_limiter_from_worker,
    _IngestNamespaceOptions as _IngestNamespaceOptions,
    _check_source_namespace as _check_source_namespace,
    _compat_route as _compat_route,
    ingest_into_serving_store as ingest_into_serving_store,
    _register_ingest_tools as _register_ingest_tools,
)
from recall_mcp.server_calibration import (  # re-exported: moved out of this module
    _register_calibration_tools as _register_calibration_tools,
)
from recall_mcp.server_memory_admin import (  # re-exported: moved out of this module
    _register_memory_admin_tools as _register_memory_admin_tools,
)
from recall_mcp.server_assistant import (  # re-exported: moved out of this module
    _HeldTools as _HeldTools,
    _caller_may_read as _caller_may_read,
    _register_assistant_tools as _register_assistant_tools,
    _register_assistant_views as _register_assistant_views,
)


class IdempotencyKeyRequired(ValueError, ToolError, RecallError):
    """A retryable mutation reached an idempotency-enforcing limiter without a key.

    `ValueError` first keeps the historical builtin for existing callers; the `ToolError` base is
    what lets MCP 2.1 deliver this actionable message instead of redacting it to "Error executing
    tool", which named neither the missing argument nor the fix.
    """


class IdempotencyReconciliation(RuntimeError, ToolError, RecallError):
    """A reserved mutation has no recoverable response and must not be executed again."""

    def __init__(
        self,
        idempotency_key: str,
        *,
        operation: str | None = None,
        request_fingerprint: str | None = None,
    ) -> None:
        self.result = json.dumps(
            {
                "status": "reconciliation_required",
                "idempotency_key": idempotency_key,
                "operation": operation,
                "request_fingerprint": request_fingerprint,
                "message": (
                    "the mutation may have committed, but neither durable nor cached result "
                    "is available; reconcile the operation before retrying"
                ),
            },
            sort_keys=True,
        )
        super().__init__(self.result)


async def _durable_replay(
    store: PgVectorStore,
    idempotency_key: str,
    operation: str | None,
    fingerprint: str | None,
) -> str | None:
    """Read a receipt without running synchronous PostgreSQL work on the event loop."""
    get_receipt = getattr(store, "get_operation_receipt", None)
    if not callable(get_receipt) or operation is None or fingerprint is None:
        return None
    try:
        return await _to_thread(
            lambda: get_receipt(
                idempotency_key,
                operation=operation,
                request_fingerprint=fingerprint,
            )
        )
    except IdempotencyConflict as conflict:
        raise ToolError(
            json.dumps(
                {"error": "idempotency_conflict", "message": str(conflict)},
                sort_keys=True,
            )
        ) from conflict


#: Which call budget each scope draws on. Keyed by scope rather than by tool name so a new tool
#: is metered the moment it declares a scope — there is no separate table to remember to update,
#: and an unmetered tool would be one that also skipped authorisation.
_SCOPE_BUDGETS = {
    SCOPE_READ: "read",
    SCOPE_WRITE: "write",
    SCOPE_FACT_WRITE: "write",
    SCOPE_FORGET: "forget",
    SCOPE_ADMIN: "admin",
}


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _transport_security_settings(
    resource_url: str, extra_hosts: str = ""
) -> TransportSecuritySettings:
    """The Host and Origin allowlist for the HTTP transports.

    The SDK compares both headers exactly (a `host:*` entry is its only pattern), so the resource
    URL is stripped and lowercased, and its host is allowed with and without the scheme's default
    port: a client that reaches `https://recall.example.com` sends `Host: recall.example.com`
    whether or not the URL spelled `:443`. `extra_hosts` (RECALL_ALLOWED_HOSTS) names further
    `host[:port]` values, for a service reached under an internal name or a proxy that rewrites
    Host; each is allowed as given, lowercased, with the resource URL's scheme (DEF-46).
    """
    parsed = urlsplit(resource_url.strip())
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        raise AuthConfigError(
            "RECALL_AUTH_RESOURCE_URL must be an absolute URL so HTTP transport security can "
            "validate Host and Origin headers"
        )
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    if ":" in host:  # an IPv6 literal goes back in brackets for a Host header
        host = f"[{host}]"
    default_port = _DEFAULT_PORTS.get(scheme)
    if parsed.port is None or parsed.port == default_port:
        netlocs = [host] + ([f"{host}:{default_port}"] if default_port else [])
    else:
        netlocs = [f"{host}:{parsed.port}"]
    for raw in extra_hosts.split(","):
        extra = raw.strip().lower()
        if not extra:
            continue
        if "/" in extra or "@" in extra or extra.startswith(":") or extra.endswith(":"):
            raise AuthConfigError(
                f"RECALL_ALLOWED_HOSTS entry {raw.strip()!r} is not a host[:port]; give names "
                f"such as recall.internal:8000, comma separated"
            )
        if extra not in netlocs:
            netlocs.append(extra)
    return TransportSecuritySettings(
        allowed_hosts=netlocs, allowed_origins=[f"{scheme}://{n}" for n in netlocs]
    )


def _answer_backend_configured(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the configured reasoning answer backend can actually be resolved."""
    try:
        return resolve_answer_provider(env) is not None
    except Exception:  # BROAD-CATCH: fail-closed
        return False


#: The tools the assistant surface serves in their compact form (`recall_mcp.assistant`).
_ASSISTANT_VIEWS: frozenset[str] = frozenset(
    {"recall_search", "recall_evidence", "recall_current_facts", "recall_apply_fact"}
)
#: The views, and `recall_withdraw` and `recall_forget`, which the assistant surface serves with
#: memo ids too, erasing a memo's file with its rows.
_ASSISTANT_HELD: frozenset[str] = _ASSISTANT_VIEWS | {"recall_withdraw", "recall_forget"}


def build_server(settings: Settings | None = None) -> MCPServer:
    """Construct the recall_mcp MCP server with its tools registered."""
    runtime_settings = settings if settings is not None else bootstrap_settings()
    if not runtime_settings.trust_policy.strict:
        _log.error(
            "RECALL_TRUST_MODE=development: the trust gate is RELAXED for this server. "
            "Uncalibrated and unbound corpora will be served instead of refused. Local work only."
        )
    runtime_env = dict(runtime_settings.env)
    # Before anything is built: a half-configured assistant surface refuses to start.
    assistant = assistant_memos_from_env(runtime_env, lite=is_lite_dsn(runtime_settings.serving_dsn))
    verifier, auth_settings, token_registry = build_auth(
        runtime_settings.transport, env=runtime_env
    )
    health = HealthController()
    mcp = MCPServer(
        "recall_mcp",
        lifespan=_make_lifespan(
            token_registry,
            health,
            runtime_settings,
        ),
        token_verifier=verifier,
        auth=auth_settings,
    )

    @mcp.custom_route("/livez", methods=["GET"], name="livez", include_in_schema=False)
    async def livez(_request: Request) -> JSONResponse:
        status, payload = await _to_thread(lambda: route_response(health, "livez"))
        return JSONResponse(payload, status_code=status)

    @mcp.custom_route("/readyz", methods=["GET"], name="readyz", include_in_schema=False)
    async def readyz(_request: Request) -> JSONResponse:
        status, payload = await _to_thread(lambda: route_response(health, "readyz"))
        return JSONResponse(payload, status_code=status)

    @mcp.custom_route("/startupz", methods=["GET"], name="startupz", include_in_schema=False)
    async def startupz(_request: Request) -> JSONResponse:
        status, payload = await _to_thread(lambda: route_response(health, "startupz"))
        return JSONResponse(payload, status_code=status)

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

    def _access_context(
        state: dict, store: PgVectorStore, *, purpose: str = "retrieval"
    ) -> AccessContext | None:
        policy = state.get("source_security_policy")
        if not isinstance(policy, SourceSecurityPolicy):
            return None
        token = get_access_token()
        claims = token.claims if token is not None else {}
        principal = str(
            (claims or {}).get("principal")
            or (token.client_id if token is not None else "local")
        )
        tenant = str((claims or {}).get("tenant") or getattr(store, "tenant", ""))
        clearance = str((claims or {}).get("clearance", "internal"))
        if clearance not in CLEARANCE_LEVELS:
            raise PermissionError("authenticated principal carries an invalid clearance")
        # Only a real True grants egress: a string such as "false" is truthy under bool().
        egress_allowed = (claims or {}).get("egress_allowed", False) is True
        return AccessContext(
            principal=principal,
            tenant=tenant,
            purpose=purpose,
            clearance=clearance,  # type: ignore[arg-type]
            egress_allowed=egress_allowed,
        )

    def _state(ctx: Context[dict, object]) -> dict:
        state = ctx.request_context.lifespan_context
        if not isinstance(state, dict) or "embedder" not in state:
            raise RuntimeError(
                "recall_mcp lifespan context is not initialized — tools must be invoked within "
                "the running server (store/embedder are opened in the lifespan)."
            )
        return state

    async def _require(
        scope: str,
        ctx: Context[dict, object],
        requested_tenant: str | None = None,
        idempotency_key: str | None = None,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> PgVectorStore:
        """Authorise this call and return the store for the caller's OWN tenant.

        Every tool body goes through here. The store it hands back is the caller's own tenant's.
        The one other route to another tenant's store is federated retrieval, which reaches the
        registry through `_retrieval_plan_for` and is gated there by `recall:federate`, not here;
        any new use of `state["stores"]` must repeat that check.

        This is also where the per-tenant call budget is debited, for the same reason: one choke
        point that a new tool cannot forget to call, because it cannot get a store without it.

        On HTTP it also writes the access line (`recall_mcp.access_log`): `denied` on a scope or
        tenant refusal, `allowed` once both pass and before the budget and idempotency checks,
        which write none.
        """
        state = _state(ctx)
        registry: StoreRegistry | None = state.get("stores")
        if registry is None:
            # Unauthenticated stdio: one caller, one tenant, a private pipe. There is no principal
            # to charge and no one to protect the local user from but themselves, so the budget
            # does not apply — matching how auth itself is scoped.
            store: PgVectorStore = state["store"]
            served = getattr(store, "tenant", None)
            if requested_tenant is not None and requested_tenant != served:
                raise PermissionError(
                    f"this single-tenant server serves tenant {served!r}, not {requested_tenant!r}"
                )
            if scope != SCOPE_READ and idempotency_key:
                durable = await _durable_replay(
                    store, idempotency_key, idempotency_operation, idempotency_fingerprint
                )
                if durable is not None:
                    raise IdempotencyReplay(durable)
            return store

        token = get_access_token()
        if token is None:
            # The SDK's bearer middleware rejects unauthenticated HTTP requests before a tool
            # runs, so this is unreachable through the normal path. It stays because the
            # alternative — falling through to some default store — would turn any future gap in
            # that middleware into a silent full-corpus read.
            raise PermissionError("this server requires authentication")
        idempotency_fingerprint = _principal_bound_fingerprint(idempotency_fingerprint)
        principal = _token_principal(token)
        session, client, tool = caller_identity(ctx)
        try:
            tenant = authorize(token.scopes, token.claims, scope)
        except PermissionError:
            await _to_thread(
                lambda: record_access(principal=principal, tenant=(token.claims or {}).get("tenant"),
                                      scope=scope, tool=tool, session=session, client=client,
                                      outcome="denied")
            )
            raise
        if requested_tenant is not None and requested_tenant != tenant:
            await _to_thread(
                lambda: record_access(principal=principal, tenant=tenant, scope=scope, tool=tool,
                                      session=session, client=client, outcome="denied")
            )
            raise PermissionError(
                f"the authenticated token is scoped to tenant {tenant!r}, not {requested_tenant!r}"
            )
        await _to_thread(
            lambda: record_access(principal=principal, tenant=tenant, scope=scope, tool=tool,
                                  session=session, client=client, outcome="allowed")
        )
        # After authorisation, so an unauthorised caller cannot burn the tenant's budget by
        # hammering a scope it does not hold.
        limiter = state.get("limiter")
        if limiter is not None:
            if scope != SCOPE_READ and getattr(limiter, "requires_idempotency", False) and not idempotency_key:
                raise IdempotencyKeyRequired(
                    "idempotency_key is required for retryable write, forget, and admin operations"
                )
            if getattr(limiter, "requires_idempotency", False):
                try:
                    await limiter.check(
                        tenant,
                        _SCOPE_BUDGETS[scope],
                        idempotency_key=idempotency_key,
                        idempotency_operation=idempotency_operation,
                        idempotency_fingerprint=idempotency_fingerprint,
                        read_only=scope == SCOPE_READ,
                    )
                except IdempotencyResultMissing as missing:
                    if scope == SCOPE_READ:
                        raise
                    store = await _to_thread(lambda: registry.get(tenant))
                    durable = await _durable_replay(
                        store,
                        missing.idempotency_key,
                        idempotency_operation,
                        idempotency_fingerprint,
                    )
                    if durable is not None:
                        raise IdempotencyReplay(durable) from missing
                    raise IdempotencyReconciliation(
                        missing.idempotency_key,
                        operation=idempotency_operation,
                        request_fingerprint=idempotency_fingerprint,
                    ) from missing
                except IdempotencyConflict as conflict:
                    raise ToolError(
                        json.dumps(
                            {"error": "idempotency_conflict", "message": str(conflict)},
                            sort_keys=True,
                        )
                    ) from conflict
            else:
                # Keep compatibility with the small synchronous test doubles and host supplied
                # local limiters. The Redis implementation is the only backend that needs the
                # richer idempotency and fallback arguments.
                await _check_limiter_async(limiter, tenant, _SCOPE_BUDGETS[scope], 1.0)
        # StoreRegistry.get holds a process-wide lock across synchronous PostgreSQL work (a
        # control-plane route refresh, a lazy tenant's store open), so it runs in a worker.
        store = await _to_thread(lambda: registry.get(tenant))
        if scope != SCOPE_READ and idempotency_key:
            durable = await _durable_replay(
                store, idempotency_key, idempotency_operation, idempotency_fingerprint
            )
            if durable is not None:
                raise IdempotencyReplay(durable)
        return store

    async def _require_mutation(
        scope: str,
        ctx: Context[dict, object],
        requested_tenant: str | None = None,
        idempotency_key: str | None = None,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> tuple[PgVectorStore | None, str | None]:
        """Authorize a mutation or return its durable replay response or status."""
        try:
            return (
                await _require(
                    scope,
                    ctx,
                    requested_tenant,
                    idempotency_key,
                    idempotency_operation,
                    idempotency_fingerprint,
                ),
                None,
            )
        except IdempotencyReplay as replay:
            return None, replay.result
        except IdempotencyReconciliation as reconciliation:
            return None, reconciliation.result

    deps = _ToolDeps(
        require=_require,
        require_mutation=_require_mutation,
        record_mutation_result=_record_mutation_result,
        state=_state,
        current_tenant=_current_tenant,
        access_context=_access_context,
        answer_backend_configured=_answer_backend_configured(runtime_env),
        assistant=assistant,
    )
    surface = resolve_tool_surface(runtime_env)
    if is_lite_dsn(runtime_settings.serving_dsn):
        surface = surface - LITE_UNSERVED_TOOLS
    if assistant is None:
        surface = surface - ASSISTANT_TOOLS
    registrar = cast(MCPServer, FilteredToolRegistrar(mcp, surface))
    # With the assistant surface on, five tools are registered in another form by
    # `_register_assistant_tools`, which reuses the held declarations' annotations and descriptions.
    held = _HeldTools(registrar, _ASSISTANT_HELD if assistant is not None else frozenset())
    tools = cast(MCPServer, held)
    _register_search_tools(tools, deps)
    _register_fact_tools(tools, deps)
    _register_reasoning_tools(tools, deps)
    _register_ingest_tools(tools, deps)
    _register_calibration_tools(tools, deps)
    _register_memory_admin_tools(tools, deps)
    _register_assistant_tools(registrar, deps, held)
    return mcp


def main() -> None:
    from recall._env import refuse_current_directory_executables

    # Launched by the client in the session's directory, a repository nobody here wrote, and the
    # indexer runs `git` there by bare name.
    refuse_current_directory_executables()
    settings = bootstrap_settings()
    environment = settings.env
    # stderr only, and propagate=False — stdout carries JSON-RPC, so a stray log line there
    # would corrupt the protocol. Before build_server: its auth and trust-mode warnings must
    # reach the configured handler and format, not Python's bare last-resort one (ENV-002).
    configure_logging(
        level=environment.get("RECALL_LOG_LEVEL", "INFO"),
        fmt=environment.get("RECALL_LOG_FORMAT", "text"),
    )
    server = build_server(settings)
    transport = settings.transport
    host = settings.host
    port = settings.port
    stateless_http = settings.mcp_stateless
    tenant = settings.tenant
    embedder = settings.embedder
    if transport in HTTP_TRANSPORTS:
        # Tenancy is per-token here, so logging a single RECALL_TENANT would be actively
        # misleading about what this process serves.
        _log.info(
            "starting %s server on %s:%s (authenticated)",
            transport,
            host,
            port,
        )
    else:
        _log.info("starting stdio server", extra={"tenant": tenant, "embedder": embedder})
    if transport == "stdio":
        server.run()
    elif transport == "sse":
        security = _transport_security_settings(
            environment["RECALL_AUTH_RESOURCE_URL"], environment.get("RECALL_ALLOWED_HOSTS", "")
        )
        server.run(
            transport="sse",
            host=host,
            port=port,
            transport_security=security,
        )
    else:
        security = _transport_security_settings(
            environment["RECALL_AUTH_RESOURCE_URL"], environment.get("RECALL_ALLOWED_HOSTS", "")
        )
        server.run(
            transport="streamable-http",
            host=host,
            port=port,
            stateless_http=stateless_http,
            transport_security=security,
        )


if __name__ == "__main__":
    main()
