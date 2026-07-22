from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import anyio.to_thread
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from pydantic import AnyHttpUrl

from recall.calibration import load_for
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
from recall_mcp.service import forget_memory, index_memory, make_embedder, memory_stats, search_memory
from recall_mcp.stores import StoreRegistry

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")
#: Transport to serve. `stdio` is a private pipe between one client and this process — there is no
#: network listener and no remote caller to authenticate, so auth is not required there. The HTTP
#: transports open a socket, and `build_auth` refuses to start them without tokens.
TRANSPORT = os.environ.get("RECALL_TRANSPORT", "stdio")
HTTP_TRANSPORTS = frozenset({"streamable-http", "sse"})
EMBEDDER_NAME = os.environ.get("RECALL_EMBEDDER", "fastembed")
#: Connections the server keeps open. This bounds concurrent in-flight tool calls at the database,
#: which is where the real limit is — more worker threads than connections just queue on the pool.
POOL_SIZE = int(os.environ.get("RECALL_POOL_SIZE", "8"))
#: Tenant this server instance serves. One store is bound to one tenant, so a
#: multi-tenant deployment runs a server (or a store) per tenant rather than switching
#: tenants on a shared connection — see PgVectorStore._prepare.
TENANT = os.environ.get("RECALL_TENANT", DEFAULT_TENANT)
#: Server-side cap on any single statement. A runaway query otherwise holds its connection until
#: the process dies, and a few of those exhaust the pool while the server still looks healthy.
STATEMENT_TIMEOUT_MS = int(os.environ.get("RECALL_STATEMENT_TIMEOUT_MS", "15000"))

_T = TypeVar("_T")

_log = get_logger("mcp")


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
            expires_at=(
                int(principal.expires_at.timestamp()) if principal.expires_at else None
            ),
            claims={"tenant": principal.tenant, "principal": principal.name},
        )


def build_auth(
    transport: str = TRANSPORT, env: dict[str, str] | None = None
) -> tuple[RecallTokenVerifier | None, AuthSettings | None, TokenRegistry | None]:
    """Resolve the auth configuration for `transport`, failing closed on the HTTP ones.

    This is the function that makes an unauthenticated network listener impossible to create by
    accident. Starting an HTTP transport without `RECALL_AUTH_TOKENS_FILE` raises instead of
    warning, because the failure mode of a warning here is a server that comes up looking healthy
    with every memory in it world-readable — and the warning lands in a journal nobody reads until
    afterwards.
    """
    e = env if env is not None else dict(os.environ)
    registry = token_registry_from_env(e)

    if transport not in HTTP_TRANSPORTS:
        if registry is not None:
            # Configured but inapplicable. Silence here would let an operator believe stdio is
            # access-controlled when the pipe itself is the only boundary.
            _log.warning(
                "RECALL_AUTH_TOKENS_FILE is set but transport is %r — stdio has no remote "
                "caller to authenticate, so the tokens are unused and the single tenant "
                "RECALL_TENANT=%r applies. Set RECALL_TRANSPORT=streamable-http to use them.",
                transport, TENANT,
            )
        return None, None, None

    if registry is None:
        raise AuthConfigError(
            f"transport {transport!r} opens a network listener, so authentication is required. "
            f"Set RECALL_AUTH_TOKENS_FILE to a JSON file of principals (see docs/AUTH.md), or "
            f"use RECALL_TRANSPORT=stdio for a private single-client pipe."
        )

    issuer = e.get("RECALL_AUTH_ISSUER_URL")
    resource = e.get("RECALL_AUTH_RESOURCE_URL")
    if not issuer or not resource:
        raise AuthConfigError(
            "RECALL_AUTH_ISSUER_URL and RECALL_AUTH_RESOURCE_URL are required for an HTTP "
            "transport. They are published in this server's protected-resource metadata so a "
            "client knows where to get a token and which audience it is for; set both to this "
            "server's own public URL if you are provisioning tokens by hand."
        )
    settings = AuthSettings(
        issuer_url=AnyHttpUrl(issuer),
        resource_server_url=AnyHttpUrl(resource),
        # Left empty on purpose: a global required_scope would reject a principal provisioned for
        # exactly one capability (a forget-only retention job holds recall:forget and nothing
        # else). Scope is enforced per tool in `_require`, against what that tool actually does.
        required_scopes=[],
    )
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


def _make_lifespan(token_registry: TokenRegistry | None):
    """Build the lifespan.

    Two shapes, decided by whether auth is on:

    - **Unauthenticated (stdio).** One store bound to `RECALL_TENANT`, exactly as before. There is
      one caller on the other end of the pipe and it gets one namespace.
    - **Authenticated (HTTP).** A `StoreRegistry` over the tenants the token file provisions.
      Nothing is opened until a request for that tenant arrives, so a server configured for ten
      tenants that only ever serves one holds one pool, not ten.
    """

    @asynccontextmanager
    async def _lifespan(_server: FastMCP):
        from recall.store import require_secure_dsn

        # FAIL CLOSED, unlike the CLI's warning: a server is unattended, so a stderr note about
        # published default credentials pointed at a remote database lands in a journal nobody
        # reads while the process comes up looking healthy. RECALL_ALLOW_INSECURE_DSN=1 opts out.
        require_secure_dsn(DEFAULT_DSN)
        store: PgVectorStore | None = None
        registry: StoreRegistry | None = None
        try:
            embedder = make_embedder(EMBEDDER_NAME)
            if token_registry is None:
                # Pooled + timed out: a server shares this store across concurrent tool calls,
                # and one connection would serialise them however many threads are available.
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
                )
        except Exception:
            _log.error(
                "startup failed (dsn=%s, embedder=%r)",
                redacted_dsn(DEFAULT_DSN), EMBEDDER_NAME, exc_info=True,
            )
            raise

        try:
            if store is not None:
                store.ensure_schema()
                probe = store
            else:
                assert registry is not None
                # Open ONE tenant eagerly. Schema creation, a missing pgvector extension and a
                # bad DSN all fail identically for every tenant, and finding that out on the
                # first client request — per tenant, at request latency — turns a startup error
                # into an intermittent runtime one.
                probe = registry.get(sorted(registry.allowed_tenants)[0])
                _log.info(
                    "auth enabled: %d tenant(s), up to %d pooled connections at full spread",
                    len(registry.allowed_tenants), registry.max_connections(),
                )
        except Exception:
            if store is not None:
                store.close()
            if registry is not None:
                registry.close()
            _log.error("schema check failed", exc_info=True)
            raise

        calibration = load_for(embedder.name)  # None -> uncalibrated fallback, flagged in results
        if calibration is None:
            _log.warning(
                "no calibration for embedder %r — using the default threshold (results will "
                "say calibrated=false). Run `recall calibrate` to fix.", embedder.name,
            )
        if not probe.check_rls_effective():
            _log.warning(
                "this database role bypasses row-level security (superuser or BYPASSRLS), so "
                "tenant isolation rests on query predicates alone. Connect as an unprivileged "
                "role for defence in depth."
            )
        try:
            yield {
                "store": store,
                "stores": registry,
                "embedder": embedder,
                "calibration": calibration,
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
    )

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
        """
        state = _state()
        registry: StoreRegistry | None = state.get("stores")
        if registry is None:
            return state["store"]  # unauthenticated stdio: one caller, one tenant

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
        return registry.get(tenant)

    @mcp.tool(
        name="recall_search",
        annotations={"title": "Search agent memory", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
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
            JSON of {query, abstained, reason, calibrated, gap_warning, stale, advice,
            hits:[{source, score, confidence, verdict, superseded_by, valid_until,
            indexed_at, text}]}.
        """
        state = _state()
        store = _require(SCOPE_READ)
        with METRICS.timer("recall_tool_latency_ms", tool="search"):
            return await _to_thread(
                lambda: search_memory(
                    store, state["embedder"], query, source=source, k=k,
                    calibration=state.get("calibration"),
                ).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_index",
        annotations={"title": "Add to agent memory", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
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
        with METRICS.timer("recall_tool_latency_ms", tool="index"):
            return await _to_thread(
                lambda: index_memory(
                    store, state["embedder"], path
                ).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_forget",
        annotations={"title": "Forget agent memory", "readOnlyHint": False,
                     "destructiveHint": True, "idempotentHint": True, "openWorldHint": False},
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
        store = _require(SCOPE_FORGET)
        with METRICS.timer("recall_tool_latency_ms", tool="forget"):
            return await _to_thread(
                lambda: forget_memory(store, sources).model_dump_json(indent=2)
            )

    @mcp.tool(
        name="recall_stats",
        annotations={"title": "Memory freshness & size", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def recall_stats() -> str:
        """Report how much memory exists and whether it is stale (freshness check).

        `stale` is True when the newest indexed content is older than 2 days.

        Returns:
            JSON of {chunks, newest_indexed_at, stale}.
        """
        store = _require(SCOPE_READ)
        return await _to_thread(
            lambda: memory_stats(store).model_dump_json(indent=2)
        )

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
            TRANSPORT, mcp.settings.host, mcp.settings.port,
        )
    else:
        _log.info("starting stdio server", extra={"tenant": TENANT, "embedder": EMBEDDER_NAME})
    mcp.run(transport=TRANSPORT)


if __name__ == "__main__":
    main()
