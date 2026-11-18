"""The ingest tools of the MCP server, and `ingest_into_serving_store` behind them.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TypedDict, cast

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations

from recall.embeddings import Embedder
from recall.errors import RecallError
from recall.index import chunk_code, chunk_text
from recall.observability import METRICS
from recall.runtime_route import RuntimeRoute
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.lite import LiteWriterBusy
from recall.store import PgVectorStore
from recall_mcp.auth import SCOPE_ADMIN, SCOPE_READ, SCOPE_WRITE
from recall_mcp.limits import AsyncRateLimiter, INDEX_BYTES_BUDGET
from recall_mcp.indexing import IndexPreflightError, index_memory
# Through the service facade on purpose: this wrapper injects the service-level retrieval seams.
from recall_mcp.service import tenant_scopes
from recall_mcp.status import JobLedger, job_status
from recall_mcp.models import IndexResult
from recall_mcp.factories import make_profile_embedder
from recall_mcp.generation_admin import generation_ingest
from recall_mcp.stores import StoreRegistry
from recall.desktop.uploads import discard_staging, stage_uploads
from recall_mcp.threads import _to_thread
from recall_mcp.server_common import (
    _ToolDeps,
    _check_limiter_async,
    _mutation_fingerprint,
    _release_mutation_reservation,
    _runtime_env_for,
)


class _MutationPreflightFailure(RuntimeError, RecallError):
    """A mutation failed before its indexing side effect began."""

    def __init__(self, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(str(cause))


def _ingest_mutation_fingerprint(
    files: list[dict[str, str]], category: str, source_namespace: str | None = None,
) -> str:
    """Fingerprint an upload without copying its complete base64 payload into canonical JSON."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(b"recall-ingest-v1\0")
    encoded_category = category.encode("utf-8")
    digest.update(len(encoded_category).to_bytes(8, "big"))
    digest.update(encoded_category)
    if source_namespace is not None:
        encoded_namespace = source_namespace.encode("utf-8")
        digest.update(b"\0source-namespace\0")
        digest.update(len(encoded_namespace).to_bytes(8, "big"))
        digest.update(encoded_namespace)
    for item in files:
        if "source_key" in item:
            encoded_source_key = str(item["source_key"]).encode("utf-8")
            digest.update(b"\0source-key\0")
            digest.update(len(encoded_source_key).to_bytes(8, "big"))
            digest.update(encoded_source_key)
        encoded_name = str(item.get("name", "")).encode("utf-8")
        encoded_content = str(item.get("content_b64", "")).encode("utf-8")
        for encoded_value in (encoded_name, encoded_content):
            digest.update(len(encoded_value).to_bytes(8, "big"))
            digest.update(encoded_value)
    return digest.hexdigest()


def _check_limiter_from_worker(
    limiter: AsyncRateLimiter,
    tenant: str,
    key: str,
    cost: float,
    *,
    read_only: bool = False,
) -> None:
    """Run the async limiter from the synchronous indexing callback worker."""
    check_sync = getattr(limiter, "check_sync", None)
    if callable(check_sync):
        check_sync(tenant, key, cost, read_only=read_only)
        return
    import asyncio
    try:
        result = limiter.check(tenant, key, cost, read_only=read_only)
    except TypeError:
        result = limiter.check(tenant, key, cost)
    if hasattr(result, "__await__"):
        asyncio.run(result)


class _IngestNamespaceOptions(TypedDict, total=False):
    source_namespace: str


def _check_source_namespace(category: str, source_namespace: str | None) -> None:
    if source_namespace is not None and (
        category != "memory" or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", source_namespace)
    ):
        raise ValueError("source_namespace requires memory uploads and 1 to 64 safe characters")


def _compat_route(state: Mapping[str, object]) -> RuntimeRoute:
    # Compatibility for direct library callers. The live server always publishes a resolved
    # RuntimeRoute; this fallback never reads the environment, so a helper call cannot drift
    # away from the state it was handed.
    return RuntimeRoute(
        mode="generation" if state.get("generation_mode") else "legacy",
        environment="development",
        source="state compatibility",
        table="chunks",
        explicit=True,
    )


def ingest_into_serving_store(
    state: Mapping[str, object],
    store: PgVectorStore,
    staged_root: str,
    category: str,
    access_context: AccessContext | None = None,
    source_namespace: str | None = None,
) -> IndexResult:
    """Index a staged upload into the store this server SERVES FROM.

    ⚠️ **`recall_ingest` used to call `generation_ingest` unconditionally**, so a server serving the
    legacy `chunks` table accepted an upload, built and activated a generation, and then could not
    find it. Measured on a project added after install:

        ingest  -> 'Built and activated generation gen_21a9... with 3 chunk(s) from 3 file(s)'
        stats   -> {'chunks': 0, 'stale': True}
        search  -> 0 hits, abstained: false

    Nothing errored. The two ends were reading different tables, and each was telling the truth
    about its own. After this, the same sequence on the same tenant reports 3 chunks and 3 hits.

    The branch is not a preference: the two paths each REFUSE the other's mode, so exactly one is
    legal for a given server. `index_memory` raises under `RECALL_ENV=production` ("local
    filesystem indexing is development-only"), and a production generation build requires an
    immutable embedder revision or artifact digest. Calling the wrong one is therefore either an
    error or, as here, a silent write to a table nobody reads.
    """
    _check_source_namespace(category, source_namespace)
    embedder = cast(Embedder, state["embedder"])
    route = state.get("route")
    if route is None:
        route = _compat_route(state)
    if not isinstance(route, RuntimeRoute):
        raise RuntimeError(
            "serving state contains an invalid route; restart the server so indexing and serving "
            "share one route decision"
        )
    if route.enterprise:
        # A preflight refusal: nothing was written, so the caller's reservation is released.
        raise IndexPreflightError(
            "enterprise control plane does not accept direct staged uploads; build an immutable "
            "generation from a manifest and cut it over through the control plane"
        )
    raw_security_policy = state.get("source_security_policy")
    security_policy = (
        raw_security_policy if isinstance(raw_security_policy, SourceSecurityPolicy) else None
    )
    if route.uses_generation:
        if security_policy is None and access_context is None:
            return generation_ingest(
                store, embedder, staged_root, category, env=_runtime_env_for(state)
            )
        return generation_ingest(
            store,
            embedder,
            staged_root,
            category,
            security_policy=security_policy,
            security_context=access_context,
            env=_runtime_env_for(state),
        )
    # ⚠️ **`category` must reach BOTH branches.** The first version of this function passed it to
    # `generation_ingest` and dropped it here, so the legacy branch fell back to `chunk_text` while
    # the generation branch chose `chunk_code` for code. Every project created by `add_project` is
    # development-mode by construction, so that was not an edge case: it was the default path, and
    # source files uploaded into a `-code` corpus were chunked as prose. Nothing errored, nothing
    # was missing, only the chunk boundaries were wrong — the quietest failure available.
    #
    # The test that accompanied that fix asserted WHICH function was called and never what it was
    # called with, which is why neither it nor its mutation run could see this.
    namespace_options: _IngestNamespaceOptions = {}
    if source_namespace is not None:
        namespace_options["source_namespace"] = source_namespace
    if security_policy is None and access_context is None:
        return index_memory(
            store,
            embedder,
            staged_root,
            chunker=chunk_code if category == "code" else chunk_text,
            env=_runtime_env_for(state),
            lane_purpose="ingest",
            **namespace_options,
        )
    return index_memory(
        store,
        embedder,
        staged_root,
        chunker=chunk_code if category == "code" else chunk_text,
        security_policy=security_policy,
        security_context=access_context,
        env=_runtime_env_for(state),
        lane_purpose="ingest",
        **namespace_options,
    )


def _register_ingest_tools(mcp: MCPServer, deps: _ToolDeps) -> None:
    _require = deps.require
    _require_mutation = deps.require_mutation
    _record_mutation_result = deps.record_mutation_result
    _state = deps.state
    _current_tenant = deps.current_tenant
    _access_context = deps.access_context

    @mcp.tool(
        name="recall_index",
        annotations=ToolAnnotations(
            title="Add to agent memory",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_index(
        path: str, ctx: Context[dict, object], idempotency_key: str | None = None
    ) -> str:
        """Index a markdown file or folder into the agent's memory so it can be recalled later.

        Re-indexing a file REPLACES its chunks completely (safe to re-run after edits; a shrunk
        file leaves no stale chunks behind).
        `path` is confined to RECALL_INDEX_ROOT (default: the server's working directory), and the
        request is refused before anything is embedded if it exceeds RECALL_INDEX_MAX_FILES or
        RECALL_INDEX_MAX_BYTES (see `recall/index_budget.py`).

        Args:
            path: a file or directory path (``**/*.md`` is indexed for directories).

        Returns:
            JSON of {files, chunks, message}.
        """
        state = _state(ctx)
        operation = "recall_index"
        fingerprint = _mutation_fingerprint(operation, {"path": path})
        store, replay = await _require_mutation(
            SCOPE_WRITE,
            ctx,
            idempotency_key=idempotency_key,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        if replay is not None:
            return replay
        assert store is not None
        limiter = state.get("limiter")
        tenant = _current_tenant(state)
        registry: StoreRegistry | None = state.get("stores")

        def _shadow_preflight() -> tuple[PgVectorStore | None, Embedder | None]:
            # Synchronous control-plane and store work (a route read is a fresh connection, and
            # an embedder may load a model), so it runs in a worker, not on the loop (PERF-002).
            shadow_store = (
                registry.get_shadow(tenant) if registry is not None and tenant is not None else None
            )
            shadow_embedder = None
            if shadow_store is not None:
                assert (
                    registry is not None
                    and registry.control_plane is not None
                    and tenant is not None
                )
                route = registry.control_plane.route(tenant)
                if route is None or route.shadow is None:
                    raise IndexPreflightError(
                        "shadow store was acquired without shadow generation metadata"
                    )
                profile_id = route.shadow.embedding_profile
                lock = state["shadow_embedder_lock"]
                with lock:
                    cache = state["shadow_embedders"]
                    shadow_embedder = cache.get(profile_id)
                    if shadow_embedder is None:
                        # The settings snapshot, not os.environ: bootstrap_settings resolves
                        # Secrets Manager keys into it only, as for the main embedder.
                        shadow_embedder = make_profile_embedder(
                            profile_id, shadow=True, env=dict(_runtime_env_for(state))
                        )
                        if shadow_embedder.dim != route.shadow.dimension:
                            raise IndexPreflightError(
                                "shadow embedder dimension does not match generation"
                            )
                        cache[profile_id] = shadow_embedder
            return shadow_store, shadow_embedder

        try:
            shadow_store, shadow_embedder = await _to_thread(_shadow_preflight)
        except Exception:  # BROAD-CATCH: cleanup-only
            await _release_mutation_reservation(state, store.tenant, idempotency_key)
            raise

        def _debit(_files: int, total_bytes: int) -> None:
            """Charge the tenant for what is about to be embedded, before it is embedded.

            The call budget alone cannot bound spend: one request may carry 20 MB and the next
            200 bytes, so counting requests prices them identically. This meters the thing that
            actually costs money, and it runs pre-flight — a refusal here has spent nothing.
            """
            if limiter is not None and tenant is not None:
                try:
                    _check_limiter_from_worker(
                        limiter, tenant, INDEX_BYTES_BUDGET, float(total_bytes), read_only=False
                    )
                except BaseException as exc:
                    raise _MutationPreflightFailure(exc) from exc

        try:
            with METRICS.timer("recall_tool_latency_ms", tool="index"):
                payload = await _to_thread(
                    lambda: index_memory(
                        store,
                        state["embedder"],
                        path,
                        on_measured=_debit,
                        shadow_store=shadow_store,
                        shadow_embedder=shadow_embedder,
                        control_plane=registry.control_plane if registry is not None else None,
                        security_policy=state.get("source_security_policy"),
                        security_context=_access_context(state, store, purpose="indexing"),
                        env=_runtime_env_for(state),
                    ).model_dump_json(indent=2)
                )
        except _MutationPreflightFailure as exc:
            await _release_mutation_reservation(state, store.tenant, idempotency_key)
            # Re-raises the ORIGINAL exception unchanged. `from` would overwrite its own __cause__.
            raise exc.cause  # noqa: B904
        except (IndexPreflightError, LiteWriterBusy):
            # The service marks path, security, filesystem, and size refusals that occur before
            # the indexer's first write. Runtime, database, and later file-read failures remain
            # reconciliation-required because they may follow a partial commit. A lite store's
            # busy write lane refuses before the lane is held, so before any write too.
            await _release_mutation_reservation(state, store.tenant, idempotency_key)
            raise
        await _record_mutation_result(
            state,
            store.tenant,
            idempotency_key,
            payload,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        return payload

    @mcp.tool(
        name="recall_tenants",
        annotations=ToolAnnotations(
            title="List available tenant scopes",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_tenants(ctx: Context[dict, object]) -> str:
        """Return the tenant scopes visible to this caller.

        An authenticated principal sees its own tenant; the full provisioned inventory is
        admin-only, because in a multi-tenant deployment tenant ids are often customer
        names and one tenant reading the customer list is cross-tenant disclosure.
        """
        state = _state(ctx)
        store = await _require(SCOPE_READ, ctx)
        registry: StoreRegistry | None = state.get("stores")
        if registry is None:
            tenants: list[str] = [store.tenant]
        else:
            token = get_access_token()
            if token is not None and SCOPE_ADMIN in (token.scopes or ()):
                tenants = sorted(getattr(registry, "allowed_tenants", {store.tenant}))
            else:
                tenants = [store.tenant]
        return json.dumps(tenant_scopes(store, tenants), indent=2)

    @mcp.tool(
        name="recall_ingest",
        annotations=ToolAnnotations(
            title="Upload and index source files",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_ingest(
        files: list[dict[str, str]],
        ctx: Context[dict, object],
        category: str = "memory",
        tenant: str | None = None,
        idempotency_key: str | None = None,
        source_namespace: str | None = None,
    ) -> str:
        """Upload bounded source files and index them in the caller's tenant.

        source_namespace optionally gives legacy memory uploads a stable destination.
        It accepts 1 to 64 ASCII letters, digits, underscores or hyphens. Names within
        it identify logical sources; use project-specific prefixes for shared tenants.
        Other files are retained across partial batches. Generation uploads retain
        their immutable staging behavior.
        Legacy namespaced replacement refuses when a source security policy is
        configured; it does not overwrite files before that refusal.
        Each file may include source_key, a lowercase SHA-256 root identity that
        qualifies its first name component without exposing the local root path.
        """
        state = _state(ctx)
        operation = "recall_ingest"
        if category not in {"documents", "code", "memory"}:
            raise ValueError("category must be documents, code, or memory")
        _check_source_namespace(category, source_namespace)
        # Hashing, decoding and writing up to 50 MiB is CPU and disk work: keep it off the loop.
        fingerprint = (
            await _to_thread(lambda: _ingest_mutation_fingerprint(files, category, source_namespace))
            if idempotency_key
            else None
        )
        store, replay = await _require_mutation(
            SCOPE_WRITE,
            ctx,
            tenant,
            idempotency_key=idempotency_key,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        if replay is not None:
            return replay
        assert store is not None
        try:
            job_id, root, total_bytes = await _to_thread(lambda: stage_uploads(store.tenant, files))
        except BaseException:
            await _release_mutation_reservation(state, store.tenant, idempotency_key)
            raise
        # Same quota `recall_index` debits, for the same reason and at the same moment:
        # after per-request caps, before any embedding spend. Without this debit an upload
        # loop under the 50 MiB per-request cap ingests unmetered.
        limiter = state.get("limiter")
        if limiter is not None:
            try:
                await _check_limiter_async(
                    limiter,
                    store.tenant,
                    INDEX_BYTES_BUDGET,
                    float(total_bytes),
                    read_only=False,
                )
            except BaseException:
                # Synchronous on purpose: this handler also sees a cancellation, and an awaited
                # worker call would hit anyio's checkpoint and skip the cleanup.
                discard_staging(root)
                await _release_mutation_reservation(state, store.tenant, idempotency_key)
                raise
        try:
            with METRICS.timer("recall_tool_latency_ms", tool="ingest"):
                result = await _to_thread(
                    lambda: ingest_into_serving_store(
                        state, store, str(root), category,
                        access_context=_access_context(state, store, purpose="indexing"),
                        **({"source_namespace": source_namespace} if source_namespace is not None else {}),
                    )
                )
        except (IndexPreflightError, LiteWriterBusy):
            # Refused before any write (a preflight refusal, or the lite store's write lane busy),
            # so nothing was indexed: drop the staged files and the reservation, so a retry can
            # run instead of being told the mutation may have committed.
            await _to_thread(lambda: discard_staging(root))
            await _release_mutation_reservation(state, store.tenant, idempotency_key)
            raise
        except Exception:  # BROAD-CATCH: fail-closed
            # Legacy mode ONLY discards here: the staged tree fed an ingest that failed, and
            # partially indexed rows become prunable, consistent with "the upload failed".
            #
            # Generation mode does NOT discard. The staged files are pinned by the manifest,
            # and a promote() that commits its transaction and then raises on the way out (a
            # commit-ack loss) leaves an ACTIVE, servable generation whose manifest still
            # points at this tree — `_reclaim_failed` cannot reclaim an ACTIVE generation, so
            # deleting the tree here would erase the source of a generation the tenant is now
            # serving, and the next upload's carry-forward would drop it permanently. Leaving
            # a rare failed-generation staging tree behind is a small, forget-cleanable leak;
            # deleting a servable generation's source is not. `except Exception`, not
            # `BaseException`: a cancellation is deferred by anyio's shielded worker thread,
            # so it cannot land mid-promote here.
            route = state.get("route")
            if not isinstance(route, RuntimeRoute):
                route = _compat_route(state)
            if route.enterprise or not route.uses_generation:
                await _to_thread(lambda: discard_staging(root))
            raise
        payload = json.loads(result.model_dump_json())
        payload.update({"job_id": job_id, "state": "completed", "category": category})
        ledger = state.get("desktop_jobs")
        if not isinstance(ledger, JobLedger):
            ledger = JobLedger()
            state["desktop_jobs"] = ledger
        ledger.put(job_id, store.tenant, payload)
        response = json.dumps(payload, indent=2)
        await _record_mutation_result(
            state,
            store.tenant,
            idempotency_key,
            response,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        return response

    @mcp.tool(
        name="recall_job_status",
        annotations=ToolAnnotations(
            title="Read indexing job status",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_job_status(job_id: str, ctx: Context[dict, object]) -> str:
        """Return the current state of one bounded indexing job."""
        state = _state(ctx)
        store = await _require(SCOPE_READ, ctx)
        result = job_status(store, job_id, state.get("desktop_jobs", {}))
        return json.dumps(result, indent=2)
