"""The MCP server's lifespan: startup configuration, the serving backend, the worker thread budget,
and `_make_lifespan`.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import cast

import anyio.to_thread
from mcp.server import MCPServer

from recall import eval_clock
from recall.calibration import load_for as calibration_load_for
from recall.answer_provider import resolve_answer_profile, resolve_answer_provider
from recall.proof_provider import resolve_proof_provider
from recall.evidence import AnswerProfile
from recall.control_plane import ControlPlane
from recall.embeddings import Embedder, embedding_profile_id
from recall.readiness import check_enterprise_readiness
from recall._env import refuse_removed_entailment, truthy
from recall.runtime_route import RuntimeRoute, resolve_runtime_route
from recall.retrieval_plan import RetrievalPlanResolver
from recall.federation import FederationConfig
from recall.security_policy import SourceSecurityPolicy, load_source_policy
from recall.backends import is_lite_dsn, lite_route_refusal, open_legacy_store
from recall.lite.refresh import refresh_access_context
from recall_mcp.lite_refresh import LiteAutoRefresher
from recall_mcp.rescue_builder import RescueBuilder
from recall.store import DEFAULT_TABLE, PgVectorStore, redacted_dsn
from recall_mcp.limits import AsyncRateLimiter, async_limiter_from_env
from recall.ops.health import HealthController
from recall_mcp.settings import (
    Settings,
    activate_runtime_settings,
    bootstrap_settings,
    reset_runtime_settings,
    unserved_atomic_rescue_tenants,
)
from recall_mcp.factories import make_embedder
from recall_mcp.retrieval import startup_retrieval_profile
from recall.profiles import RetrievalProfile
from recall_mcp.stores import StoreRegistry
from recall_mcp.translation import provider_from_env
from recall_mcp.server_auth import TenantProvisioning, require_effective_rls
from recall_mcp.threads import _to_thread
from recall_mcp.server_common import _log


def table_override_refusal(
    table: str, *, generation_mode: bool, authenticated: bool
) -> str | None:
    """Explain why a custom legacy table cannot be used by generation-aware serving."""
    if table == DEFAULT_TABLE or not (generation_mode or authenticated):
        return None
    where = "generation mode" if generation_mode else "authenticated tenant routing"
    return (
        f"RECALL_TABLE={table!r} cannot be honoured under {where}: that store reads the "
        "generation table 'recall_chunks_v1'. Unset RECALL_TABLE, or serve the legacy table "
        "with RECALL_ENV unset."
    )


def benchmark_generation_setting(
    generation_id: str | None,
    *,
    benchmark_pin: bool,
    generation_mode: bool,
    authenticated: bool,
) -> str | None:
    """Validate the explicit retired-snapshot pin used by reproducible stdio benchmarks."""
    value = (generation_id or "").strip()
    if not value:
        return None
    if not benchmark_pin:
        raise RuntimeError("RECALL_PINNED_GENERATION_ID requires RECALL_BENCHMARK_PIN=1")
    if not generation_mode or authenticated:
        raise RuntimeError(
            "RECALL_PINNED_GENERATION_ID is allowed only for unauthenticated generation-mode "
            "stdio serving"
        )
    return value


#: Worker threads held back for everything that is not a queued search.
#:
#: Every tool body runs through `_to_thread`, and so does the OIDC bearer-token validator. A
#: request parked in `RetrievalAdmission` is holding one of those threads for as long as it
#: waits, so without reserved headroom a saturated search path starves `recall_index`,
#: `recall_forget`, `recall_stats` and, worst, authentication itself.
RESERVED_WORKER_THREADS = 8


def worker_thread_budget(profile: RetrievalProfile) -> int:
    """How many worker threads this process needs for `profile` to be the binding constraint.

    The admission gate is entered INSIDE a worker thread, so its capacity is denominated in
    threads whether or not it says so. anyio's default limiter is 40 tokens, and the fast
    profile's 8 + 32 is also 40: the 41st concurrent search would never reach `__enter__` at
    all. It would wait in anyio's limiter, which has no timeout, no budget and no counter, so
    the `queue_full` shed that the whole design exists to produce could not fire, and
    `recall_retrieval_rejected_total` would sit at zero while clients waited unboundedly.

    A guard that reads as protection and cannot fire is worse than no guard, so the pool is
    sized from the profile rather than the profile being trusted to fit the pool.
    """
    return profile.max_concurrency + profile.queue_capacity + RESERVED_WORKER_THREADS


def apply_worker_thread_budget(profile: RetrievalProfile) -> None:
    """Raise the default worker pool to `worker_thread_budget`, never lower it.

    Only raises: a host that has already sized its pool generously is not ours to shrink, and
    lowering it would strand threads already checked out. Must be called from inside the event
    loop, which is why it lives in the lifespan and not at import.
    """
    limiter = anyio.to_thread.current_default_thread_limiter()
    required = worker_thread_budget(profile)
    if limiter.total_tokens < required:
        limiter.total_tokens = required


def bounded_tenant_probe_ids(tenants: frozenset[str], limit: int) -> tuple[str, ...]:
    """Choose a deterministic, bounded readiness sample from configured tenants."""
    if limit < 1:
        raise ValueError("tenant readiness probe limit must be >= 1")
    return tuple(sorted(tenants)[:limit])


@dataclass
class _StartupConfig:
    """What the lifespan resolves from configuration before it opens anything."""

    runtime_settings: Settings
    runtime_env: dict[str, str]
    runtime_secret_versions: dict[str, str]
    serving_dsn: str
    embedder_name: str
    table: str
    tenant: str
    enterprise: bool
    runtime_route: RuntimeRoute
    source_policy: SourceSecurityPolicy | None
    pool_size: int
    connection_budget: int
    max_tenants: int
    readiness_tenant_probes: int
    statement_timeout_ms: int
    retrieval_plan_resolver: RetrievalPlanResolver
    federation_config: FederationConfig
    translation_provider: object


@dataclass
class _ServingBackend:
    """What the lifespan opened: the store or registry, the embedder, the providers."""

    store: PgVectorStore | None
    registry: StoreRegistry | None
    embedder: Embedder
    answer_provider: object
    answer_profile: AnswerProfile
    proof_provider: object
    generation_mode: bool
    pinned_generation_id: str | None
    lite: bool
    clock_installed: bool


def _make_lifespan(
    token_registry: TenantProvisioning | None,
    health: HealthController | None = None,
    settings: Settings | None = None,
) -> Callable[[MCPServer], AbstractAsyncContextManager[dict]]:
    """Build the lifespan.

    Two shapes, decided by whether auth is on:

    - **Unauthenticated (stdio).** One store bound to `RECALL_TENANT`, exactly as before. There is
      one caller on the other end of the pipe and it gets one namespace.
    - **Authenticated (HTTP).** A `StoreRegistry` over the tenants the deployment provisions —
      the token file's principals, or `RECALL_OIDC_TENANTS`. This function does not know which.
      A bounded readiness sample is opened at startup; every tenant outside that sample stays
      lazy until its first authenticated request.
    """

    def _resolve_startup_config() -> _StartupConfig:
        """Phase 1: resolve and validate configuration and the route, before any I/O."""
        from recall.store import require_secure_dsn

        runtime_settings = settings if settings is not None else bootstrap_settings()
        runtime_env = dict(runtime_settings.env)
        runtime_secret_versions = dict(runtime_settings.secret_versions)
        serving_dsn = runtime_settings.serving_dsn
        embedder_name = runtime_settings.embedder
        table = runtime_settings.table
        tenant = runtime_settings.tenant
        enterprise = runtime_settings.enterprise_control_plane
        runtime_route = resolve_runtime_route(runtime_env, enterprise=enterprise)
        source_policy = load_source_policy(runtime_env)
        if enterprise and source_policy is None:
            raise RuntimeError(
                "enterprise control plane requires RECALL_SOURCE_POLICY_FILE; "
                "source authorization and redaction must be configured"
            )
        pool_size = runtime_settings.pool_size
        connection_budget = runtime_settings.connection_budget
        max_tenants = runtime_settings.max_tenants
        readiness_tenant_probes = runtime_settings.readiness_tenant_probes
        statement_timeout_ms = runtime_settings.statement_timeout_ms

        if health is not None:
            health.mark_starting()

        # FIRST, before any I/O. Resolving the cost profile is pure environment parsing, so a
        # contradictory RECALL_RETRIEVAL_PROFILE / RECALL_RERANK pair, or a quality profile whose
        # reranker artifact is not the pinned one, costs nothing to detect and must not be
        # discovered on the first client request. A server that starts clean and then refuses
        # every search is a server whose configuration error reads as an outage.
        retrieval_profile = startup_retrieval_profile(runtime_env)
        retrieval_plan_resolver = RetrievalPlanResolver.from_env(runtime_env)
        federation_config = FederationConfig.from_env(runtime_env)
        # The entailment judge was removed. A deployment that still sets RECALL_ENTAILMENT to a
        # true value relied on its stricter filter, so it is refused here, before any I/O, rather
        # than served wider results it never asked for.
        refuse_removed_entailment(runtime_env)
        # Validate localization configuration at startup as well. Constructing the provider is
        # pure configuration work and performs no network request; delaying this until a client
        # asks for a locale would turn a deployment error into a request-time surprise.
        translation_provider = provider_from_env(runtime_env)
        # Size the worker pool from the profile, so the admission gate is the binding constraint
        # rather than a coincidence. See `worker_thread_budget`.
        apply_worker_thread_budget(retrieval_profile)

        # FAIL CLOSED, unlike the CLI's warning: a server is unattended, so a stderr note about
        # published default credentials pointed at a remote database lands in a journal nobody
        # reads while the process comes up looking healthy. RECALL_ALLOW_INSECURE_DSN=1 opts out.
        require_secure_dsn(serving_dsn)
        _log.info(
            "retrieval profile %s (candidates %d/leg, returns %d, reranker %s, "
            "budget %d ms, %d concurrent + %d queued)",
            retrieval_profile.name,
            retrieval_profile.candidate_k,
            retrieval_profile.returned_k,
            retrieval_profile.reranker,
            retrieval_profile.latency_budget_ms,
            retrieval_profile.max_concurrency,
            retrieval_profile.queue_capacity,
        )
        return _StartupConfig(
            runtime_settings=runtime_settings,
            runtime_env=runtime_env,
            runtime_secret_versions=runtime_secret_versions,
            serving_dsn=serving_dsn,
            embedder_name=embedder_name,
            table=table,
            tenant=tenant,
            enterprise=enterprise,
            runtime_route=runtime_route,
            source_policy=source_policy,
            pool_size=pool_size,
            connection_budget=connection_budget,
            max_tenants=max_tenants,
            readiness_tenant_probes=readiness_tenant_probes,
            statement_timeout_ms=statement_timeout_ms,
            retrieval_plan_resolver=retrieval_plan_resolver,
            federation_config=federation_config,
            translation_provider=translation_provider,
        )

    def _open_serving_backend(config: _StartupConfig) -> _ServingBackend:
        """Phase 2: build the embedder and providers, refuse a bad route, open the store or registry.

        Also installs the evaluation clock. A failure here marks health failed and re-raises
        without uninstalling it, exactly as before the lifespan was split into phases.
        """
        runtime_env = config.runtime_env
        runtime_route = config.runtime_route
        serving_dsn = config.serving_dsn
        embedder_name = config.embedder_name
        table = config.table
        tenant = config.tenant
        enterprise = config.enterprise
        pool_size = config.pool_size
        connection_budget = config.connection_budget
        max_tenants = config.max_tenants
        statement_timeout_ms = config.statement_timeout_ms
        clock_installed = False
        store: PgVectorStore | None = None
        registry: StoreRegistry | None = None
        try:
            embedder = make_embedder(embedder_name, env=runtime_env)
            answer_provider = resolve_answer_provider(runtime_env)
            answer_profile = resolve_answer_profile(runtime_env)
            proof_provider = resolve_proof_provider(runtime_env)
            generation_mode = runtime_route.uses_generation
            pinned_generation_id = benchmark_generation_setting(
                runtime_env.get("RECALL_PINNED_GENERATION_ID"),
                benchmark_pin=truthy(runtime_env.get("RECALL_BENCHMARK_PIN")),
                generation_mode=generation_mode,
                authenticated=token_registry is not None,
            )
            refusal = table_override_refusal(
                table,
                generation_mode=generation_mode,
                authenticated=token_registry is not None or enterprise,
            )
            if refusal:
                raise RuntimeError(refusal)
            # One SQLite file serves one local client on the legacy route. Everything below
            # that needs PostgreSQL (migrations, row-level security, generations, the control
            # plane, a tenant registry) is either skipped for it or refused here, by name.
            lite = is_lite_dsn(serving_dsn)
            if lite:
                lite_refusal = lite_route_refusal(serving_dsn, runtime_route)
                if lite_refusal is None and (token_registry is not None or enterprise):
                    lite_refusal = (
                        "the lite store (one SQLite file) serves one local client over stdio; "
                        "authentication and the enterprise control plane need the full install"
                    )
                if lite_refusal is not None:
                    raise RuntimeError(lite_refusal)
            # An evaluation clock (`recall.eval_clock`): lite only, read before anything is
            # written, so no row of this run carries the wall clock's time.
            clock = eval_clock.from_env(runtime_env, lite=lite)
            if clock is not None:
                eval_clock.install(clock)
                clock_installed = True
            # Inspect migration state before PgVectorStore prepares a pgvector codec. On a fresh
            # database the extension deliberately does not exist yet; reporting "migrations
            # pending" is more useful than leaking the driver's missing-type error. This path is
            # SELECT-only and uses the serving credential.
            from recall.schema import SchemaTooOld, schema_status

            probe_table = DEFAULT_TABLE if (generation_mode or token_registry is not None) else table
            schema = None if lite else schema_status(serving_dsn, table=probe_table, dim=embedder.dim)
            if schema is not None and not schema.compatible:
                pending = [m.version for m in schema.pending]
                raise SchemaTooOld(
                    f"database migrations pending: {pending}; run `recall schema apply`"
                )
            if enterprise and token_registry is None:
                raise RuntimeError("enterprise control plane requires authenticated tenant routing")
            unserved = unserved_atomic_rescue_tenants(
                runtime_env, (tenant,) if token_registry is None else token_registry.tenants
            )
            if unserved:
                _log.warning(
                    "RECALL_ATOMIC_RESCUE_MODE_BY_TENANT names tenant(s) this server does not "
                    "serve, which keep no override: %s",
                    ", ".join(unserved),
                )
            if token_registry is None:
                # Pooled + timed out: a server shares this store across concurrent tool calls,
                # and one connection would serialise them however many threads are available.
                if generation_mode:
                    from recall.generation_store import GenerationStore

                    store = GenerationStore(
                        serving_dsn,
                        dim=embedder.dim,
                        tenant=tenant,
                        pool_size=pool_size,
                        statement_timeout_ms=statement_timeout_ms,
                    )
                else:
                    store = open_legacy_store(
                        serving_dsn,
                        dim=embedder.dim,
                        table=table,
                        tenant=tenant,
                        pool_size=pool_size,
                        statement_timeout_ms=statement_timeout_ms,
                    )
            else:
                if len(token_registry.tenants) > max_tenants:
                    raise RuntimeError(
                        f"configured tenant count {len(token_registry.tenants)} exceeds "
                        f"RECALL_MAX_TENANTS={max_tenants}"
                    )
                registry = StoreRegistry(
                    dsn=serving_dsn,
                    dim=embedder.dim,
                    allowed_tenants=token_registry.tenants,
                    pool_size=pool_size,
                    connection_budget=connection_budget,
                    max_tenants=max_tenants,
                    statement_timeout_ms=statement_timeout_ms,
                    generation_mode=generation_mode and not enterprise,
                    control_plane=(
                        ControlPlane(serving_dsn, statement_timeout_ms=statement_timeout_ms)
                        if enterprise else None
                    ),
                    embedding_profile=embedding_profile_id(embedder),
                )
        except Exception as exc:  # BROAD-CATCH: fail-closed
            if health is not None:
                health.mark_failed(exc)
            _log.error(
                "startup failed (dsn=%s, embedder=%r)",
                redacted_dsn(serving_dsn),
                embedder_name,
                exc_info=True,
            )
            raise
        return _ServingBackend(
            store=store,
            registry=registry,
            embedder=embedder,
            answer_provider=answer_provider,
            answer_profile=answer_profile,
            proof_provider=proof_provider,
            generation_mode=generation_mode,
            pinned_generation_id=pinned_generation_id,
            lite=lite,
            clock_installed=clock_installed,
        )

    def _check_schema_and_probes(
        config: _StartupConfig, backend: _ServingBackend
    ) -> tuple[PgVectorStore, list[PgVectorStore]]:
        """Phase 3: check the schema, apply a generation pin, open the bounded readiness sample.

        A failure closes the store or registry and re-raises; it does not uninstall the
        evaluation clock, exactly as before the lifespan was split into phases.
        """
        store = backend.store
        registry = backend.registry
        pinned_generation_id = backend.pinned_generation_id
        readiness_tenant_probes = config.readiness_tenant_probes
        max_tenants = config.max_tenants
        try:
            if store is not None:
                store.check_schema()
                if pinned_generation_id is not None:
                    set_fixed_generation = getattr(store, "set_fixed_generation", None)
                    if not callable(set_fixed_generation):
                        raise RuntimeError(
                            "RECALL_PINNED_GENERATION_ID requires a generation-mode store"
                        )
                    set_fixed_generation(pinned_generation_id)
                probe = store
            else:
                assert registry is not None
                # Open only a bounded, deterministic sample. The shared pool and the common
                # control plane catch process-wide failures once, while stores for the remaining
                # tenants stay lazy until their first authenticated request.
                probe_tenants = bounded_tenant_probe_ids(
                    registry.allowed_tenants, readiness_tenant_probes
                )
                health_probes = [registry.get(tenant_id) for tenant_id in probe_tenants]
                probe = health_probes[0]
                _log.info(
                    "auth enabled: %d tenant(s) of max %d, probing %d tenant(s), up to %d "
                    "pooled connections within budget %d",
                    len(registry.allowed_tenants),
                    max_tenants,
                    len(health_probes),
                    registry.max_connections(),
                    registry.connection_budget,
                )
            if registry is None:
                health_probes = [probe]
        except Exception as exc:  # BROAD-CATCH: cleanup-only
            if health is not None:
                health.mark_failed(exc)
            if store is not None:
                store.close()
            if registry is not None:
                registry.close()
            _log.error("schema check failed", exc_info=True)
            raise
        return probe, health_probes

    async def _gate_readiness(
        config: _StartupConfig, backend: _ServingBackend, probe: PgVectorStore
    ) -> tuple[AsyncRateLimiter | None, bool]:
        """Phase 4: the RLS gate, enterprise readiness, and the per-tenant limiter.

        Returns the limiter and whether enterprise readiness passed. A failure closes everything
        opened so far, uninstalls the evaluation clock, and re-raises.
        """
        store = backend.store
        registry = backend.registry
        embedder = backend.embedder
        lite = backend.lite
        clock_installed = backend.clock_installed
        enterprise = config.enterprise
        runtime_env = config.runtime_env
        limiter: AsyncRateLimiter | None = None
        try:
            try:
                # A SQLite file has no roles and holds one tenant, so there is no policy to check.
                rls_warning = None if lite else require_effective_rls(
                    rls_effective=probe.check_rls_effective(), multi_tenant=registry is not None
                )
            except RuntimeError:
                _log.error("refusing to serve: row-level security is not effective for this role")
                raise
            if rls_warning is not None:
                _log.warning("%s", rls_warning)
            enterprise_readiness_ok = True
            if enterprise:
                # The calibration argument is supplied again. #182 removed it, and because the
                # parameter defaults to None every enterprise boot since then took the
                # `calibration is None` path: a permanent degraded-readiness warning that said
                # nothing about the real calibration state, and an identity-mismatch FAILURE that
                # could not be reached from this call site at all. A check that reads as a gate
                # and cannot fail is worse than no check, so the argument comes back rather than
                # the branch being deleted. `load_for` returns None for a calibration belonging to
                # a different embedder, so the warning now means "there is none", which is
                # actionable, instead of "this call site does not pass one", which was not.
                #
                # Stated precisely, because the first version of this comment overclaimed: what
                # is repaired is the WARNING, not the identity-mismatch FAILURE. `load_for(P)`
                # returns None exactly when the stored calibration names another embedder and
                # otherwise constructs `Calibration(embedder=P)`, so `calibration.embedder != P`
                # is still unreachable FROM HERE. That branch guards direct library callers, who
                # may pass any Calibration, and `tests/test_enterprise_readiness.py` exercises it
                # there. Making it reachable from startup needs an identity-agnostic loader, which
                # is a separate change.
                readiness = check_enterprise_readiness(
                    probe,
                    embedder,
                    control_plane=registry.control_plane if registry is not None else None,
                    calibration=calibration_load_for(embedding_profile_id(embedder)),
                )
                if not readiness.ready:
                    enterprise_readiness_ok = False
                    raise RuntimeError(
                        "enterprise readiness failed: " + "; ".join(readiness.failures)
                    )
                if readiness.degraded:
                    _log.warning(
                        "enterprise readiness degraded: %s", "; ".join(readiness.warnings)
                    )
            # Built only for the authenticated shape: buckets are keyed by tenant, and stdio has
            # no principal to attribute a call to. Reported at startup so the effective budget is
            # visible in the journal rather than inferred from which requests started failing.
            limiter = async_limiter_from_env(runtime_env) if registry is not None else None
            if limiter is not None:
                limits = getattr(limiter, "limits", {})
                _log.info(
                    "per-tenant budgets: %s",
                    ", ".join(f"{k}={v.capacity:,.0f}" for k, v in sorted(limits.items()))
                    or "(all disabled)",
                )
        except Exception as exc:  # BROAD-CATCH: cleanup-only
            # Every refusal between the schema check and the yield (RLS, enterprise readiness,
            # calibration load, limiter construction) leaves the same resources open as the
            # handlers above, and the main `finally` below is not entered yet.
            if health is not None:
                health.mark_failed(exc)
            if limiter is not None:
                await limiter.close()
            if store is not None:
                store.close()
            if registry is not None:
                registry.close()
            if clock_installed:
                eval_clock.uninstall()
            raise
        return limiter, enterprise_readiness_ok

    def _assemble_runtime_state(
        config: _StartupConfig,
        backend: _ServingBackend,
        probe: PgVectorStore,
        health_probes: list[PgVectorStore],
        limiter: AsyncRateLimiter | None,
        enterprise_readiness_ok: bool,
    ) -> dict[str, object]:
        """Phase 5: read the active generation and build the state every tool reads.

        Called inside the lifespan's main `try`, so a failure here reaches its `finally`.
        """
        store = backend.store
        registry = backend.registry
        embedder = backend.embedder
        answer_provider = backend.answer_provider
        answer_profile = backend.answer_profile
        proof_provider = backend.proof_provider
        generation_mode = backend.generation_mode
        runtime_route = config.runtime_route
        source_policy = config.source_policy
        translation_provider = config.translation_provider
        readiness_tenant_probes = config.readiness_tenant_probes
        max_tenants = config.max_tenants
        connection_budget = config.connection_budget
        runtime_secret_versions = config.runtime_secret_versions
        runtime_settings = config.runtime_settings
        retrieval_plan_resolver = config.retrieval_plan_resolver
        federation_config = config.federation_config
        active_generation = None
        if generation_mode:
            try:
                # `probe` is typed as the common store because authenticated registries can
                # return either implementation. Generation mode supplies this capability,
                # while enterprise routing may deliberately keep a base store here.
                active_generation_reader = cast(
                    Callable[[], str], getattr(probe, "active_generation_id")
                )
                active_generation = active_generation_reader()
            except Exception:  # BROAD-CATCH: fail-closed, readiness reports missing generation
                _log.warning("no active generation is available during startup")
        runtime_state: dict[str, object] = {
            "store": store,
            "stores": registry,
            "embedder": embedder,
            "answer_provider": answer_provider,
            "answer_profile": answer_profile,
            "proof_provider": proof_provider,
            "route": runtime_route,
            "route_identity": runtime_route.identity(),
            "source_security_policy": source_policy,
            # Which store this server READS from, so a write can be routed to the same place.
            # `recall_ingest` used to build a generation unconditionally, including on a server
            # serving the legacy `chunks` table, so an upload succeeded and then could not be
            # found. Recorded here rather than re-derived in the tool, because two readings of
            # `RECALL_ENV` are two chances to disagree, and the whole defect was a disagreement
            # about which store was in play.
            "generation_mode": generation_mode,
            "limiter": limiter,
            "translation_provider": translation_provider,
            "shadow_embedders": {},
            "shadow_embedder_lock": threading.Lock(),
            "health_probe": probe,
            "health_probes": tuple(health_probes),
            "control_plane": registry.control_plane if registry is not None else None,
            "tenant_count": len(registry.allowed_tenants) if registry is not None else 1,
            "tenant_probe_limit": readiness_tenant_probes,
            "max_tenants": max_tenants,
            "connection_budget": connection_budget,
            "active_generation": active_generation,
            "enterprise_readiness_ok": enterprise_readiness_ok,
            "secret_versions": runtime_secret_versions,
            "settings": runtime_settings,
            "trust_policy": runtime_settings.trust_policy,
            "retrieval_plan_resolver": retrieval_plan_resolver,
            "federation_config": federation_config,
            "federation_embedders": {},
            "federation_embedder_lock": threading.Lock(),
        }
        return runtime_state

    @asynccontextmanager
    async def _lifespan(_server: MCPServer) -> AsyncIterator[dict]:
        config = _resolve_startup_config()
        backend = _open_serving_backend(config)
        probe, health_probes = _check_schema_and_probes(config, backend)
        limiter, enterprise_readiness_ok = await _gate_readiness(config, backend, probe)
        store = backend.store
        registry = backend.registry
        clock_installed = backend.clock_installed
        source_policy = config.source_policy
        settings_token = None
        lite_refresher: LiteAutoRefresher | None = None
        rescue_builder: RescueBuilder | None = None
        try:
            runtime_state = _assemble_runtime_state(
                config, backend, probe, health_probes, limiter, enterprise_readiness_ok
            )
            # A lite store follows its memo folder on its own: a catch-up refresh now, and a
            # debounced check before each retrieval (`recall_mcp.lite_refresh`). None otherwise.
            # It applies the source policy with the context `recall_index` gets on a stdio server
            # (`_access_context` with no token), so both write the same rows (CCA SEC-001); the CLI
            # refresh uses the same one (`refresh_access_context`).
            lite_refresher = LiteAutoRefresher.from_env(
                store,
                backend.embedder,
                config.runtime_env,
                security_policy=source_policy,
                security_context=(
                    None
                    if source_policy is None
                    else refresh_access_context(str(getattr(store, "tenant", "")))
                ),
            )
            if lite_refresher is not None:
                lite_refresher.start()
            runtime_state["lite_refresher"] = lite_refresher
            # A PostgreSQL store whose rescue answers without a missing artifact (`skip`, a desktop
            # install) builds that artifact in the background after a search (`recall_mcp.rescue_builder`).
            rescue_builder = RescueBuilder.from_env(store, backend.embedder, config.runtime_env)
            runtime_state["rescue_builder"] = rescue_builder
            settings_token = activate_runtime_settings(config.runtime_settings)
            if health is not None:
                health.mark_started(runtime_state)
            yield runtime_state
        finally:
            if lite_refresher is not None:
                # Before the store closes: a refresh cut off by the close would record a
                # closed-database failure as the last refresh (CCA BUG-006).
                await _to_thread(lite_refresher.stop)
            if rescue_builder is not None:
                # Before the store closes, for the same reason.
                await _to_thread(rescue_builder.stop)
            if limiter is not None:
                await limiter.close()
            if store is not None:
                store.close()
            if registry is not None:
                registry.close()
            if health is not None:
                health.mark_stopped()
            if clock_installed:
                # Last: a refresh finishing during shutdown still writes the evaluation time.
                eval_clock.uninstall()
            if settings_token is not None:
                reset_runtime_settings(settings_token)

    return _lifespan
