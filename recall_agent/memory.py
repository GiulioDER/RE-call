"""`RecallAgentMemory`: RE-call as in-process memory for a Claude Agent SDK application.

This is the supported way to consume RE-call from an Agent SDK app without an MCP transport: the
same `recall_mcp.service` functions the MCP server calls, wrapped as in-process SDK tools, with
the same tool names, the same model-facing descriptions, and byte-identical result rendering, so
everything written against the MCP surface (skills, prompts, measured search-rate results)
transfers unchanged.

Two properties are load-bearing and deliberate:

- **Trust semantics survive the wrapping.** The trust policy is applied per call and a
  `TrustRefusal` is rendered as its wire form rather than raised through the SDK, so an outage
  and an empty result never collapse into the same shape (see `recall_agent.rendering`).
- **The sync service layer is serialised, not parallelised.** Every store-touching call runs in a
  worker thread under one `asyncio.Lock` — the same arrangement `recall_interop`'s benchmark
  backend uses — because agent tool traffic is bursty but effectively concurrency-1, and the
  simple correctness argument is worth more here than idle parallelism.

This module is importable without `claude_agent_sdk`; only the methods that produce SDK objects
(`sdk_mcp_server`, `session_start_hook`, `options`) touch it, via `recall_agent._sdk`.
"""
from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.generation_store import GenerationStore
from recall.observability import get_logger
from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, PgVectorStore
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall_agent.rendering import render_refusal, render_result, render_tool_error
from recall_mcp.factories import make_embedder
from recall_mcp.service import (
    evidence_memory,
    forget_memory,
    index_memory,
    memory_stats,
    search_memory,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; the SDK is an optional extra
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, McpSdkServerConfig

#: The MCP server's own default, kept identical so pointing both surfaces at one corpus needs no
#: configuration.
DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"

_log = get_logger("agent.memory")


def resolve_dsn(env: Mapping[str, str], explicit: str | None = None) -> str:
    """Explicit wins, then `RECALL_SERVING_DSN`, then `RECALL_DSN`, then the local default."""
    if explicit:
        return explicit
    return env.get("RECALL_SERVING_DSN") or env.get("RECALL_DSN") or DEFAULT_DSN


def resolve_embedder(env: Mapping[str, str], explicit: Embedder | str | None = None) -> Embedder:
    """An instance passes through; a name (explicit, else `RECALL_EMBEDDER`) goes to the factory.

    ⚠️ Constructing an embedder is not cheap and not always local: fastembed loads an ONNX model
    and voyage makes a blocking probe request. Callers on an event loop must reach this through
    `RecallAgentMemory`, which defers it to a worker thread, rather than calling it directly.
    """
    if isinstance(explicit, Embedder):
        return explicit
    name = explicit or env.get("RECALL_EMBEDDER", "fastembed")
    return make_embedder(str(name), dict(env))


class RecallAgentMemory:
    """One tenant's RE-call memory, packaged for a Claude Agent SDK application.

    Construct it, then hand its pieces to `ClaudeAgentOptions` — or let `options()` assemble
    them:

        memory = RecallAgentMemory(dsn=..., tenant=...)
        options = memory.options(model="claude-sonnet-5")

    Lifecycle: the store is created lazily on first use and owned by this object unless one was
    injected via `store=`; `close()` (or the context manager) releases only what it owns.
    `use_generation_store=True` serves the generation-bound table — a constructor decision,
    deliberately not an environment variable, because which store class answers is something the
    host application must be able to see at the construction site.
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        tenant: str | None = None,
        table: str | None = None,
        embedder: Embedder | str | None = None,
        policy: TrustPolicy | None = None,
        calibration: Calibration | None = None,
        use_generation_store: bool = False,
        pool_size: int = 2,
        server_name: str = "recall",
        store: PgVectorStore | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if store is not None and (dsn is not None or table is not None or use_generation_store):
            raise ValueError(
                "pass either an existing store or (dsn/table/use_generation_store), not both: "
                "an injected store already decided all three"
            )
        # ONE mapping decides every environment-derived setting, including the embedder resolved
        # later. `from_env(env=...)` passes its mapping straight through, so a caller who supplies
        # one is never silently topped up from the host process: a test or a multi-tenant host
        # would otherwise get the process's RECALL_TENANT or RECALL_EMBED_PROFILE mixed into a
        # configuration it thought it had stated in full.
        env = dict(os.environ if env is None else env)
        self._env = env
        self._dsn = resolve_dsn(env, dsn)
        # Held unresolved until first use: constructing an embedder loads an ONNX model (fastembed)
        # or makes a blocking probe request (voyage), and a host naturally builds this object on
        # its event loop, right before assembling ClaudeAgentOptions. Every other blocking call in
        # this class is moved off the loop by `_call`; resolving here would make the constructor
        # the one exception, stalling the loop for the whole model load.
        self._embedder_spec = embedder
        self._embedder: Embedder | None = embedder if isinstance(embedder, Embedder) else None
        self._policy = policy if policy is not None else TrustPolicy.from_env(dict(env))
        self._calibration = calibration
        # Both mirror `recall_mcp.server`: RECALL_TENANT, and RECALL_TABLE with
        # empty-means-unset. The server additionally REFUSES a non-identifier table at import
        # and refuses RECALL_TABLE together with generation mode; here a bad value surfaces
        # later, from `PgVectorStore`'s own validation, and `table` is simply unused when
        # `use_generation_store` is set. Parity on the values, not yet on the refusals.
        self._tenant = (
            tenant if tenant is not None else env.get("RECALL_TENANT") or DEFAULT_TENANT
        )
        self._table = (
            table
            if table is not None
            else env.get("RECALL_TABLE", "").strip() or DEFAULT_TABLE
        )
        self._use_generation_store = use_generation_store
        self._pool_size = pool_size
        self._server_name = server_name
        self._store: PgVectorStore | None = store
        self._owns_store = store is None
        self._closed = False
        self._lock = asyncio.Lock()
        # A THREAD lock, not the asyncio one: the `embedder` property is reachable from a host
        # thread while a tool call is resolving the same embedder in a worker, and two fastembed
        # model loads for one object is a real cost even though either result would be correct.
        self._embedder_lock = threading.Lock()

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, **overrides: Any
    ) -> "RecallAgentMemory":
        """Construct from an explicit environment mapping (default `os.environ`); overrides win.

        The mapping is handed to the constructor, which derives every environment-backed setting
        from it and from nothing else.
        """
        values = dict(os.environ if env is None else env)
        overrides.setdefault("env", values)
        if "dsn" not in overrides and "store" not in overrides:
            # Not when a store is injected: the constructor refuses `store` together with `dsn`,
            # so defaulting one here made `from_env(store=...)` raise unconditionally, which is
            # the same shape as the embedder defect fixed in e7ae27c1.
            overrides["dsn"] = resolve_dsn(values)
        return cls(**overrides)

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def embedder(self) -> Embedder:
        """The resolved embedder, constructing it on first access if it was named rather than given.

        ⚠️ On an event loop, prefer letting a tool call resolve it in the worker thread; reading
        this property directly pays the model load on the calling thread.
        """
        return self._embedder_or_create()

    def close(self) -> None:
        """Release the store this object owns. Sticky: a closed memory refuses to reopen.

        The stickiness matters because `_store is None` is also the not-yet-created state, so
        without the flag a tool call arriving after close (a SessionStart hook firing during
        teardown, say) would silently open a fresh pool that nothing is left to close.
        `PgVectorStore.close` is deliberately sticky for the same reason; this mirrors it.

        Not synchronised with in-flight calls: it must stay callable from `__exit__`, which is
        sync, while `_call` holds an async lock. Close a memory when its session is over, not
        underneath one.
        """
        if not self._owns_store:
            # An injected store is the caller's to close, so close() releases nothing and must
            # not disable this object either: refusing afterwards would brick a memory whose
            # store is still perfectly usable by its owner.
            return
        self._closed = True
        if self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self) -> "RecallAgentMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- SDK-producing surface (the only paths that import claude_agent_sdk) ------------------

    def sdk_mcp_server(self, *, write_tools: bool = False) -> "McpSdkServerConfig":
        """An in-process MCP server carrying the read tools, plus writes when opted into."""
        from recall_agent import _sdk

        return _sdk.build_sdk_mcp_server(self, write_tools=write_tools)

    def session_start_hook(self) -> "HookMatcher":
        """A SessionStart hook injecting the memory digest (fail-open, like the CLI hook)."""
        from recall_agent import _sdk

        return _sdk.build_session_start_matcher(self)

    def allowed_tools(self, *, write_tools: bool = False) -> list[str]:
        """Fully qualified tool names for `ClaudeAgentOptions.allowed_tools`."""
        names = ["recall_search", "recall_evidence"]
        if write_tools:
            names += ["recall_index", "recall_forget"]
        return [f"mcp__{self._server_name}__{name}" for name in names]

    def options(self, *, write_tools: bool = False, **overrides: Any) -> "ClaudeAgentOptions":
        """A ready `ClaudeAgentOptions`: server, allowed tools, and hook, merged with overrides.

        Merge, never replace: an overrides `mcp_servers` is dict-merged (a collision on this
        memory's own server name raises), `allowed_tools` is extended, and `hooks` lists are
        appended per event. Everything else passes through untouched.
        """
        from recall_agent import _sdk

        return _sdk.build_options(self, write_tools=write_tools, overrides=overrides)

    # -- store plumbing ------------------------------------------------------------------------

    def _embedder_or_create(self) -> Embedder:
        """Resolve the named embedder once, on whichever thread first needs it.

        Resolved against `self._env`, the mapping this object was constructed with, because
        `make_embedder` reads far more than the name from it (`RECALL_EMBED_PROFILE`, model
        digests, API keys), and a caller who passed an explicit mapping must not have the host
        process's variables mixed into the result.
        """
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = resolve_embedder(self._env, self._embedder_spec)
            return self._embedder

    def _make_store(self) -> PgVectorStore:
        dim = self._embedder_or_create().dim
        if self._use_generation_store:
            return GenerationStore(
                self._dsn, dim, tenant=self._tenant, pool_size=self._pool_size
            )
        return PgVectorStore(
            self._dsn,
            dim,
            table=self._table,
            tenant=self._tenant,
            pool_size=self._pool_size,
        )

    def _store_or_create(self) -> PgVectorStore:
        """Runs inside the worker thread, under the lock, so creation cannot race."""
        if self._closed:
            raise RuntimeError("this RecallAgentMemory is closed; construct a new one")
        if self._store is None:
            store = self._make_store()
            try:
                store.check_schema()
            except BaseException:
                # The pool opens eagerly in the constructor, so a rejected schema would otherwise
                # drop a store holding live connections and a maintenance thread, and the next
                # tool call would build another (`_store` is assigned only on success, and a
                # schema mismatch is persistent). `recall_mcp.server` closes on this same failure
                # for the same reason.
                #
                # `BaseException`, where that precedent catches `Exception`: this runs in a worker
                # thread a cancelled tool call can abandon, and a pool leaked on cancellation is
                # leaked just as thoroughly as one leaked on error. A deliberate deviation from
                # the precedent rather than a copy of it.
                store.close()
                raise
            self._store = store
        return self._store

    async def _call(self, fn: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(fn)

    # -- tool implementations (plain async callables; the SDK wrapping lives in _sdk) ----------

    def _retrieval_kwargs(self, args: dict[str, Any]) -> dict[str, Any]:
        """Forward only the arguments the caller actually supplied.

        Nothing here restates the service layer's defaults (`k`, `related_relation`,
        `related_max_items`). Copying them would make `recall_mcp.service` and this wrapper two
        owners of one default, and the wrapper would keep passing yesterday's value after the
        service changed its mind. An absent key is simply not forwarded.
        """
        kwargs: dict[str, Any] = {"calibration": self._calibration, "policy": self._policy}
        if args.get("source") is not None:
            kwargs["source"] = str(args["source"])
        for name in ("k", "related_max_items"):
            if args.get(name) is not None:
                kwargs[name] = int(args[name])
        for name in ("explain", "include_related"):
            if args.get(name) is not None:
                kwargs[name] = bool(args[name])
        if args.get("related_relation") is not None:
            kwargs["related_relation"] = str(args["related_relation"])
        return kwargs

    async def _retrieve(self, run: Any) -> dict[str, Any]:
        """Run a retrieval and render it, turning a refusal into its wire form rather than an error.

        Scoped to `TrustRefusal` alone. What the service layer raises (a production refusal, a
        path outside the index root, a budget cap) is its own signal and must not be flattened
        into "invalid tool arguments"; argument coercion is guarded at each call site instead,
        before the call, where the error really is the caller's.
        """
        try:
            result = await self._call(run)
        except TrustRefusal as refusal:
            return render_refusal(refusal)
        return render_result(result)

    async def _recall_search(self, args: dict[str, Any]) -> dict[str, Any]:
        # Coerced before the call: model-supplied arguments are untrusted, and `int(None)` or a
        # missing `query` would otherwise leave this method as a raw exception through the SDK's
        # tool channel, indistinguishable from the transport failing.
        try:
            query = str(args["query"])
            kwargs = self._retrieval_kwargs(args)
        except (TypeError, ValueError, KeyError) as error:
            return render_tool_error(f"{type(error).__name__}: {error}")

        def run() -> Any:
            return search_memory(
                self._store_or_create(), self._embedder_or_create(), query, **kwargs
            )

        return await self._retrieve(run)

    async def _recall_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            query = str(args["query"])
            kwargs = self._retrieval_kwargs(args)
            if args.get("max_items") is not None:
                kwargs["max_items"] = int(args["max_items"])
        except (TypeError, ValueError, KeyError) as error:
            return render_tool_error(f"{type(error).__name__}: {error}")

        def run() -> Any:
            return evidence_memory(
                self._store_or_create(), self._embedder_or_create(), query, **kwargs
            )

        return await self._retrieve(run)

    async def _recall_index(self, args: dict[str, Any]) -> dict[str, Any]:
        # No `glob`: see the INDEX_SCHEMA comment in `recall_agent._sdk`. A model-chosen glob
        # switches off the exclusion list that keeps `tokens.json` out of the corpus, so the
        # argument is not read here even if a caller sends one the schema does not advertise.
        try:
            path = str(args["path"])
        except (TypeError, ValueError, KeyError) as error:
            return render_tool_error(f"{type(error).__name__}: {error}")

        def run() -> Any:
            return index_memory(self._store_or_create(), self._embedder_or_create(), path)

        return render_result(await self._call(run))

    async def _recall_forget(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = args["sources"]
        except KeyError as error:
            return render_tool_error(f"KeyError: {error}")
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            # A bare string is iterable, so `[str(x) for x in "notes.md"]` would ask to erase
            # eight one-character sources: the requested erasure silently does nothing while a
            # one-character source elsewhere in the tenant is erased instead. On the
            # right-to-erasure path that must be a refusal, not a shrug.
            return render_tool_error("sources must be a list of source identifiers")
        sources = [str(item) for item in raw]

        def run() -> Any:
            return forget_memory(self._store_or_create(), sources)

        return render_result(await self._call(run))

    async def _session_start(
        self, input_data: Any, tool_use_id: Any, context: Any
    ) -> dict[str, Any]:
        """Digest injection, fail-open: a hook must never be the reason a session does not start.

        Fail-open, but never fail-SILENT. A bare `return {}` made every reason for a missing
        digest look identical: an empty corpus, an unreachable database, and a schema the serving
        checkout cannot read all produced the same nothing. A persistent failure here (a schema
        mismatch is persistent) then recurs once per session with no trace anywhere, which is how
        it becomes expensive to diagnose later rather than obvious now. The session still starts.
        """
        try:
            stats = await self._call(lambda: memory_stats(self._store_or_create()))
        except Exception:
            _log.warning(
                "RE-call memory digest skipped: memory is unavailable for tenant %r",
                self._tenant,
                exc_info=True,
            )
            return {}
        if not stats.chunks:
            return {}
        text = (
            f"RE-call memory is available: {stats.chunks} indexed chunks. "
            "Call `recall_search` before proposing an idea, forming a hypothesis, or repeating "
            "past work, and treat an `abstained: true` result as 'no supported answer' rather "
            "than as an empty one."
        )
        if stats.stale:
            text += " The newest memory is older than the freshness window."
        return {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": text,
            }
        }
