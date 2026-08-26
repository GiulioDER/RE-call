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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.generation_store import GenerationStore
from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, PgVectorStore
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall_agent.rendering import render_refusal, render_result
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


def resolve_dsn(env: Mapping[str, str], explicit: str | None = None) -> str:
    """Explicit wins, then `RECALL_SERVING_DSN`, then `RECALL_DSN`, then the local default."""
    if explicit:
        return explicit
    return env.get("RECALL_SERVING_DSN") or env.get("RECALL_DSN") or DEFAULT_DSN


def resolve_embedder(env: Mapping[str, str], explicit: Embedder | str | None = None) -> Embedder:
    """An instance passes through; a name (explicit, else `RECALL_EMBEDDER`) goes to the factory."""
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
    ) -> None:
        if store is not None and (dsn is not None or table is not None or use_generation_store):
            raise ValueError(
                "pass either an existing store or (dsn/table/use_generation_store), not both: "
                "an injected store already decided all three"
            )
        env = os.environ
        self._dsn = resolve_dsn(env, dsn)
        self._embedder = resolve_embedder(env, embedder)
        self._policy = policy if policy is not None else TrustPolicy.from_env(dict(env))
        self._calibration = calibration
        self._tenant = tenant if tenant is not None else DEFAULT_TENANT
        self._table = table if table is not None else DEFAULT_TABLE
        self._use_generation_store = use_generation_store
        self._pool_size = pool_size
        self._server_name = server_name
        self._store: PgVectorStore | None = store
        self._owns_store = store is None
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, **overrides: Any
    ) -> "RecallAgentMemory":
        """Resolve dsn/embedder/policy from `env` (default `os.environ`); overrides win."""
        values = dict(os.environ if env is None else env)
        if "dsn" not in overrides:
            overrides["dsn"] = resolve_dsn(values)
        if "embedder" not in overrides:
            # Guarded, not setdefault: resolving an embedder can load a model, and a caller who
            # supplied one must not pay for a second.
            overrides["embedder"] = resolve_embedder(values)
        if "policy" not in overrides:
            overrides["policy"] = TrustPolicy.from_env(values)
        return cls(**overrides)

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def close(self) -> None:
        if self._owns_store and self._store is not None:
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

    def _make_store(self) -> PgVectorStore:
        if self._use_generation_store:
            return GenerationStore(
                self._dsn, self._embedder.dim, tenant=self._tenant, pool_size=self._pool_size
            )
        return PgVectorStore(
            self._dsn,
            self._embedder.dim,
            table=self._table,
            tenant=self._tenant,
            pool_size=self._pool_size,
        )

    def _store_or_create(self) -> PgVectorStore:
        """Runs inside the worker thread, under the lock, so creation cannot race."""
        if self._store is None:
            store = self._make_store()
            store.check_schema()
            self._store = store
        return self._store

    async def _call(self, fn: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(fn)

    # -- tool implementations (plain async callables; the SDK wrapping lives in _sdk) ----------

    async def _recall_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        source = None if args.get("source") is None else str(args["source"])
        k = int(args.get("k", 5))
        explain = bool(args.get("explain", False))
        include_related = bool(args.get("include_related", False))
        related_relation = str(args.get("related_relation", "source"))
        related_max_items = int(args.get("related_max_items", 3))

        def run() -> Any:
            return search_memory(
                self._store_or_create(),
                self._embedder,
                query,
                source=source,
                k=k,
                calibration=self._calibration,
                policy=self._policy,
                explain=explain,
                include_related=include_related,
                related_relation=related_relation,
                related_max_items=related_max_items,
            )

        try:
            result = await self._call(run)
        except TrustRefusal as refusal:
            return render_refusal(refusal)
        return render_result(result)

    async def _recall_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        source = None if args.get("source") is None else str(args["source"])
        k = int(args.get("k", 5))
        max_items = None if args.get("max_items") is None else int(args["max_items"])
        explain = bool(args.get("explain", False))
        include_related = bool(args.get("include_related", False))
        related_relation = str(args.get("related_relation", "source"))
        related_max_items = int(args.get("related_max_items", 3))

        def run() -> Any:
            return evidence_memory(
                self._store_or_create(),
                self._embedder,
                query,
                source=source,
                k=k,
                max_items=max_items,
                calibration=self._calibration,
                policy=self._policy,
                explain=explain,
                include_related=include_related,
                related_relation=related_relation,
                related_max_items=related_max_items,
            )

        try:
            result = await self._call(run)
        except TrustRefusal as refusal:
            return render_refusal(refusal)
        return render_result(result)

    async def _recall_index(self, args: dict[str, Any]) -> dict[str, Any]:
        path = str(args["path"])
        glob = None if args.get("glob") is None else str(args["glob"])

        def run() -> Any:
            return index_memory(self._store_or_create(), self._embedder, path, glob=glob)

        return render_result(await self._call(run))

    async def _recall_forget(self, args: dict[str, Any]) -> dict[str, Any]:
        sources = [str(item) for item in args["sources"]]

        def run() -> Any:
            return forget_memory(self._store_or_create(), sources)

        return render_result(await self._call(run))

    async def _session_start(
        self, input_data: Any, tool_use_id: Any, context: Any
    ) -> dict[str, Any]:
        """Digest injection, fail-open: a hook must never be the reason a session does not start."""
        try:
            stats = await self._call(lambda: memory_stats(self._store_or_create()))
        except Exception:
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
