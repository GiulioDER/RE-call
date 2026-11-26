"""The assistant surface of the MCP server: the held tool declarations and their views.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic.json_schema import WithJsonSchema

from recall.observability import METRICS
from recall.lite import LiteWriterBusy
from recall.lite.refresh import refresh_access_context
from recall_mcp.assistant import (
    FORGET_DESCRIPTION,
    REMEMBER_DESCRIPTION,
    STATED_FACT_SCHEMA,
    AssistantMemos,
    dumps as assistant_dumps,
    memo_ref,
)
from recall.store import PgVectorStore
from recall.declared_keys import subject_view
from recall_mcp.auth import SCOPE_FORGET, SCOPE_READ, SCOPE_WRITE, authorize
from recall_mcp.provenance import current_facts_memory
from recall_mcp.threads import _to_thread
from recall_mcp.tool_surface import ToolRegistrar
from recall_mcp.server_common import _ToolDeps, _parse_as_of


class _HeldTools:
    """Holds the named `@mcp.tool(...)` declarations back instead of registering them.

    The assistant surface serves five existing tools in another form. Holding their declarations
    keeps the registration functions untouched (a held tool's body stays callable, and what is not
    held passes straight through), and `register` publishes the replacement under the held
    declaration's annotations and description, so only the parameters and the answer change.
    """

    def __init__(self, mcp: ToolRegistrar, names: frozenset[str]) -> None:
        self._mcp = mcp
        self._names = names
        self.held: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}

    def tool(self, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        name = kwargs.get("name")
        if name not in self._names:
            return self._mcp.tool(*args, **kwargs)

        def hold(fn: Any) -> Any:
            self.held[name] = (kwargs, fn)
            return fn

        return hold

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        kwargs, declared = self.held[name]
        description = kwargs.get("description") or declared.__doc__
        self._mcp.tool(**{**kwargs, "description": description})(fn)

    def __getattr__(self, item: str) -> Any:
        if item == "_mcp":
            raise AttributeError("_HeldTools._mcp is unset; __init__ did not complete")
        return getattr(self._mcp, item)


def _caller_may_read(state: dict) -> bool:
    """Whether this call's caller holds SCOPE_READ, decided as `_require` decides it, debiting nothing.

    `recall_remember` needs only SCOPE_WRITE, while its receipt's `fact_key` reports values and memos
    `recall_current_facts` would serve under SCOPE_READ, so a writer without it gets its own key and
    no more (security review of PR 990). An unauthenticated stdio server has one local caller.
    """
    if state.get("stores") is None:
        return True
    token = get_access_token()
    if token is None:
        return False
    try:
        authorize(token.scopes, token.claims, SCOPE_READ)
    except PermissionError:
        return False
    return True


def _register_assistant_tools(mcp: ToolRegistrar, deps: _ToolDeps, held: _HeldTools) -> None:
    """The assistant surface: `recall_remember`, `recall_forget_memo`, and the compact forms.

    Registered only when RECALL_ASSISTANT_MEMOS names a memo folder (`recall_mcp.assistant`). The
    compact `recall_search` and `recall_evidence` offer only the
    parameters an assistant uses; `recall_current_facts` and `recall_apply_fact` keep theirs. All
    seven answer in `assistant_dumps` form except `recall_withdraw`, which answers `WithdrawResult`
    JSON as the plain tool does; their descriptions are the held declarations'.
    """
    assistant = deps.assistant
    if assistant is None:
        return
    _require = deps.require
    _state = deps.state
    _access_context = deps.access_context

    def _policy(state: dict, store: PgVectorStore) -> dict[str, Any]:
        # The source policy, with the context the server's own refresh uses (CCA SEC-001).
        policy = state.get("source_security_policy")
        context = None if policy is None else refresh_access_context(str(getattr(store, "tenant", "")))
        return {"security_policy": policy, "security_context": context}

    @mcp.tool(
        name="recall_remember",
        description=REMEMBER_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Remember one fact",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    async def recall_remember(
        text: str,
        ctx: Context[dict, object],
        kind: str = "note",
        valid_from: str = "",
        valid_until: str = "",
        supersedes: str = "",
        fact: Annotated[dict[str, Any] | None, WithJsonSchema(STATED_FACT_SCHEMA)] = None,
        request_id: str = "",
        expected_revision: str = "",
        include_receipt: bool = False,
        settle: bool = False,
    ) -> str:
        state = _state(ctx)
        store = await _require(SCOPE_WRITE, ctx)
        with METRICS.timer("recall_tool_latency_ms", tool="remember"):
            result = await _to_thread(
                lambda: assistant.remember(
                    store,
                    state["embedder"],
                    text,
                    kind or "note",
                    valid_from or None,
                    valid_until or None,
                    memo_ref(supersedes),
                    fact or None,
                    request_id=request_id or None,
                    expected_revision=expected_revision or None,
                    include_receipt=include_receipt,
                    access_context=_access_context(state, store),
                    read_facts=_caller_may_read(state),
                    settle=settle,
                    **_policy(state, store),
                )
            )
        return assistant_dumps(result)

    @mcp.tool(
        name="recall_forget_memo",
        description=FORGET_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Forget one memo",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_forget_memo(memo_id: str, ctx: Context[dict, object]) -> str:
        # This single-memo tool has no idempotency key.
        state = _state(ctx)
        store = await _require(SCOPE_FORGET, ctx)
        refresh = _policy(state, store)
        try:
            with METRICS.timer("recall_tool_latency_ms", tool="forget_memo"):
                result = await _to_thread(
                    lambda: assistant.forget(
                        store,
                        state["embedder"],
                        memo_ref(memo_id) or "",
                        security_policy=refresh["security_policy"],
                        # The caller's context for the erasure, as `recall_forget` passes it.
                        security_context=_access_context(state, store, purpose="erasure"),
                        refresh_context=refresh["security_context"],
                    )
                )

        except LiteWriterBusy as busy:
            raise ToolError(json.dumps({"error": "lite_writer_busy", "message": str(busy)})) from busy
        return assistant_dumps(result)

    _register_assistant_views(held, deps, assistant)


def _register_assistant_views(held: _HeldTools, deps: _ToolDeps, assistant: AssistantMemos) -> None:
    """The four existing tools in compact form, each a projection of the held tool's own answer."""
    _require = deps.require

    if "recall_search" in held.held:
        full_search = held.held["recall_search"][1]

        async def recall_search(query: str, ctx: Context[dict, object], k: int = 5) -> str:
            full = await full_search(query, ctx, k=k)
            return assistant_dumps(assistant.render_search(json.loads(full)))

        held.register("recall_search", recall_search)

    if "recall_evidence" in held.held:
        full_evidence = held.held["recall_evidence"][1]

        async def recall_evidence(
            query: str,
            ctx: Context[dict, object],
            k: int | None = None,
            max_items: int | None = None,
        ) -> str:
            full = await full_evidence(query, ctx, k=k, max_items=max_items)
            return assistant_dumps(assistant.render_evidence(json.loads(full)))

        held.register("recall_evidence", recall_evidence)

    if "recall_current_facts" in held.held:

        async def recall_current_facts(
            ctx: Context[dict, object], as_of: str | None = None, subject: str = "", predicate: str = ""
        ) -> str:
            # From the library, not the held tool's text: that one sorts every key, nested facts
            # included, and the compact form keeps the order the ledger writes.
            store = await _require(SCOPE_READ, ctx)
            instant = _parse_as_of(as_of)
            state = deps.state(ctx)

            def build() -> str:
                # The view and the serialisation are O(ledger) too, so they stay off the loop.
                result = current_facts_memory(
                    store, as_of=instant, security_policy=state.get("source_security_policy"),
                    access_context=deps.access_context(state, store),
                )
                if subject.strip():
                    result = subject_view(result, subject, predicate.strip() or None)
                return assistant_dumps(assistant.render_current_facts(result))

            return await _to_thread(build)

        held.register("recall_current_facts", recall_current_facts)

    if "recall_apply_fact" in held.held:
        full_apply = held.held["recall_apply_fact"][1]

        async def recall_apply_fact(
            namespace: str,
            subject: str,
            predicate: str,
            object: object,
            evidence_card_ids: list[str],
            request_id: str,
            ctx: Context[dict, object],
            context: dict[str, object] | None = None,
            valid_from: str | None = None,
            valid_until: str | None = None,
        ) -> str:
            full = await full_apply(
                namespace=namespace,
                subject=subject,
                predicate=predicate,
                object=object,
                evidence_card_ids=evidence_card_ids,
                request_id=request_id,
                ctx=ctx,
                context=context,
                valid_from=valid_from,
                valid_until=valid_until,
            )
            return assistant_dumps(assistant.render_apply_fact(json.loads(full)))

        held.register("recall_apply_fact", recall_apply_fact)
