"""The assistant surface of the MCP server: the held tool declarations and their views.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, cast

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations
from pydantic.json_schema import WithJsonSchema

from recall.observability import METRICS
from recall.lite import LiteStore
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
from recall_mcp.models import ForgetResult, WithdrawResult
from recall_mcp.threads import _to_thread
from recall_mcp.server_common import _ToolDeps, _parse_as_of


class _HeldTools:
    """Holds the named `@mcp.tool(...)` declarations back instead of registering them.

    The assistant surface serves four existing tools in another form. Holding their declarations
    keeps the registration functions untouched (a held tool's body stays callable, and what is not
    held passes straight through), and `register` publishes the replacement under the held
    declaration's annotations and description, so only the parameters and the answer change.
    """

    def __init__(self, mcp: MCPServer, names: frozenset[str]) -> None:
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

    def register(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        kwargs, declared = self.held[name]
        description = kwargs.get("description") or declared.__doc__
        return self._mcp.tool(**{**kwargs, "description": description})(fn)

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


def _register_assistant_tools(mcp: MCPServer, deps: _ToolDeps, held: _HeldTools) -> None:
    """The assistant surface: `recall_remember`, `recall_forget_memo`, and the compact forms.

    Registered only when RECALL_ASSISTANT_MEMOS names a memo folder on a lite store
    (`recall_mcp.assistant`). The compact `recall_search` and `recall_evidence` offer only the
    parameters an assistant uses; `recall_current_facts` and `recall_apply_fact` keep theirs. All
    six answer in `assistant_dumps` form, and their descriptions are the held declarations'.
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
    ) -> str:
        state = _state(ctx)
        store = await _require(SCOPE_WRITE, ctx)
        with METRICS.timer("recall_tool_latency_ms", tool="remember"):
            result = await _to_thread(
                lambda: assistant.remember(
                    cast(LiteStore, store),
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
        # `_require`, not `_require_mutation`: the difference is idempotency keys, which a lite
        # server (one local client over stdio) never receives, and the tool offers none.
        state = _state(ctx)
        store = await _require(SCOPE_FORGET, ctx)
        refresh = _policy(state, store)
        with METRICS.timer("recall_tool_latency_ms", tool="forget_memo"):
            result = await _to_thread(
                lambda: assistant.forget(
                    cast(LiteStore, store),
                    state["embedder"],
                    memo_ref(memo_id) or "",
                    security_policy=refresh["security_policy"],
                    # The caller's context for the erasure, as `recall_forget` passes it.
                    security_context=_access_context(state, store, purpose="erasure"),
                    refresh_context=refresh["security_context"],
                )
            )
        return assistant_dumps(result)

    if "recall_withdraw" in held.held:
        plain_withdraw = held.held["recall_withdraw"][1]

        async def recall_withdraw(
            sources: list[str], ctx: Context[dict, object], idempotency_key: str | None = None
        ) -> str:
            # A memo of the folder, named by its id (as hits name it here) or by its file, is
            # withdrawn as `recall_forget_memo` forgets one, file included. The plain tool matched
            # no id, and for a path it left the file, which the next refresh indexed again
            # (a live agent run, 2026-11-18). Every other source goes to it unchanged.
            memos = {source: memo for source in sources if (memo := assistant.folder_memo(source)) is not None}
            rest = [source for source in sources if source not in memos]
            done: list[tuple[str, WithdrawResult]] = []
            if memos:
                state = _state(ctx)
                store = await _require(SCOPE_FORGET, ctx)
                refresh = _policy(state, store)
                for source, memo in memos.items():
                    with METRICS.timer("recall_tool_latency_ms", tool="withdraw"):
                        # Awaited before the loop moves on, so the closure reads this `memo`.
                        result = await _to_thread(
                            lambda: assistant.withdraw(
                                cast(LiteStore, store),
                                state["embedder"],
                                memo,
                                security_policy=refresh["security_policy"],
                                security_context=_access_context(state, store, purpose="erasure"),
                                refresh_context=refresh["security_context"],
                            )
                        )
                    if result is None:
                        rest.append(str(assistant.path_of(memo)))  # no file: by the name it had
                    else:
                        done.append((source, result))
            if not done:
                return cast(str, await plain_withdraw(rest, ctx, idempotency_key))
            parts = [result for _, result in done]
            if rest:
                parts.append(WithdrawResult.model_validate_json(await plain_withdraw(rest, ctx, idempotency_key)))

            def total(field: str) -> int:
                values = [getattr(part, field) for part in parts]
                return -1 if -1 in values else sum(values)

            return WithdrawResult(
                chunks_removed=total("chunks_removed"),
                sources_withdrawn=[source for source, _ in done] + [s for part in parts[len(done):] for s in part.sources_withdrawn],
                sources_not_found=[s for part in parts[len(done):] for s in part.sources_not_found],
                message=" ".join(part.message for part in parts),
                outbox_events_scrubbed=total("outbox_events_scrubbed"),
                staged_files_removed=total("staged_files_removed"),
                facts_withdrawn=total("facts_withdrawn"),
                facts_redacted=total("facts_redacted"),
            ).model_dump_json(indent=2)

        held.register("recall_withdraw", recall_withdraw)

    if "recall_forget" in held.held:
        plain_forget = held.held["recall_forget"][1]

        async def recall_forget(
            sources: list[str], ctx: Context[dict, object], idempotency_key: str | None = None
        ) -> str:
            # A memo of the folder, named by its id or by its file, is forgotten as
            # `recall_forget_memo` forgets one, file included: the plain tool left the file, which
            # the next refresh indexed again, within one check while the rescue is on (CCA
            # STAKES-003, owner decision 2026-11-19, as #1354 did for withdraw). Every other
            # source goes to it unchanged.
            memos = {source: memo for source in sources if (memo := assistant.folder_memo(source)) is not None}
            rest = [source for source in sources if source not in memos]
            done: list[tuple[str, ForgetResult]] = []
            if memos:
                state = _state(ctx)
                store = await _require(SCOPE_FORGET, ctx)
                refresh = _policy(state, store)
                for source, memo in memos.items():
                    with METRICS.timer("recall_tool_latency_ms", tool="forget"):
                        # Awaited before the loop moves on, so the closure reads this `memo`.
                        result = await _to_thread(
                            lambda: assistant.forget_result(
                                cast(LiteStore, store),
                                state["embedder"],
                                memo,
                                security_policy=refresh["security_policy"],
                                security_context=_access_context(state, store, purpose="erasure"),
                                refresh_context=refresh["security_context"],
                            )
                        )
                    if result is None:
                        rest.append(str(assistant.path_of(memo)))  # no file: by the name it had
                    else:
                        done.append((source, result))
            if not done:
                return cast(str, await plain_forget(rest, ctx, idempotency_key))
            parts = [result for _, result in done]
            if rest:
                parts.append(ForgetResult.model_validate_json(await plain_forget(rest, ctx, idempotency_key)))

            def total(field: str) -> int:
                values = [getattr(part, field) for part in parts]
                return -1 if -1 in values else sum(values)

            return ForgetResult(
                chunks_removed=total("chunks_removed"),
                sources_removed=[source for source, _ in done] + [s for part in parts[len(done):] for s in part.sources_removed],
                sources_not_found=[s for part in parts[len(done):] for s in part.sources_not_found],
                message=" ".join(part.message for part in parts),
                outbox_events_scrubbed=total("outbox_events_scrubbed"),
                staged_files_removed=total("staged_files_removed"),
                facts_retracted=total("facts_retracted"),
                facts_redacted=total("facts_redacted"),
            ).model_dump_json(indent=2)

        held.register("recall_forget", recall_forget)

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
