"""The fact tools of the MCP server: current facts and stated facts.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import json

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations

from recall.declared_keys import subject_view
from recall_mcp.auth import SCOPE_FACT_WRITE, SCOPE_READ
from recall_mcp.provenance import apply_fact_memory, current_facts_memory
from recall_mcp.threads import _to_thread
from recall_mcp.server_common import (
    _META_REQUIRED_SCOPE,
    _ToolDeps,
    _mutation_fingerprint,
    _parse_as_of,
    _release_mutation_reservation,
    _trust_policy_for,
)


def _register_fact_tools(mcp: MCPServer, deps: _ToolDeps) -> None:
    _require = deps.require
    _require_mutation = deps.require_mutation
    _record_mutation_result = deps.record_mutation_result
    _state = deps.state
    _access_context = deps.access_context

    @mcp.tool(
        name="recall_current_facts",
        annotations=ToolAnnotations(
            title="Read current structured facts",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_current_facts(
        ctx: Context[dict, object], as_of: str | None = None, subject: str = "", predicate: str = ""
    ) -> str:
        """Return the tenant scoped current projection of the append only fact ledger.

        Facts a memo states in its `recall_graph` frontmatter appear here, with the memo in
        `sources`, once the store has recorded them; a lite store does so whenever it indexes.
        Values that disagree with no supersession between their memos are listed under
        `contested`, not `facts`: memory holds both and says neither is current.
        `served_state` is `degraded` when a check against the served memos failed, naming each in
        `served_lookups_failed`; nothing is settled then. After an `admission` or `content`
        failure stated facts may be missing; after a `presence` or `supersession` failure a
        listed fact may be retired or no longer served.

        With `subject`, only that subject's facts (matched ignoring case and spacing) and, with
        `predicate` too, only that predicate's; `similar_subjects` then names subjects whose names
        contain or are contained in it. Look a subject up before stating a fact about it, and
        repeat an existing subject and predicate exactly to write under the same key.
        """
        store = await _require(SCOPE_READ, ctx)
        instant = _parse_as_of(as_of)
        state = _state(ctx)

        def build() -> str:
            # The view and the serialisation are O(ledger) too, so they stay off the loop.
            result = current_facts_memory(
                store, as_of=instant, security_policy=state.get("source_security_policy"),
                access_context=_access_context(state, store),
            )
            if subject.strip():
                result = subject_view(result, subject, predicate.strip() or None)
            return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)

        return await _to_thread(build)

    @mcp.tool(
        name="recall_apply_fact",
        annotations=ToolAnnotations(
            title="Apply a structured fact",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        meta={_META_REQUIRED_SCOPE: SCOPE_FACT_WRITE},
    )
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
        """Apply one structured atomic fact through the deterministic provenance controller.

        The only evidence input is an opaque list of card identifiers returned by
        ``recall_evidence``. Trust verdicts, timestamps, ranks, generation identifiers, links,
        approval fields, and writer identity are server-owned and cannot be supplied here.
        Unsupported prose claims are refused or trigger one controller-generated fresh search.
        """
        state = _state(ctx)
        operation = "recall_apply_fact"
        fingerprint = _mutation_fingerprint(
            operation,
            {
                "namespace": namespace,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "evidence_card_ids": evidence_card_ids,
                "context": context,
                "valid_from": valid_from,
                "valid_until": valid_until,
            },
        )
        store, replay = await _require_mutation(
            SCOPE_FACT_WRITE,
            ctx,
            idempotency_key=request_id,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        if replay is not None:
            return replay
        assert store is not None
        token = get_access_token()
        writer = "stdio"
        if token is not None:
            claims = token.claims or {}
            writer = str(claims.get("principal", token.client_id or "authenticated"))
        try:
            result = await _to_thread(
                lambda: apply_fact_memory(
                    store,
                    state["embedder"],
                    claim={
                        "namespace": namespace,
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "context": context or {},
                        "valid_from": valid_from,
                        "valid_until": valid_until,
                    },
                    evidence_card_ids=evidence_card_ids,
                    request_id=request_id,
                    writer=writer,
                    policy=_trust_policy_for(state),
                    security_policy=state.get("source_security_policy"),
                    access_context=_access_context(state, store),
                )
            )
        except Exception:  # BROAD-CATCH: cleanup-only
            # No answer to keep: the erasure lock could not be taken (a busy lite lane, a
            # PostgreSQL lock timeout) or the call failed. The reservation is released so the same
            # request id can be sent again, which the ledger answers by its own record of the id
            # (a replay or a refusal), so a retry never appends twice (2026-11-04 audit,
            # DEPLOY-002).
            await _release_mutation_reservation(state, store.tenant, request_id)
            raise
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        await _record_mutation_result(
            state,
            store.tenant,
            request_id,
            payload,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        return payload
