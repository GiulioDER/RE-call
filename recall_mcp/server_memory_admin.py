"""The memory administration tools of the MCP server.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import PurePath

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from recall.backends import write_session
from recall.embedding_core import profile_lineage_mismatch
from recall.embeddings import FastEmbedEmbedder
from recall.errors import IdempotencyConflict
from recall.generation_store import GenerationStore
from recall.idempotency_receipts import check_result_size
from recall.lite import LiteStore, LiteWriterBusy
from recall.lite.refresh import client_reason, refresh_access_context
from recall.lite.refresh import read_state as read_refresh_state
from recall.observability import METRICS
from recall_mcp.assistant import WITHDRAW_DESCRIPTION, memo_ref
from recall_mcp.auth import SCOPE_FORGET, SCOPE_READ
from recall_mcp.lifecycle import (
    ErasurePreflightRefused,
    forget_memory,
    withdraw_memory,
    memory_inventory,
    memory_stats,
    resolve_served_reference,
)
from recall_mcp.models import ForgetResult, WithdrawResult
from recall_mcp.stores import StoreRegistry
from recall_mcp.threads import _to_thread
from recall_mcp.tool_surface import ToolRegistrar
from recall_mcp.server_common import _ToolDeps, _mutation_fingerprint, _release_mutation_reservation


def _register_memory_admin_tools(mcp: ToolRegistrar, deps: _ToolDeps) -> None:
    _require = deps.require
    _require_mutation = deps.require_mutation
    _record_mutation_result = deps.record_mutation_result
    _state = deps.state
    _current_tenant = deps.current_tenant
    _access_context = deps.access_context

    async def _erase(
        ctx: Context[dict, object],
        sources: list[str],
        idempotency_key: str | None,
        *,
        operation: str,
        tool: str,
        eraser: Callable[..., ForgetResult | WithdrawResult],
    ) -> str:
        """The body `recall_forget` and `recall_withdraw` share; they differ only in these three."""
        state = _state(ctx)
        assistant = deps.assistant
        arguments: dict[str, object] = {"sources": sources}
        if assistant is not None and idempotency_key:
            # Bind the scope for possible assistant aliases using syntax only. A receipt replay
            # must not resolve caller-selected filesystem paths. Names that cannot be aliases
            # keep their plain receipt identity.
            for source in sources:
                ref = memo_ref(source)
                if not ref:
                    continue
                path = PurePath(ref)
                if assistant.path_of(path.name) is None:
                    continue
                arguments["assistant_folder"] = str(assistant.folder)
                if len(path.parts) > 1 and not path.is_absolute():
                    arguments["assistant_index_root"] = str(assistant.index_root)
        fingerprint = _mutation_fingerprint(operation, arguments)
        store, replay = await _require_mutation(
            SCOPE_FORGET,
            ctx,
            idempotency_key=idempotency_key,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        if replay is not None:
            return replay
        assert store is not None
        registry: StoreRegistry | None = state.get("stores")
        tenant = _current_tenant(state)
        try:
            shadow = (
                # get_shadow holds the registry lock across a control-plane read (PERF-002).
                await _to_thread(lambda: registry.get_shadow(tenant))
                if registry is not None and tenant is not None
                else None
            )
        except Exception as exc:  # noqa: BLE001  # BROAD-CATCH: cleanup-only; release the unused reservation
            try:
                await _release_mutation_reservation(state, store.tenant, idempotency_key)
            except Exception as cleanup_error:  # noqa: BLE001  # BROAD-CATCH: cleanup-only; preserve the preflight error
                exc.add_note(f"could not release the erasure reservation ({type(cleanup_error).__name__})")
            raise
        control = registry.control_plane if registry is not None else None
        try:
            security_context = _access_context(state, store, purpose="erasure")
        except PermissionError:
            # An invalid clearance claim is refused before anything is read or deleted.
            await _release_mutation_reservation(state, store.tenant, idempotency_key)
            raise
        policy = state.get("source_security_policy")

        def erase_sources(names: list[str]) -> ForgetResult | WithdrawResult:
            return eraser(
                store,
                names,
                shadow,
                control,
                security_policy=policy,
                security_context=security_context,
                # An authenticated server (a store registry) keeps host paths in its log.
                name_rescue_roots=registry is None,
            )

        def folder_memos() -> dict[str, str]:
            if assistant is None:
                return {}
            return {source: memo for source in sources if (memo := assistant.folder_memo(source)) is not None}

        def erase_request(memos: dict[str, str]) -> str:
            if not memos or assistant is None:
                return erase_sources(sources).model_dump_json(indent=2)

            # Folder memos must lose their files too. Keep the requested names in the response.
            rest = [source for source in sources if source not in memos]
            done = []
            parts = []
            erase_memo = assistant.forget_result if operation == "recall_forget" else assistant.withdraw
            refresh_context = refresh_access_context(store.tenant) if policy is not None else None
            for source, memo in memos.items():
                result = erase_memo(
                    store,
                    state["embedder"],
                    memo,
                    security_policy=policy,
                    security_context=security_context,
                    refresh_context=refresh_context,
                )
                if result is None:
                    rest.append(str(assistant.path_of(memo)))
                else:
                    done.append(source)
                    parts.append(result)

            plain = erase_sources(rest) if rest or not parts else None
            if not parts:
                assert plain is not None
                return plain.model_dump_json(indent=2)
            if plain is not None:
                parts.append(plain)

            def total(field: str) -> int:
                values = [getattr(part, field) for part in parts]
                return -1 if -1 in values else sum(values)

            removed = "sources_removed" if operation == "recall_forget" else "sources_withdrawn"
            facts = "facts_retracted" if operation == "recall_forget" else "facts_withdrawn"
            values = {
                removed: done + (getattr(plain, removed) if plain is not None else []),
                "sources_not_found": plain.sources_not_found if plain is not None else [],
                "message": " ".join(part.message for part in parts),
                **{field: total(field) for field in (
                    "chunks_removed", "outbox_events_scrubbed", "staged_files_removed", facts, "facts_redacted",
                )},
            }
            return parts[0].model_copy(update=values).model_dump_json(indent=2)

        def keyed_lite_erasure(lite_store: LiteStore, key: str) -> str:
            # Cover separate clients/processes sharing this file, not just callers on one server.
            # The initial authorizer could have looked up the receipt before another call won
            # the lane. Check again and store the response before releasing it.
            with write_session(lite_store, "forget"):
                try:
                    existing = lite_store.get_operation_receipt(
                        key, operation=operation, request_fingerprint=fingerprint
                    )
                except IdempotencyConflict as conflict:
                    raise ToolError(
                        json.dumps({"error": "idempotency_conflict", "message": str(conflict)}, sort_keys=True)
                    ) from conflict
                if existing is not None:
                    return existing
                memos = folder_memos()
                result = erase_request(memos)
                if memos:
                    try:
                        check_result_size(result)
                    except ValueError:
                        # The effects already ran. Keep a bounded response under the same key
                        # so a client can reconcile instead of applying the erasure again.
                        result = json.dumps({
                            "status": "reconciliation_required",
                            "operation": operation,
                            "request_fingerprint": fingerprint,
                            "message": "the erasure result exceeds the replay limit; inspect the requested sources",
                        }, sort_keys=True)
                lite_store.record_operation_receipt(
                    key,
                    result,
                    operation=operation,
                    request_fingerprint=fingerprint,
                    # Only the confirmed ordinary route may also replay on a server with no
                    # assistant. A different assistant scope still presents its own scoped hash.
                    ordinary_erasure_fingerprint=(
                        _mutation_fingerprint(operation, {"sources": sources})
                        if not memos and "assistant_folder" in arguments else None
                    ),
                )
                return result

        try:
            with METRICS.timer("recall_tool_latency_ms", tool=tool):
                if registry is None and isinstance(store, LiteStore) and idempotency_key:
                    lite_store = store
                    key = idempotency_key
                    return await _to_thread(lambda: keyed_lite_erasure(lite_store, key))
                payload = await _to_thread(lambda: erase_request(folder_memos()))
        except (ErasurePreflightRefused, LiteWriterBusy):
            # Refused before the first write (a malformed, ambiguous or unauthorized request, or
            # the lite store's write lane busy), so nothing was deleted and a retry may run.
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
        name="recall_forget",
        annotations=ToolAnnotations(
            title="Forget agent memory",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_forget(
        sources: list[str], ctx: Context[dict, object], idempotency_key: str | None = None
    ) -> str:
        """Permanently delete indexed memory for the given source(s). IRREVERSIBLE.

        This is the right-to-erasure path: use it to make the agent forget a memory that should
        no longer be recalled (e.g. it indexed something it should not have retained). Deletion
        is scoped to the caller's tenant (this server's tenant on stdio, the token's tenant on an
        authenticated server) and cannot reach another tenant's memory. A source
        that does not exist is reported in `sources_not_found` rather than silently counted as
        "removed" — check that list before assuming a name was actually forgotten. A fact
        applied with `recall_apply_fact` whose only evidence was in these sources is retracted
        with them, so `recall_current_facts` stops listing it, and its value is erased: the fact
        ledger and the evidence cards citing these sources are rewritten in place under fresh ids,
        keeping the shape of the history but not the value, and the receipts naming them are deleted. A
        fact with evidence in a memo that survives stays. On a lite store or the legacy table a
        memo whose file was deleted from disk, and pruned by the next index, counts as surviving:
        its fact is hidden until the file comes back, unless this call names that memo too (a
        memo pruned before 0.15.0 left no record and counts as forgotten). A name that matches nothing still
        completes the erasure of facts it supplied, so re-run a forget that reported a failure.
        Whatever a forgotten memory superseded stays retired; for a memory that was a mistake,
        and should give back what it replaced, use `recall_withdraw` instead.

        Args:
            sources: one or more source values to forget, exactly as they appear in
                `recall_search` hits (the `source` field).

        Returns:
            JSON of {chunks_removed, sources_removed, sources_not_found, message,
            facts_retracted, facts_redacted}.
        """
        return await _erase(
            ctx, sources, idempotency_key, operation="recall_forget", tool="forget", eraser=forget_memory
        )

    @mcp.tool(
        name="recall_withdraw",
        # Beside the assistant surface, search names a hit by its memo_id and shows no source.
        description=None if deps.assistant is None else WITHDRAW_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Withdraw a mistaken memory",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_withdraw(
        sources: list[str], ctx: Context[dict, object], idempotency_key: str | None = None
    ) -> str:
        """Withdraw memory that was a MISTAKE, and give back what it replaced. IRREVERSIBLE.

        Use this, not `recall_forget`, when a memory should never have been believed: a wrong
        correction, a mistaken "this replaces that". It erases the memory exactly as
        `recall_forget` does, and then the two differ in one thing. `recall_forget` keeps retired
        whatever the forgotten memory superseded (forgetting "moved to Rome" does not make Paris
        current again); `recall_withdraw` voids its `supersedes:` claims, so what it replaced is
        current again, and whatever superseded IT now supersedes what it replaced instead (in
        A -> B -> C, withdrawing B leaves A retired by C). In the fact ledger, each fact it alone
        supported is withdrawn, which also undoes the supersession that fact made, and the
        values it supplied are erased as `recall_forget` erases them. A memory forgotten earlier
        can be withdrawn by the name it had. Scoped to the caller's tenant.

        Args:
            sources: one or more source values to withdraw, exactly as they appear in
                `recall_search` hits (the `source` field).

        Returns:
            JSON of {chunks_removed, sources_withdrawn, sources_not_found, message,
            facts_withdrawn, facts_redacted}.
        """
        return await _erase(
            ctx, sources, idempotency_key, operation="recall_withdraw", tool="withdraw", eraser=withdraw_memory
        )

    @mcp.tool(
        name="recall_inventory",
        annotations=ToolAnnotations(
            title="List what this tenant holds",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_inventory(ctx: Context[dict, object], limit: int = 5000) -> str:
        """List every source in memory with its raw content digest for client-side sync."""
        state = _state(ctx)
        store = await _require(SCOPE_READ, ctx)
        return await _to_thread(
            lambda: memory_inventory(
                store,
                limit=limit,
                security_policy=state.get("source_security_policy"),
                access_context=_access_context(state, store),
            ).model_dump_json(indent=2)
        )

    @mcp.tool(
        name="recall_resolve_reference",
        annotations=ToolAnnotations(
            title="Resolve a supersedes reference before writing it",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_resolve_reference(ctx: Context[dict, object], reference: str, written_in: str = "") -> str:
        """Which memo a `supersedes:` reference names here, by search's own rule, before you write it.

        Call it before writing `supersedes:` into a memo file. `status` is `resolved` (write the
        `write` value), `ambiguous` (several memos share the name: write the one you mean with its
        folder, from `candidates`), `dangling` or `self`. `written_in` is the memo you will write
        it in. Writes nothing.
        """
        state = _state(ctx)
        store = await _require(SCOPE_READ, ctx)
        return await _to_thread(
            lambda: resolve_served_reference(
                store,
                reference,
                written_in=written_in,
                security_policy=state.get("source_security_policy"),
                access_context=_access_context(state, store),
            ).model_dump_json(indent=2)
        )

    @mcp.tool(
        name="recall_stats",
        annotations=ToolAnnotations(
            title="Memory freshness & size",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_stats(ctx: Context[dict, object]) -> str:
        """Report how much memory exists and whether it is stale (freshness check).

        `stale` is True when the newest indexed content is older than 2 days. On a lite store,
        `refresh` holds the last refresh of the memo folder (outcome and reason), which explains a
        memo that is not found yet.

        Returns:
            JSON of {chunks, newest_indexed_at, stale, metrics, refresh}.
            Stdio FastEmbed generation stores also report `serving_lineage` as
            `match`, `unverified`, or `LINEAGE_MISMATCH`.
        """
        store = await _require(SCOPE_READ, ctx)
        state = _state(ctx)
        embedder = state["embedder"]

        def stats() -> str:
            if (
                isinstance(store, GenerationStore)
                and isinstance(embedder, FastEmbedEmbedder)
                and state.get("stores") is None
            ):
                with store.snapshot():
                    binding = store.generation_binding()
                    payload = memory_stats(store).model_dump(mode="json")
                expected_model = binding.get("embedder_model")
                expected_dim = binding.get("embedder_dimension")
                profile_id = binding.get("embedder_profile_id")
                fingerprint = binding.get("embedder_profile_fingerprint")
                if (
                    (expected_model is not None and embedder.name != expected_model)
                    or (expected_dim is not None and str(embedder.dim) != expected_dim)
                    or profile_lineage_mismatch(embedder, profile_id, fingerprint) is not None
                ):
                    payload["serving_lineage"] = "LINEAGE_MISMATCH"
                elif profile_id and fingerprint:
                    payload["serving_lineage"] = "match"
                else:
                    payload["serving_lineage"] = "unverified"
                return json.dumps(payload, indent=2)
            result = memory_stats(store)
            if isinstance(store, LiteStore):
                # The last automatic refresh, so an agent can see why a memo it just wrote is not
                # found yet (failed, over the budget, or a calibration still below its floor).
                last = read_refresh_state(store.path).get("last")
                if isinstance(last, dict):
                    last = {**last, "reason": client_reason(str(last.get("reason", "")), str(last.get("error_type") or ""))}
                result.refresh = last
            return result.model_dump_json(indent=2)

        return await _to_thread(stats)
