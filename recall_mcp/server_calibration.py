"""The calibration tools of the MCP server.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import json

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations

from recall_mcp.auth import SCOPE_ADMIN, SCOPE_READ, SCOPE_WRITE
from recall_mcp.status import calibration_status
from recall_mcp.generation_admin import publish_calibration, run_calibration
from recall_mcp.threads import _to_thread
from recall_mcp.server_common import (
    _META_REQUIRED_SCOPE,
    _ToolDeps,
    _mutation_fingerprint,
    _runtime_env_for,
)


def _register_calibration_tools(mcp: MCPServer, deps: _ToolDeps) -> None:
    _require = deps.require
    _require_mutation = deps.require_mutation
    _record_mutation_result = deps.record_mutation_result
    _state = deps.state

    @mcp.tool(
        name="recall_calibration_status",
        annotations=ToolAnnotations(
            title="Read corpus calibration status",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_calibration_status(
        ctx: Context[dict, object], tenant: str | None = None
    ) -> str:
        """Return the latest calibration artifact bound to the caller's generation."""
        store = await _require(SCOPE_READ, ctx, tenant)
        result = await _to_thread(lambda: calibration_status(store))
        return json.dumps(result, indent=2, default=str)

    @mcp.tool(
        name="recall_calibration_run",
        annotations=ToolAnnotations(
            title="Run corpus calibration",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    async def recall_calibration_run(
        ctx: Context[dict, object],
        generation_id: str | None = None,
        queries: list[dict[str, object]] | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Fit corpus calibration: PostgreSQL creates a draft; Lite activates completed fits immediately.

        A completed uncertified Lite fit replaces the prior calibration and blocks strict answers.
        Attempts that cannot complete retain the prior artifact. Lite has no separate publish step.
        """
        state = _state(ctx)
        operation = "recall_calibration_run"
        fingerprint = _mutation_fingerprint(
            operation, {"generation_id": generation_id, "queries": queries}
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
        result = await _to_thread(
            lambda: run_calibration(
                store, state["embedder"], generation_id, queries, env=_runtime_env_for(state)
            )
        )
        payload = json.dumps(result, indent=2, default=str)
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
        name="recall_calibration_publish",
        annotations=ToolAnnotations(
            title="Publish corpus calibration",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
        # Publishing flips what the whole tenant serves — the blast radius the admin scope
        # was written for (auth.py SCOPE_ADMIN). The annotation hints cannot express an
        # escalation above write, so it travels in _meta where clients can read it.
        meta={_META_REQUIRED_SCOPE: SCOPE_ADMIN},
    )
    async def recall_calibration_publish(
        calibration_id: str,
        ctx: Context[dict, object],
        tenant: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Publish one certified calibration artifact for the caller's tenant.

        Under an active atomic rescue it also rebuilds the generation's rescue artifact for the new
        calibration; the result's `atomic_rescue` field reports the build, or a WARNING with the
        retry command when it failed, and the publication stands.

        Requires the `recall:admin` scope: publishing an ARBITRARY existing calibration
        changes the serve/abstain decision for every query the tenant runs, which is
        deliberately not implied by write. Note the invariant this enforces is exactly that
        and no wider: a write-scoped `recall_ingest` in generation mode can still
        certify-and-activate the calibration for ITS OWN upload (via `_certify_upload`), so
        write scope is not "can never change what the tenant serves" — it is "cannot publish
        a calibration the caller did not just produce".
        """
        state = _state(ctx)
        operation = "recall_calibration_publish"
        fingerprint = _mutation_fingerprint(operation, {"calibration_id": calibration_id})
        store, replay = await _require_mutation(
            SCOPE_ADMIN,
            ctx,
            tenant,
            idempotency_key=idempotency_key,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        if replay is not None:
            return replay
        assert store is not None
        result = await _to_thread(
            lambda: publish_calibration(
                store, calibration_id, embedder=state["embedder"], env=_runtime_env_for(state)
            )
        )
        payload = json.dumps(result, indent=2, default=str)
        await _record_mutation_result(
            state,
            store.tenant,
            idempotency_key,
            payload,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
        return payload
