"""What every tool family of the MCP server shares: the tool dependencies, the mutation receipts, the
as-of and trust-policy helpers, and the server's logger.

Moved out of `recall_mcp.server` (server split). `recall_mcp.server` re-imports every name, so
`recall_mcp.server.<name>` keeps resolving.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from recall.idempotency_receipts import ANY_OPERATION
from recall.observability import get_logger
from recall.security_policy import AccessContext
from recall_mcp.assistant import AssistantMemos
from recall.store import PgVectorStore
from recall.lineage import canonical_sha256
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall_mcp.limits import AsyncRateLimiter
from recall_mcp.settings import Settings
from recall_mcp.stores import StoreRegistry
from recall_mcp.threads import _to_thread


def _mutation_fingerprint(operation: str, arguments: Mapping[str, object]) -> str:
    """Hash the exact mutation operation and JSON arguments used for idempotency."""
    return canonical_sha256({"operation": operation, "arguments": arguments})


def _principal_bound_fingerprint(fingerprint: str | None) -> str | None:
    """Bind an idempotency fingerprint to the authenticated principal.

    Receipts are keyed by tenant and request key, so without this another principal of the same
    tenant who sends the same key and arguments would be handed the first principal's response,
    whatever the source policy grants it. Bound, that reuse is an idempotency conflict. A call
    with no authenticated principal (stdio) keeps the bare fingerprint.
    """
    if fingerprint is None:
        return None
    token = get_access_token()
    if token is None:
        return fingerprint
    principal = str((getattr(token, "claims", None) or {}).get("principal") or token.client_id or "")
    if not principal:
        return fingerprint
    return canonical_sha256({"principal": principal, "request": fingerprint})


async def _record_mutation_result(
    state: dict[str, object],
    tenant: str,
    idempotency_key: str | None,
    result: str,
    *,
    idempotency_operation: str | None = None,
    idempotency_fingerprint: str | None = None,
) -> None:
    """Persist a mutation receipt before attempting the Redis replay-cache write."""
    if not idempotency_key:
        return
    idempotency_fingerprint = _principal_bound_fingerprint(idempotency_fingerprint)
    store = state.get("store")
    if store is None:
        registry = state.get("stores")
        if isinstance(registry, StoreRegistry):
            store = await _to_thread(lambda: registry.get(tenant))
    durable = getattr(store, "record_operation_receipt", None)
    limiter = state.get("limiter")
    record = getattr(limiter, "store_idempotency_result", None)
    if callable(durable):
        try:
            await _to_thread(
                lambda: durable(
                    idempotency_key,
                    result,
                    operation=idempotency_operation or ANY_OPERATION,
                    request_fingerprint=idempotency_fingerprint,
                )
            )
        except BaseException as durable_error:
            # The mutation has committed but its durable receipt did not. The Redis copy is then
            # the only replay a retry can find, so still try it, and surface the durable failure.
            if callable(record):
                try:
                    await record(
                        tenant,
                        idempotency_key,
                        result,
                        operation=idempotency_operation,
                        request_fingerprint=idempotency_fingerprint,
                    )
                except Exception:  # BROAD-CATCH: fail-open, the durable failure below is reported
                    _log.warning(
                        "idempotency replay cache write failed after the durable receipt failed"
                    )
            raise durable_error
    if callable(record):
        try:
            await record(
                tenant,
                idempotency_key,
                result,
                operation=idempotency_operation,
                request_fingerprint=idempotency_fingerprint,
            )
        except Exception:  # BROAD-CATCH: fail-open, the durable receipt replays a retry
            # The mutation has committed. When the durable receipt exists, a retry with the same
            # key replays it, so a Redis failure here must not turn a success into an error.
            if not callable(durable):
                raise
            _log.warning("idempotency replay cache write failed; durable receipt retained")


async def _release_mutation_reservation(
    state: dict[str, object], tenant: str, idempotency_key: str | None
) -> None:
    """Release a reservation only when a mutation failed before any side effect."""
    if not idempotency_key:
        return
    limiter = state.get("limiter")
    release = getattr(limiter, "release_idempotency_reservation", None)
    if callable(release):
        await release(tenant, idempotency_key)


#: `_meta` key under which a tool advertises a required scope the annotation hints cannot
#: express. Published to clients in tools/list, so a downgrade has to lie publicly; the
#: authorisation test imports this constant rather than restating the literal.
_META_REQUIRED_SCOPE = "recall/requiredScope"


_T = TypeVar("_T")


_log = get_logger("mcp")


async def _check_limiter_async(
    limiter: AsyncRateLimiter,
    tenant: str,
    key: str,
    cost: float,
    *,
    read_only: bool = False,
) -> None:
    """Call either the production async limiter or a legacy local test double."""
    if getattr(limiter, "requires_idempotency", False):
        await limiter.check(tenant, key, cost, read_only=read_only)
        return
    check = getattr(limiter, "check")
    result = check(tenant, key, cost)
    if hasattr(result, "__await__"):
        await result


class _Require(Protocol):
    """The authorise-and-debit choke point `build_server` constructs per process."""

    async def __call__(
        self,
        scope: str,
        ctx: Context[dict, object],
        requested_tenant: str | None = None,
        idempotency_key: str | None = None,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> PgVectorStore: ...


class _RequireMutation(Protocol):
    async def __call__(
        self,
        scope: str,
        ctx: Context[dict, object],
        requested_tenant: str | None = None,
        idempotency_key: str | None = None,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> tuple[PgVectorStore | None, str | None]: ...


class _RecordMutationResult(Protocol):
    async def __call__(
        self,
        state: dict[str, object],
        tenant: str,
        idempotency_key: str | None,
        result: str,
        *,
        idempotency_operation: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class _ToolDeps:
    """What a tool body needs from `build_server`'s closures.

    Most of the callables are `build_server` closures over `_state` and over each other, so
    they cannot be module globals; `record_mutation_result` is the module-level helper. Passing
    them explicitly is what lets each tool family register from a module-level function while
    the bodies stay byte-identical.
    """

    require: _Require
    require_mutation: _RequireMutation
    record_mutation_result: _RecordMutationResult
    state: Callable[[Context[dict, object]], dict]
    current_tenant: Callable[[dict], str | None]
    access_context: Callable[..., AccessContext | None]
    answer_backend_configured: bool
    #: The assistant surface (`recall_mcp.assistant`), when RECALL_ASSISTANT_MEMOS turns it on.
    assistant: AssistantMemos | None = None


def _parse_as_of(as_of: str | None) -> datetime | None:
    if not as_of:
        return None
    try:
        instant = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("as_of must be an ISO 8601 timestamp") from exc
    if instant.tzinfo is None:
        # Graph validity bounds are timezone aware (bare dates are read as UTC), and comparing
        # them with a naive instant raises TypeError.
        instant = instant.replace(tzinfo=UTC)
    return instant


def _trust_policy_for(state: Mapping[str, object]) -> TrustPolicy:
    return _runtime_settings_for(state).trust_policy


def _runtime_settings_for(state: Mapping[str, object]) -> Settings:
    settings = state.get("settings")
    if not isinstance(settings, Settings):
        raise RuntimeError("MCP runtime state is missing its Settings snapshot")
    return settings


def _runtime_env_for(state: Mapping[str, object]) -> Mapping[str, str]:
    return _runtime_settings_for(state).env


def _tool_error_for_trust_refusal(refusal: TrustRefusal, *, refresh_warning: str | None = None) -> ToolError:
    """Keep the stable trust refusal payload visible through MCP's error channel.

    `refresh_warning` rides along when the lite store's refresh is failing: an empty or
    uncalibrated store is the usual reason for the refusal then, and the warning is the cause.
    """
    payload: dict[str, object] = {"error": "trust_refusal", **refusal.to_dict()}
    if refresh_warning is not None:
        payload["refresh_warning"] = refresh_warning
    return ToolError(json.dumps(payload, sort_keys=True))


def _token_principal(token: Any) -> str:
    """The principal an access line names: the token's `principal` claim, else its client id."""
    return str((token.claims or {}).get("principal", token.client_id))
