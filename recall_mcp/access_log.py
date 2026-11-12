"""One access line per authorization decision: who called, from which session, for which tool.

A shared HTTP server sees every session of every client, and the bearer token already tells it
the principal and tenant; until this module nothing recorded either for a call that succeeded,
so a call could not be traced back to the session that made it.

The session is `_meta.threadId` when the call carries one (Codex sends its chat id there with
every tool call), else the `X-Recall-Session` header, which a client's headers helper sets per
session (`scripts/mcp_headers_helper.py`); the client is the `X-Recall-Client` header. Both are
claims by whoever holds the token: they attribute a call, they never authorize one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from recall.observability import get_logger

_log = get_logger("mcp.access")

#: Header values are client-supplied text bound for a log line: anything outside this set (a
#: newline that would forge a second line, a control character, an unbounded length) is recorded
#: as `invalid` rather than echoed.
_SAFE_VALUE = re.compile(r"[A-Za-z0-9._:@-]{1,128}")


def _safe(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if _SAFE_VALUE.fullmatch(text) else "invalid"


def _field(container: object, name: str) -> object:
    # The SDK hands typed params on some versions and plain mappings on others.
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def caller_identity(ctx: Any) -> tuple[str | None, str | None, str | None]:
    """`(session, client, tool)` for the request `ctx` belongs to; None where it carries none.

    Every field is read with a default: this is a witness on the authorization path, and the
    request context's shape differs across SDK releases, so a missing field must cost the record
    a value, never the caller its call.
    """
    headers = getattr(ctx, "headers", None) or {}
    request_context = getattr(ctx, "request_context", None)
    # The per-call thread id is finer than the header: a client that caches its headers helper
    # (Codex does, per process) sends one header for every chat of that process.
    session = _field(getattr(request_context, "meta", None) or {}, "threadId")
    if session is None:
        session = headers.get("x-recall-session")
    tool = None
    if getattr(request_context, "method", None) == "tools/call":
        tool = _field(getattr(request_context, "params", None), "name")
    return _safe(session), _safe(headers.get("x-recall-client")), _safe(tool)


def record_access(
    *,
    principal: str,
    tenant: str | None,
    scope: str,
    tool: str | None,
    session: str | None,
    client: str | None,
    outcome: Literal["allowed", "denied"],
) -> None:
    """Write the access line.

    `session`, `client` and `tool` come from `caller_identity`, already reduced to the safe set or
    `invalid`; `None` logs as `-`. The principal and tenant are the server's own, from the verified
    token, and may legitimately hold any character an identity provider issues (a URL-shaped
    subject), so they are written with `%r`, which escapes a newline instead of refusing the name.
    """
    _log.info(
        "access outcome=%s principal=%r tenant=%r scope=%s tool=%s session=%s client=%s",
        outcome,
        principal,
        tenant,
        scope,
        tool or "-",
        session or "-",
        client or "-",
    )
