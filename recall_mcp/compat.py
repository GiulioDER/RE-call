"""Stable compatibility helpers shared by serving adapters.

The service module historically carried serialization needed by MCP and Agent SDK clients.  This
module owns that presentation contract so compatibility consumers do not need to import the full
service hub.  ``recall_mcp.service.serving_json`` remains available as a forwarding import.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast


def serving_json(result: object) -> str:
    """Serialize a service result while omitting empty additive fields."""
    dump = cast(Callable[..., str], getattr(result, "model_dump_json"))
    exclude: set[str] = set()
    if getattr(result, "explanation", None) is None:
        exclude.add("explanation")
    if not getattr(result, "related_items", ()):
        exclude.add("related_items")
    if not getattr(result, "related_diagnostics", ()):
        exclude.add("related_diagnostics")
    return dump(indent=2, exclude=exclude)


__all__ = ["serving_json"]
