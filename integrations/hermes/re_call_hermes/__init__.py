"""RE-call memory for Hermes Agent: a MemoryProvider over a local RE-call MCP server."""

from __future__ import annotations

from .provider import ReCallMemoryProvider

__all__ = ["ReCallMemoryProvider", "register"]
__version__ = "0.1.0"


def register(ctx) -> None:
    """Hermes calls this once per load; a new provider instance every time (multiplexing contract)."""
    ctx.register_memory_provider(ReCallMemoryProvider())
