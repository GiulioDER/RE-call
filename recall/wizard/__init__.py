"""Installation wizard: the parts that take a machine from nothing to a calibrated corpus.

Kept separate from `recall.setup`, which is the existing configuration interview and stays the
authority on embedder/reranker/judge selection. This package adds what that interview cannot do on
its own: turn a directory into an immutable generation, produce a labelled query set, and wire the
result into an MCP client.

Nothing here is Windows-only. `probe` and `wiring` branch on the platform; everything else is
ordinary Python so it can be tested without a display and reused on any host.
"""

from __future__ import annotations

__all__ = ["inventory"]
