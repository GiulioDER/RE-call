"""Hermes loads this directory as a memory provider plugin and calls ``register(ctx)``.

The implementation is the ``re_call_hermes`` package beside this file, which is also what a pip
install provides (entry point ``hermes_agent.memory_providers: re-call``).
"""

from __future__ import annotations

if __package__:
    from .re_call_hermes import ReCallMemoryProvider, register
else:  # imported as a bare module (pytest collecting the repository root)
    from re_call_hermes import ReCallMemoryProvider, register

__all__ = ["ReCallMemoryProvider", "register"]
