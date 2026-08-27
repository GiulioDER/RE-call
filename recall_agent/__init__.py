"""RE-call as in-process memory for Claude Agent SDK applications.

The supported entry point is `RecallAgentMemory` (see `recall_agent.memory`). This package is
importable without `claude_agent_sdk` installed; only the methods that produce SDK objects need
the `agent` extra, and they say so when it is missing.
"""
from __future__ import annotations

from recall_agent.memory import RecallAgentMemory

__all__ = ["RecallAgentMemory"]
