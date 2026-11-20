"""The parts of Hermes's `agent/memory_provider.py` (read at e0754412, 2026-11-08) the provider uses.

Signatures and abstract methods are copied from that file; bodies are reduced to what a test needs.
The contract tests (HERMES_SRC) check the provider against the real module instead.
"""

from __future__ import annotations

import contextvars
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

INDICATOR_GLYPH = "🧠"
SPAWNED: list[str] = []  # thread names, so a test can see every background thread went through here


def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
    SPAWNED.append(name)
    ctx = contextvars.copy_context()
    return threading.Thread(target=lambda *a, **k: ctx.run(target, *a, **k), args=args, kwargs=kwargs,
                            name=name, daemon=daemon)


@dataclass(frozen=True)
class RecallStatus:
    provider_label: str
    count: int
    glyph: str = INDICATOR_GLYPH


TRIVIAL_PROMPT_RE = re.compile(r"^(yes|no|ok|okay|thanks|hi|hello|continue|done)[\s!?.]*$", re.IGNORECASE)


def is_trivial_prompt(text):
    stripped = (text or "").strip()
    return not stripped or stripped.startswith("/") or bool(TRIVIAL_PROMPT_RE.match(stripped))


class MemoryProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None: ...

    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]: ...

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def recall_status(self):
        return None

    def handle_tool_call(self, tool_name, args, **kwargs):
        raise NotImplementedError

    def shutdown(self) -> None:  # noqa: B027 - copied from Hermes, an optional hook
        pass

    # Copied from Hermes v0.23.0 agent/memory_provider.py:236: called on an uninitialized instance.
    def on_profile_renamed(self, old_name: str, new_name: str, hermes_home: str) -> None:  # noqa: B027
        """A profile whose config selects this provider was renamed; ``hermes_home`` is its new
        directory. Re-key anything the provider stores under the old profile name."""
