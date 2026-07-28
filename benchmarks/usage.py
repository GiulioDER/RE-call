"""Process-wide LLM token metering, to measure the cost DIFFERENCE between the memory systems.

The benchmark's own generator + judge cost the same for every arm, so they cancel. What differs is
what the memory layer itself sends to an LLM: RE-call sends NOTHING (no LLM in its retrieval path),
Mem0 runs a fact-extraction call per session at ingest. Metering that difference is the whole point.

Each arm runs in its own process, so a process-global counter is enough. `install_openai_meter`
wraps the OpenAI SDK's sync chat-completion so EVERY call in the process is counted — ours AND
Mem0's, since Mem0 talks to the same OpenRouter endpoint through the same SDK. The run then records:

    total   = every LLM call in the process        (harness generator/judge + Mem0 internal)
    harness = the benchmark's own generator/judge   (tracked separately by OpenRouterLLM)
    memory  = total - harness                       (the memory layer's own LLM cost)

For RE-call, `memory` is ~0 by construction — which is the published claim, now measured rather
than asserted. For Mem0, it is the extraction overhead.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_lock = threading.Lock()
_state = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_installed = False


def reset() -> None:
    with _lock:
        _state.update(calls=0, prompt_tokens=0, completion_tokens=0)


def record(prompt_tokens: int, completion_tokens: int) -> None:
    with _lock:
        _state["calls"] += 1
        _state["prompt_tokens"] += prompt_tokens
        _state["completion_tokens"] += completion_tokens


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_state)


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a create() that records `usage` off each response, then returns it unchanged.

    Split out from `install_openai_meter` so the recording logic is unit-testable without the real
    SDK: a streaming or usage-less response (``usage is None``) is simply not counted.
    """

    def wrapped(this: Any, *args: Any, **kwargs: Any) -> Any:
        resp = original(this, *args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            record(
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        return resp

    wrapped.__metered__ = True  # type: ignore[attr-defined]
    return wrapped


def install_openai_meter() -> bool:
    """Patch the OpenAI SDK's sync chat-completion to count every call in this process.

    Idempotent, and defensive: any failure (an SDK layout this doesn't recognise) leaves the SDK
    untouched and returns False, so metering can never break a paid run. Returns True when active.
    """
    global _installed
    if _installed:
        return True
    try:
        import openai.resources.chat.completions as completions

        original = completions.Completions.create
        if getattr(original, "__metered__", False):
            _installed = True
            return True
        completions.Completions.create = _wrap(original)
        _installed = True
        return True
    except Exception:
        return False
