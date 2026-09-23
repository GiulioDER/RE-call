"""Helpers shared by the memory-tenant atomizer study harnesses.

Every definition here is copied verbatim from the C8 study harnesses on the private branch
`claude/atomizer-measure` (`scripts/c8_atomizer_reference.py`, `scripts/c8_atomizer_reasoning.py`
and `scripts/c8_atomizer_round3.py`), which the memory harnesses imported when they were run.
Those harnesses are not published, so the pieces the memory harnesses use live here instead, with
unchanged behaviour: the question writer's response format and model routing, the span size
constants, the frozen rule that rejects a generated question, the OpenRouter call, and the spend
meter. `window_bounds` is the public one from `recall.atomizer`, as it was there.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Mapping
import urllib.request

from recall.atomizer import window_bounds

__all__ = [
    "Budget",
    "MIN_SPAN_CONTENT_WORDS",
    "RESPONSE_FORMAT",
    "SPAN_MAX_WORDS",
    "SPAN_MIN_WORDS",
    "WRITER_MODEL",
    "WRITER_ROUTING",
    "copied_run",
    "probe_rejection",
    "window_bounds",
]

# From scripts/c8_atomizer_reference.py.
SPAN_MIN_WORDS = 12
SPAN_MAX_WORDS = 30
MIN_SPAN_CONTENT_WORDS = 6
QUESTION_MIN_WORDS = 5
QUESTION_MAX_WORDS = 45
MAX_COPIED_RUN = 5
PRICE_IN, PRICE_OUT = 0.15e-6, 0.60e-6
_WORD = re.compile(r"[0-9a-z]+")

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "retrieval_probe",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answerable", "question"],
            "properties": {
                "answerable": {"type": "boolean"},
                "question": {"type": "string"},
            },
        },
    },
}

# From scripts/c8_atomizer_round3.py.
WRITER_MODEL = "meta-llama/llama-3.3-70b-instruct"
WRITER_ROUTING = {"require_parameters": True, "allow_fallbacks": True}


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def copied_run(question: str, span: str) -> int:
    """Longest run of consecutive span tokens that the question repeats verbatim."""

    q = _tokens(question)
    s = _tokens(span)
    best = 0
    previous = [0] * (len(s) + 1)
    for q_token in q:
        current = [0] * (len(s) + 1)
        for index, s_token in enumerate(s, start=1):
            if q_token == s_token:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def probe_rejection(answerable: bool, question: str, span: str) -> str | None:
    """Return the frozen reason a generated probe is rejected, or None to keep it."""

    if not answerable:
        return "not_answerable"
    words = question.split()
    if not QUESTION_MIN_WORDS <= len(words) <= QUESTION_MAX_WORDS:
        return "question_length"
    if copied_run(question, span) >= MAX_COPIED_RUN:
        return "copied_span"
    return None


def _openrouter(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:  # noqa: S310
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("OpenRouter returned a non-object response")
    return decoded


# From scripts/c8_atomizer_reasoning.py.
class Budget:
    """A thread-safe spend meter that refuses new calls once the cap is reached."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.spent = 0.0
        self.served: set[str] = set()
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.spent >= self.cap:
                raise RuntimeError(f"budget exhausted: USD {self.spent:.4f} of {self.cap:.2f}")

    def charge(self, response: Mapping[str, Any]) -> None:
        usage = response.get("usage") or {}
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)):
            cost = PRICE_IN * usage.get("prompt_tokens", 0) + PRICE_OUT * usage.get(
                "completion_tokens", 0
            )
        with self._lock:
            self.spent += float(cost)
            self.served.add(f"{response.get('model')}|{response.get('provider')}")
