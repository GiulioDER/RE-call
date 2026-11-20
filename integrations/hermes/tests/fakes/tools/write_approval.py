"""A fake of Hermes's `tools/write_approval.py` (v0.23.0 to v0.23.3 share this surface).

Tests set `DECISION` to the GateDecision `evaluate_gate` returns, or `FAIL` to make it raise, and
read `CALLS`. The default allows, as Hermes does with the gate off."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MEMORY = "memory"
SKILLS = "skills"


@dataclass(slots=True, kw_only=True)
class GateDecision:
    allow: bool = False
    blocked: bool = False
    stage: bool = False
    message: str = ""


DECISION: GateDecision = GateDecision(allow=True)
FAIL: Exception | None = None
ENABLED: Any = False  # what write_approval_enabled returns, or an exception it raises
CALLS: list[dict[str, Any]] = []


def evaluate_gate(subsystem: str, *, inline_summary: str = "", inline_detail: str = "") -> GateDecision:
    CALLS.append({"subsystem": subsystem, "summary": inline_summary, "detail": inline_detail})
    if FAIL is not None:
        raise FAIL
    return DECISION


def write_approval_enabled(subsystem: str) -> bool:
    if isinstance(ENABLED, Exception):
        raise ENABLED
    return bool(ENABLED)
