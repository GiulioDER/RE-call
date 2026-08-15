"""The answer key: every defect the fleet certifies, and what each member does NOT catch.

Read this file to learn what the eval harness is guaranteed against. See
docs/EVAL_CALIBRATION_FLEET_DESIGN.md for why it exists.

⚠️ EXPECTED VALUES ARE DERIVED, NEVER CAPTURED. Every number below is written from the formula
in its comment. This is the line between a fleet and a snapshot test: a snapshot blesses
whatever the code does today, so it detects only CHANGE; a fleet asserts what the code MUST do,
so it detects change and pre-existing wrongness alike. If an expected value ever has to be
edited to make a test pass, that is a finding to investigate, not a chore. Re-recording is how
this kind of suite rots.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FleetMember:
    """One defect class, the system that embodies it, and the result that must come back."""

    #: Stable identifier, used as the parametrised test id.
    name: str
    #: What this member embodies, in one sentence.
    defect: str
    #: Produces the scripted inputs for this member's surface.
    build: Callable[[], Any]
    #: The closed-form result, derived by hand, with the derivation in a comment.
    expected: Any
    #: What this member does NOT certify. Required.
    does_not_catch: str

    def __post_init__(self) -> None:
        if not self.does_not_catch.strip():
            raise ValueError(
                f"{self.name}: does_not_catch must name this member's blind spot. An optional "
                f"field would be empty on every member within a month, and a fleet that does "
                f"not state what it misses invites being read as covering more than it does."
            )
