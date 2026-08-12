"""What a corpus audit reports. Pure dataclasses, so the renderer needs no database driver."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimDrift:
    """One line whose surrounding words held steady while its numbers changed."""

    path: str
    old_sha: str
    new_sha: str
    old_date: str
    new_date: str
    old_line: str
    new_line: str


@dataclass(frozen=True)
class StaleAnswer:
    """One question whose nearest match is text the corpus has already replaced."""

    question: str
    plain_top_file: str
    plain_top_superseded_by: str  # "" when the nearest match was not superseded
    trusted_verdict: str
    trusted_abstained: bool
