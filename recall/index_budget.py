"""How much a request may index before anything is read or embedded: one rule for every caller.

`recall_index` (an agent's request) and the lite store's automatic refresh (an unattended server
thread) both index a folder named in configuration. Neither may embed an unbounded tree, because
`RECALL_INDEX_ROOT` can be set wide (a project, a home directory) and a hosted embedder bills per
token. The check measures the candidate list the indexer would read, so the measured set and the
indexed set are the same list.

`RECALL_INDEX_MAX_FILES` and `RECALL_INDEX_MAX_BYTES` set the limits. A person running
`recall lite refresh` by hand is not bound by them; the command warns instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "DEFAULT_MAX_INDEX_BYTES",
    "DEFAULT_MAX_INDEX_FILES",
    "IndexBudget",
    "budget_from_env",
    "budget_refusal",
]

DEFAULT_MAX_INDEX_FILES = 2000
DEFAULT_MAX_INDEX_BYTES = 20_000_000


@dataclass(frozen=True)
class IndexBudget:
    max_files: int = DEFAULT_MAX_INDEX_FILES
    max_bytes: int = DEFAULT_MAX_INDEX_BYTES


def budget_from_env(env: Mapping[str, str]) -> IndexBudget:
    """The configured budget. A value that is not a whole number raises ValueError, naming it."""

    def read(name: str, default: int) -> int:
        raw = env.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
        return value

    return IndexBudget(
        max_files=read("RECALL_INDEX_MAX_FILES", DEFAULT_MAX_INDEX_FILES),
        max_bytes=read("RECALL_INDEX_MAX_BYTES", DEFAULT_MAX_INDEX_BYTES),
    )


def budget_refusal(subject: str, files: int, total_bytes: int, budget: IndexBudget) -> str | None:
    """Why `files` files of `total_bytes` bytes exceed `budget`, or None when they fit.

    `subject` opens the sentence, e.g. "index request for 'notes'". The wording names the measured
    value, the limit and the variable that raises it.
    """
    if files > budget.max_files:
        return (
            f"{subject} exceeds the file-count budget: {files} candidate file(s) > limit "
            f"{budget.max_files}; set RECALL_INDEX_MAX_FILES to raise it."
        )
    if total_bytes > budget.max_bytes:
        return (
            f"{subject} exceeds the byte budget: {total_bytes} candidate byte(s) > limit "
            f"{budget.max_bytes}; set RECALL_INDEX_MAX_BYTES to raise it."
        )
    return None
