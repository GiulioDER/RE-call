"""E-1: chronological order, only for a question that asks about order.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-gated-chronological-order.md. The
pattern list below is the one that record fixed before any census, and it is not edited after
one: a gate that misses or over-fires is a result, not something to tune. Each phrase is matched
as written, case-insensitive, anywhere in the question; ``.`` does not cross a line break.
"""

from __future__ import annotations

import re

ORDER_PATTERNS: tuple[str, ...] = (
    r"in what order",
    r"in which order",
    r"what order",
    r"chronological",
    r"chronologically",
    r"sequence of",
    r"the order (in which|that|of)",
    r"order (did|do|were|was)",
    r"timeline",
    r"which came first",
    r"which happened first",
    r"what happened (first|next|last|before|after)",
    r"list .* in (the )?order",
    r"rank .* by (date|time)",
    r"earliest to latest",
    r"oldest to newest",
    r"first to last",
)
_COMPILED = tuple(re.compile(pattern, re.IGNORECASE) for pattern in ORDER_PATTERNS)


def matched_patterns(question: str) -> list[str]:
    """Every pattern of ``ORDER_PATTERNS`` the question matches, in list order."""
    return [pattern.pattern for pattern in _COMPILED if pattern.search(question)]


def asks_for_order(question: str) -> bool:
    """Whether the question asks for a sequence, by the fixed pattern list alone."""
    return any(pattern.search(question) for pattern in _COMPILED)


#: E-2's revision (docs/preregistrations/2026-09-25-aml-c9-revised-ordering-gate.md), fixed there
#: before any held-out question was read: bare ``timeline`` becomes ``timeline of``, the list
#: pattern also matches "list in order", and seven phrasings are added. E-1's list above is unchanged.
ORDER_PATTERNS_V2: tuple[str, ...] = (
    r"in what order",
    r"in which order",
    r"what order",
    r"chronological",
    r"chronologically",
    r"sequence of",
    r"the order (in which|that|of)",
    r"order (did|do|were|was)",
    r"timeline of",
    r"which came first",
    r"which happened first",
    r"what happened (first|next|last|before|after)",
    r"list (.* )?in (the )?order",
    r"rank .* by (date|time)",
    r"earliest to latest",
    r"oldest to newest",
    r"first to last",
    r"correct order",
    r"right order",
    r"nearest to farthest",
    r"farthest to nearest",
    r"newest to oldest",
    r"latest to earliest",
    r"last to first",
)
_COMPILED_V2 = tuple(re.compile(pattern, re.IGNORECASE) for pattern in ORDER_PATTERNS_V2)


def matched_patterns_v2(question: str) -> list[str]:
    """Every pattern of ``ORDER_PATTERNS_V2`` the question matches, in list order."""
    return [pattern.pattern for pattern in _COMPILED_V2 if pattern.search(question)]


def asks_for_order_v2(question: str) -> bool:
    """E-2's gate: whether the question asks for a sequence, by the revised pattern list."""
    return any(pattern.search(question) for pattern in _COMPILED_V2)
