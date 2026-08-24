"""Deterministic query classes and fixed routing decisions.

The classifier is deliberately small and versioned.  It uses only the query text, never corpus
content or benchmark labels, so routing measurements remain reproducible and cannot leak answers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

QueryClass = Literal[
    "lookup",
    "list",
    "temporal",
    "causal",
    "comparative",
    "status",
    "entity",
    "unknown",
]
RoutingProfile = Literal["fast", "quality"]
RoutingMode = Literal["shadow", "active"]

QUERY_CLASS_VERSION = "query-class-v1"
ROUTING_POLICY_VERSION = "routing-v1"

_RULES: tuple[tuple[QueryClass, tuple[str, ...]], ...] = (
    (
        "temporal",
        (
            r"\bwhen\b",
            r"\bdate\b",
            r"\bdated\b",
            r"\bbefore\b",
            r"\bafter\b",
            r"\byesterday\b",
            r"\btoday\b",
            r"\blast\s+(?:week|month|year|time)\b",
        ),
    ),
    (
        "causal",
        (
            r"\bwhy\b",
            r"\bhow\s+did\b",
            r"\breason\b",
            r"\bcaused?\b",
            r"\bled\s+to\b",
            r"\bconsequence\b",
        ),
    ),
    (
        "comparative",
        (
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bdifference\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bbetter\b",
            r"\bworse\b",
        ),
    ),
    (
        "list",
        (
            r"\blist\b",
            r"\ball\b",
            r"\bevery\b",
            r"\bwhich\s+(?:ones|items|things)\b",
            r"\bwhat\s+(?:were|are)\s+the\b",
        ),
    ),
    (
        "status",
        (
            r"\bcurrent\b",
            r"\bstatus\b",
            r"\bstate\b",
            r"\bowner\b",
            r"\bactive\b",
            r"\blatest\b",
            r"\bdecided\b",
        ),
    ),
    (
        "entity",
        (
            r"\bwho\b",
            r"\bwhose\b",
            r"\bwhere\b",
            r"\bwhich\s+(?:person|team|company|project)\b",
        ),
    ),
    (
        "lookup",
        (
            r"\bwhat\b",
            r"\bwhich\b",
            r"\bfind\b",
            r"\btell\s+me\b",
            r"\bhow\b",
        ),
    ),
)


@dataclass(frozen=True)
class QueryClassification:
    """Deterministic class, matched rules, and classifier version for one query."""

    query_class: QueryClass
    matched_rules: tuple[str, ...] = ()
    classifier_version: str = QUERY_CLASS_VERSION


@dataclass(frozen=True)
class RoutingDecision:
    """Fixed routing arm and optional structural expansion derived without corpus access."""

    query_class: QueryClass
    profile: RoutingProfile
    related_expansion: bool = False
    expansion_mode: str | None = None
    matched_rules: tuple[str, ...] = ()
    policy_version: str = ROUTING_POLICY_VERSION


def classify_query(query: str) -> QueryClassification:
    """Classify one query using fixed precedence and no external state."""
    normalized = " ".join(query.casefold().split())
    if not normalized:
        return QueryClassification("unknown")
    for query_class, patterns in _RULES:
        matched = tuple(pattern for pattern in patterns if re.search(pattern, normalized))
        if matched:
            return QueryClassification(query_class, matched)
    return QueryClassification("unknown")


def route_query(query: str) -> RoutingDecision:
    """Return the preregistered fixed routing arm for ``query``."""
    classification = classify_query(query)
    query_class = classification.query_class
    if query_class in {"temporal", "status"}:
        profile: RoutingProfile = "quality"
        related = False
        expansion_mode = None
    elif query_class in {"causal", "comparative", "entity"}:
        profile = "quality"
        related = True
        expansion_mode = "structure"
    else:
        profile = "fast"
        related = False
        expansion_mode = None
    return RoutingDecision(
        query_class=query_class,
        profile=profile,
        related_expansion=related,
        expansion_mode=expansion_mode,
        matched_rules=classification.matched_rules,
    )


def routing_mode(value: str | None = None) -> RoutingMode:
    """Resolve the opt in routing mode without changing the default serving behavior."""
    selected = (value if value is not None else "shadow").strip().casefold()
    if selected not in {"shadow", "active"}:
        raise ValueError("routing mode must be shadow or active")
    return selected  # type: ignore[return-value]


__all__ = [
    "QUERY_CLASS_VERSION",
    "ROUTING_POLICY_VERSION",
    "QueryClass",
    "QueryClassification",
    "RoutingDecision",
    "RoutingMode",
    "RoutingProfile",
    "classify_query",
    "route_query",
    "routing_mode",
]
