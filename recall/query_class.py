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
GraphExpansionMode = Literal["off", "one_hop"]
GraphExpansionRequest = Literal["auto", "off", "one_hop"]
GraphActivationCategory = Literal[
    "number",
    "single_hop",
    "multi_hop",
    "temporal",
    "list_completion",
    "explicit_comparison",
    "other",
]

QUERY_CLASS_VERSION = "query-class-v1"
ROUTING_POLICY_VERSION = "routing-v1"
GRAPH_ACTIVATION_POLICY_VERSION = "graph-activation-v1"

_GRAPH_NUMBER_PATTERNS: tuple[str, ...] = (
    r"\bhow\s+many\b",
    r"\bhow\s+much\b",
    r"\b(?:number|count|total)\s+of\b",
    r"\b(?:percentage|percent|amount|quantity|rate)\b",
)
_GRAPH_SINGLE_HOP_PATTERNS: tuple[str, ...] = (r"\bsingle[- ]hop\b",)
_GRAPH_MULTI_HOP_PATTERNS: tuple[str, ...] = (
    r"\bmulti[- ]hop\b",
    r"\bchain\b",
    r"\bpath\s+from\b",
    r"\bconnect(?:ed|s)?\b.*\bto\b",
    r"\brelate(?:d|s)?\b.*\bto\b",
)
_GRAPH_LIST_PATTERNS: tuple[str, ...] = (
    r"\blist\b",
    r"\ball\b",
    r"\bevery\b",
    r"\bwhich\s+(?:ones|items|things)\b",
    r"\bwhich\s+\w+s\b",
    r"\bwhat\s+(?:were|are)\s+the\b",
    r"\bwhat\s+else\b",
    r"\b(?:complete|fill\s+in)\s+(?:the\s+)?list\b",
    r"\bremaining\b",
)
_GRAPH_EXPLICIT_COMPARISON_PATTERNS: tuple[str, ...] = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bbetter\b",
    r"\bworse\b",
)

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
    graph_expansion: GraphExpansionMode = "off"
    graph_activation_category: GraphActivationCategory = "other"
    graph_activation_reason: str | None = None


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
    graph_category = classify_graph_activation(query)
    graph_expansion: GraphExpansionMode = (
        "one_hop"
        if graph_category
        in {"multi_hop", "temporal", "list_completion", "explicit_comparison"}
        else "off"
    )
    graph_reason = (
        f"category:{graph_category}" if graph_expansion == "one_hop" else "category_not_selected"
    )
    return RoutingDecision(
        query_class=query_class,
        profile=profile,
        related_expansion=related,
        expansion_mode=expansion_mode,
        matched_rules=classification.matched_rules,
        graph_expansion=graph_expansion,
        graph_activation_category=graph_category,
        graph_activation_reason=graph_reason,
    )


def classify_graph_activation(query: str) -> GraphActivationCategory:
    """Classify the query for conservative semantic graph activation.

    This policy is intentionally separate from the retrieval classifier. The older classifier is
    part of frozen routing artifacts, while graph activation needs explicit safeguards for numeric
    and single hop questions and a narrower comparison rule than the broad retrieval class.
    """
    normalized = " ".join(query.casefold().split())
    if not normalized:
        return "other"
    temporal_patterns = (
        r"\bwhen\b",
        r"\bdate\b",
        r"\bdated\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\byesterday\b",
        r"\btoday\b",
        r"\blast\s+(?:week|month|year|time)\b",
    )
    for category, patterns in (
        ("number", _GRAPH_NUMBER_PATTERNS),
        ("single_hop", _GRAPH_SINGLE_HOP_PATTERNS),
        ("multi_hop", _GRAPH_MULTI_HOP_PATTERNS),
        ("temporal", temporal_patterns),
        ("list_completion", _GRAPH_LIST_PATTERNS),
        ("explicit_comparison", _GRAPH_EXPLICIT_COMPARISON_PATTERNS),
    ):
        if any(re.search(pattern, normalized) for pattern in patterns):
            return category  # type: ignore[return-value]
    return "other"


def resolve_graph_expansion(
    query: str, requested: GraphExpansionRequest = "auto"
) -> GraphExpansionMode:
    """Resolve explicit graph control or category aware automatic activation."""
    if requested == "auto":
        return route_query(query).graph_expansion
    if requested not in {"off", "one_hop"}:
        raise ValueError("graph expansion must be auto, off, or one_hop")
    return requested


def routing_mode(value: str | None = None) -> RoutingMode:
    """Resolve the opt in routing mode without changing the default serving behavior."""
    selected = (value if value is not None else "shadow").strip().casefold()
    if selected not in {"shadow", "active"}:
        raise ValueError("routing mode must be shadow or active")
    return selected  # type: ignore[return-value]


__all__ = [
    "GRAPH_ACTIVATION_POLICY_VERSION",
    "QUERY_CLASS_VERSION",
    "ROUTING_POLICY_VERSION",
    "GraphActivationCategory",
    "GraphExpansionMode",
    "GraphExpansionRequest",
    "QueryClass",
    "QueryClassification",
    "RoutingDecision",
    "RoutingMode",
    "RoutingProfile",
    "classify_graph_activation",
    "classify_query",
    "resolve_graph_expansion",
    "route_query",
    "routing_mode",
]
