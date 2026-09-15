"""Deterministic query-to-evidence anchor features for spare-slot admission."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence, cast


QUERY_ANCHOR_LIMIT = 3
QUERY_ANCHOR_MIN_TOKEN_LENGTH = 4
QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR = 2.0 / 3.0
QUERY_ANCHOR_POLICY = "query_anchor_empty_base_v1"
QUERY_ANCHOR_STOPWORDS = frozenset(
    {
        "according",
        "also",
        "answer",
        "applied",
        "does",
        "from",
        "have",
        "into",
        "item",
        "matter",
        "memory",
        "record",
        "recorded",
        "revision",
        "should",
        "statement",
        "that",
        "their",
        "there",
        "these",
        "this",
        "under",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
    }
)
_TOKEN = re.compile(r"[a-z0-9]+")


def query_anchor_tokens(value: object) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(str(value).casefold())
        if len(token) >= QUERY_ANCHOR_MIN_TOKEN_LENGTH and token not in QUERY_ANCHOR_STOPWORDS
    }


def query_anchor_features(
    query: str,
    proposal: Mapping[str, object],
    pool: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Measure rare query anchor coverage by the proposed chunk and source."""

    source_tokens: dict[str, set[str]] = {}
    for item in pool:
        source_tokens.setdefault(str(item["source"]), set()).update(
            query_anchor_tokens(item["text"])
        )
    query_tokens = query_anchor_tokens(query)
    document_frequency = {
        token: sum(token in tokens for tokens in source_tokens.values())
        for token in query_tokens
    }
    anchors = sorted(
        query_tokens,
        key=lambda token: (
            document_frequency[token],
            hashlib.sha256(token.encode()).hexdigest(),
        ),
    )[:QUERY_ANCHOR_LIMIT]
    proposal_chunk_tokens = query_anchor_tokens(proposal["text"])
    proposal_source_tokens = source_tokens[str(proposal["source"])]
    covered_chunk = sum(token in proposal_chunk_tokens for token in anchors)
    covered_source = sum(token in proposal_source_tokens for token in anchors)
    count = len(anchors)
    return {
        "anchor_count": count,
        "anchor_document_frequencies": [document_frequency[token] for token in anchors],
        "covered_by_proposal_chunk": covered_chunk,
        "covered_by_proposal_source": covered_source,
        "chunk_coverage_fraction": covered_chunk / count if count else 0.0,
        "source_coverage_fraction": covered_source / count if count else 0.0,
        "zero_document_frequency_anchors": sum(
            document_frequency[token] == 0 for token in anchors
        ),
        "uncovered_zero_document_frequency_anchors": sum(
            document_frequency[token] == 0 and token not in proposal_source_tokens
            for token in anchors
        ),
    }


def query_anchor_candidate_eligible(
    base_count: int,
    features: Mapping[str, object],
) -> bool:
    """Return whether the first guarded proposal may fill an empty result."""

    return (
        base_count == 0
        and int(cast(Any, features["anchor_count"])) == QUERY_ANCHOR_LIMIT
        and int(cast(Any, features["zero_document_frequency_anchors"])) == 0
        and float(cast(Any, features["chunk_coverage_fraction"]))
        >= QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR
    )


def direct_query_anchor_candidate(
    query: str,
    pool: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Select one pool chunk directly with the frozen anchor ordering."""

    eligible: list[tuple[float, float, int, Mapping[str, object]]] = []
    for position, item in enumerate(pool):
        features = query_anchor_features(query, item, pool)
        if query_anchor_candidate_eligible(0, features):
            eligible.append(
                (
                    float(cast(Any, features["chunk_coverage_fraction"])),
                    float(cast(Any, features["source_coverage_fraction"])),
                    position,
                    item,
                )
            )
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (-row[0], -row[1], row[2]))[0][3]


__all__ = [
    "QUERY_ANCHOR_LIMIT",
    "QUERY_ANCHOR_MIN_TOKEN_LENGTH",
    "QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR",
    "QUERY_ANCHOR_POLICY",
    "query_anchor_candidate_eligible",
    "direct_query_anchor_candidate",
    "QUERY_ANCHOR_STOPWORDS",
    "query_anchor_features",
    "query_anchor_tokens",
]
