"""Retrieval metrics (pure functions over id lists) + the guard false-confident rate."""
from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant."""
    if k <= 0:
        return 0.0
    rel = set(relevant_ids)
    topk = retrieved_ids[:k]
    return sum(1 for r in topk if r in rel) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the relevant items found in the top-k."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(topk & rel) / len(rel)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant item (0 if none)."""
    rel = set(relevant_ids)
    for i, r in enumerate(retrieved_ids):
        if r in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    rel = set(relevant_ids)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(retrieved_ids[:k]) if r in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def fraction_true(flags: list[bool]) -> float:
    """Mean of boolean flags; NaN on empty input — a rate with no data is NOT a score.

    (0.0-on-empty would read as a PERFECT superseded-trust rate and a CATASTROPHIC accuracy at
    the same time; NaN forces publishers to render 'n/a' instead of a fake number.)
    """
    if not flags:
        return float("nan")
    return sum(1 for f in flags if f) / len(flags)


def superseded_trust_rate(stale_trusted_flags: list[bool]) -> float:
    """Fraction of trust-sensitive queries where a superseded/expired memory was presented as a
    trustworthy answer (flag True = the system trusted a stale memory). Lower is better — this
    is the false-positive-retrieval failure mode the trust layer exists to kill. NaN on empty.
    """
    return fraction_true(stale_trusted_flags)


def successor_accuracy(successor_hit_flags: list[bool]) -> float:
    """Fraction of supersession queries where the top trusted answer was the current successor.
    NaN on empty."""
    return fraction_true(successor_hit_flags)


def abstention_accuracy(abstained_flags: list[bool]) -> float:
    """Fraction of expected-abstain queries where the system actually abstained. NaN on empty."""
    return fraction_true(abstained_flags)


def near_miss_false_confident_rate(confident_flags: list[bool]) -> float:
    """Fraction of NEAR-MISS queries (high-similarity distractor, no answer in the corpus) the
    system answered confidently (flag True = did not abstain). Lower is better — this is the
    failure class a cosine threshold passes by construction: the distractor's similarity clears
    any calibrated threshold, so only a judgment beyond the retriever's own score can catch it.
    NaN on empty.
    """
    return fraction_true(confident_flags)


def gap_false_confident_rate(confident_flags: list[bool]) -> float:
    """Fraction of FAR-GAP (off-topic, unanswerable) queries answered confidently (flag True =
    did not abstain). Same polarity as `near_miss_false_confident_rate`; distinct from
    `false_confident_rate`, which takes raw gap_warning flags instead. NaN on empty.
    """
    return fraction_true(confident_flags)


def false_abstain_rate(abstained_flags: list[bool]) -> float:
    """Fraction of ANSWERABLE queries the system wrongly abstained on (flag True = abstained).
    Lower is better — the regression check for any abstention mechanism: killing near-misses is
    worthless if it also kills real answers. NaN on empty.
    """
    return fraction_true(abstained_flags)


def false_confident_rate(gap_warnings: list[bool]) -> float:
    """Given the `gap_warning` flags for the UNANSWERABLE queries, the fraction where the system
    was (wrongly) confident — i.e. gap_warning was False. Lower is better; this is the guard's job.
    """
    if not gap_warnings:
        return 0.0
    return sum(1 for g in gap_warnings if not g) / len(gap_warnings)
