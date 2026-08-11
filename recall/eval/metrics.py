"""Retrieval metrics (pure functions over id lists) + the guard false-confident rate."""
from __future__ import annotations

from collections.abc import ItemsView, KeysView, Mapping, Set as AbstractSet
import math
import random
from statistics import NormalDist
from typing import Any

from recall.observability import percentile


#: The quantiles both eval harnesses publish, named once so the two cannot drift apart.
LATENCY_QUANTILES = (0.50, 0.95)


def latency_report(samples_ms: list[float]) -> dict[str, float]:
    """p50/p95 of a latency sample, nearest-rank, in milliseconds to 0.1 ms.

    `int(q*n)` reads like a percentile index and is not — it IS the 1-based nearest rank, so
    subscripting a 0-based list with it reports the next sample up, one whole rank too high,
    every time. That defect was fixed in `recall.observability.percentile` and again in
    `recall.eval.scale`, and `tests/test_percentiles.py` records why a partial fix is the
    dangerous one: "fixing one copy and publishing from the other is exactly how the wrong number
    reached results/*.md".

    Both eval harnesses carried it again (`lat[int(0.95 * len(lat))]` for p95 and
    `lat[len(lat) // 2]` for p50 — the same off-by-one, since `len(lat) // 2` is also a 1-based
    rank whenever n is even). They now call this instead. Measured: p50 moves for EVERY even n,
    p95 only when n is a multiple of 20, and the old value was always the higher of the two —
    see the CHANGELOG entry, which names the published figures that move.

    Not yet one copy repo-wide: `benchmarks/latency.py:_percentile` is a separate implementation
    on a DIFFERENT convention (`int(round(q*(n-1)))`, which interpolates the rank rather than
    taking the ceiling) and also backs that module's bootstrap CI quantiles. It returns 51.0
    where this returns 50.0 for p50 of 1..100. Left alone deliberately rather than silently
    swept in, and named here so the next reader does not have to discover it.

    Empty sample -> empty dict, matching what both harnesses already published for a run with no
    timed queries: a latency report with no data is not a latency of zero.
    """
    if not samples_ms:
        return {}
    ordered = sorted(samples_ms)
    # `ndigits=None`: round ONCE, here, at the precision this report publishes. Reading through
    # `percentile`'s 3-dp default and rounding that to 0.1 ms disagrees with a single round on
    # ~0.5% of values, which would make "to 0.1 ms" above a label the function does not honour.
    return {f"p{int(q * 100)}": round(percentile(ordered, q, ndigits=None), 1)
            for q in LATENCY_QUANTILES}


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant.

    NOTE on this eval set: every query has exactly ONE relevant document, so P@5 is
    mechanically capped at 1/5 = 0.20 — a "perfect" retriever that puts the single answer in the
    top 5 scores 0.20, not 1.0. Read P@5 here as a rescaled "answer present in top 5" indicator,
    not as classical precision. R@5 / MRR / nDCG are the informative ranking metrics; see
    `bootstrap_ci` for uncertainty on the headline rates.
    """
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


def nan_to_null(value: Any, seen: frozenset[int] = frozenset()) -> Any:
    """Recursively replace non-finite floats with None, so `allow_nan=False` can stay on.

    The NaN convention above is what makes this necessary: every rate here reports NaN rather
    than a fake score on no data, and `json.dumps` renders that as a bare `NaN` token which is
    not valid JSON and which no non-Python parser will read. Sanitise, then serialise strictly,
    so a missed one is a loud failure instead of an unparseable artifact.

    Matches on the ABCs, not on `dict`/`list`. `proposal_precision_recall` returns a
    `MappingProxyType`, which is a Mapping but NOT a dict, so a `dict`-only test would hand the
    proxy straight back with its NaN intact and `json.dumps` would then die on the proxy rather
    than on the number. Containers are normalised to plain dict/list on the way out.

    `seen` is path scoped, so it records ancestors only: the same object referenced twice as
    SIBLINGS is not a cycle and survives on both paths. A real cycle RAISES rather than nulling
    the repeated member away. Returning None there would be the worst outcome available, a
    valid looking artifact with data silently replaced, and `pending_datasets` treats an
    existing file as a finished corpus. Stack safety must not be bought with a wrong file.

    Unordered sets are emitted in sorted order. Their iteration order varies per process under
    hash randomisation, and both callers write git tracked artifacts that would otherwise
    rewrite themselves on every run. The test is "is this an ordered view", not "is this a
    `set`": any hash backed `collections.abc.Set` has the same problem. `dict.keys()` and
    `dict.items()` register as Sets but are already ORDERED, so they keep their order, because
    sorting them would buy no determinism they did not have and would discard insertion order.

    Mapping KEYS are deliberately out of scope. A non-finite key survives to `json.dumps`, which
    rejects it under `allow_nan=False`, and a non-scalar key raises there too; both are the loud
    failure we want. Note `json.dumps` silently stringifies int, float, bool and None keys, and
    that coercion is accepted here rather than pre-empted.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, (Mapping, list, tuple, AbstractSet)):
        if id(value) in seen:
            raise ValueError("Circular reference detected")
        seen = seen | {id(value)}
        if isinstance(value, Mapping):
            return {k: nan_to_null(v, seen) for k, v in value.items()}
        if isinstance(value, AbstractSet) and not isinstance(value, (KeysView, ItemsView)):
            return [nan_to_null(v, seen) for v in sorted(value, key=repr)]
        return [nan_to_null(v, seen) for v in value]
    return value


def bootstrap_ci(
    flags: list[bool],
    *,
    n: int = 1000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> tuple[float, float]:
    """Bootstrapped confidence interval for the mean of boolean flags (a rate).

    Resamples ``flags`` WITH replacement ``n`` times, takes the mean of each resample, and
    returns the (lower, upper) percentiles for the given confidence level. This puts an honest
    uncertainty band on the headline rates (answerable/unanswerable/trust accuracy) which, on a
    ~25-query eval set, are otherwise reported as deceptively precise point estimates.

    Returns ``(nan, nan)`` on empty input (no data -> no interval), matching `fraction_true`.
    ``seed`` makes the interval reproducible run-to-run. Dependency-free (stdlib ``random``),
    so it works in the offline test suite without numpy.
    """
    if not flags:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(flags)
    means: list[float] = []
    for _ in range(n):
        sample = [flags[rng.randrange(m)] for _ in range(m)]
        means.append(sum(1 for f in sample if f) / m)
    means.sort()
    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    lo = means[min(len(means) - 1, int(lo_q * len(means)))]
    hi = means[min(len(means) - 1, int(hi_q * len(means)))]
    return (lo, hi)


def wilson_ci(
    flags: list[bool], *, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for the mean of boolean flags (a binomial proportion).

    Preferred over `bootstrap_ci` for every rate this eval publishes, because the eval's
    per-class samples are tiny (n=2 for abstention, n=4 for successor accuracy) and often
    degenerate. A percentile bootstrap of an all-True sample resamples all-True every time and
    returns ``[1.00, 1.00]`` — it reports CERTAINTY from two observations, which is the opposite
    of what a confidence interval is for. Wilson is derived from the normal approximation to the
    binomial rather than from resampling, so it stays bounded in [0, 1], never collapses at
    p=0 or p=1, and widens as n shrinks.

    Returns ``(nan, nan)`` on empty input (no data -> no interval), matching `fraction_true`.
    """
    if not flags:
        return (float("nan"), float("nan"))
    n = len(flags)
    p = sum(1 for f in flags if f) / n
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


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

    NaN on empty, like every other rate here: a config with no unanswerable queries has not
    earned a 0.00 "the guard never failed", and publishing one beside genuinely measured rates
    is exactly the unearned-pass this module's NaN convention exists to prevent.
    """
    return fraction_true([not g for g in gap_warnings])
