"""The dedup collapse must survive being made fast.

`collapse()` was a pure-Python O(k²) cosine that recomputed both norms on every call and rebuilt
the identical similarity matrix once per threshold — measured at ~47 s per question at the
documented defaults (k=200, 1536 dims, four thresholds), inside a probe whose header advertises
"$0 … embeddings and cosines only". It is now one numpy matrix multiply, reused across thresholds.

A speedup that changes which chunks survive is not a speedup, it is a different experiment — and
§9j is published off this curve. So the test that matters is not "is it fast" but "does it return
exactly what the old implementation returned", checked against an independent reimplementation of
the old arithmetic rather than against the new code's own output.
"""
from __future__ import annotations

import random

import pytest

from benchmarks.beam.dedup_probe import DEDUP_COSINES, collapse, similarity_matrix

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def _reference_cosine(a: list[float], b: list[float]) -> float:
    """The original inline implementation, kept here as the oracle."""
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def _reference_collapse(
    chunks: list[dict[str, str]], vectors: list[list[float]], threshold: float
) -> list[int]:
    kept: list[int] = []
    for i in range(len(chunks)):
        dup_of = None
        for slot, j in enumerate(kept):
            if _reference_cosine(vectors[i], vectors[j]) >= threshold:
                dup_of = slot
                break
        if dup_of is None:
            kept.append(i)
            continue
        j = kept[dup_of]
        di, dj = chunks[i].get("created_at", ""), chunks[j].get("created_at", "")
        if di and di > dj:
            kept[dup_of] = i
    return kept


def _corpus(rng: random.Random, n: int, dim: int = 48) -> tuple[list[dict], list[list[float]]]:
    """Near-duplicate clusters plus noise, so the threshold actually bites."""
    seeds = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(max(1, n // 3))]
    vectors, chunks = [], []
    for _ in range(n):
        src = seeds[rng.randrange(len(seeds))]
        spread = rng.choice([0.001, 0.3])
        vectors.append([v + rng.gauss(0, spread) for v in src])
        chunks.append({"created_at": rng.choice(["", "2024-01-01", "2024-06-01", "2025-02-02"])})
    return chunks, vectors


@pytest.mark.parametrize("n", [1, 2, 5, 20, 60])
@pytest.mark.parametrize("threshold", DEDUP_COSINES)
def test_the_fast_collapse_returns_exactly_the_old_survivor_set(n: int, threshold: float) -> None:
    rng = random.Random(1234 + n)
    chunks, vectors = _corpus(rng, n)
    sims = similarity_matrix(vectors)
    assert collapse(chunks, vectors, threshold, sims) == _reference_collapse(
        chunks, vectors, threshold
    )


def test_a_zero_vector_does_not_divide_by_zero_and_matches_the_old_guard() -> None:
    # The old code's `if na and nb else 0.0` returned 0.0 for a zero vector. Normalising rows
    # must not turn that into a NaN, which would compare False against every threshold and
    # silently change the survivor set.
    chunks = [{"created_at": ""}, {"created_at": ""}]
    vectors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sims = similarity_matrix(vectors)
    assert sims[0][0] == 0.0
    assert collapse(chunks, vectors, 0.9, sims) == _reference_collapse(chunks, vectors, 0.9)


def test_collapse_still_works_without_a_precomputed_matrix() -> None:
    # The matrix argument is an optimisation for the threshold sweep, not a new requirement.
    rng = random.Random(99)
    chunks, vectors = _corpus(rng, 12)
    assert collapse(chunks, vectors, 0.95) == _reference_collapse(chunks, vectors, 0.95)


def test_an_identical_pair_collapses_and_the_newer_one_wins() -> None:
    # The probe's headline mechanism, pinned directly rather than only through the oracle.
    vectors = [[1.0, 0.0], [1.0, 0.0]]
    chunks = [{"created_at": "2024-01-01"}, {"created_at": "2025-01-01"}]
    assert collapse(chunks, vectors, 0.99) == [1], "the newer restatement must take the slot"
