"""Portable exact-count property and deterministic work bound from the PR905 audit.

CCA Hypothesis found (n=209,rejected=188) through the actual near_miss_coverage API.
Its assert_bounded projection observed 0 where the independently specified exact
90% relation required 1. The same relation is generated here using native bounded
counts, including the counterexample, without adding the audit framework to CI.
Red proof against e9ce4a5ac: the exact-count node fails at 188/209, and the work-bound
node counts 131209 full-corpus predicates for 512 repeated-heading chunks.
"""
from __future__ import annotations

from collections.abc import Iterable

import pytest

from recall import near_miss
from tests.test_unknown_term_boundaries import TinyEmbedder


class _Scores:
    def __init__(self, rejected: int) -> None:
        self.remaining = rejected

    def unknown_terms(self, tokens: list[str]) -> list[str]:
        return []

    def top_cosine(self, vector: list[float]) -> float:
        if self.remaining:
            self.remaining -= 1
            return 0.1
        return 0.9


def test_exact_provisional_boundary() -> None:
    cases = [(209, 188)]
    for n in range(10, 221):
        boundary = 9 * n // 10
        cases.extend((n, r) for r in {0, n, boundary, max(0, boundary - 1), min(n, boundary + 1)})
    for n, rejected in cases:
        probes = {"unknown_entity": [], "in_vocabulary": [f"question {i}" for i in range(n)]}
        report = near_miss.near_miss_coverage(_Scores(rejected), TinyEmbedder(), 0.5, probes)
        assert report["in_vocabulary"]["rejected"] == rejected
        assert report["provisional_in_vocabulary"] == (10 * rejected < 9 * n), (n, rejected, report)


def _alpha(i: int) -> str:
    chars = []
    while True:
        chars.append(chr(97 + i % 26))
        i //= 26
        if not i:
            return "".join(reversed(chars))


def test_probe_generation_avoids_full_corpus_scan_per_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 512
    chunks = [f"# Shared Topic\n\nThe topicword{_alpha(i)} needs servicing." for i in range(n)]
    comparisons = 0

    def counted_any(values: Iterable[bool]) -> bool:
        nonlocal comparisons
        for value in values:
            comparisons += 1
            if value:
                return True
        return False

    monkeypatch.setattr(near_miss, "any", counted_any, raising=False)
    probes = near_miss.near_miss_probes(chunks, [], per_class=40, seed=0)
    assert probes["in_vocabulary"] == [], "the fixture did not exercise rejected candidates"
    assert comparisons <= 8 * n, f"probe generation rescanned the corpus {comparisons} times"
