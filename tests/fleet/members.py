"""The answer key: every defect the fleet certifies, and what each member does NOT catch.

Read this file to learn what the eval harness is guaranteed against. See
docs/EVAL_CALIBRATION_FLEET_DESIGN.md for why it exists.

⚠️ EXPECTED VALUES ARE DERIVED, NEVER CAPTURED. Every number below is written from the formula
in its comment. This is the line between a fleet and a snapshot test: a snapshot blesses
whatever the code does today, so it detects only CHANGE; a fleet asserts what the code MUST do,
so it detects change and pre-existing wrongness alike. If an expected value ever has to be
edited to make a test pass, that is a finding to investigate, not a chore. Re-recording is how
this kind of suite rots.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FleetMember:
    """One defect class, the system that embodies it, and the result that must come back."""

    #: Stable identifier, used as the parametrised test id.
    name: str
    #: What this member embodies, in one sentence.
    defect: str
    #: Produces the scripted inputs for this member's surface.
    build: Callable[[], Any]
    #: The closed-form result, derived by hand, with the derivation in a comment.
    expected: Any
    #: What this member does NOT certify. Required.
    does_not_catch: str

    def __post_init__(self) -> None:
        if not self.does_not_catch.strip():
            raise ValueError(
                f"{self.name}: does_not_catch must name this member's blind spot. An optional "
                f"field would be empty on every member within a month, and a fleet that does "
                f"not state what it misses invites being read as covering more than it does."
            )


#: Above DEFAULT_GAP_THRESHOLD (0.50), so `gap_warning` is False: the system was CONFIDENT.
SCORE_CONFIDENT = 0.80
#: Below it, so `gap_warning` is True: the guard fired.
SCORE_GAP = 0.20

#: Retrieval depth every surface A member scripts. `_score_config` searches with k=10.
DEPTH = 10


def _rows(gold: str, rank: int | None, prefix: str, score: float) -> list[tuple[str, float]]:
    """`DEPTH` rows with `gold` at 1-based `rank`, or absent from the list when `rank` is None."""
    ids = [f"{prefix}_{i}.md:0" for i in range(DEPTH)]
    if rank is not None:
        ids[rank - 1] = gold
    return [(cid, score) for cid in ids]


def _answerable(rank: int | None, n: int) -> tuple[list[dict], dict]:
    """`n` answerable queries, each with its gold document at 1-based `rank` (None = absent)."""
    queries, script = [], {}
    for i in range(n):
        text = f"answerable query {i}"
        gold = f"gold_{i}.md:0"
        queries.append(
            {"id": f"a{i}", "query": text, "relevant_ids": [gold], "answerable": True}
        )
        script[text] = _rows(gold, rank, prefix=f"filler{i}", score=SCORE_CONFIDENT)
    return queries, script


def _unanswerable(scores: list[float]) -> tuple[list[dict], dict]:
    """One unanswerable query per score. A score below 0.50 makes the guard fire on that query."""
    queries, script = [], {}
    for i, score in enumerate(scores):
        text = f"unanswerable query {i}"
        queries.append({"id": f"u{i}", "query": text, "relevant_ids": [], "answerable": False})
        script[text] = _rows(gold="", rank=None, prefix=f"noise{i}", score=score)
    return queries, script


def _combine(*parts: tuple[list[dict], dict]) -> tuple[list[dict], dict]:
    queries: list[dict] = []
    script: dict[str, list[tuple[str, float]]] = {}
    for part_queries, part_script in parts:
        queries.extend(part_queries)
        script.update(part_script)
    return queries, script


def _at_rank(rank: int | None, n: int = 4) -> Callable[[], tuple[list[dict], dict]]:
    """`n` answerable queries at `rank`, plus four unanswerable ones the guard fires on."""
    return lambda: _combine(_answerable(rank, n), _unanswerable([SCORE_GAP] * 4))


def _dropper_half() -> tuple[list[dict], dict]:
    """Five queries with gold at rank 1, five with gold absent, plus the same four unanswerable."""
    present_q, present_s = _answerable(rank=1, n=5)
    absent_q, absent_s = _answerable(rank=None, n=5)
    # `_answerable` reuses ids and query text across both calls, so rename the absent half.
    absent_q = [{**q, "id": f"absent-{q['id']}", "query": f"absent {q['query']}"} for q in absent_q]
    absent_s = {f"absent {text}": rows for text, rows in absent_s.items()}
    return _combine((present_q, present_s), (absent_q, absent_s), _unanswerable([SCORE_GAP] * 4))


SURFACE_A: tuple[FleetMember, ...] = (
    FleetMember(
        name="perfect-rank-1",
        defect="none: the clean twin every other member must differ from",
        build=_at_rank(1),
        # r=1: R@5=1.0, P@5=0.2, MRR=1/1=1.0, nDCG=1/log2(2)=1.0. All four unanswerable
        # queries score 0.20 < 0.50, so the guard fires on every one and fcr = 0.0.
        expected={
            "p_at_5": 0.2, "r_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0, "fcr_with_guard": 0.0
        },
        does_not_catch="anything about real retrieval: the store is scripted, so a perfect "
                       "score here says nothing about whether the retriever can find documents",
    ),
    FleetMember(
        name="gold-at-rank-3",
        defect="ranking degraded while retrieval stays intact",
        build=_at_rank(3),
        # r=3: R@5=1.0 (unchanged), P@5=0.2 (unchanged), MRR=1/3, nDCG=1/log2(4)=0.5.
        # The two unchanged fields are the point: a scorer that conflated set metrics with
        # ranked ones passes r_at_5 and fails here.
        expected={
            "p_at_5": 0.2, "r_at_5": 1.0, "mrr": 1 / 3, "ndcg_at_10": 0.5,
            "fcr_with_guard": 0.0,
        },
        does_not_catch="a rank error larger than the k=5 window, which boundary-rank-6 covers",
    ),
    FleetMember(
        name="boundary-rank-5",
        defect="gold sits exactly on the inclusive edge of the k=5 window",
        build=_at_rank(5),
        # r=5: R@5=1.0 (5 <= 5), P@5=0.2, MRR=1/5=0.2, nDCG=1/log2(6).
        expected={
            "p_at_5": 0.2, "r_at_5": 1.0, "mrr": 0.2,
            "ndcg_at_10": 1 / math.log2(6), "fcr_with_guard": 0.0,
        },
        does_not_catch="an off-by-one in the OTHER direction; it is paired with boundary-rank-6",
    ),
    FleetMember(
        name="boundary-rank-6",
        defect="gold sits one position past the k=5 window",
        build=_at_rank(6),
        # r=6: R@5=0.0 (6 > 5), P@5=0.0, MRR=1/6, nDCG=1/log2(7). MRR and nDCG stay non-zero
        # because both read the whole k=10 list. This repo has a documented history of 1-based
        # and 0-based rank confusion; see `metrics.latency_report`'s docstring.
        expected={
            "p_at_5": 0.0, "r_at_5": 0.0, "mrr": 1 / 6,
            "ndcg_at_10": 1 / math.log2(7), "fcr_with_guard": 0.0,
        },
        does_not_catch="an off-by-one at k=10, the nDCG depth, which no member currently pins",
    ),
    FleetMember(
        name="gold-dropper-half",
        defect="gold absent entirely from the results on half the questions",
        build=_dropper_half,
        # 10 answerable queries: 5 at rank 1 scoring (0.2, 1.0, 1.0, 1.0) and 5 with gold
        # absent scoring (0, 0, 0, 0). Means over 10: P@5 = 5*0.2/10 = 0.1, R@5 = 5/10 = 0.5,
        # MRR = 5/10 = 0.5, nDCG = 5/10 = 0.5.
        expected={
            "p_at_5": 0.1, "r_at_5": 0.5, "mrr": 0.5, "ndcg_at_10": 0.5,
            "fcr_with_guard": 0.0,
        },
        does_not_catch="whether the mean is taken over questions or pooled over hits when a "
                       "question carries MORE than one relevant document; every fleet query "
                       "has exactly one",
    ),
    FleetMember(
        name="guard-never-fires",
        defect="every unanswerable query scores above the gap threshold",
        build=lambda: _combine(
            _answerable(rank=1, n=4), _unanswerable([SCORE_CONFIDENT] * 4)
        ),
        # 0.80 >= 0.50 on all four, so gap_warning is False on all four, so the system was
        # confident on every unanswerable query: false_confident_rate = 4/4 = 1.0, the worst
        # possible value. Paired with guard-fires-on-half to pin the `not g` negation's
        # polarity inside `false_confident_rate`.
        expected={
            "p_at_5": 0.2, "r_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0,
            "fcr_with_guard": 1.0,
        },
        does_not_catch="whether the threshold itself is well chosen; it pins the plumbing "
                       "between a dense score and the published rate, not the calibration",
    ),
    FleetMember(
        name="guard-fires-on-half",
        defect="the guard fires on half the unanswerable queries",
        build=lambda: _combine(
            _answerable(rank=1, n=4),
            _unanswerable([SCORE_GAP, SCORE_GAP, SCORE_CONFIDENT, SCORE_CONFIDENT]),
        ),
        # Two below threshold (guard fires) and two above (it does not): fcr = 2/4 = 0.5.
        # A scorer computing any() or all() instead of a mean passes both extremes and fails
        # here, which is why the interior point exists rather than a second extreme.
        expected={
            "p_at_5": 0.2, "r_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0,
            "fcr_with_guard": 0.5,
        },
        does_not_catch="rounding of the published rate; 0.5 is exactly representable",
    ),
    FleetMember(
        name="no-answerable-queries",
        defect="DECLARED BLIND SPOT: a config with no answerable queries publishes a fake 0.0",
        build=lambda: _unanswerable([SCORE_GAP] * 4),
        # ⚠️ THIS PINS A VALUE I BELIEVE IS WRONG, deliberately.
        #
        # `recall/eval/metrics.py` states the convention: a rate with no data is NaN, because
        # 0.0 "would read as a PERFECT superseded-trust rate and a CATASTROPHIC accuracy at the
        # same time" and NaN "forces publishers to render 'n/a' instead of a fake number".
        # `_score_config` does not honour it: `harness.py:168` uses `mean(ps) if ps else 0.0`
        # for p_at_5, r_at_5, mrr and ndcg_at_10, while fcr_with_guard on the SAME return object
        # IS NaN-on-empty via `false_confident_rate`. One object, two conventions.
        #
        # Latent, not active: the shipped eval corpus has answerable queries. Pinned as CURRENT
        # BEHAVIOUR rather than fixed here, because changing the empty-case semantics of four
        # published metrics deserves its own reviewed diff and not a quiet ride inside a
        # test-only change. Whoever fixes it changes these four zeros to NaN and this comment.
        expected={
            "p_at_5": 0.0, "r_at_5": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0,
            "fcr_with_guard": 0.0,
        },
        does_not_catch="the defect it documents. It asserts today's wrong value, so it will "
                       "go red when the inconsistency is FIXED, which is the intended signal",
    ),
)
