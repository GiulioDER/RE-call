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

from recall.eval.harness import ARM_ENTAIL_ONLY, ARM_STACKED, ARM_THRESHOLD
from recall.eval.promotion.aggregate import UnpairedArms
from recall.eval.promotion.manifest import FrozenQuestion
from recall.eval.promotion.records import QuestionRecord


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
    """`DEPTH` rows with `gold` at 1-based `rank`, or absent from the list when `rank` is None.

    `rank` must be `None` or within `1..DEPTH`. Without this check, `rank <= 0` would silently
    index `ids[-1]` via Python's negative-index wraparound and place `gold` at the LAST row
    instead of raising, which is a fixture bug that would read exactly like a retrieval defect.
    `rank > DEPTH` already fails loudly with `IndexError`; this makes both directions fail the
    same way, on purpose, rather than leaving one of them to an accident of list indexing.
    """
    if rank is not None and not (1 <= rank <= DEPTH):
        raise ValueError(f"rank must be None or within 1..{DEPTH}, got {rank}")
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
        defect="a config with no answerable queries must publish NaN, not a fake 0.0",
        build=lambda: _unanswerable([SCORE_GAP] * 4),
        # `recall/eval/metrics.py` states the convention: a rate with no data is NaN, because
        # 0.0 "would read as a PERFECT superseded-trust rate and a CATASTROPHIC accuracy at the
        # same time" and NaN "forces publishers to render 'n/a' instead of a fake number".
        # `_score_config` now honours it via `_mean_or_nan` (harness.py) for p_at_5, r_at_5,
        # mrr and ndcg_at_10, matching fcr_with_guard on the SAME return object, which was
        # already NaN-on-empty via `false_confident_rate`. One object, one convention.
        #
        # `test_surface_a_member_reports_its_closed_form` compares NaN fields with `math.isnan`
        # rather than `pytest.approx` (which does not match NaN without `nan_ok=True`), so this
        # member still fails loudly if `_score_config` ever regresses back to a literal 0.0.
        expected={
            "p_at_5": float("nan"), "r_at_5": float("nan"), "mrr": float("nan"),
            "ndcg_at_10": float("nan"),
            "fcr_with_guard": 0.0,
        },
        does_not_catch="whether a DOWNSTREAM renderer (results_to_markdown, save_charts) "
                       "actually turns this NaN into 'n/a' or a blank bar rather than the "
                       "literal string 'nan'; that is a separate rendering-level assertion, "
                       "not this surface-A scoring check",
    ),
)

#: Enough paired questions that `_paired_sign_p`'s ONE-sided exact permutation test over
#: all-positive deltas clears 0.05 with room to spare: 0.5**12 = 0.000244. Eight would already
#: do it (0.5**8 = 0.0039); twelve is chosen so the member does not sit near the boundary of a
#: test it does not otherwise pin.
N_PAIRED = 12

#: One corpus, so Holm correction over a single p-value leaves it unchanged and the member is
#: about the gate's decision rather than about the correction.
CORPUS = "fleet"

GATE_KWARGS = {
    "manifest_digest": "fleet-digest",
    "baseline_label": "baseline",
    "candidate_label": "candidate",
    "security_green": True,
    "latency_budget_ms": 1000.0,
    "certified_latency_p95_ms": 100.0,
    # 200 rather than the 10_000 default: no member asserts on a bootstrap-derived interval
    # BOUND, only on whether the lower bound clears zero, and with every delta identical at
    # +1.0 or 0.0 the interval is degenerate at that value for any sample count.
    "bootstrap_samples": 200,
}


@dataclass(frozen=True)
class GateExpectation:
    """What the gate must do. `raises` and the other two are mutually exclusive."""

    promoted: bool | None = None
    failure_contains: str | None = None
    raises: type[Exception] | None = None
    #: Substring the raised exception's message must contain. Optional, but strongly
    #: recommended whenever more than one code path can raise the same `raises` type: a type
    #: check alone cannot distinguish two failure modes that happen to share a base class.
    raises_contains: str | None = None

    def __post_init__(self) -> None:
        if self.raises is not None and (
            self.promoted is not None or self.failure_contains is not None
        ):
            raise ValueError(
                "raises is set together with promoted or failure_contains. The test's raises "
                "branch returns before checking either, so a stale value left on the other two "
                "would silently never be checked; setting only one family makes that visible "
                "instead of hiding it behind a dead field."
            )
        if self.raises_contains is not None and self.raises is None:
            raise ValueError(
                "raises_contains is set without raises. There is no exception to read a "
                "message from without raises, so this field alone can never be checked."
            )
        if self.raises is None and self.promoted is None and self.failure_contains is None:
            raise ValueError(
                "none of raises, promoted or failure_contains is set. A GateExpectation that "
                "asserts nothing would pass no matter what the gate did, which is the exact "
                "silent-pass failure mode this fleet exists to catch."
            )


def _record(
    qid: str,
    gold: str | None,
    retrieved: tuple[str, ...],
    *,
    verdict: str = "ok",
    input_hash: str = "fleet-hash",
) -> QuestionRecord:
    """One arm's outcome for one question.

    `gold=None` means unanswerable: `QuestionRecord.answerable` is
    `bool(expected_relevance_labels)`, so an empty tuple IS the unanswerable label.
    `verdict="abstained"` is what `.abstained` reads, and it must not be `"unverified"` on
    every row of an arm or `build_safety` reports superseded_trust_rate as NOT MEASURED.
    """
    return QuestionRecord(
        question_id=qid,
        corpus=CORPUS,
        expected_relevance_labels=(gold,) if gold else (),
        retrieved_chunk_ids=retrieved,
        rank_positions=tuple(range(1, len(retrieved) + 1)),
        dense_cosine=0.8,
        confidence=0.8,
        trust_verdict=verdict,
        embedding_profile_id="scripted",
        retrieval_profile="legacy",
        generation="legacy",
        candidate_pool=20,
        reranking_status="not_configured",
        stage_timings={"total": 10.0},
        input_hash=input_hash,
        output_hash=f"out-{qid}",
    )


def _frozen(qid: str, gold: str | None) -> FrozenQuestion:
    return FrozenQuestion(
        question_id=qid,
        corpus=CORPUS,
        input_hash="fleet-hash",
        expected_relevance_labels=(gold,) if gold else (),
    )


def _hit(qid: str) -> tuple[str, ...]:
    """Gold at rank 1, so hit@5 is 1.0."""
    return (f"gold_{qid}.md:0",) + tuple(f"filler_{qid}_{i}.md:0" for i in range(9))


def _miss(qid: str) -> tuple[str, ...]:
    """Gold at rank 6, so hit@5 is 0.0 while the id is still retrieved."""
    return (
        tuple(f"filler_{qid}_{i}.md:0" for i in range(5))
        + (f"gold_{qid}.md:0",)
        + tuple(f"tail_{qid}_{i}.md:0" for i in range(4))
    )


def _arms(
    *,
    candidate_hits: bool,
    baseline_hits: bool,
    candidate_abstains_on_answerable: bool = False,
    include_unanswerable: bool = True,
    drop_from_candidate: str | None = None,
):
    """Build a paired pair of arms plus the frozen manifest they must both cover."""
    frozen, baseline, candidate = [], [], []
    for index in range(N_PAIRED):
        qid = f"q{index:02d}"
        gold = f"gold_{qid}.md:0"
        frozen.append(_frozen(qid, gold))
        baseline.append(_record(qid, gold, _hit(qid) if baseline_hits else _miss(qid)))
        if qid == drop_from_candidate:
            continue
        candidate.append(
            _record(
                qid,
                gold,
                _hit(qid) if candidate_hits else _miss(qid),
                verdict="abstained" if candidate_abstains_on_answerable else "ok",
            )
        )
    if include_unanswerable:
        # Both arms need at least one unanswerable question or `build_safety` raises on a NaN
        # false_confidence. `verdict="abstained"` is the CORRECT behaviour on an unanswerable
        # question, so gap_false_confident_rate is 0.0 on both arms.
        for index in range(2):
            qid = f"u{index}"
            frozen.append(_frozen(qid, None))
            baseline.append(_record(qid, None, (), verdict="abstained"))
            candidate.append(_record(qid, None, (), verdict="abstained"))
    return baseline, candidate, frozen, dict(GATE_KWARGS)


SURFACE_B: tuple[FleetMember, ...] = (
    FleetMember(
        name="treatment-strictly-better",
        defect="none: the clean twin, a candidate that must be allowed through",
        build=lambda: _arms(baseline_hits=False, candidate_hits=True),
        # Every paired delta is +1.0 (baseline gold at rank 6 = hit@5 0.0, candidate at rank 1
        # = 1.0), so the bootstrap interval is degenerate at 1.0 and its lower bound clears
        # zero. `_paired_sign_p` is a ONE-sided exact permutation test: with every delta
        # identical and maximal, exactly 1 of the 2**12 = 4096 sign permutations (all-positive)
        # ties the observed mean, giving p = 0.5**12 = 0.000244 < 0.05; both safety deltas are
        # 0.0; superseded_trust_rate is exactly 0.0; security is green; latency is 100.0
        # against a 1000.0 budget. No failure remains, so promoted is True.
        expected=GateExpectation(promoted=True),
        does_not_catch="whether the bootstrap interval is correctly WIDE; every delta here is "
                       "identical, so the interval is degenerate and its width is never tested",
    ),
    FleetMember(
        name="identical-arms",
        defect="candidate identical to baseline, so there is no improvement to promote",
        build=lambda: _arms(baseline_hits=True, candidate_hits=True),
        # Every delta is exactly 0.0, so the interval does not clear zero and no corpus
        # improves. The gate must refuse.
        expected=GateExpectation(
            promoted=False, failure_contains="bootstrap interval does not clear zero"
        ),
        does_not_catch="a candidate that is better on MRR but not on hit@5; the gate's primary "
                       "axis is hit@5 and no member exercises the MRR reported beside it",
    ),
    FleetMember(
        name="safety-regressed",
        defect="quality improves while the candidate wrongly abstains on every answerable question",
        build=lambda: _arms(
            baseline_hits=False, candidate_hits=True, candidate_abstains_on_answerable=True
        ),
        # false_abstention goes 0.0 -> 1.0, a regression far beyond the 0.02 tolerance, so the
        # safety axis must veto a candidate that wins on quality. This is the member that
        # proves the safety axis can FIRE rather than merely being reported.
        expected=GateExpectation(
            promoted=False, failure_contains="false abstention regresses"
        ),
        does_not_catch="a regression just BELOW the two-point tolerance; it tests that the "
                       "check fires, not where its boundary sits",
    ),
    FleetMember(
        name="nan-safety-class",
        defect="an arm with no unanswerable questions, so a safety rate has no data",
        build=lambda: _arms(
            baseline_hits=False, candidate_hits=True, include_unanswerable=False
        ),
        # `build_safety` computes gap_false_confident_rate over an empty class, which is NaN,
        # and raises rather than letting it through. This is the 08-09 failure shape exactly:
        # the gate compares with `>`, every comparison against NaN is False, so an empty class
        # would read identically to a passed check. Documented in aggregate.py; executed here.
        # `raises_contains` pins the NaN/empty-class reason specifically, so a future change
        # that made this member's arms UNPAIRED or VACUOUS instead (both also ValueError,
        # raised earlier in build_gate_input than this NaN check) cannot pass by accident.
        expected=GateExpectation(
            raises=ValueError,
            raises_contains="are NaN because their class is empty",
        ),
        does_not_catch="a PARTIALLY empty class; one unanswerable question is enough to make "
                       "the rate non-NaN while still being far too few to mean anything",
    ),
    FleetMember(
        name="unpaired-arms",
        defect="a frozen question missing from the candidate arm",
        build=lambda: _arms(
            baseline_hits=False, candidate_hits=True, drop_from_candidate="q00"
        ),
        # A paired test over a set that quietly shrank is not the test that was declared.
        # `raises_contains` pins the manifest-coverage reason specifically, distinguishing it
        # from `nan-safety-class`'s NaN reason above (also a ValueError subclass).
        expected=GateExpectation(
            raises=UnpairedArms,
            raises_contains="does not cover the frozen manifest exactly",
        ),
        does_not_catch="a question present in both arms but with a DIFFERENT input_hash, which "
                       "aggregate.py also refuses and which no member exercises",
    ),
    FleetMember(
        name="latency-pending",
        defect="an otherwise promotable candidate with no certified latency measurement",
        build=lambda: (
            *_arms(baseline_hits=False, candidate_hits=True)[:3],
            {**GATE_KWARGS, "certified_latency_p95_ms": None},
        ),
        # `certified_latency_p95_ms` defaults to None, which the gate treats as PENDING, and
        # PENDING FAILS. It is the single explicit door named in aggregate.py's docstring, and
        # a door that silently stopped blocking would look exactly like one never opened.
        expected=GateExpectation(promoted=False, failure_contains="PENDING"),
        does_not_catch="a NaN latency, which the gate refuses on a different branch and which "
                       "no member exercises",
    ),
)


@dataclass(frozen=True)
class TrustEvalFixture:
    """Everything `test_fleet.run_surface_c` needs to call `run_trust_eval` against a store
    built by `tests.fleet.scripted.QueryKeyedTrustStore`."""

    queries: list[dict]
    script: dict[str, list[tuple[str, float]]]
    supersession_edges: dict[str, str]


@dataclass(frozen=True)
class NearMissEvalFixture:
    """Everything `test_fleet.run_surface_d` needs to call `run_nearmiss_eval` against a store
    built by `tests.fleet.scripted.QueryKeyedTrustStore`."""

    queries: list[dict]
    nearmiss: list[dict]
    script: dict[str, list[tuple[str, float]]]


#: Shared plain (non-trust) queries every SURFACE_C / SURFACE_D member calibrates against: one
#: answerable sample at SCORE_CONFIDENT and one unanswerable sample at SCORE_GAP.
#:
#: `recall.calibration.best_threshold` fits from these via its `_quantile` (lower tail) and
#: `_upper_quantile` (mirrored upper tail) helpers, both of which at n=1
#: collapse to index 0 for ANY percentile: so the 5th-percentile answerable floor is just
#: SCORE_CONFIDENT and the 95th-percentile unanswerable ceiling is just SCORE_GAP. The fitted
#: threshold is therefore the exact midpoint: `floor((0.80 + 0.20) / 2 * 1000) / 1000 == 0.5`.
#: Every closed form in SURFACE_C and SURFACE_D assumes this threshold.
_PLAIN_QUERIES: list[dict] = [
    {"id": "p1", "query": "plain query", "answerable": True, "relevant_ids": ["plain_gold.md:0"]},
    {"id": "p2", "query": "plain gap query", "answerable": False, "relevant_ids": []},
]
_PLAIN_SCRIPT: dict[str, list[tuple[str, float]]] = {
    "plain query": [("plain_gold.md:0", SCORE_CONFIDENT)],
    "plain gap query": [("plain_noise.md:0", SCORE_GAP)],
}


def _trust_catches_scripted_supersession() -> TrustEvalFixture:
    trust_queries = [
        {
            "id": "t1", "query": "trust successor query", "trust": True, "expect": "successor",
            "stale_ids": ["stale.md:0"], "successor_ids": ["successor.md:0"],
        },
        {
            "id": "t2", "query": "trust abstain query", "trust": True, "expect": "abstain",
            "stale_ids": ["stale2.md:0"], "successor_ids": [],
        },
    ]
    script = dict(_PLAIN_SCRIPT)
    # successor scores BELOW threshold on its own: it only reaches "ok" via the promotion the
    # stale hit's own confident score licenses (see the member's derivation comment).
    script["trust successor query"] = [("stale.md:0", SCORE_CONFIDENT), ("successor.md:0", 0.40)]
    script["trust abstain query"] = [("stale2.md:0", SCORE_GAP)]
    return TrustEvalFixture(
        queries=_PLAIN_QUERIES + trust_queries,
        script=script,
        supersession_edges={"stale.md": "successor.md"},
    )


def _trust_misses_unscripted_supersession() -> TrustEvalFixture:
    trust_queries = [
        {
            "id": "t1", "query": "trust query with missing edge", "trust": True,
            "expect": "successor", "stale_ids": ["stale.md:0"],
            "successor_ids": ["successor.md:0"],
        },
    ]
    script = dict(_PLAIN_SCRIPT)
    script["trust query with missing edge"] = [("stale.md:0", SCORE_CONFIDENT)]
    return TrustEvalFixture(
        queries=_PLAIN_QUERIES + trust_queries,
        script=script,
        supersession_edges={},  # the point of this member: no edge recorded for "stale.md"
    )


#: `run_trust_eval` publishes these on `TrustEvalResult`. Its Wilson-CI companions
#: (`*_ci` fields) and `n_*` sample counts are pure functions of the same flags these members
#: already drive and are declared, not silently skipped, in `test_fleet.py`'s roll-up.
SURFACE_C: tuple[FleetMember, ...] = (
    FleetMember(
        name="trust-catches-scripted-supersession",
        defect="none: the clean twin — a supersession edge IS recorded, so the trust arm must "
               "refuse the stale hit and promote its successor instead",
        build=_trust_catches_scripted_supersession,
        # t1 (expect=successor): "stale.md:0" scores 0.80 (>= threshold 0.5) with a scripted
        # edge stale.md -> successor.md. `recall/trust.py:_verdict` resolves the edge BEFORE it
        # ever reads the score, so the stale hit's verdict is "superseded", not "ok" —
        # str_trust does not count it. Because the stale hit's OWN score cleared the threshold,
        # `evaluate`'s promotion step treats its successor ("successor.md:0", scored 0.40,
        # itself below threshold) as though it had been the confident answer and promotes it
        # from low_confidence to ok. That promoted hit is the query's only "ok" hit, so
        # trust_coverage counts it and successor_acc sees ok_keys[0] equal the successor.
        # t2 (expect=abstain): "stale2.md:0" scores 0.20 (< threshold); no edge is needed — it
        # is low_confidence on score alone, ok=[], so the query abstains as expected.
        # str_baseline / str_recency read no verdicts at all, only the top/newest SCRIPTED hit,
        # which is the stale doc on both trust queries here -> 2/2 = 1.0 for both.
        # str_trust: neither t1 nor t2 counts a stale id as "ok" (t1's ok hit is the SUCCESSOR,
        # not the stale id; t2 has no ok hit at all) -> 0/2 = 0.0.
        # trust_coverage: t1 has one ok hit, t2 has none -> 1/2 = 0.5.
        # successor_acc (n=1, only t1 has expect=="successor"): ok_keys[0] is the successor ->
        # 1/1 = 1.0.
        # abstain_acc (n=1, only t2 has expect!="successor"): tres.abstained is True -> 1/1 = 1.0.
        # mrr_answerable_(baseline|trust): the one plain answerable query's gold scores 0.80,
        # clears threshold, keeps rank 1 under both baseline and trust search -> 1.0 / 1.0.
        expected={
            "str_baseline": 1.0, "str_recency": 1.0, "str_trust": 0.0, "trust_coverage": 0.5,
            "successor_acc": 1.0, "abstain_acc": 1.0,
            "mrr_answerable_baseline": 1.0, "mrr_answerable_trust": 1.0,
        },
        does_not_catch="whether the SAME queries still behave when the corpus never recorded "
                       "the edge at all; paired with trust-misses-unscripted-supersession, "
                       "which is exactly that case. Also does not catch str_recency's real "
                       "tie-break: every scripted hit here carries indexed_at=None, so the "
                       "'prefer the newest' comparison never distinguishes two real timestamps, "
                       "and touch_stale is passed False, so the re-sync simulation is unreached",
    ),
    FleetMember(
        name="trust-misses-unscripted-supersession",
        defect="a stale memory whose supersedes: edge was never recorded reaches str_trust as "
               "an undetected false positive — the trust layer has no signal beyond the score",
        build=_trust_misses_unscripted_supersession,
        # Same threshold (0.5). "stale.md:0" scores 0.80 with NO edge scripted (supersession()
        # returns {}), so `resolve_successor` returns None immediately and `_verdict` falls
        # through to score >= threshold -> "ok". The eval's OWN label still says this id is
        # stale (`stale_ids=["stale.md:0"]`), so trust_flags counts it: str_trust = 1/1 = 1.0.
        # This member's build also differs from the paired member's in more than the edge (one
        # trust query and a one-row script here, vs two trust queries and a two-row script
        # there), so its 1.0 next to the paired member's 0.0 is not itself a single-variable
        # proof. Toggling ONLY supersession_edges on a fixture held otherwise fixed is: measured
        # directly, this member's own fixture with an edge added ("stale.md" -> "successor.md")
        # moves str_trust from 1.0 to 0.0, and the paired member's fixture with its edge removed
        # moves str_trust from 0.0 to 0.5 (not to 1.0 — that fixture has a second, abstain-only
        # trust query whose str_trust contribution the edge alone cannot touch). successor_acc:
        # ok_keys[0] is "stale.md:0", not the successor -> 0/1 = 0.0. trust_coverage: one ok hit
        # -> 1/1 = 1.0. No expect=="abstain" query exists in this fixture, so abst_flags is
        # empty and abstain_acc is NaN by `fraction_true`'s own empty-input convention
        # (`recall/eval/metrics.py`) — asserted as NaN, not skipped.
        expected={
            "str_baseline": 1.0, "str_recency": 1.0, "str_trust": 1.0, "trust_coverage": 1.0,
            "successor_acc": 0.0, "abstain_acc": float("nan"),
            "mrr_answerable_baseline": 1.0, "mrr_answerable_trust": 1.0,
        },
        does_not_catch="an edge that exists but is written with the wrong basename or points "
                       "at an ambiguous target (a basename two files share); this member's edge "
                       "is simply absent, not malformed",
    ),
)


def _nearmiss_above_threshold_fools_cosine_guard() -> NearMissEvalFixture:
    script = dict(_PLAIN_SCRIPT)
    script["near miss query"] = [("distractor.md:0", SCORE_CONFIDENT)]
    return NearMissEvalFixture(
        queries=list(_PLAIN_QUERIES),
        nearmiss=[{"id": "nm1", "query": "near miss query"}],
        script=script,
    )


def _nearmiss_below_threshold_reads_as_an_ordinary_gap() -> NearMissEvalFixture:
    script = dict(_PLAIN_SCRIPT)
    script["near miss query"] = [("distractor.md:0", SCORE_GAP)]
    return NearMissEvalFixture(
        queries=list(_PLAIN_QUERIES),
        nearmiss=[{"id": "nm1", "query": "near miss query"}],
        script=script,
    )


#: `run_nearmiss_eval` publishes these on `NearMissEvalResult`, one row per `recall.eval.harness`
#: ARM. `entail_latency_ms_mean` / `query_latency_ms_mean` are wall-clock timings with no closed
#: form to derive — declared, not covered, in `test_fleet.py`'s roll-up.
SURFACE_D: tuple[FleetMember, ...] = (
    FleetMember(
        name="nearmiss-above-threshold-fools-the-cosine-guard",
        defect="none: the clean twin — a near-miss distractor scores ABOVE the gap threshold, "
               "exactly the failure class near-miss testing exists to name",
        build=_nearmiss_above_threshold_fools_cosine_guard,
        # threshold=0.5 (same derivation as SURFACE_C). The near-miss distractor scores 0.80: it
        # clears the threshold on cosine alone, so ARM_THRESHOLD never abstains on it ->
        # nearmiss_fcr = 1/1 = 1.0 — a cosine threshold cannot tell a confident distractor from a
        # confident answer, which is the whole reason entailment exists.
        # ARM_STACKED uses `scripted.AlwaysEntailJudge`, which never demotes an ok hit, so it
        # reproduces ARM_THRESHOLD exactly on every rate — the same degeneracy
        # `tests/test_eval_nearmiss.py`'s `test_accept_all_judge_cannot_change_the_threshold_arm`
        # pins against a real database; this member pins it without one.
        # ARM_ENTAIL_ONLY's threshold is -1.0 (passes any real cosine), so the distractor is
        # "ok" there too, regardless of its score -> nearmiss_fcr = 1.0.
        # gap_fcr / false_abstain / mrr_answerable come from the SHARED plain/gap queries and do
        # not depend on the near-miss score at all: the far-gap query (0.20) stays below every
        # non-permissive threshold and abstains under THRESHOLD/STACKED (gap_fcr 0.0), but
        # ENTAIL_ONLY's permissive threshold passes it too -> gap_fcr 1.0 there. The answerable
        # query (0.80) is never wrongly abstained on in any arm -> false_abstain 0.0 throughout,
        # and its gold is the query's only hit -> mrr_answerable 1.0 throughout.
        expected={
            ARM_THRESHOLD: {
                "nearmiss_fcr": 1.0, "gap_fcr": 0.0, "false_abstain": 0.0, "mrr_answerable": 1.0,
            },
            ARM_STACKED: {
                "nearmiss_fcr": 1.0, "gap_fcr": 0.0, "false_abstain": 0.0, "mrr_answerable": 1.0,
            },
            ARM_ENTAIL_ONLY: {
                "nearmiss_fcr": 1.0, "gap_fcr": 1.0, "false_abstain": 0.0, "mrr_answerable": 1.0,
            },
        },
        does_not_catch="whether a judge that can actually DISAGREE demotes this near-miss; "
                       "AlwaysEntailJudge never does, so ARM_STACKED's equality with "
                       "ARM_THRESHOLD here proves the judge is wired in and inert when it "
                       "agrees, not that it can rescue a confident near-miss when it disagrees",
    ),
    FleetMember(
        name="nearmiss-below-threshold-reads-as-an-ordinary-gap",
        defect="the near-miss distractor happens to score low, so the cosine guard catches it "
               "for the wrong reason — indistinguishable from an ordinary far gap",
        build=_nearmiss_below_threshold_reads_as_an_ordinary_gap,
        # Only the near-miss score changes (0.20 instead of 0.80): it now falls below the 0.5
        # threshold like an ordinary gap, so ARM_THRESHOLD and ARM_STACKED abstain on it ->
        # nearmiss_fcr = 0/1 = 0.0, the OPPOSITE of the paired member, driven by nothing but
        # what the store returns for one query. ENTAIL_ONLY's threshold is still permissive
        # (-1.0), so it passes the distractor regardless of its now-low score, unchanged from
        # the paired member -> nearmiss_fcr 1.0 there too — a real, derived consequence of a
        # permissive threshold plus an agreeable judge, not a second independent proof (see
        # does_not_catch below). gap_fcr / false_abstain / mrr_answerable are unaffected by the
        # near-miss score -> identical to the paired member in every arm.
        expected={
            ARM_THRESHOLD: {
                "nearmiss_fcr": 0.0, "gap_fcr": 0.0, "false_abstain": 0.0, "mrr_answerable": 1.0,
            },
            ARM_STACKED: {
                "nearmiss_fcr": 0.0, "gap_fcr": 0.0, "false_abstain": 0.0, "mrr_answerable": 1.0,
            },
            ARM_ENTAIL_ONLY: {
                "nearmiss_fcr": 1.0, "gap_fcr": 1.0, "false_abstain": 0.0, "mrr_answerable": 1.0,
            },
        },
        does_not_catch="ARM_ENTAIL_ONLY ever telling a near-miss apart from a genuine gap: its "
                       "permissive threshold plus this fleet's agreeable judge makes it "
                       "confident on both members here, so no member shows entail-only "
                       "distinguishing the two classes it exists to separate",
    ),
)
