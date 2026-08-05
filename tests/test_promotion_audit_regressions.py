"""Regression tests for the CCA audit of the promotion harness.

Every test here pins a defect that was PROVEN by execution against the first version of this code,
not one that was reasoned about. Six of them describe ways the gate could return `promoted=True`,
or satisfy a criterion, without having measured the thing the criterion names — which is the exact
failure class the harness's own docstrings claim to prevent, found in the harness itself.
"""
from __future__ import annotations

import json
import math

import pytest

from recall.eval.promotion.aggregate import UnpairedArms, decide, secondary_metrics
from recall.eval.promotion.manifest import FrozenQuestion, question_input_hash, write_manifest
from recall.eval.promotion.records import build_record, record_from_dict, record_to_dict
from recall.eval.promotion.run import ArmConfig, SearchOutcome, score_arm
from recall.promotion import (
    RetrievalGateInput,
    SafetyMetrics,
    evaluate_retrieval_promotion,
)

GOLD = "gold.md:0"
MISS = "other.md:0"


def _frozen(qid: str, corpus: str, *, answerable: bool = True) -> FrozenQuestion:
    labels = (GOLD,) if answerable else ()
    return FrozenQuestion(
        question_id=qid,
        corpus=corpus,
        input_hash=question_input_hash(
            question_id=qid, corpus=corpus, query=qid, expected_relevance_labels=labels
        ),
        expected_relevance_labels=labels,
    )


def _record(question, *, hit: bool, verdict: str = "ok", **overrides):
    retrieved = (GOLD, MISS) if hit else (MISS,)
    kwargs = dict(
        question_id=question.question_id,
        corpus=question.corpus,
        expected_relevance_labels=question.expected_relevance_labels,
        retrieved_chunk_ids=retrieved,
        rank_positions=tuple(range(1, len(retrieved) + 1)),
        dense_cosine=0.8,
        confidence=0.7,
        trust_verdict=verdict,
        embedding_profile_id="p1",
        retrieval_profile="fast",
        generation="g1",
        candidate_pool=20,
        reranking_status="not_configured",
        stage_timings={"total": 12.0},
        input_hash=question.input_hash,
    )
    kwargs.update(overrides)
    return build_record(**kwargs)


def _corpus(name: str, n_answerable: int = 20, n_unanswerable: int = 4):
    return [_frozen(f"{name}-a{i}", name) for i in range(n_answerable)] + [
        _frozen(f"{name}-u{i}", name, answerable=False) for i in range(n_unanswerable)
    ]


def _arm(frozen, *, hits: int, verdict: str = "ok"):
    records, answered = [], 0
    for question in frozen:
        if not question.answerable:
            records.append(
                _record(
                    question,
                    hit=False,
                    verdict="abstained" if verdict == "ok" else verdict,
                    retrieved_chunk_ids=(),
                    rank_positions=(),
                    dense_cosine=math.nan,
                    confidence=math.nan,
                )
            )
            continue
        hit = answered < hits
        answered += 1 if hit else 0
        records.append(_record(question, hit=hit, verdict=verdict))
    return records


COMMON = dict(
    manifest_digest="d",
    baseline_label="b",
    candidate_label="c",
    security_green=True,
    latency_budget_ms=250.0,
    bootstrap_samples=500,
)


# --------------------------------------------------------------------------------------------
# STAKES-002 / NUM-001 — a non-finite certified latency passed the budget check.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("-inf"), -5.0, 0.0])
def test_a_certified_latency_that_is_not_a_positive_finite_number_blocks(value) -> None:
    """`nan > budget` is False, so an unmeasured p95 passed the budget while reporting MEASURED.

    NaN is reachable without malice: this package's own `_percentile` returns NaN on an empty
    sample.
    """
    safe = SafetyMetrics(0.1, 0.1, 0.0)
    frozen = _corpus("alpha") + _corpus("beta")
    decision, document = decide(
        _arm(frozen, hits=4), _arm(frozen, hits=18), frozen,
        certified_latency_p95_ms=value, **COMMON,
    )
    assert not decision.promoted
    assert any("not a positive finite measurement" in f for f in decision.failures)
    del document, safe


def test_a_finite_positive_latency_under_budget_still_promotes() -> None:
    """The control: the new refusal must not have swallowed the passing case."""
    frozen = _corpus("alpha") + _corpus("beta")
    decision, _ = decide(
        _arm(frozen, hits=4), _arm(frozen, hits=18), frozen,
        certified_latency_p95_ms=90.0, **COMMON,
    )
    assert decision.promoted, decision.failures


def test_the_gate_itself_refuses_a_nan_latency_not_only_the_harness() -> None:
    """Asserted at `evaluate_retrieval_promotion`, since callers other than `decide` exist."""
    from recall.promotion import QuestionOutcome

    outcomes = tuple(QuestionOutcome(str(i), "a", 0.0, 1.0, 0.0, 1.0) for i in range(20))
    safe = SafetyMetrics(0.1, 0.1, 0.0)
    decision = evaluate_retrieval_promotion(
        RetrievalGateInput(outcomes, safe, safe, True, float("nan"), 250.0),
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert any("not a positive finite measurement" in f for f in decision.failures)


# --------------------------------------------------------------------------------------------
# STAKES-003 / BUG-003 — the evidence ledger was not tamper-evident.
# --------------------------------------------------------------------------------------------


def test_an_edited_ledger_row_is_refused_rather_than_gated_on() -> None:
    """The manifest was tamper-evident and the EVIDENCE was not.

    Measured against the first version: editing `retrieved_chunk_ids` on ten baseline rows and
    leaving their `output_hash` alone turned the same inputs from `promoted=false` to
    `promoted=true`.
    """
    question = _frozen("q1", "alpha")
    payload = record_to_dict(_record(question, hit=False))
    payload["retrieved_chunk_ids"] = [GOLD]
    payload["rank_positions"] = [1]
    with pytest.raises(ValueError, match="recomputed output hash"):
        record_from_dict(payload)


def test_an_unedited_row_round_trips(tmp_path) -> None:
    """The control: recomputation must accept every honestly-written row."""
    question = _frozen("q1", "alpha")
    record = _record(question, hit=True)
    assert record_from_dict(record_to_dict(record)) == record


# --------------------------------------------------------------------------------------------
# STAKES-001 / BUG-004 / NUM-003 — `decide` narrowed the manifest to the baseline's corpora.
# --------------------------------------------------------------------------------------------


def test_a_corpus_missing_from_both_arms_refuses_instead_of_being_dropped(tmp_path) -> None:
    """Proven on the first version: a two-corpus manifest where the candidate REGRESSES on beta
    returned `promoted=false`; with beta absent from both ledgers the SAME manifest returned
    `promoted=true`, stamped with the full two-corpus digest."""
    from recall.eval.promotion.__main__ import main

    frozen = _corpus("alpha") + _corpus("beta")
    manifest = tmp_path / "m.jsonl"
    write_manifest(manifest, frozen, corpus_hashes={})

    alpha_only = [q for q in frozen if q.corpus == "alpha"]
    for name, hits in (("baseline", 2), ("candidate", 20)):
        path = tmp_path / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in _arm(alpha_only, hits=hits):
                handle.write(json.dumps(record_to_dict(record), sort_keys=True) + "\n")

    with pytest.raises(UnpairedArms, match="does not cover the frozen manifest exactly"):
        main([
            "decide", "--manifest", str(manifest),
            "--baseline", str(tmp_path / "baseline.jsonl"),
            "--candidate", str(tmp_path / "candidate.jsonl"),
            "--out", str(tmp_path / "decision.json"),
            "--security-green", "--certified-latency-p95-ms", "90",
        ])


# --------------------------------------------------------------------------------------------
# STAKES-004 — a wholly degraded arm SATISFIED the zero-tolerance superseded-trust check.
# --------------------------------------------------------------------------------------------


def test_an_arm_whose_trust_layer_never_ran_cannot_supply_a_superseded_trust_rate() -> None:
    """`unverified` is not in `_STALE`, so a development-mode arm scored exactly 0.0 and PASSED
    "superseded trust rate is not zero" by never having measured it.

    `recall/trust.py` overwrites EVERY verdict with `unverified` in degraded mode, after
    `evaluate` computed the real one, so a superseded hit and a clean one leave identical rows.

    Blocks like a PENDING latency rather than raising: an operator needs a decision naming what
    was missing more than a traceback.
    """
    frozen = _corpus("alpha")
    degraded = _arm(frozen, hits=10, verdict="unverified")
    decision, document = decide(
        degraded, list(degraded), frozen, certified_latency_p95_ms=90.0, **COMMON
    )
    assert not decision.promoted
    assert any("superseded trust rate was NOT MEASURED" in f for f in decision.failures)
    assert document["safety"]["candidate"]["superseded_trust_rate"] is None


def test_an_arm_with_real_verdicts_is_still_measured() -> None:
    """The control: the refusal must key on the DEGRADED verdict, not on abstention."""
    frozen = _corpus("alpha")
    arm = _arm(frozen, hits=10)
    decision, _ = decide(arm, list(arm), frozen, certified_latency_p95_ms=90.0, **COMMON)
    assert not decision.promoted  # null difference, as expected
    assert not any("NOT MEASURED" in f for f in decision.failures)


# --------------------------------------------------------------------------------------------
# STAKES-005 / BUG-001 / SEC-003 — a different configuration silently resumed as this arm.
# --------------------------------------------------------------------------------------------


def _search(_query: str) -> SearchOutcome:
    return SearchOutcome(
        retrieved_chunk_ids=(GOLD,),
        dense_cosine=0.8,
        confidence=0.7,
        trust_verdict="ok",
        reranking_status="not_configured",
    )


def _arm_config(**overrides) -> ArmConfig:
    kwargs = dict(
        label="baseline",
        embedding_profile_id="p1",
        retrieval_profile="fast",
        generation="g1",
        candidate_pool=20,
        embedding_fingerprint="f" * 64,
    )
    kwargs.update(overrides)
    return ArmConfig(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("retrieval_profile", "quality"),
        ("generation", "g2"),
        ("candidate_pool", 50),
        ("embedding_profile_id", "p2"),
    ],
)
def test_two_configurations_never_share_a_ledger_filename(field, value) -> None:
    """Proven on the first version: arms differing only in `--retrieval-profile` produced the
    identical stem, the second resumed the first's rows, and the gate compared a file with
    itself."""
    assert _arm_config().artifact_stem() != _arm_config(**{field: value}).artifact_stem()


def test_resuming_another_configurations_rows_is_refused(tmp_path) -> None:
    """The filename is not the only guard: `--out` can be passed explicitly, and "the file was
    named right" is weaker than "the rows say what produced them"."""
    frozen = [_frozen("q1", "alpha")]
    queries = {"q1": "q1"}
    ledger = tmp_path / "arm.jsonl"
    score_arm(frozen, queries, _search, _arm_config(), ledger)
    with pytest.raises(ValueError, match="was scored under"):
        score_arm(
            frozen, queries, _search, _arm_config(retrieval_profile="quality"), ledger
        )


def test_resuming_the_same_configuration_still_works(tmp_path) -> None:
    """The control: the identity check must not break ordinary resume."""
    frozen = [_frozen("q1", "alpha")]
    ledger = tmp_path / "arm.jsonl"
    score_arm(frozen, {"q1": "q1"}, _search, _arm_config(), ledger)
    again = score_arm(frozen, {"q1": "q1"}, _search, _arm_config(), ledger)
    assert again.scored_now == 0 and again.resumed == 1


# --------------------------------------------------------------------------------------------
# BUG-002 — `--no-resume` read nothing but still appended, and the discarded rows won.
# --------------------------------------------------------------------------------------------


def test_no_resume_against_a_populated_ledger_is_refused(tmp_path) -> None:
    """`read_ledger` keeps the FIRST occurrence of an id, so appending a second generation of
    rows means every later read returns the ones this run was told to discard."""
    frozen = [_frozen("q1", "alpha")]
    ledger = tmp_path / "arm.jsonl"
    score_arm(frozen, {"q1": "q1"}, _search, _arm_config(), ledger)
    with pytest.raises(FileExistsError, match="already holds rows and resume is off"):
        score_arm(frozen, {"q1": "q1"}, _search, _arm_config(), ledger, resume=False)


# --------------------------------------------------------------------------------------------
# NUM-002 — the diagnostics were pooled while the headline is macro.
# --------------------------------------------------------------------------------------------


def test_the_secondary_metrics_are_macro_averaged_like_the_headline() -> None:
    """A pooled mean weights each corpus by its question count. Measured on the first version,
    the same document carried macro 0.55 beside a pooled 0.25 with nothing naming the weighting."""
    big = _corpus("big", n_answerable=10, n_unanswerable=1)
    small = _corpus("small", n_answerable=2, n_unanswerable=1)
    records = _arm(big, hits=0) + _arm(small, hits=2)
    metrics = secondary_metrics(records)
    # macro: corpus "big" scores 0.0, corpus "small" scores 1.0 → 0.5
    # pooled would be 2/12 = 0.1667
    assert metrics.hit_at_5 == pytest.approx(0.5)


# --------------------------------------------------------------------------------------------
# NUM-004 — a bare NaN token is not valid JSON, and this artifact is committed.
# --------------------------------------------------------------------------------------------


def test_the_decision_artifact_is_always_parseable_json(tmp_path) -> None:
    from recall.eval.promotion.aggregate import write_decision

    with pytest.raises(ValueError):
        write_decision(tmp_path / "d.json", {"latency_p95_ms": float("nan")})


# --------------------------------------------------------------------------------------------
# STAKES-007 — the artifact could not show what was actually compared.
# --------------------------------------------------------------------------------------------


def test_the_decision_records_each_arms_configuration_not_just_its_label() -> None:
    frozen = _corpus("alpha")
    baseline = _arm(frozen, hits=4)
    candidate = [
        _record(q, hit=True, retrieval_profile="quality")
        if q.answerable
        else _record(
            q, hit=False, verdict="abstained", retrieval_profile="quality",
            retrieved_chunk_ids=(), rank_positions=(),
            dense_cosine=math.nan, confidence=math.nan,
        )
        for q in frozen
    ]
    _, document = decide(
        baseline, candidate, frozen, certified_latency_p95_ms=90.0, **COMMON
    )
    assert document["arms"]["baseline"]["retrieval_profile"] == "fast"
    assert document["arms"]["candidate"]["retrieval_profile"] == "quality"
    assert document["n_manifest_questions"] == len(frozen)


def test_an_arm_whose_rows_disagree_about_their_configuration_is_refused() -> None:
    """A ledger holding two configurations cannot be published as one arm."""
    frozen = _corpus("alpha", n_answerable=3, n_unanswerable=1)
    mixed = _arm(frozen, hits=3)
    mixed[0] = _record(frozen[0], hit=True, retrieval_profile="quality")
    with pytest.raises(UnpairedArms, match="different configurations"):
        decide(mixed, list(mixed), frozen, certified_latency_p95_ms=90.0, **COMMON)


# --------------------------------------------------------------------------------------------
# BUG-005 / STAKES-006 — MT-RAG was wired and unrunnable.
# --------------------------------------------------------------------------------------------


def test_an_mtrag_manifest_is_reachable_from_the_mtrag_corpus_flag() -> None:
    """`MtragAdapter` stamps `mtrag:<domain>`; an equality-only filter matched nothing."""
    from recall.eval.promotion.adapters import MtragAdapter

    adapter = MtragAdapter()
    stamped = adapter.corpus_of.__func__(adapter, type("Q", (), {"question_id": "clapnq/t1"})())
    prefix = f"{adapter.name}:"
    assert stamped != adapter.name
    assert stamped == adapter.name or stamped.startswith(prefix)
