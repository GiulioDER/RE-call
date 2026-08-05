"""From question records to a promotion decision, and every way that must refuse to pass.

The gap matrix recorded that `tests/test_promotion.py` held ONE test and it asserted
`not decision.promoted`: "a gate that has only ever been shown to fail is not evidence that it can
pass; it is compatible with a gate that refuses everything." Both directions are exercised here,
and the pass path is exercised FIRST, because every refusal below is only meaningful against a
configuration that would otherwise have promoted.
"""
from __future__ import annotations

import math

import pytest

from recall.eval.promotion.aggregate import (
    UnpairedArms,
    VacuousArm,
    build_gate_input,
    build_outcomes,
    build_safety,
    decide,
    secondary_metrics,
)
from recall.eval.promotion.manifest import FrozenQuestion, question_input_hash
from recall.eval.promotion.records import build_record
from recall.promotion import (
    QuestionOutcome,
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


def _record(question: FrozenQuestion, *, hit: bool, abstained: bool = False, verdict: str = "ok"):
    retrieved = (GOLD, MISS) if hit else (MISS,)
    if abstained:
        retrieved = ()
    return build_record(
        question_id=question.question_id,
        corpus=question.corpus,
        expected_relevance_labels=question.expected_relevance_labels,
        retrieved_chunk_ids=retrieved,
        rank_positions=tuple(range(1, len(retrieved) + 1)),
        dense_cosine=0.8 if retrieved else math.nan,
        confidence=0.7 if retrieved else math.nan,
        trust_verdict="abstained" if abstained else verdict,
        embedding_profile_id="bge-small-symmetric-v1",
        retrieval_profile="fast",
        generation="g1",
        candidate_pool=20,
        reranking_status="not_configured",
        stage_timings={"total": 12.0},
        input_hash=question.input_hash,
    )


def _corpus(name: str, n_answerable: int = 20, n_unanswerable: int = 4):
    frozen = [_frozen(f"{name}-a{i}", name) for i in range(n_answerable)]
    frozen += [
        _frozen(f"{name}-u{i}", name, answerable=False) for i in range(n_unanswerable)
    ]
    return frozen


def _arm(frozen, *, hits: int, abstains_on_unanswerable: bool = True, verdict: str = "ok"):
    records = []
    answered = 0
    for question in frozen:
        if not question.answerable:
            records.append(
                _record(question, hit=False, abstained=abstains_on_unanswerable, verdict=verdict)
            )
            continue
        hit = answered < hits
        answered += 1 if hit else 0
        records.append(_record(question, hit=hit, verdict=verdict))
    return records


# --------------------------------------------------------------------------------------------
# The pass path, first.
# --------------------------------------------------------------------------------------------


def test_the_gate_promotes_a_real_improvement_when_every_other_condition_is_met() -> None:
    """The `promoted=True` branch, which had never executed in this repository."""
    frozen = _corpus("alpha") + _corpus("beta")
    baseline = _arm(frozen, hits=4)
    candidate = _arm(frozen, hits=18)
    decision, document = decide(
        baseline,
        candidate,
        frozen,
        manifest_digest="d",
        baseline_label="base",
        candidate_label="cand",
        security_green=True,
        latency_budget_ms=250.0,
        certified_latency_p95_ms=90.0,
        bootstrap_samples=500,
    )
    assert decision.promoted, decision.failures
    assert decision.failures == ()
    assert document["latency_status"] == "MEASURED"
    assert document["promoted"] is True


# --------------------------------------------------------------------------------------------
# Every refusal, each one flipped off the configuration above.
# --------------------------------------------------------------------------------------------


def test_the_gate_refuses_to_promote_a_null_difference() -> None:
    """baseline == candidate. The macro delta is exactly zero, so the interval cannot clear it."""
    frozen = _corpus("alpha") + _corpus("beta")
    arm = _arm(frozen, hits=12)
    decision, document = decide(
        arm,
        list(arm),
        frozen,
        manifest_digest="d",
        baseline_label="base",
        candidate_label="cand",
        security_green=True,
        latency_budget_ms=250.0,
        certified_latency_p95_ms=90.0,
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert decision.macro_hit5_delta == 0.0
    assert any("bootstrap interval does not clear zero" in f for f in decision.failures)
    assert any("Holm corrected paired significance" in f for f in decision.failures)


def test_pending_latency_blocks_promotion_rather_than_passing_silently() -> None:
    """The one gate input this program cannot supply. PENDING must fail, not be skipped."""
    frozen = _corpus("alpha") + _corpus("beta")
    decision, document = decide(
        _arm(frozen, hits=4),
        _arm(frozen, hits=18),
        frozen,
        manifest_digest="d",
        baseline_label="base",
        candidate_label="cand",
        security_green=True,
        latency_budget_ms=250.0,
        certified_latency_p95_ms=None,
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert any("latency is PENDING" in failure for failure in decision.failures)
    assert document["latency_status"] == "PENDING"
    assert document["latency"]["gate_input_p95_ms"] is None
    # The observed figure is still recorded, and labelled as what it is.
    assert document["latency"]["observed_diagnostic_only"]["candidate_p95_ms"] == 12.0


def test_pending_latency_is_the_only_difference_between_a_pass_and_this_failure() -> None:
    """Proves the PENDING branch is what fired, not some other condition of the fixture."""
    frozen = _corpus("alpha") + _corpus("beta")
    baseline, candidate = _arm(frozen, hits=4), _arm(frozen, hits=18)
    common = dict(
        manifest_digest="d",
        baseline_label="b",
        candidate_label="c",
        security_green=True,
        latency_budget_ms=250.0,
        bootstrap_samples=500,
    )
    measured, _ = decide(
        baseline, candidate, frozen, certified_latency_p95_ms=90.0, **common
    )
    pending, _ = decide(
        baseline, candidate, frozen, certified_latency_p95_ms=None, **common
    )
    assert measured.promoted
    assert not pending.promoted
    assert len(pending.failures) == 1


def test_a_measured_p95_over_budget_still_fails_the_way_it_always_did() -> None:
    """The PENDING extension must not have replaced the budget check."""
    frozen = _corpus("alpha") + _corpus("beta")
    decision, _ = decide(
        _arm(frozen, hits=4),
        _arm(frozen, hits=18),
        frozen,
        manifest_digest="d",
        baseline_label="b",
        candidate_label="c",
        security_green=True,
        latency_budget_ms=50.0,
        certified_latency_p95_ms=90.0,
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert any("p95 exceeds its profile budget" in failure for failure in decision.failures)


def test_an_ungreen_security_verification_blocks_promotion() -> None:
    frozen = _corpus("alpha") + _corpus("beta")
    decision, _ = decide(
        _arm(frozen, hits=4),
        _arm(frozen, hits=18),
        frozen,
        manifest_digest="d",
        baseline_label="b",
        candidate_label="c",
        security_green=False,
        latency_budget_ms=250.0,
        certified_latency_p95_ms=90.0,
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert any("security verification is not green" in f for f in decision.failures)


def test_a_stale_memory_served_as_the_answer_blocks_promotion() -> None:
    frozen = _corpus("alpha") + _corpus("beta")
    candidate = _arm(frozen, hits=18)
    candidate[0] = _record(frozen[0], hit=True, verdict="superseded")
    decision, _ = decide(
        _arm(frozen, hits=4),
        candidate,
        frozen,
        manifest_digest="d",
        baseline_label="b",
        candidate_label="c",
        security_green=True,
        latency_budget_ms=250.0,
        certified_latency_p95_ms=90.0,
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert any("superseded trust rate is not zero" in f for f in decision.failures)


# --------------------------------------------------------------------------------------------
# Refusals to BUILD a gate input at all.
# --------------------------------------------------------------------------------------------


def test_an_arm_missing_a_question_refuses_rather_than_dropping_the_pair() -> None:
    frozen = _corpus("alpha")
    baseline = _arm(frozen, hits=10)
    with pytest.raises(UnpairedArms, match="does not cover the frozen manifest exactly"):
        build_outcomes(baseline[:-1], baseline, frozen)


def test_a_record_scored_against_a_different_input_hash_is_refused() -> None:
    """The corpus drifted between the two arms; the pairing is no longer a pairing."""
    frozen = _corpus("alpha")
    baseline = _arm(frozen, hits=10)
    candidate = _arm(frozen, hits=10)
    tampered = frozen[0]
    candidate[0] = _record(
        FrozenQuestion(
            question_id=tampered.question_id,
            corpus=tampered.corpus,
            input_hash="0" * 64,
            expected_relevance_labels=tampered.expected_relevance_labels,
        ),
        hit=True,
    )
    with pytest.raises(UnpairedArms, match="was scored against input hash"):
        build_outcomes(baseline, candidate, frozen)


def test_empty_safety_classes_refuse_instead_of_reading_as_a_passed_check() -> None:
    """`nan - nan > 0.02` is False, so a class with no data looks exactly like a clean one."""
    frozen = [_frozen(f"a{i}", "alpha") for i in range(5)]  # no unanswerable questions
    with pytest.raises(ValueError, match="are NaN because their class is empty"):
        build_safety(_arm(frozen, hits=3), arm="candidate")


def test_a_manifest_with_no_answerable_question_has_no_quality_signal() -> None:
    frozen = [_frozen(f"u{i}", "alpha", answerable=False) for i in range(5)]
    arm = _arm(frozen, hits=0)
    with pytest.raises(UnpairedArms, match="no answerable question"):
        build_outcomes(arm, arm, frozen)


def test_unanswerable_questions_do_not_dilute_the_paired_quality_delta() -> None:
    """hit@5 on a question with no relevant document is 0.0 for every system that ever existed."""
    frozen = _corpus("alpha", n_answerable=10, n_unanswerable=90)
    outcomes = build_outcomes(_arm(frozen, hits=2), _arm(frozen, hits=2), frozen)
    assert len(outcomes) == 10
    assert all(outcome.corpus == "alpha" for outcome in outcomes)


def test_safety_metrics_use_the_repositorys_own_rate_definitions() -> None:
    frozen = _corpus("alpha", n_answerable=4, n_unanswerable=4)
    records = _arm(frozen, hits=4, abstains_on_unanswerable=False)
    safety = build_safety(records, arm="x")
    # every unanswerable question was answered confidently, none of the answerable abstained
    assert safety.false_confidence == 1.0
    assert safety.false_abstention == 0.0
    assert safety.superseded_trust_rate == 0.0


def test_secondary_metrics_are_recorded_and_gate_nothing() -> None:
    frozen = _corpus("alpha", n_answerable=10, n_unanswerable=2)
    metrics = secondary_metrics(_arm(frozen, hits=5))
    assert metrics.n_answerable == 10 and metrics.n_unanswerable == 2
    assert metrics.hit_at_1 == 0.5 and metrics.hit_at_5 == 0.5 and metrics.hit_at_20 == 0.5
    assert 0.0 < metrics.ndcg_at_5 <= 1.0
    assert metrics.latency_p95_ms == 12.0


def test_the_gate_input_marks_latency_pending_by_default() -> None:
    """The default must be the honest state, not the convenient one."""
    frozen = _corpus("alpha")
    arm = _arm(frozen, hits=5)
    gate = build_gate_input(
        arm, arm, frozen, security_green=True, latency_budget_ms=250.0
    )
    assert gate.latency_p95_ms is None


def test_an_arm_that_hits_nothing_at_all_is_refused_as_a_label_space_mismatch() -> None:
    """The defect the first real run of this harness had, and nothing raised.

    Two arms that both compare `file.md:0` labels against content-hash chunk ids differ by
    exactly zero and produce a clean-looking refusal. The numbers mean nothing, and the gate
    cannot tell the difference — so the harness has to.
    """
    frozen = _corpus("alpha")
    broken = _arm(frozen, hits=0)
    with pytest.raises(VacuousArm, match="label-space mismatch"):
        build_outcomes(broken, broken, frozen)


def test_the_vacuity_refusal_names_the_labels_and_the_ids_it_compared() -> None:
    """Diagnosing this without the two id spaces side by side is the slow part."""
    frozen = _corpus("alpha")
    broken = _arm(frozen, hits=0)
    with pytest.raises(VacuousArm) as excinfo:
        build_outcomes(broken, broken, frozen)
    assert GOLD in str(excinfo.value) and MISS in str(excinfo.value)
    assert "label_kind" in str(excinfo.value)


def test_one_hit_anywhere_in_the_pool_is_enough_to_clear_the_vacuity_check() -> None:
    """It is a mismatch detector, not a quality bar: a genuinely weak arm must still be gateable."""
    frozen = _corpus("alpha")
    weak = _arm(frozen, hits=1)
    assert len(build_outcomes(weak, weak, frozen)) == 20


def test_a_search_must_declare_which_id_space_its_labels_live_in() -> None:
    from recall.eval.promotion.search import LABEL_KEYS, StoreSearch

    assert set(LABEL_KEYS) == {
        "chunk_id", "source", "file_ord", "locomo_dia", "ladder_doc"
    }
    with pytest.raises(ValueError, match="label_kind must be one of"):
        StoreSearch(
            store=None, embedder=None, k=5, candidate_k=20,
            retrieval_profile="fast", label_kind="whatever",
        )


def test_the_decision_records_which_trust_verdicts_produced_it() -> None:
    """`false_confidence: 1.0` from a degraded run must not read as a property of the retriever."""
    frozen = _corpus("alpha")
    # One real verdict, so the arm is not WHOLLY degraded — a wholly degraded arm reports its
    # superseded rate as NOT MEASURED and is covered by its own test.
    arm = [
        _record(question, hit=question.answerable, verdict="unverified")
        for question in frozen[:-1]
    ] + [_record(frozen[-1], hit=False, abstained=True)]
    _, document = decide(
        arm, list(arm), frozen,
        manifest_digest="d", baseline_label="b", candidate_label="c",
        security_green=True, latency_budget_ms=250.0, bootstrap_samples=200,
    )
    assert document["trust_verdicts"]["baseline"] == {"abstained": 1, "unverified": 23}


def test_the_original_regression_test_still_holds() -> None:
    """`tests/test_promotion.py`'s case, re-asserted here against the extended dataclass."""
    outcomes = tuple(
        [QuestionOutcome(str(i), "a", 0.0, 1.0, 0.0, 1.0) for i in range(20)]
        + [QuestionOutcome(str(i), "b", 1.0, 0.0, 1.0, 0.0) for i in range(20)]
    )
    safe = SafetyMetrics(0.1, 0.1, 0.0)
    decision = evaluate_retrieval_promotion(
        RetrievalGateInput(outcomes, safe, safe, True, 100, 250), bootstrap_samples=500
    )
    assert not decision.promoted
    assert any("b hit@5 regresses" in failure for failure in decision.failures)
