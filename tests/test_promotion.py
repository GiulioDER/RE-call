from recall.promotion import (
    QuestionOutcome, RetrievalGateInput, SafetyMetrics, evaluate_retrieval_promotion,
)


def test_promotion_gate_rejects_a_corpus_regression() -> None:
    outcomes = tuple(
        [QuestionOutcome(str(i), "a", 0.0, 1.0, 0.0, 1.0) for i in range(20)]
        + [QuestionOutcome(str(i), "b", 1.0, 0.0, 1.0, 0.0) for i in range(20)]
    )
    safe = SafetyMetrics(0.1, 0.1, 0.0)
    decision = evaluate_retrieval_promotion(
        RetrievalGateInput(outcomes, safe, safe, True, 100, 250),
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert any("b hit@5 regresses" in failure for failure in decision.failures)
