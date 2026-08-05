from benchmarks.mtrag.run import ndcg_at, query_text, recall_at, score_predictions


def test_query_modes() -> None:
    task = {
        "task_id": "q",
        "input": [
            {"speaker": "user", "text": "first"},
            {"speaker": "agent", "text": "reply"},
            {"speaker": "user", "text": "second"},
            {"speaker": "user", "text": "third"},
            {"speaker": "user", "text": "fourth"},
        ],
    }
    assert query_text(task, "last") == "fourth"
    assert query_text(task, "recent3") == "second\nthird\nfourth"


def test_binary_metrics() -> None:
    relevant = {"a", "b"}
    assert recall_at(["a", "x", "b"], relevant, 1) == 0.5
    assert recall_at(["a", "x", "b"], relevant, 3) == 1.0
    assert ndcg_at(["a", "b"], relevant, 2) == 1.0
    assert 0.0 < ndcg_at(["x", "a", "b"], relevant, 3) < 1.0


def test_score_predictions_overall_is_pooled_not_domain_macro() -> None:
    """`overall` weights domains by judged-query count, and this fixture proves which one it is.

    The domains are deliberately UNBALANCED: three judged queries in clapnq (all hits) against
    one in cloud (a miss). Pooled gives 3/4 = 0.75; an unweighted mean of the two non-empty
    domain figures gives (1.0 + 0.0)/2 = 0.5. A balanced fixture cannot tell those apart, which
    is how the previous version of this test passed under either definition while its name
    asserted one of them.
    """
    predictions = [
        {"task_id": "q1", "contexts": [{"document_id": "a"}]},
        {"task_id": "q2", "contexts": [{"document_id": "b"}]},
        {"task_id": "q3", "contexts": [{"document_id": "c"}]},
        {"task_id": "q4", "contexts": [{"document_id": "x"}]},
    ]
    qrels = {
        "clapnq": {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}},
        "cloud": {"q4": {"d"}},
        "fiqa": {},
        "govt": {},
    }
    scores = score_predictions(predictions, qrels)

    assert scores["overall"]["count"] == 4
    assert scores["overall"]["nDCG@5"] == 0.75
    assert scores["overall"]["Recall@5"] == 0.75

    # The per-domain figures stay unweighted within a domain.
    assert scores["domains"]["clapnq"]["nDCG@5"] == 1.0
    assert scores["domains"]["cloud"]["nDCG@5"] == 0.0
    assert scores["domains"]["fiqa"]["nDCG@5"] is None

    # Pin the choice itself: the pooled figure must NOT equal the domain macro. Without this,
    # switching the aggregation to a macro average would still satisfy every assertion above
    # that happens to hold under both.
    populated = [
        scores["domains"][d]["nDCG@5"]
        for d in ("clapnq", "cloud", "fiqa", "govt")
        if scores["domains"][d]["nDCG@5"] is not None
    ]
    domain_macro = sum(populated) / len(populated)
    assert domain_macro == 0.5
    assert scores["overall"]["nDCG@5"] != domain_macro
