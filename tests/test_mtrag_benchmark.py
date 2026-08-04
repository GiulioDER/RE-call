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


def test_score_predictions_is_macro_average() -> None:
    predictions = [
        {"task_id": "q1", "contexts": [{"document_id": "a"}]},
        {"task_id": "q2", "contexts": [{"document_id": "x"}]},
    ]
    qrels = {
        "clapnq": {"q1": {"a"}},
        "cloud": {"q2": {"b"}},
        "fiqa": {},
        "govt": {},
    }
    scores = score_predictions(predictions, qrels)
    assert scores["overall"]["count"] == 2
    assert scores["overall"]["nDCG@5"] == 0.5
    assert scores["overall"]["Recall@5"] == 0.5
