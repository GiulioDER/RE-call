from __future__ import annotations

from scripts.train_task_specific_selector import flatten_training_pairs, order_from_scores


def test_flatten_training_pairs_emits_one_positive_and_four_negatives() -> None:
    payload = {
        "train_rows": [
            {
                "id": "q1",
                "query": "question",
                "positive_text": "right",
                "negative_texts": ["n1", "n2", "n3", "n4"],
            }
        ]
    }

    queries, responses, labels = flatten_training_pairs(payload, enforce_row_count=False)

    assert queries == ["question"] * 5
    assert responses == ["right", "n1", "n2", "n3", "n4"]
    assert labels == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_order_from_scores_breaks_exact_tie_by_dense_rank() -> None:
    candidates = [
        {"chunk_id": "a", "dense_rank": 1},
        {"chunk_id": "b", "dense_rank": 2},
        {"chunk_id": "c", "dense_rank": 3},
    ]

    order = order_from_scores(candidates, {"a": 0.2, "b": 0.8, "c": 0.8})

    assert order == ["b", "c", "a"]

