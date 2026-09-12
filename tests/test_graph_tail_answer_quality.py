from __future__ import annotations

import json

from scripts.run_voyage4_graph_tail_answer_quality import _load


def test_load_category_filter_keeps_paired_rows_and_excludes_other_categories(tmp_path) -> None:
    artifact = {
        "arms": {
            "baseline": {"rows": [{"id": "a", "category": 4}, {"id": "b", "category": 2}]},
            "selective_margin_005": {
                "rows": [{"id": "a", "category": 4}, {"id": "b", "category": 2}]
            },
        }
    }
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    filtered = _load(path, "selective_margin_005", category=4)

    assert filtered["arms"]["baseline"]["rows"] == [{"id": "a", "category": 4}]
    assert filtered["arms"]["selective_margin_005"]["rows"] == [{"id": "a", "category": 4}]
