from __future__ import annotations

import json
from pathlib import Path

from scripts.enterprise_rag_make_slices import make_slices


def test_make_slices_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps({"question_id": f"qst_{index:04d}", "question_type": "project_related"})
            for index in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    first = make_slices(questions, tmp_path / "one", seed="fixed")
    second = make_slices(questions, tmp_path / "two", seed="fixed")

    assert first == second
    dev = (tmp_path / "one" / "dev.ids").read_text(encoding="utf-8").splitlines()
    confirmation = (tmp_path / "one" / "confirmation.ids").read_text(encoding="utf-8").splitlines()
    assert set(dev).isdisjoint(confirmation)
    assert sorted(dev + confirmation) == [f"qst_{index:04d}" for index in range(20)]
