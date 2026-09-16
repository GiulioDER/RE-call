from __future__ import annotations

import json

from scripts.build_atomic_fact_blind_source_census import generate_census_rows


def test_census_attempts_every_candidate_after_a_validation_rejection(tmp_path) -> None:
    """A rejected question must not truncate the exhaustive source census.

    Red proof: before the first green run, the deliberate mutation replaced the rejection branch's
    `continue` with `break` in `generate_census_rows`. This node failed at the intended assertion
    with an attempt count of two instead of three.
    """

    class FakeWriter:
        def __init__(self) -> None:
            self.responses = iter(
                [
                    json.dumps(
                        {"question": "When may operators change the alpha cutoff in production?"}
                    ),
                    "not-json",
                    json.dumps(
                        {"question": "Which condition permits the beta setting to change safely?"}
                    ),
                ]
            )

        def complete(self, system: str, user: str) -> str:
            return next(self.responses)

    candidates = []
    for index, content in enumerate(
        (
            "The alpha calibration policy requires certification before any threshold change occurs.",
            "The rejected middle fact still consumes exactly one exhaustive generation attempt.",
            "The beta setting changes only when a certified quality evaluation approves it.",
        )
    ):
        path = tmp_path / f"source-{index}.md"
        path.write_text(content, encoding="utf-8")
        candidates.append(
            {
                "source": f"recall/source-{index}.md",
                "path": path,
                "view": {
                    "content": content,
                    "parent_ordinal": 0,
                    "construction": "extractive_fallback",
                },
            }
        )

    rows, rejections, attempts = generate_census_rows(candidates, FakeWriter())
    assert attempts == 3
    assert len(rows) == 2
    assert rejections == {"invalid_json": 1}
