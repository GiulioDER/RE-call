from __future__ import annotations

import json

from scripts.build_atomic_fact_release_confirmation import generate_confirmation_rows


def test_confirmation_advances_after_rejection_and_stops_at_target(tmp_path) -> None:
    """Use one call per source, advance after rejection, and stop at the frozen target.

    Red proof: the plausible mutation replaced the rejection branch's `continue` with `break`.
    This test failed at the intended row and attempt assertions with one accepted row and two
    attempts instead of two accepted rows and three attempts.
    """

    class FakeWriter:
        def __init__(self) -> None:
            self.responses = iter(
                [
                    json.dumps(
                        {"question": "When can operators change the alpha cutoff safely?"}
                    ),
                    "not-json",
                    json.dumps(
                        {"question": "Which condition permits the beta setting to change?"}
                    ),
                    json.dumps(
                        {"question": "This fourth response must never be requested by construction?"}
                    ),
                ]
            )

        def complete(self, system: str, user: str) -> str:
            return next(self.responses)

    candidates = []
    facts = (
        "The alpha cutoff changes only after a certified paired evaluation approves it.",
        "This deliberately rejected fact consumes one and only one writer call in sequence.",
        "The beta setting changes only when an independent quality gate approves promotion.",
        "The fourth candidate must remain untouched after the fixed target is reached safely.",
    )
    for index, content in enumerate(facts):
        path = tmp_path / f"source-{index}.md"
        path.write_text(content, encoding="utf-8")
        candidates.append(
            {
                "source": f"recall/source-{index}.md",
                "source_sha256": f"source-{index}",
                "gold_ordinal": 0,
                "answer_span_sha256": f"answer-{index}",
                "construction": "extractive_fallback",
                "order": f"order-{index}",
                "path": path,
                "view": {"content": content},
            }
        )

    rows, rejections, attempts = generate_confirmation_rows(
        candidates, FakeWriter(), target_rows=2, max_attempts=4
    )

    assert len(rows) == 2
    assert attempts == 3
    assert rejections == {"invalid_json": 1}
