from __future__ import annotations

import scripts.capture_query_anchor_dev_features as capture


def test_registered_guarded_parity_mismatch_keeps_no_features(monkeypatch) -> None:
    """A repeated-order mismatch must be counted without retaining a feature row.

    Red proof targets ``try_analyze_query``. The deliberate mutation marked the caught mismatch
    false, so the intended mismatch assertion failed.
    """
    monkeypatch.setattr(
        capture,
        "analyze_query",
        lambda *args: (_ for _ in ()).throw(
            ValueError("local guarded proposal differs from live shadow")
        ),
    )

    row = capture.try_analyze_query(
        {"expected_answerability": "answerable"}, {}, {}, {}, object()
    )

    assert row == {
        "parity_mismatch": True,
        "expected_answerability": "answerable",
    }
