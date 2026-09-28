"""`scripts/aml_w2w4_answers.py`: the rendering and scoring that decide the W2/W4 numbers, offline.

Each test names the mutation of the harness it was watched to fail on (the red proof).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w2w4_answers as h  # noqa: E402

WORDS = "Caroline: I went hiking yesterday with Mel and next week we camp by the lake again".split()


def _row(route: str) -> dict:
    def window(item_id: str, start: int, end: int, score: float) -> dict:
        return {
            "id": item_id, "content": " ".join(WORDS[start:end]), "created_at": "2023-05-08T13:56:00+00:00",
            "session_id": "s1", "kind": "raw", "score": score,
            "render_facts": {"word_start": start, "speaker_ranges": [["user", 0, len(WORDS)]], "add_digest": "a1"},
        }

    return {"id": "q1", "category": "2", "route": route,
            "items": [window("w2", 6, 16, 0.9), window("w1", 0, 9, 0.4)]}


def _texts(arm: str, row: dict) -> list[str]:
    return [str(item.content) for item in h.render(arm, row)]


def test_the_served_arm_resolves_only_off_the_code_route_and_the_content_gate_everywhere() -> None:
    """Invariant: B applies T-1 v1 only when the question took a non-code route; W4a applies it on
    every route; W4b renders week expressions relative (v2).

    Red proof: dropping the ``row["route"] != "code"`` condition in `render` resolves B on the code
    route and fails the first assertion.
    """
    code = _row("code")
    assert not any("[=" in text for text in _texts("B", code))
    assert any("yesterday [= 2023-05-07]" in text for text in _texts("W4a", code))
    assert any("next week [= the week after 2023-05-08]" in text for text in _texts("W4b", code))
    assert any("yesterday [= 2023-05-07]" in text for text in _texts("B", _row("context")))


def test_w2c_coalesces_the_windows_of_one_add_before_dating() -> None:
    """Invariant: W2c returns the Add's windows as one dated item in source order, overlap removed.

    Red proof: calling `compose_items` with ``coalesce=False`` in `render` returns two items and
    fails the length assertion.
    """
    texts = _texts("W2c", _row("code"))
    assert len(texts) == 1
    assert texts[0] == "[2023-05-08 13:56 UTC] " + " ".join(WORDS[0:16])


def test_the_round_trip_strips_both_renders() -> None:
    """Invariant: removing T-1's brackets, v1 or v2, gives the undecorated text back, so the
    mechanism's round-trip check can compare every T-1 arm with B.

    Red proof: restricting `BRACKET` to digits inside the brackets (the T-1 harness's v1-only form)
    leaves the v2 bracket in place and fails the equality.
    """
    code = _row("code")
    base = [h.strip_resolutions(t) for t in _texts("B", code)]
    assert [h.strip_resolutions(t) for t in _texts("W4b", code)] == base
    assert h.mechanism({"dataset": "locomo", "rows": [code]})["round_trip_failures"] == {}


def _records(dataset: str, arm: str, labels: list[bool], category: str = "2") -> list[dict]:
    return [{"dataset": dataset, "id": f"q{i}", "arm": arm, "category": category,
             "label": "CORRECT" if ok else "WRONG"} for i, ok in enumerate(labels)]


def test_the_decision_needs_a_positive_primary_the_guard_and_the_noise_floor() -> None:
    """Invariant: W4a is recommended when its primary effect is positive, its all-question interval
    stays above -1.5 points, and it beats the replicate; a clear loss is not recommended.

    Red proof: comparing the guard against the point estimate instead of the interval's lower
    bound (``ci95_points[0]`` to ``diff_points``) recommends the noisy arm and fails the last
    assertion.
    """
    base = [True] * 60 + [False] * 40
    better = [True] * 70 + [False] * 30
    records = (_records("locomo", "B", base) + _records("locomo", "B2", base)
               + _records("locomo", "W4a", better) + _records("locomo", "W4b", better))
    decided = h.score(records)["decisions"]
    assert decided["W4a"]["recommended"] is True
    noisy = [True] * 60 + [False] * 40
    for i in range(10):
        noisy[i] = False
    for i in range(60, 71):
        noisy[i] = True
    records = _records("locomo", "B", base) + _records("locomo", "B2", base) + _records("locomo", "W4a", noisy)
    assert h.score(records)["decisions"]["W4a"]["recommended"] is False


def test_the_collect_captures_the_service_create_app_is_given() -> None:
    """Invariant: the captured object is the service, create_app's second argument.

    Red proof: storing ``settings`` instead of ``service`` in `capturing`'s wrapper (the first
    collect's defect) fails the identity assertion.
    """
    import inspect

    from recall_aml.app import create_app

    assert list(inspect.signature(create_app).parameters)[:2] == ["settings", "service"]
    settings, service = object(), object()
    captured: dict = {}
    wrapped = h.capturing(lambda s, v, **options: (s, v, options), captured)
    assert wrapped(settings, service, shutdown=None) == (settings, service, {"shutdown": None})
    assert captured["service"] is service
