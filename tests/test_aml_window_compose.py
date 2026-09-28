"""W2: speaker marks and session coalescing of returned windows (``recall_aml.window_compose``).

Round two, 2026-09-28. Both options are off by default; with them off every item is returned as
it came. Each test names the mutation of the production code it was watched to fail on (the red
proof).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from recall_aml.code4 import word_windows
from recall_aml.models import SearchItem
from recall_aml.speaker_render import message_word_ranges
from recall_aml.variants import variant
from recall_aml.window_compose import GAP, compose_items
from tests.test_aml_relative_dates_and_adjacency import _search
from tests.test_aml_specialist_fusion import _service

MESSAGES = [
    ("user", "I moved to Lisbon in March and started a new job"),
    ("assistant", "Congratulations, a move and a new job at once is a lot"),
    ("user", "Yes and I adopted a cat named Miso"),
]
WORDS = " ".join(content for _, content in MESSAGES).split()
RANGES = [[role, start, end] for role, start, end in message_word_ranges(MESSAGES)]


def _window(item_id: str, start: int, end: int, *, add: str = "add-1", score: float = 0.5) -> SearchItem:
    return SearchItem(
        id=item_id, content=" ".join(WORDS[start:end]), created_at=datetime(2024, 3, 1, tzinfo=UTC),
        source="raw", session_id="s1", kind="raw", score=score,
        render_facts={"word_start": start, "speaker_ranges": RANGES, "add_digest": add},
    )


def _record(item_id: str) -> SearchItem:
    return SearchItem(id=item_id, content="kind: procedure", source="m", session_id="s1", kind="procedure", score=0.4)


def test_with_both_options_off_every_item_is_returned_as_it_came() -> None:
    """Invariant: the served default changes nothing.

    Red proof: removing the early return in `compose_items` and always rendering joins the two
    windows of one Add into one item and fails the equality.
    """
    items = [_window("w2", 6, 20), _record("r"), _window("w1", 0, 12)]
    assert compose_items(items, speakers=False, coalesce=False) == items


def test_windows_of_one_add_become_one_item_in_source_order_without_the_overlap() -> None:
    """Invariant: coalesced windows read in source order with their shared words once, at the
    position of the best-ranked window, keeping its id and score; other items keep their places.

    Red proof: appending each window's words whole (dropping ``[overlap:]``) in `_render` repeats
    the shared words and fails the content equality.
    """
    items = [_window("w2", 6, 20, score=0.9), _record("r"), _window("w1", 0, 12, score=0.3)]
    out = compose_items(items, speakers=False, coalesce=True)
    assert [item.id for item in out] == ["w2", "r"]
    assert out[0].content == " ".join(WORDS[0:20])
    assert out[0].score == 0.9


def test_a_gap_between_windows_is_marked_and_other_adds_stay_apart() -> None:
    """Invariant: non-contiguous windows of one Add are joined with the gap marker; windows of a
    different Add are never merged into it.

    Red proof: keying the groups by ``item.session_id`` alone (dropping the Add digest) in
    `compose_items` merges the other Add's window and fails the length assertion.
    """
    items = [_window("a", 0, 6), _window("b", 20, 26), _window("c", 8, 12, add="add-2")]
    out = compose_items(items, speakers=False, coalesce=True)
    assert len(out) == 2
    assert out[0].content == " ".join(WORDS[0:6]) + GAP + " ".join(WORDS[20:26])
    assert out[1].id == "c"


def test_speaker_marks_follow_the_stored_message_ranges() -> None:
    """Invariant: a window spanning two roles is marked where each message begins; a one-role
    window is left alone (K-1's rule).

    Red proof: passing the window's offset as 0 instead of its ``word_start`` in `_render` puts the
    marks at the wrong words and fails the equality.
    """
    two_roles = compose_items([_window("w", 6, 26)], speakers=True, coalesce=False)[0]
    assert two_roles.content == (
        "[user] and started a new job [assistant] Congratulations, a move and a new job at once is a lot "
        "[user] Yes and I"
    )
    one_role = _window("u", 0, 8)
    assert compose_items([one_role], speakers=True, coalesce=False)[0] == one_role


def test_the_render_facts_never_reach_a_client() -> None:
    """Invariant: the position facts stay server side.

    Red proof: removing ``exclude=True`` from `SearchItem.render_facts` puts them in the dump and
    fails the assertion.
    """
    assert "render_facts" not in _window("w", 0, 8).model_dump(mode="json")


def test_build_chunks_stores_roles_ranges_and_the_add_for_each_window() -> None:
    """Invariant: each raw window records the role and word range of every message it overlaps,
    aligned with its message ordinals, and a digest of its Add.

    Red proof: storing ``speaker_ranges`` without the overlap filter (every message of the Add)
    fails the per-window equality.
    """
    from recall_aml.models import AddRequest
    from recall_aml.service import build_chunks

    request = AddRequest.model_validate(
        {
            "request_id": "w2-add",
            "user_id": "u",
            "session_id": "s1",
            "messages": [{"role": role, "content": text, "timestamp": 1_700_000_000_000} for role, text in MESSAGES],
        }
    )
    chunks = build_chunks(request, [], word_window_size=8, word_window_stride=6, content_only_windows=True)
    raws = [c for c in chunks if c.metadata.get("record_type") == "raw"]
    windows = word_windows(" ".join(t for _, t in MESSAGES), size=8, stride=6)
    assert [c.text for c in raws] == windows
    for chunk in raws:
        start, end = chunk.metadata["word_start"], chunk.metadata["word_end"]
        expected = [r for r in RANGES if r[1] < end and r[2] > start]
        assert chunk.metadata["speaker_ranges"] == expected
    assert len({c.metadata["add_digest"] for c in raws}) == 1


def test_the_service_composes_only_when_asked_and_c9_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: the service applies W2 when its flags are on, reports it in the search-content
    profile, and the served C9 has both off.

    Red proof: dropping the `compose_items` call from `HostedService.search` leaves the returned
    window unmarked and fails the ``[assistant]`` assertion.
    """
    c9 = variant("C9_routed_specialists_grounded_graph_atomic")
    assert (c9.speaker_marks, c9.session_coalesce) == (False, False)
    monkeypatch.setenv("RECALL_AML_SPEAKER_MARKS", "1")
    monkeypatch.setenv("RECALL_AML_SESSION_COALESCE", "1")
    service, _, _, _, _ = _service("C7_routed_specialists")
    import asyncio

    from recall_aml.models import AddRequest

    asyncio.run(service.add(AddRequest.model_validate({
        "request_id": "w2-s", "user_id": "w2-user", "session_id": "w2-sess",
        "messages": [{"role": role, "content": text, "timestamp": 1_700_000_000_000} for role, text in MESSAGES],
    })))
    response = _search(service, "w2-user", "when did I adopt the cat?")
    assert any("[assistant]" in str(item.content) for item in response.data)
    assert "+speaker-marks-v1+session-coalesce-v1" in service.search_content_profile


def test_a_bad_w2_value_stops_service_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: an unreadable flag is refused at startup.

    Red proof: removing ``self.session_coalesce`` from the startup read in `HostedService.__init__`
    lets the service start and fails the ``pytest.raises``.
    """
    monkeypatch.setenv("RECALL_AML_SESSION_COALESCE", "yes")
    with pytest.raises(ValueError, match="RECALL_AML_SESSION_COALESCE"):
        _service("C7_routed_specialists")
