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
    """A raw window as `build_chunks` stores it: only the ranges of the messages it overlaps."""
    overlapping = [r for r in RANGES if r[1] < end and r[2] > start]
    return SearchItem(
        id=item_id, content=" ".join(WORDS[start:end]), created_at=datetime(2024, 3, 1, tzinfo=UTC),
        source="raw", session_id="s1", kind="raw", score=score,
        render_facts={"word_start": start, "speaker_ranges": overlapping, "add_digest": add},
    )


def _record(item_id: str) -> SearchItem:
    return SearchItem(id=item_id, content="kind: procedure", source="m", session_id="s1", kind="procedure", score=0.4)


def test_with_both_options_off_every_item_is_returned_as_it_came() -> None:
    """Invariant: the served default changes nothing.

    Red proof (corrected by the audit of #809, which found the old one stayed green): removing
    the early return in `compose_items` renders each window again, which re-joins a window's
    words on single spaces, and fails the equality on the double-spaced window.
    """
    spaced = _window("w3", 0, 6).model_copy(update={"content": "I  moved to Lisbon in March"})
    items = [_window("w2", 6, 20), _record("r"), _window("w1", 0, 12), spaced]
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


@pytest.mark.parametrize("name", ["RECALL_AML_SESSION_COALESCE", "RECALL_AML_SPEAKER_MARKS"])
def test_a_bad_w2_value_stops_service_startup(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Invariant: an unreadable flag is refused at startup.

    Red proofs: removing ``self.session_coalesce``, then ``self.speaker_marks`` (audit of #809:
    that half was uncovered), from the startup read in `HostedService.__init__` lets the service
    start and fails the ``pytest.raises`` for that flag.
    """
    monkeypatch.setenv(name, "yes")
    with pytest.raises(ValueError, match=name):
        _service("C7_routed_specialists")


# Audit of #809 (2026-10-01). Each red proof ran on the testbench host against the named mutation,
# failing at the named assertion, then green with the line restored.


def test_coalesced_speaker_marks_use_every_window_s_ranges() -> None:
    """Invariant: a coalesced item is marked from the union of its windows' stored ranges, since
    each window stores only the messages it overlaps; the composed item's facts are cleared.

    Red proofs: `_ranges` reading only the best window (``members[:1]``) loses every mark and fails
    the content equality; the composed copy keeping ``render_facts`` fails the last assertion.
    """
    items = [_window("b", 12, 20, score=0.9), _window("a", 0, 12), _window("c", 20, 31)]
    out = compose_items(items, speakers=True, coalesce=True)
    assert len(out) == 1 and out[0].id == "b"
    assert out[0].content == (
        "[user] " + " ".join(WORDS[0:11]) + " [assistant] " + " ".join(WORDS[11:23])
        + " [user] " + " ".join(WORDS[23:31])
    )
    assert out[0].render_facts is None


def test_windows_stored_before_w2_are_never_merged() -> None:
    """Invariant: a window without an Add digest (stored before this change) passes through, even
    beside another such window of the same session.

    Red proof: dropping the ``isinstance(add, str) and add`` guard in `compose_items` keys both on
    ``(session, None)``, merges them, and fails the equality.
    """
    legacy = [
        _window(name, start, end).model_copy(
            update={"render_facts": {"word_start": start, "speaker_ranges": None, "add_digest": None}}
        )
        for name, start, end in (("x", 0, 6), ("y", 20, 26))
    ]
    assert compose_items(legacy, speakers=True, coalesce=True) == legacy


def test_only_raw_windows_carry_render_facts() -> None:
    """Invariant: an atomic view (which also stores an integer ``word_start``) and a compiled
    record carry no render facts; only a raw window does.

    Red proof: the ``record_type != "raw"`` guard removed from `window_compose.render_facts` gives
    the atomic view facts and fails the first assertion.
    """
    from recall_aml.window_compose import render_facts

    assert render_facts({"record_type": "atomic_view", "word_start": 3}) is None
    assert render_facts({"record_type": "procedure"}) is None
    assert render_facts({"record_type": "raw", "word_start": 3, "add_digest": "d"}) == {
        "word_start": 3, "speaker_ranges": None, "add_digest": "d",
    }


def test_each_add_gets_its_own_digest() -> None:
    """Invariant: the digest identifies the Add, so two Adds of one session never coalesce.

    Red proof: `build_chunks` digesting ``request.session_id`` instead of ``request.request_id``
    gives both Adds one digest and fails the inequality.
    """
    from recall_aml.identity import canonical_digest
    from recall_aml.models import AddRequest
    from recall_aml.service import build_chunks
    from recall_aml.window_compose import ADD_DIGEST_CHARS

    digests = {}
    for request_id in ("add-one", "add-two"):
        request = AddRequest.model_validate({
            "request_id": request_id, "user_id": "u", "session_id": "same-session",
            "messages": [{"role": role, "content": text} for role, text in MESSAGES],
        })
        chunks = build_chunks(request, [], word_window_size=8, word_window_stride=6, content_only_windows=True)
        digests[request_id] = {c.metadata["add_digest"] for c in chunks if c.metadata.get("record_type") == "raw"}
    assert digests["add-one"] != digests["add-two"]
    assert digests["add-one"] == {canonical_digest("add-one")[:ADD_DIGEST_CHARS]}


#: Twelve alternating turns of 30 words: several 160-word windows, each spanning two roles.
LONG = [
    ("user" if turn % 2 == 0 else "assistant",
     " ".join(f"t{turn}w{word}" for word in range(30)) + (" fountain pens" if turn == 3 else ""))
    for turn in range(12)
]


class _VisualStore:
    def query_dense(self, vector, k, **_kw):  # type: ignore[no-untyped-def]
        return []

    def chunks_by_ids(self, ids):  # type: ignore[no-untyped-def]
        return {}


def _c9_service():  # type: ignore[no-untyped-def]
    """C9 as served, over the forget tests' in-memory doubles, with a text-only visual store."""
    from recall_aml.retrieval import HostedRetriever
    from recall_aml.service import HostedService
    from tests.test_aml_forget import C9, _Compiler, _Embedder, _Repository, _Reranker
    from tests.test_aml_specialist_fusion import _MultimodalEmbedder

    class _Repo(_Repository):
        def multimodal_store(self, tenant):  # type: ignore[no-untyped-def]
            return _VisualStore()

        def media_store(self, tenant):  # type: ignore[no-untyped-def]
            return _VisualStore()

    service = HostedService(
        _Repo(), _Compiler(), HostedRetriever(_Embedder("code"), _Reranker()),  # type: ignore[arg-type]
        behavior=C9, multimodal_embedder=_MultimodalEmbedder(),  # type: ignore[arg-type]
        specialist_retrievers={
            C9.context_embedding_profile: HostedRetriever(_Embedder("context"), _Reranker())  # type: ignore[arg-type]
        },
    )
    import asyncio

    from recall_aml.models import AddRequest

    asyncio.run(service.add(AddRequest.model_validate({
        "request_id": "w2-long", "user_id": "w2-user", "session_id": "w2-sess",
        "messages": [{"role": role, "content": text, "timestamp": 1_700_000_000_000} for role, text in LONG],
    })))
    return service


@pytest.mark.parametrize(("speakers", "coalesce"), [(False, False), (True, False), (False, True), (True, True)])
def test_c9_applies_exactly_the_flags_it_is_given(monkeypatch: pytest.MonkeyPatch, speakers: bool, coalesce: bool) -> None:
    """Invariant, at the service on C9: each flag does its own thing and nothing else, the profile
    names exactly the flags on, and the facts never reach the response body.

    Red proofs (audit of #809: the one service test ran both flags on, so neither survived it):
    the flags passed swapped to `compose_items` in `HostedService.search` fails the speakers-only
    row at the mark assertion; composition forced on whatever the flags (guard and arguments)
    fails the both-off row at the mark assertion. Forcing the guard alone is an equivalent
    mutation, since `compose_items` returns its input when both flags are off.
    """
    import recall_aml.service as service_module

    # The baseline is the rendering with composition patched out, so it cannot depend on the code
    # under test.
    with monkeypatch.context() as patched:
        patched.setattr(service_module, "compose_items", lambda items, **_kwargs: list(items))
        baseline = _search(_c9_service(), "w2-user", "when did I talk about fountain pens?")
    assert len(baseline.data) > 2, "precondition: several windows of the one Add are served"
    monkeypatch.setenv("RECALL_AML_SPEAKER_MARKS", "1" if speakers else "0")
    monkeypatch.setenv("RECALL_AML_SESSION_COALESCE", "1" if coalesce else "0")
    service = _c9_service()
    response = _search(service, "w2-user", "when did I talk about fountain pens?")

    marked = any("[assistant]" in str(item.content) for item in response.data)
    assert marked == speakers
    if coalesce:
        assert len(response.data) < len(baseline.data)
    else:
        assert len(response.data) == len(baseline.data)
    if not speakers and not coalesce:
        assert response.model_dump_json() == baseline.model_dump_json()
    assert ("+speaker-marks-v1" in service.search_content_profile) == speakers
    assert ("+session-coalesce-v1" in service.search_content_profile) == coalesce
    assert "render_facts" not in response.model_dump_json()


def test_c9_marks_windows_on_the_multimodal_route_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: a plain-text query with a visual word (C9 routes it to multimodal, rendered by
    `render_preserved`) gets W2 like any other (audit of #809: there it did nothing).

    Red proof: `multimodal._text_item` without ``render_facts`` serves the windows unmarked and
    fails the mark assertion.
    """
    from recall_aml.specialists import route_query

    monkeypatch.setenv("RECALL_AML_SPEAKER_MARKS", "1")
    query = "show me the chart of fountain pens"
    response = _search(_c9_service(), "w2-user", query)
    assert route_query(query) == "multimodal" and response.specialist_route == "multimodal"
    assert response.data, "precondition: the text windows are served on this route"
    assert any("[assistant]" in str(item.content) for item in response.data)


def test_a_composition_failure_serves_the_windows_as_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: W2 fails open; a malformed stored fact never turns a Search into an error.

    Red proof: the ``try`` around `compose_items` in `HostedService.search` removed; the error
    escapes and fails at the outcome assertion.
    """
    import recall_aml.service as service_module

    def broken(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise TypeError("malformed speaker_ranges")

    monkeypatch.setenv("RECALL_AML_SPEAKER_MARKS", "1")
    monkeypatch.setattr(service_module, "compose_items", broken)
    service = _c9_service()
    try:
        outcome: object = _search(service, "w2-user", "when did I talk about fountain pens?")
    except Exception as error:  # the property under test is that nothing escapes
        outcome = error
    assert not isinstance(outcome, Exception), outcome
    assert not any("[assistant]" in str(item.content) for item in outcome.data)  # type: ignore[attr-defined]
