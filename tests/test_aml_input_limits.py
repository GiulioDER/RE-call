"""AML Coding inputs beyond the old caps are accepted, and accepting them stays linear.

The 2026-09-24 pre-run audit found three limits that refused a whole Add or Search with a
permanent 422: a query over 20,000 characters, a message over 200,000 characters, and more than
256 messages. Lifting the last two exposed two per-window scans that were quadratic in the size
of one Add, measured at 23 s of atomic views for 4 M characters and 3 s of chunking for 8,000
messages; both are now bisected. Each behaviour test records the mutation it was proved red
against, with this file unchanged.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import ValidationError

import recall.atomizer as atomizer
from recall_aml.models import AddRequest, SearchRequest, bounded_query
from recall_aml.service import build_chunks
from recall_aml.variants import variant


C9 = variant("C9_routed_specialists_grounded_graph_atomic")


def add_request(messages: list[dict[str, Any]]) -> AddRequest | None:
    try:
        return AddRequest.model_validate(
            {"request_id": "r", "user_id": "u", "session_id": "s", "messages": messages}
        )
    except ValidationError:
        return None


def test_a_long_search_query_keeps_both_ends_instead_of_a_422():
    """Red proof: `SearchRequest.validate_query` with the old refusal restored
    (`if len(value) > 20_000: raise ValueError(...)`). This test then failed on
    `assert request is not None`."""
    query = "HEAD " + "x" * 30_000 + " TAIL: how do I fix the parser?"
    try:
        request = SearchRequest.model_validate({"query": query, "user_id": "u"})
    except ValidationError:
        request = None

    assert request is not None
    assert isinstance(request.query, str)
    assert len(request.query) == 20_000
    assert request.query.startswith("HEAD ")
    assert request.query.endswith("TAIL: how do I fix the parser?")


def test_a_query_within_the_bound_is_unchanged():
    """Nonbehavioural control for the bound: identity below 20,000 characters, so every query
    that was accepted before reaches Search byte for byte."""
    assert bounded_query("q" * 20_000) == "q" * 20_000
    assert bounded_query("short") == "short"


def test_a_message_over_200000_characters_is_accepted():
    """Red proof: `Message.validate_content` with the 200,000 character refusal restored. This
    test then failed on `assert request is not None`."""
    request = add_request([{"role": "tool", "content": "log line\n" * 30_000}])

    assert request is not None
    assert len(request.messages[0].content) == 270_000


def test_more_than_256_messages_are_accepted():
    """Red proof: `AddRequest.messages` with `max_length=256` restored. This test then failed on
    `assert request is not None`."""
    request = add_request([{"role": "user", "content": f"step {i}"} for i in range(300)])

    assert request is not None
    assert len(request.messages) == 300


def test_window_message_ordinals_match_a_full_scan():
    """The bisected message lookup returns exactly what scanning every message returned.

    Red proof: `build_chunks`, `stop = bisect_left(range_starts, word_end)` mutated to
    `bisect_left(range_starts, word_start)`. This test then failed on the ordinal equality
    (messages starting inside a window were dropped).
    """
    rng = random.Random(7)
    messages = [
        {"role": "user", "content": " ".join(f"w{j}" for j in range(rng.choice([1, 3, 40, 170])))}
        for _ in range(400)
    ]
    request = add_request(messages)
    assert request is not None
    chunks = build_chunks(
        request,
        [],
        word_window_size=C9.word_window_size,
        word_window_stride=C9.word_window_stride,
        content_only_windows=C9.content_only_windows,
        stable_window_identity=C9.stable_window_order,
    )

    ranges, cursor = [], 0
    for ordinal, message in enumerate(messages):
        count = len(message["content"].split())
        ranges.append((ordinal, cursor, cursor + count))
        cursor += count
    for chunk in chunks:
        start, end = chunk.metadata["word_start"], chunk.metadata["word_end"]
        expected = [o for o, s, e in ranges if s < end and e > start]
        assert chunk.metadata["message_ordinals"] == expected


def random_bounds(rng: random.Random) -> list[tuple[int, int]]:
    return atomizer.window_bounds(
        rng.randint(0, 700), size=rng.choice([16, 40, 160]), stride=rng.choice([8, 12, 30, 120])
    )


def test_the_sorted_parent_lookup_matches_the_full_scan_on_every_range():
    """Red proof: `_sorted_parent_window`, `stop = bisect_right(starts, start)` mutated to
    `bisect_left(starts, start)`. This test then failed on the equality (a window that starts
    exactly where the view starts was skipped)."""
    rng = random.Random(11)
    checked = 0
    for _ in range(40):
        bounds = random_bounds(rng)
        if bounds[0][1] < bounds[0][0]:
            continue
        starts = [s for s, _ in bounds]
        ends = [e for _, e in bounds]
        last = bounds[-1][1]
        for start in range(0, last + 1, 3):
            for width in (1, 6, 24, 40):
                end = start + width
                assert atomizer._sorted_parent_window(
                    start, end, bounds, starts, ends
                ) == atomizer.parent_window(start, end, bounds)
                checked += 1
    assert checked > 1_000


class CountingBounds(list):
    touched = 0

    def __iter__(self):
        for item in super().__iter__():
            CountingBounds.touched += 1
            yield item

    def __getitem__(self, index):
        CountingBounds.touched += 1
        return super().__getitem__(index)


def test_window_views_touches_a_constant_number_of_windows_per_view(monkeypatch):
    """Atomic views must stay linear in the size of one Add.

    Red proof: `window_views` reverted to call `parent_window(start, end, bounds)`. This test
    then failed on the touch bound: every view scanned every window (about 2,000 per view here).
    """
    real = atomizer.window_bounds
    monkeypatch.setattr(
        atomizer, "window_bounds", lambda *a, **k: CountingBounds(real(*a, **k))
    )
    CountingBounds.touched = 0
    rng = random.Random(3)
    words = " ".join(f"tok{rng.randrange(5_000)}" for _ in range(250_000))

    views = atomizer.window_views(words, window_size=160, window_stride=120, strategy="micro")

    assert len(views) > 15_000
    # Two list comprehensions read every bound once; each view then reads at most a few.
    assert CountingBounds.touched < 4 * len(views) + 3 * 2_100
