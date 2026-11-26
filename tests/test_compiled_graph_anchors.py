"""Anchors are the only text a compile may cite, so their offsets and ids decide every edge.

Red proof, recorded 2026-11-26:

- ``test_anchors_cover_body_with_overlap``: mutation ``step = ANCHOR_CHARS`` (no overlap) in
  ``recall.compiled_graph.anchors.build_source_anchors`` fails on the anchor starts
  ``[0, 1600, 3200] != [0, 1440, 2880]``.
- ``test_anchor_ids_bind_source_and_text``: mutation dropping ``source`` from the hashed payload
  fails on ``a.id != b.id``.
- ``test_whitespace_body_has_no_anchors``: mutation removing the ``quote.strip()`` check fails on
  one anchor returned for a whitespace body.
- ``test_fit_payload_keeps_head_and_tail``: mutation keeping only the head fails on the kept ids.
"""

from __future__ import annotations

import re

from recall.compiled_graph.anchors import (
    ANCHOR_CHARS,
    ANCHOR_OVERLAP_CHARS,
    build_source_anchors,
    compile_payload,
    encode_stored_data,
    fit_payload,
)


def _body(length: int) -> str:
    return "".join(chr(ord("a") + i % 26) for i in range(length))


def test_anchors_cover_body_with_overlap() -> None:
    body = _body(4_000)
    anchors = build_source_anchors("notes/a.md", body)
    assert ANCHOR_CHARS == 1_600 and ANCHOR_OVERLAP_CHARS == 160
    assert [anchor.start for anchor in anchors] == [0, 1_440, 2_880]
    assert anchors[-1].end == 4_000
    for anchor in anchors:
        assert anchor.quote == body[anchor.start : anchor.end]
    assert [anchor.id[:5] for anchor in anchors] == ["a000_", "a001_", "a002_"]
    assert all(re.fullmatch(r"a\d{3}_[0-9a-f]{16}", anchor.id) for anchor in anchors)


def test_anchor_ids_bind_source_and_text() -> None:
    (a,) = build_source_anchors("notes/a.md", "same text")
    (b,) = build_source_anchors("notes/b.md", "same text")
    (c,) = build_source_anchors("notes/a.md", "other text")
    assert a.id != b.id
    assert a.id != c.id
    assert build_source_anchors("notes/a.md", "same text")[0].id == a.id


def test_whitespace_body_has_no_anchors() -> None:
    assert build_source_anchors("notes/a.md", "  \n\t") == []
    assert build_source_anchors("notes/a.md", "") == []


def test_fit_payload_keeps_head_and_tail() -> None:
    anchors = build_source_anchors("notes/a.md", _body(16_000))
    payload = compile_payload("notes/a.md", anchors)
    assert fit_payload(payload, len(encode_stored_data(payload))) is None
    budget = len(encode_stored_data(payload)) // 2
    fitted = fit_payload(payload, budget)
    assert fitted is not None
    assert len(encode_stored_data(fitted)) <= budget
    ids = [anchor["id"] for anchor in fitted["anchors"]]
    all_ids = [anchor.id for anchor in anchors]
    assert ids[0] == all_ids[0]
    assert ids[-1] == all_ids[-1]
    assert ids == [anchor_id for anchor_id in all_ids if anchor_id in ids]
    assert fit_payload(payload, 10) is None
