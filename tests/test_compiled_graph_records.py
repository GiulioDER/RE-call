"""A compiled record may keep only text copied verbatim from the anchors it cites.

Red proof, recorded 2026-11-26 (production symbols in ``recall.compiled_graph.records``):

- ``test_fields_kept_only_when_verbatim_in_cited_anchor``: ``_supported_text`` returning ``value``
  unconditionally fails on ``records[0].fields``.
- ``test_entities_kept_only_when_in_quoted_evidence``: keeping every nonempty proposed entity fails
  on ``records[0].entities``.
- ``test_entity_spanning_two_cited_anchors_dropped``: the pre-fix check against the cited quotes
  joined by a newline (``e in "\\n".join(quotes)``) fails on ``records[0].entities``.
- ``test_unknown_anchor_ids_dropped_and_bare_indices_resolved``: never resolving a bare index (the
  ``_BARE_ANCHOR_INDEX`` branch made ``if False``) fails on ``resolved_bare_anchor_ids == 1``.
- ``test_record_without_valid_anchor_dropped``: an unknown id falling back to the first anchor
  (``anchor_by_id.get(anchor_id, anchors[0])``) fails on ``records == []``.
- ``test_bare_index_resolves_only_against_sent_anchors``: resolving bare indices against every
  anchor (``sent_ids = [anchor.id for anchor in anchors]``) fails on ``resolved_bare_anchor_ids``.
- ``test_value_recovers_anchor_when_cited_id_unknown``: the recovery branch made ``if False`` fails
  on the recovered anchor ids.
- ``test_recovery_ignores_entities``: recovery values that include the entities
  (``(summary, detail, *entities)``) fail on ``records == []``.
- ``test_whitespace_summary_does_not_recover_an_anchor`` and ``test_whitespace_is_never_grounded``:
  the blank test in ``_supported_text`` reduced to ``if not value:`` (recovery calls the same
  function) fails on ``records == []`` and on ``records[0].fields``.
- ``test_invisible_summary_is_never_grounded`` and
  ``test_invisible_summary_does_not_recover_an_anchor``: the blank test as
  ``all(char.isspace() or unicodedata.category(char) == "Cf" ...)`` fails for every parameter
  but U+200B, on ``records[0].fields`` and on ``records == []``; reduced to
  ``if not value.strip()`` it fails for all six; ``0x1D159`` dropped from ``_BLANK_GLYPHS`` fails
  for U+1D159.
- ``test_value_in_no_anchor_is_never_scanned_for_visible_text``: the category scan run before the
  containment test in ``_supported_text`` fails on ``calls == 0`` (it scans the whole value once
  per anchor).
- ``test_more_than_eight_records_truncated``: dropping the ``[:MAX_RECORDS]`` slice fails on
  ``len(records) == 8`` (12).
- ``test_unknown_kind_dropped``: the kind check made ``if False`` fails on ``records == []``.
- ``test_empty_fields_backfilled_from_first_anchor``: backfilling ``("", "")`` fails on the fields.
- ``test_malformed_record_items_are_skipped``: ``_strings`` accepting any value fails on
  ``records == []`` (a string ``entities`` was accepted).
- ``test_prompt_digest_is_stable``: one extra space in ``COMPILE_SYSTEM_PROMPT`` fails on the
  pinned digest.
"""

from __future__ import annotations

import pytest

from recall.compiled_graph import records as records_module
from recall.compiled_graph.anchors import ANCHOR_CHARS, SourceAnchor, build_source_anchors
from recall.compiled_graph.records import (
    COMPILE_SYSTEM_PROMPT,
    MAX_RECORDS,
    RECORD_KINDS,
    prompt_digest,
    validate_reply,
)

FIRST = "The deploy uses port 18015. Alice owns the release checklist."
SECOND = "x" * 1_300 + " The cache moved to SQLite in October."


def _anchors() -> list[SourceAnchor]:
    return build_source_anchors("notes/a.md", FIRST + " " * (1_600 - len(FIRST)) + SECOND)


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "kind": "fact",
        "summary": "",
        "detail": "",
        "entities": [],
        "evidence_anchor_ids": [],
    }
    record.update(overrides)
    return record


def test_fields_kept_only_when_verbatim_in_cited_anchor() -> None:
    anchors = _anchors()
    assert len(anchors) == 2
    reply = {
        "records": [
            _record(
                summary="The deploy uses port 18015.",
                detail="The cache moved to SQLite in October.",
                evidence_anchor_ids=[anchors[0].id],
            )
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert len(records) == 1
    assert records[0].fields == ("The deploy uses port 18015.", "")
    assert diagnostics.removed_fields == 1
    assert records[0].spans == ((anchors[0].start, anchors[0].end),)


def test_entities_kept_only_when_in_quoted_evidence() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            _record(
                summary="Alice owns the release checklist.",
                entities=["Alice", "Bob", "release checklist"] + [f"e{i}" for i in range(40)],
                evidence_anchor_ids=[anchors[0].id],
            )
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert records[0].entities == ("Alice", "release checklist")
    assert diagnostics.removed_entities == 41


def test_entity_spanning_two_cited_anchors_dropped() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            _record(
                summary="Alice owns the release checklist.",
                entities=["Alice", "  \n  ", "\n"],
                evidence_anchor_ids=[anchors[0].id, anchors[1].id],
            )
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert records[0].entities == ("Alice",)
    assert diagnostics.removed_entities == 2


def test_unknown_anchor_ids_dropped_and_bare_indices_resolved() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            _record(
                summary="The cache moved to SQLite in October.",
                evidence_anchor_ids=["a001", "a999_ffffffffffffffff"],
            )
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert records[0].anchor_ids == (anchors[1].id,)
    assert diagnostics.resolved_bare_anchor_ids == 1
    assert diagnostics.invalid_anchor_references == 1


def test_bare_index_resolves_only_against_sent_anchors() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            _record(summary="invented", evidence_anchor_ids=["a001"]),
            _record(summary="invented", evidence_anchor_ids=[anchors[1].id]),
        ]
    }
    records, diagnostics = validate_reply(reply, anchors, sent_anchor_ids=[anchors[0].id])
    assert diagnostics.resolved_bare_anchor_ids == 0
    assert [record.anchor_ids for record in records] == [(anchors[1].id,)]


def test_record_without_valid_anchor_dropped() -> None:
    anchors = _anchors()
    reply = {"records": [_record(summary="nothing here", evidence_anchor_ids=["a777_0000"])]}
    records, diagnostics = validate_reply(reply, anchors)
    assert records == []
    assert diagnostics.rejected_anchor_ids == 1


def test_value_recovers_anchor_when_cited_id_unknown() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            _record(summary="Alice owns the release checklist.", evidence_anchor_ids=["a777_0000"])
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert [record.anchor_ids for record in records] == [(anchors[0].id,)]
    assert diagnostics.recovered_anchor_records == 1


def test_recovery_ignores_entities() -> None:
    anchors = _anchors()
    reply = {"records": [_record(entities=["Alice"], evidence_anchor_ids=["a777_0000"])]}
    records, diagnostics = validate_reply(reply, anchors)
    assert records == []
    assert diagnostics.recovered_anchor_records == 0
    assert diagnostics.rejected_anchor_ids == 1


def test_whitespace_summary_does_not_recover_an_anchor() -> None:
    anchors = _anchors()
    reply = {"records": [_record(summary="   ", evidence_anchor_ids=["a777_0000"])]}
    records, diagnostics = validate_reply(reply, anchors)
    assert records == []
    assert diagnostics.recovered_anchor_records == 0


def test_whitespace_is_never_grounded() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            _record(summary="   ", entities=["  ", "Alice"], evidence_anchor_ids=[anchors[0].id])
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert records[0].fields == ("", anchors[0].quote[:700])
    assert records[0].entities == ("Alice",)
    assert diagnostics.evidence_backfilled_records == 1


#: Zero-width space (Cf), variation selector (Mn), Hangul filler (Lo), blank Braille (So), a
#: private-use character (Co) and the musical null notehead (So).
INVISIBLE = [
    pytest.param(chr(code), id=f"U+{code:04X}")
    for code in (0x200B, 0xFE0F, 0x3164, 0x2800, 0xE000, 0x1D159)
]


def _anchors_holding(char: str) -> list[SourceAnchor]:
    return build_source_anchors("notes/z.md", f"Alice{char} owns the release checklist.")


@pytest.mark.parametrize("char", INVISIBLE)
def test_invisible_summary_is_never_grounded(char: str) -> None:
    anchors = _anchors_holding(char)
    reply = {"records": [_record(summary=char, evidence_anchor_ids=[anchors[0].id])]}
    records, diagnostics = validate_reply(reply, anchors)
    assert records[0].fields == ("", anchors[0].quote)
    assert diagnostics.evidence_backfilled_records == 1


@pytest.mark.parametrize("char", INVISIBLE)
def test_invisible_summary_does_not_recover_an_anchor(char: str) -> None:
    anchors = _anchors_holding(char)
    reply = {"records": [_record(summary=char, evidence_anchor_ids=["a777_0000"])]}
    records, diagnostics = validate_reply(reply, anchors)
    assert records == []
    assert diagnostics.recovered_anchor_records == 0


def test_value_in_no_anchor_is_never_scanned_for_visible_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    category = records_module.unicodedata.category

    def counting_category(char: str) -> str:
        nonlocal calls
        calls += 1
        return category(char)

    monkeypatch.setattr(records_module.unicodedata, "category", counting_category)
    long_value = chr(0x200B) * (ANCHOR_CHARS * 4) + "a"
    reply = {"records": [_record(summary=long_value, evidence_anchor_ids=["a777_0000"])]}
    records, _ = validate_reply(reply, _anchors())
    assert records == []
    assert calls == 0


def test_more_than_eight_records_truncated() -> None:
    anchors = _anchors()
    one = _record(summary="The deploy uses port 18015.", evidence_anchor_ids=[anchors[0].id])
    records, diagnostics = validate_reply({"records": [one] * 12}, anchors)
    assert MAX_RECORDS == 8
    assert len(records) == 8
    assert diagnostics.proposed_records == 8


def test_unknown_kind_dropped() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            _record(kind="opinion", summary="The deploy uses port 18015.",
                    evidence_anchor_ids=[anchors[0].id])
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert records == []
    assert diagnostics.rejected_kind == 1
    assert "fact" in RECORD_KINDS


def test_empty_fields_backfilled_from_first_anchor() -> None:
    anchors = _anchors()
    reply = {"records": [_record(summary="invented", evidence_anchor_ids=[anchors[0].id])]}
    records, diagnostics = validate_reply(reply, anchors)
    assert records[0].fields == ("", anchors[0].quote[:700])
    assert diagnostics.evidence_backfilled_records == 1


@pytest.mark.parametrize("reply", [None, [], {"records": "x"}, {"nope": []}])
def test_malformed_reply_raises(reply: object) -> None:
    with pytest.raises(ValueError):
        validate_reply(reply, _anchors())  # type: ignore[arg-type]  # the malformed input is the point


def test_malformed_record_items_are_skipped() -> None:
    anchors = _anchors()
    reply = {
        "records": [
            "text",
            {"kind": 3},
            _record(summary=["x"], evidence_anchor_ids=[anchors[0].id]),
            _record(summary="The deploy uses port 18015.", entities="Alice",
                    evidence_anchor_ids=[anchors[0].id]),
        ]
    }
    records, diagnostics = validate_reply(reply, anchors)
    assert records == []
    assert diagnostics.rejected_malformed == 4


def test_prompt_digest_is_stable() -> None:
    assert "untrusted data" in COMPILE_SYSTEM_PROMPT
    assert prompt_digest() == PINNED_DIGEST


PINNED_DIGEST = "5d1a743aeed7b7e10ca72798fc4bc8ba71f44e3bd7da736bb5e5b9445febbd63"
