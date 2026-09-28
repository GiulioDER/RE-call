"""W2: show returned windows as conversation, at Search time: speaker marks and session coalescing.

Round two, 2026-09-28. On AML Textual, C9 returns up to 100 content-only 160-word windows in
relevance order: no speaker, 40 words repeated between neighbours, and the windows of one
exchange scattered through the list. Two options, each off by default:

* **Speaker marks** (K-1, pre-registered 2026-09-26): ``[user]`` or ``[assistant]`` where each
  message begins, in any span that covers two or more roles (`recall_aml.speaker_render`).
* **Session coalescing**: the returned windows of one Add become one item, in source order, with
  the overlaps removed and `` … `` at each gap. The item keeps the position, id, score and date of
  its best-ranked window. Order between Adds stays relevance order; only order within one Add is
  source order (global chronological order was measured and hurts every type but ordering).

Both work from facts `build_chunks` stores with each raw window (word offsets, the roles and word
ranges of the messages it overlaps, a digest of its Add) and `render_full_evidence` carries on
`SearchItem.render_facts`, a field never sent to a client. An item without them (compiled records,
images, windows stored before this change) passes through untouched, and with both options off
every item is returned as it came. Nothing stored, embedded or ranked changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from recall_aml.models import SearchItem
from recall_aml.speaker_render import mark_window

GAP = " … "


def _facts(item: SearchItem) -> dict[str, Any] | None:
    facts = item.render_facts
    if not isinstance(item.content, str) or not facts or not isinstance(facts.get("word_start"), int):
        return None
    return facts


def _ranges(members: Sequence[SearchItem]) -> list[tuple[str, int, int]]:
    seen: set[tuple[str, int, int]] = set()
    for member in members:
        for role, start, end in (member.render_facts or {}).get("speaker_ranges") or ():
            seen.add((str(role), int(start), int(end)))
    return sorted(seen, key=lambda item: (item[1], item[2]))


def _render(members: Sequence[SearchItem], *, speakers: bool) -> SearchItem:
    """One item from the windows of one Add: stitched in source order, optionally speaker-marked."""
    best = members[0]
    spans: list[tuple[int, list[str]]] = []
    for member in sorted(members, key=lambda item: int((item.render_facts or {})["word_start"])):
        start = int((member.render_facts or {})["word_start"])
        words = str(member.content).split()
        if spans and start <= spans[-1][0] + len(spans[-1][1]):
            span_start, span_words = spans[-1]
            overlap = span_start + len(span_words) - start
            span_words.extend(words[overlap:])
        else:
            spans.append((start, list(words)))
    ranges = _ranges(members) if speakers else []
    texts = [
        mark_window(" ".join(words), start, ranges) if speakers else " ".join(words)
        for start, words in spans
    ]
    content = GAP.join(texts)
    return best if content == best.content else best.model_copy(update={"content": content})


def compose_items(items: Sequence[SearchItem], *, speakers: bool, coalesce: bool) -> list[SearchItem]:
    """Speaker-mark and/or coalesce raw windows; everything else keeps its place unchanged."""
    if not speakers and not coalesce:
        return list(items)
    slots: list[SearchItem | list[SearchItem]] = []
    by_add: dict[tuple[str, str], list[SearchItem]] = {}
    for item in items:
        facts = _facts(item)
        if facts is None:
            slots.append(item)
            continue
        add = facts.get("add_digest")
        if coalesce and isinstance(add, str) and add:
            key = (item.session_id, add)
            if key in by_add:
                by_add[key].append(item)
                continue
            by_add[key] = [item]
            slots.append(by_add[key])
        else:
            slots.append([item])
    return [slot if isinstance(slot, SearchItem) else _render(slot, speakers=speakers) for slot in slots]
