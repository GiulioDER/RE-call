"""Whether a sentence that names a supersession ASSERTS it.

Two readers turn prose into a proposed `supersedes:` edge: `recall.fix`, which `recall lint --fix`
offers to write, and the deterministic proposal rules behind `recall rewrite plan`. Both matched a
cue word and a reference and nothing else, so "This never supersedes [[a]]" and "If the canary
fails, this would be replaced by [[b]]" each became an edge, and an applied edge demotes the memo
it names in search (supersession audit of 2026-10-21, findings 6 and 25). The refusal lives here so
the two readers cannot disagree about what a negated or conditional cue is.

Each word is judged over the narrowest span that still governs the cue:

* negation and modals (``not``, ``never``, ``no longer``, ``n't``, ``would``, ``might``, ``could``,
  ``should``): the clause that holds the cue, from the nearest comma, colon, parenthesis or dash
  before it up to the reference. "This memo is no longer current, superseded by [[b]]" stays a real
  cue, because its negation sits in the clause before;
* ``if`` and ``unless``: the whole sentence before the cue, because a leading condition governs the
  clause after its comma, and the rest of the cue's own clause after the reference ("replaced by
  [[b]] if the canary fails", "replaced by [[b]] should the canary fail").

A sentence ends at terminal punctuation, a semicolon, a blank line, a list item or a heading; a
single newline does not end one, because memos wrap their prose. References are masked before any
word is looked for, so a memo named ``never-again.md`` does not negate the sentence that names it.

The words are deliberately few. A refusal sends the cue back to a human, who can still declare the
edge; a false edge demotes a current memo with nobody told.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

#: Document references every caller has in common: wikilinks, markdown link targets, inline code
#: and bare ``name.md``. A caller adds its own forms through ``references``.
_REFERENCE = re.compile(r"\[\[[^\]\n]+\]\]|\]\([^)\s]+\)|`[^`\n]+`|[\w./\-]+\.md\b")
_SENTENCE_BREAK = re.compile(
    r"[.!?][\"')\]*]*(?=\s|$)|;|\n[ \t]*\n|\n(?=[ \t]*(?:[-*+]|\d+[.)])[ \t])|\n(?=[ \t]*#)"
)
_CLAUSE_BREAK = re.compile(r"[,:()–—]|\s-\s")
_NEGATION = re.compile(
    r"\b(?:not|never|cannot|neither|nor|no\s+longer)\b|n['’]t\b", re.IGNORECASE
)
_MODAL = re.compile(r"\b(?:would|might|could|should)\b", re.IGNORECASE)
_CONDITION = re.compile(r"\b(?:if|unless)\b", re.IGNORECASE)
#: "replaced by [[b]] should the canary fail": an inverted condition right after the reference.
_INVERTED_CONDITION = re.compile(r"\s*(?:should|were|had)\b", re.IGNORECASE)

Span = tuple[int, int]


def _masked(text: str, spans: Iterable[Span]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = "\0" * (end - start)
    return "".join(chars)


def _last_break_end(pattern: re.Pattern[str], text: str, start: int, end: int) -> int:
    found = start
    for match in pattern.finditer(text, start, end):
        found = match.end()
    return found


def clause_head(text: str, cue_start: int, references: Iterable[Span] = ()) -> Span:
    """The span of the clause before the cue at ``cue_start``: where its subject is written."""
    masked = _masked(text, [*references, *((m.start(), m.end()) for m in _REFERENCE.finditer(text))])
    sentence_start = _last_break_end(_SENTENCE_BREAK, masked, 0, cue_start)
    return _last_break_end(_CLAUSE_BREAK, masked, sentence_start, cue_start), cue_start


def unasserted(text: str, cue: Span, reference: Span, references: Iterable[Span] = ()) -> str | None:
    """Why the cue at ``cue``, naming the reference at ``reference``, asserts no supersession.

    Returns ``"negated"`` or ``"conditional"``, or None when the sentence asserts the relation.
    ``references`` are further spans to mask, the caller's own reference forms.
    """
    spans = [reference, *references, *((m.start(), m.end()) for m in _REFERENCE.finditer(text))]
    masked = _masked(text, spans)
    cue_start, cue_end = cue
    ref_end = reference[1]
    sentence_start = _last_break_end(_SENTENCE_BREAK, masked, 0, cue_start)
    clause_start = _last_break_end(_CLAUSE_BREAK, masked, sentence_start, cue_start)
    gap = masked[cue_end:ref_end]
    clause = masked[clause_start:cue_start] + " " + gap
    if _NEGATION.search(clause):
        return "negated"
    if _MODAL.search(clause):
        return "conditional"
    following = _SENTENCE_BREAK.search(masked, ref_end)
    sentence_end = following.start() if following else len(masked)
    clause_after = _CLAUSE_BREAK.search(masked, ref_end, sentence_end)
    tail = masked[ref_end: clause_after.start() if clause_after else sentence_end]
    if (
        _CONDITION.search(masked[sentence_start:cue_start] + " " + gap)
        or _CONDITION.search(tail)
        or _INVERTED_CONDITION.match(tail)
    ):
        return "conditional"
    return None
