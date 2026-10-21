"""Whether a sentence that names a supersession ASSERTS it.

Three readers turn prose into a proposed `supersedes:` edge: `recall.fix`, which `recall lint --fix`
offers to write; the deterministic extraction engine behind `recall rewrite plan`
(`recall.truth_extraction._engine`); and the deterministic proposal rules behind `recall reasoning
proposals` and the MCP `recall_rewrite_plan` tool. Each matched a cue word and a reference and
nothing else, so "This never supersedes [[a]]" and "If the canary fails, this would be replaced by
[[b]]" each became an edge, and an applied edge demotes the memo it names in search. The refusal
lives here so the three readers cannot disagree about what a negated or conditional cue is.

Each word is judged over the narrowest span that still governs the cue:

* negations (``not``, ``never``, ``cannot``, ``neither``, ``nor``, ``nothing``, ``none``,
  ``nobody``, ``no``, ``no longer``, ``n't``) and modals (``would``, ``might``, ``could``,
  ``should``, and a lower-case ``may``, so the month does not count): the clause that holds the cue,
  from the nearest comma, colon, parenthesis or dash before it up to the reference. "This memo is
  no longer current, superseded by [[b]]" stays a real cue, because its negation sits in the clause
  before;
* ``if`` and ``unless``: the whole sentence, before the cue and after the reference, because a
  condition governs the clause it is attached to whichever side of a comma it sits. Before the cue
  a semicolon does not end that sentence ("If the canary fails; this is replaced by [[b]]" is
  conditional); after the reference it does ("superseded by [[b]]; see it if you need the
  history" keeps its cue);
* ``should``, ``were`` or ``had`` right after the reference: an inverted condition ("replaced by
  [[b]], should the canary fail").

A sentence ends at terminal punctuation, a semicolon (except for a leading condition, above), a
blank line, a list item, or either end of a heading line; a single newline inside prose does not
end one, because memos wrap their prose.

The span judged here is not the sentence a reviewer is shown: `recall.fix` shows
`recall.lint.sentence_around`, which ignores semicolons and masks nothing, so it is usually the
wider of the two. That is deliberate. The reviewer is the last gate, and the wider sentence is
where they see anything this module did not read.

References are masked before any word is looked for, so a memo named ``never-again.md`` does not
negate the sentence that names it.

Only a window of about `_WINDOW` characters on each side of the cue is read, opened at a line start
so it never begins inside a code span, and the reference patterns are anchored to the start of a
token: a memo with one very long token, or with many cues, costs time linear in its length rather
than quadratic. A cue whose line began more than `_WINDOW` characters before it is refused as
``"unjudged"``.

The words are deliberately few. A refused cue is not proposed, and its edge is left for a person to
declare by hand; a false edge demotes a current memo with nobody told.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

#: Characters read on each side of a cue. A sentence longer than this is judged on its nearest part.
_WINDOW = 2000

#: Document references every caller has in common: wikilinks, markdown link targets, inline code
#: and bare ``name.md``. Bounded and anchored to a token start, so a long token is scanned once.
#: Inline code is left unbounded on purpose: a bound would end a long span early and pair every
#: later backtick on the line the wrong way round, masking the prose between them.
_REFERENCE = re.compile(
    r"\[\[[^\]\n]{1,200}\]\]|\]\([^)\s]{1,400}\)|`[^`\n]+`|(?<![\w./\-])[\w./\-]+\.md\b"
)
#: Where a sentence ends, semicolons aside. The last two alternatives end one before and after an
#: ATX heading line (one to six `#`, then a space or the end of the line), so a wrapped prose line
#: that starts with `#470` is not a heading at either end.
_BREAKS = (
    r"[.!?][\"')\]*]*(?=\s|$)",
    r"\n[ \t]*\n",
    r"\n(?=[ \t]*(?:[-*+]|\d+[.)])[ \t])",
    r"\n(?=[ \t]{0,3}#{1,6}(?:[ \t]|$))",
    r"^[ \t]{0,3}#{1,6}(?:[ \t][^\n]*)?\n",
)
#: The sentence that bounds a negation, a modal and the text after the reference: a semicolon ends
#: it, so "Not now; superseded by [[b]]" keeps its cue.
_SENTENCE_BREAK = re.compile("|".join((";", *_BREAKS)), re.MULTILINE)
#: The sentence a leading `if` or `unless` governs: a semicolon does not end it, so "If the canary
#: fails; this is replaced by [[b]]" is conditional.
_CONDITION_BREAK = re.compile("|".join(_BREAKS), re.MULTILINE)
_CLAUSE_BREAK = re.compile(r"[,:()\u2013\u2014]|\s-\s")
_NEGATION = re.compile(
    r"\b(?:not|never|cannot|neither|nor|nothing|none|nobody|no\s+longer)\b|\bno\b(?!-)"
    r"|n['\u2019]t\b",
    re.IGNORECASE,
)
_MODAL = re.compile(r"\b(?:would|might|could|should)\b|(?-i:\bmay\b)", re.IGNORECASE)
_CONDITION = re.compile(r"\b(?:if|unless)\b", re.IGNORECASE)
#: "replaced by [[b]], should the canary fail": an inverted condition right after the reference,
#: past any punctuation or masked delimiter that closes it.
_INVERTED_CONDITION = re.compile(r"[\s\0,(\u2013\u2014-]*(?:should|were|had)\b", re.IGNORECASE)

Span = tuple[int, int]
#: "unjudged": the cue's line began more than `_WINDOW` characters before it, so the window could
#: not open at a line start; refused rather than judged on a window that may begin inside a span.
Unasserted = Literal["negated", "conditional", "unjudged"]


def _window(
    text: str, start: int, end: int, references: re.Pattern[str] | None, keep: Iterable[Span]
) -> tuple[int, str] | None:
    """``(offset, masked)``: the text around ``start:end`` with every reference masked, or None.

    The window opens just after a newline, because a code span or a wikilink never crosses one:
    opened mid-line it could begin inside a span and pair the backticks after it the wrong way
    round, masking the prose between them. None when no newline falls in the `_WINDOW` characters
    before ``start``.
    """
    lo = 0
    if start > _WINDOW:
        newline = text.find("\n", start - _WINDOW, start)
        if newline < 0:
            return None
        lo = newline + 1
    chunk = text[lo:min(len(text), end + _WINDOW)]
    spans = [(max(0, a - lo), min(len(chunk), max(0, b - lo))) for a, b in keep]
    for pattern in (_REFERENCE, references):
        if pattern is not None:
            spans.extend(m.span() for m in pattern.finditer(chunk))
    chars = list(chunk)
    for a, b in spans:
        chars[a:b] = "\0" * (b - a)
    return lo, "".join(chars)


def _last_break_end(pattern: re.Pattern[str], text: str, start: int, end: int) -> int:
    found = start
    for match in pattern.finditer(text, start, end):
        found = match.end()
    return found


def sentence_head(
    text: str, cue_start: int, references: re.Pattern[str] | None = None
) -> Span | None:
    """The span of the sentence before the cue at ``cue_start``, where its subject is written.

    None when the window could not open at a line start (see `_window`).
    """
    window = _window(text, cue_start, cue_start, references, ())
    if window is None:
        return None
    lo, masked = window
    return lo + _last_break_end(_SENTENCE_BREAK, masked, 0, cue_start - lo), cue_start


def unasserted(
    text: str, cue: Span, reference: Span, references: re.Pattern[str] | None = None
) -> Unasserted | None:
    """Why the cue at ``cue``, naming the reference at ``reference``, asserts no supersession.

    Returns ``"negated"``, ``"conditional"`` or ``"unjudged"``, or None when the sentence asserts
    the relation. ``references`` is the caller's own reference pattern, masked like the shared ones.
    """
    window = _window(text, cue[0], reference[1], references, (reference,))
    if window is None:
        return "unjudged"
    lo, masked = window
    cue_start, cue_end, ref_end = cue[0] - lo, cue[1] - lo, reference[1] - lo
    sentence_start = _last_break_end(_SENTENCE_BREAK, masked, 0, cue_start)
    clause_start = _last_break_end(_CLAUSE_BREAK, masked, sentence_start, cue_start)
    gap = masked[cue_end:ref_end]
    clause = masked[clause_start:cue_start] + " " + gap
    if _NEGATION.search(clause):
        return "negated"
    if _MODAL.search(clause):
        return "conditional"
    following = _SENTENCE_BREAK.search(masked, ref_end)
    tail = masked[ref_end: following.start() if following else len(masked)]
    condition_start = _last_break_end(_CONDITION_BREAK, masked, 0, cue_start)
    if (
        _CONDITION.search(masked[condition_start:cue_start] + " " + gap + " " + tail)
        or _INVERTED_CONDITION.match(tail)
    ):
        return "conditional"
    return None
