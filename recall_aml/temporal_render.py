"""T-1: resolve relative time expressions in returned items against each item's own date.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md

AML's reader sees each item's date (``dated_items``) but still has to do the arithmetic from
"last Friday" to a calendar day, and about half of the audited LoCoMo temporal errors were exactly
that step. This transform inserts the resolved date in brackets right AFTER each matched phrase.
The phrase itself stays, because AML's LoCoMo judge wants the gold's own granularity and relative
form, and removing every inserted bracket gives back the original text byte for byte.

It is a Search-time render step only: nothing stored, embedded or ranked changes. The anchor is
the item's ``created_at`` (its Add's latest message time) as a UTC calendar day; an item without
one, or with multimodal content, is returned unchanged. The pattern table is the pre-registered one
and is not tuned on any benchmark.

W4 (round two, 2026-09-28) adds two options, both off by default so the served render is unchanged:

* ``render="v2"`` keeps week expressions relative and anchored. AML's LoCoMo and LongMemEval answer
  prompt keeps "week-based expressions relative", and its judge accepts a relative answer only when
  the anchor date and the unit match, so "last week" becomes ``[= the week before 2023-05-08]``
  rather than a computed Monday, and "two weeks ago" ``[= 2 weeks before 2023-05-08]`` rather than
  a day. Every other expression renders as in v1.
* ``skip_code=True`` is what lets T-1 run on every route (the content gate) without touching code:
  an item that looks like code is left alone, and inside prose a phrase that is part of code is
  left alone too: glued to code punctuation (``date.today()``, ``$yesterday``), an operand or a
  call argument (``today = now()``, ``f(since="yesterday")``), a shell flag's value
  (``date -d yesterday``), or inside a fence or a backtick span. The route gate, the default,
  protects code by never resolving on the code route (K6, 2026-09-26).

  🔁 Audit 2026-09-30: the first version annotated code holding fewer than two of its substring
  signals (``today = datetime.now()``, SQL ``'yesterday'``, ``--since=yesterday``), and read prose
  such as "by myself." and "I will return" as code. Every AML Coding Search takes the code route,
  and the content gate was measured on LoCoMo and LongMemEval-S only, so it has not been shown safe
  for Coding: YAML and INI ``key: value`` lines, for one, still look like prose here.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
import re

from recall_aml.models import SearchItem

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "a couple of": 2,
    "a few": 3,
}
_COUNT = r"\d{1,3}|a couple of|a few|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"

#: One alternation, longest phrases first, so "the day before yesterday" is never read as
#: "yesterday" and "last night" never as "last".
_PATTERN = re.compile(
    r"\b(?:"
    r"(?P<before>the day before yesterday)"
    r"|(?P<ago>(?P<count>" + _COUNT + r")\s+(?P<unit>day|week|month|year)s?\s+ago)"
    r"|(?P<dayword>today|tonight|this morning|this afternoon|this evening|yesterday|last night|tomorrow)"
    r"|(?P<rel>last|this|next)\s+(?P<span>weekend|week|month|year|" + "|".join(_WEEKDAYS) + r")"
    # Skip a phrase this transform already resolved, so applying it twice changes nothing; any
    # other bracket after a phrase (LoCoMo's "[shared image: ...]") does not block it.
    r")\b(?!\s*\[(?:=|≈|week of|weekend of) )",
    re.IGNORECASE,
)


RENDER_VERSIONS = ("v1", "v2")

#: ``re.IGNORECASE`` matches these to an ASCII letter of the pattern, but ``str.lower`` does not
#: map them to that letter, so a lookup by the lowered phrase failed: "laſt week" raised KeyError
#: and failed the whole Search, "yeſterday" resolved to the anchor day. These are every such
#: character in Python 3.12, enumerated over all code points.
_FOLD = str.maketrans({"\u0130": "i", "\u0131": "i", "\u017f": "s", "\u212a": "k"})
#: Every match contains one of these, so ASCII text holding none of them cannot match and skips
#: the pattern (most returned items hold no relative time). Non-ASCII text always runs it, because
#: of the characters in ``_FOLD``.
_KEYWORDS = ("yesterday", "today", "tonight", "tomorrow", "this", "last", "next", "ago")

#: Kinds of syntax that conversation rarely produces; an item holding two or more distinct kinds
#: is code to the content gate. Matched as syntax rather than as substrings, so "by myself.",
#: "I will return it", "Paris -> Rome" and a lone "{name}" stay prose.
_CODE_SYNTAX = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bdef \w+\s*\(",
        r"\bclass \w+\s*[:(]",
        r"\bfrom [\w.]+ import \w",
        r"\bimport \w+(?:\.\w+)+",
        r"(?<![A-Za-z])self\.\w",
        r"\w\(\)",
        r"\w\s*(?:==|!=)\s*\S",
        r"[{}]",
        r"\w\);",
        r"```",
        r"</\w",
        r"#include\b",
        r"\bconsole\.\w",
        r"\bprint\(",
        r"\[\]",
        r"&&|\|\|",
        r"\w->\w|\)\s*->",
        r"\)\s*=>",
        r"\b(?:let|const|var)\s+\w+\s*=",
        r"\b\w+\s*=\s*[\w.]+\(",
    )
)
#: A phrase touching one of these on its left is part of code (``$yesterday``, `` `today``).
_CODE_LEFT = frozenset("$@#\\`")
#: ... or on its right (``today()``, ``yesterday=True``, ``today[0]``).
_CODE_RIGHT = frozenset("(=[")
_QUOTES = frozenset("'\"")
#: A shell flag, whose value is code (``date -d yesterday``, ``--since yesterday``); "-meh-" is
#: prose.
_FLAG = re.compile(r"--?[A-Za-z](?:[\w-]*\w)?")
#: A fenced block or an inline backtick span. A backtick between letters is an apostrophe
#: ("didn`t"), never a span's edge.
_CODE_SPAN = re.compile(r"```.*?```|(?<![\w`])`[^`]*`(?![\w`])", re.DOTALL)
#: How far either side of a phrase `_glued_to_code` looks.
_LOOK = 64


def looks_like_code(text: str) -> bool:
    """Two or more distinct kinds of code syntax: code for the content gate, never prose."""
    kinds = 0
    for pattern in _CODE_SYNTAX:
        if pattern.search(text):
            kinds += 1
            if kinds == 2:
                return True
    return False


def _identifier(character: str) -> bool:
    return character.isalnum() or character == "_"


def _call_name(character: str) -> bool:
    """An ASCII identifier character, which ends a call's name (a CJK gloss "昨天(" is prose)."""
    return character.isascii() and _identifier(character)


def _token_sides(text: str, start: int, end: int) -> tuple[str, str]:
    """The non-space text glued to ``text[start:end]`` on each side, up to ``_LOOK`` characters."""
    low, high = start, end
    while low > max(0, start - _LOOK) and not text[low - 1].isspace():
        low -= 1
    while high < min(len(text), end + _LOOK) and not text[high].isspace():
        high += 1
    return text[low:start], text[end:high]


def _glued_to_code(text: str, start: int, end: int) -> bool:
    """Whether the phrase at ``text[start:end]`` is part of code rather than of the prose."""
    left = text[start - 1] if start > 0 else ""
    before = text[start - 2] if start > 1 else ""
    right = text[end] if end < len(text) else ""
    after = text[end + 1] if end + 1 < len(text) else ""
    if left in _CODE_LEFT or right in _CODE_RIGHT:
        return True
    # Member access, never the end of a sentence or an ellipsis: ``date.today``,
    # ``today.strftime``, ``Cls::today``, ``$this->today``. Prose writes "yesterday.",
    # "Well...yesterday", "Note:yesterday" and "tomorrow:10am".
    if (left == "." and _identifier(before)) or (right == "." and _identifier(after)):
        return True
    if (left == ":" == before) or (right == ":" == after) or (left == ">" and before == "-"):
        return True
    # A path or a URL, not an alternative: "logs/2023/today.txt", "/tmp/today", but not
    # "today/tomorrow" or "today/this week/this month".
    if "/" in (left, right):
        prefix, suffix = _token_sides(text, start, end)
        # A token that runs past the window (a sha256 path segment) is read as a path, the
        # side that keeps code as written.
        if (
            len(prefix) >= _LOOK
            or len(suffix) >= _LOOK
            or prefix.count("/") >= 2
            or suffix.count("/") >= 2
            or prefix.startswith(("/", "./", "~/"))
        ):
            return True
    # Only the neighbourhood decides, so the cost per phrase is bounded: slicing the whole text
    # before and after every phrase made a long item quadratic (a 120 KB item took 310 s).
    head = text[max(0, start - _LOOK):start]
    tail = text[end:end + _LOOK]
    opened = head[-1:] in _QUOTES
    if opened:
        head = head[:-1]
    if tail[:1] in _QUOTES:
        tail = tail[1:]
        # A literal in a list, an object or a cast: ``["today", ...]``, ``{"today": 1}``,
        # ``'2 weeks ago'::interval``.
        if tail.startswith(("]", "::")) or (opened and head.endswith(("[", "{"))):
            return True
        # A JSON or dict value: ``"when": "yesterday"``. Prose quotes after a word: 'She said:
        # "tomorrow"'.
        colon = head.rstrip()
        if opened and colon.endswith(":") and colon[-2:-1] in _QUOTES:
            return True
    # A call argument: ``f(yesterday)``, ``new Date(today)``, ``print("today")``. Prose opens a
    # parenthesis after a space: "(yesterday)".
    if head.endswith("(") and _call_name(head[-2:-1]):
        return True
    # An operand: ``since=yesterday``, ``day = 'yesterday'``, ``today = datetime.now()``. On the
    # same line only: a Markdown underline (``=====``) is not an assignment.
    stripped = head.rstrip(" \t")
    following = tail.lstrip(" \t")
    if stripped.endswith("=") or (following.startswith("=") and not following.startswith("=>")):
        return True
    line = stripped.rsplit("\n", 1)[-1]
    words = line.split()
    # A word the window cut into is not known to start with its first character here.
    cut = start > _LOOK and line == stripped and len(words) == 1 and not line[:1].isspace()
    return bool(words) and not cut and _FLAG.fullmatch(words[-1]) is not None


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _shift_months(day: date, months: int) -> tuple[int, int]:
    index = day.year * 12 + (day.month - 1) + months
    return index // 12, index % 12 + 1


def _key(phrase: str) -> str:
    """``phrase`` lowered the way the pattern matched it (``_FOLD``)."""
    return phrase.translate(_FOLD).lower()


def _resolve(match: re.Match[str], anchor: date, render: str = "v1") -> str:
    if render == "v2":
        anchored = _resolve_v2_week(match, anchor)
        if anchored is not None:
            return anchored
    if match.group("before"):
        return f"[= {anchor - timedelta(days=2):%Y-%m-%d}]"
    if match.group("ago"):
        raw = _key(match.group("count"))
        count = int(raw) if raw.isdigit() else _NUMBERS[raw]
        unit = _key(match.group("unit"))
        if unit == "day":
            return f"[≈ {anchor - timedelta(days=count):%Y-%m-%d}]"
        if unit == "week":
            return f"[≈ {anchor - timedelta(weeks=count):%Y-%m-%d}]"
        if unit == "month":
            year, month = _shift_months(anchor, -count)
            return f"[≈ {year:04d}-{month:02d}]"
        return f"[≈ {anchor.year - count:04d}]"
    word = match.group("dayword")
    if word:
        word = _key(word)
        if word in {"yesterday", "last night"}:
            return f"[= {anchor - timedelta(days=1):%Y-%m-%d}]"
        if word == "tomorrow":
            return f"[= {anchor + timedelta(days=1):%Y-%m-%d}]"
        return f"[= {anchor:%Y-%m-%d}]"
    step = {"last": -1, "this": 0, "next": 1}[_key(match.group("rel"))]
    span = _key(match.group("span"))
    if span == "week":
        return f"[week of {_monday(anchor) + timedelta(weeks=step):%Y-%m-%d}]"
    if span == "weekend":
        saturday = _monday(anchor) + timedelta(days=5)
        return f"[weekend of {saturday + timedelta(weeks=step):%Y-%m-%d}]"
    if span == "month":
        year, month = _shift_months(anchor, step)
        return f"[= {year:04d}-{month:02d}]"
    if span == "year":
        return f"[= {anchor.year + step:04d}]"
    target = _WEEKDAYS.index(span)
    if step == 0:
        day = _monday(anchor) + timedelta(days=target)
    elif step < 0:
        back = (anchor.weekday() - target) % 7 or 7
        day = anchor - timedelta(days=back)
    else:
        ahead = (target - anchor.weekday()) % 7 or 7
        day = anchor + timedelta(days=ahead)
    return f"[= {day:%Y-%m-%d}]"


def _resolve_v2_week(match: re.Match[str], anchor: date) -> str | None:
    """Week-based expressions, kept relative and anchored on the item's own day (v2 only)."""
    day = f"{anchor:%Y-%m-%d}"
    if match.group("ago") and _key(match.group("unit")) == "week":
        raw = _key(match.group("count"))
        count = int(raw) if raw.isdigit() else _NUMBERS[raw]
        return f"[= {count} week{'s' if count != 1 else ''} before {day}]"
    if match.group("rel") and _key(match.group("span")) in {"week", "weekend"}:
        span = _key(match.group("span"))
        where = {"last": "before", "this": "of", "next": "after"}[_key(match.group("rel"))]
        return f"[= the {span} {where} {day}]"
    return None


def resolve_text(text: str, anchor: date, *, render: str = "v1", skip_code: bool = False) -> str:
    """Insert `` [resolution]`` after every matched relative expression in ``text``.

    ``render`` picks the rendering (``RENDER_VERSIONS``); with ``skip_code`` a phrase that is part
    of code (`_glued_to_code`, or inside a fence or a backtick span) is left as written, and so is
    the whole of a text whose fences do not pair up, since it was cut inside one.
    """
    if render not in RENDER_VERSIONS:
        raise ValueError(f"unknown relative-time render {render!r}")
    if text.isascii():
        lowered = text.lower()
        if not any(keyword in lowered for keyword in _KEYWORDS):
            return text
    spans: list[tuple[int, int]] = []
    if skip_code and "`" in text:
        if text.count("```") % 2:
            return text
        spans = [found.span() for found in _CODE_SPAN.finditer(text)]
    # Spans never overlap, so the one that could hold a phrase is the last to open before it.
    opens = [low for low, _ in spans]

    def in_span(start: int, end: int) -> bool:
        index = bisect_right(opens, start) - 1
        return index >= 0 and end <= spans[index][1]

    def replace(match: re.Match[str]) -> str:
        start, end = match.span()
        if skip_code and (_glued_to_code(text, start, end) or in_span(start, end)):
            return match.group(0)
        return f"{match.group(0)} {_resolve(match, anchor, render)}"

    return _PATTERN.sub(replace, text)


def _anchor(created: datetime) -> date:
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(timezone.utc).date()


def resolve_relative_times(
    items: Sequence[SearchItem], *, render: str = "v1", skip_code: bool = False
) -> list[SearchItem]:
    """Apply ``resolve_text`` to every text item that has a ``created_at``; keep the rest.

    With ``skip_code`` (the content gate) an item that `looks_like_code` is kept as written.
    """
    output: list[SearchItem] = []
    for item in items:
        if item.created_at is None or not isinstance(item.content, str):
            output.append(item)
            continue
        if skip_code and looks_like_code(item.content):
            output.append(item)
            continue
        try:
            resolved = resolve_text(
                item.content, _anchor(item.created_at), render=render, skip_code=skip_code
            )
        except OverflowError:
            # An anchor at the edge of the calendar has no day before or after it: keep the item
            # as written rather than fail the whole Search.
            output.append(item)
            continue
        output.append(item if resolved == item.content else item.model_copy(update={"content": resolved}))
    return output
