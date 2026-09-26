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
"""

from __future__ import annotations

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


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _shift_months(day: date, months: int) -> tuple[int, int]:
    index = day.year * 12 + (day.month - 1) + months
    return index // 12, index % 12 + 1


def _resolve(match: re.Match[str], anchor: date) -> str:
    if match.group("before"):
        return f"[= {anchor - timedelta(days=2):%Y-%m-%d}]"
    if match.group("ago"):
        raw = match.group("count").lower()
        count = int(raw) if raw.isdigit() else _NUMBERS[raw]
        unit = match.group("unit").lower()
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
        word = word.lower()
        if word in {"yesterday", "last night"}:
            return f"[= {anchor - timedelta(days=1):%Y-%m-%d}]"
        if word == "tomorrow":
            return f"[= {anchor + timedelta(days=1):%Y-%m-%d}]"
        return f"[= {anchor:%Y-%m-%d}]"
    step = {"last": -1, "this": 0, "next": 1}[match.group("rel").lower()]
    span = match.group("span").lower()
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


def resolve_text(text: str, anchor: date) -> str:
    """Insert `` [resolution]`` after every matched relative expression in ``text``."""
    return _PATTERN.sub(lambda match: f"{match.group(0)} {_resolve(match, anchor)}", text)


def _anchor(created: datetime) -> date:
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(timezone.utc).date()


def resolve_relative_times(items: Sequence[SearchItem]) -> list[SearchItem]:
    """Apply ``resolve_text`` to every text item that has a ``created_at``; keep the rest."""
    output: list[SearchItem] = []
    for item in items:
        if item.created_at is None or not isinstance(item.content, str):
            output.append(item)
            continue
        resolved = resolve_text(item.content, _anchor(item.created_at))
        output.append(item if resolved == item.content else item.model_copy(update={"content": resolved}))
    return output
