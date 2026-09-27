"""T-2: the calendar range a question names, for a date-restricted extra retrieval leg.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-query-time-leg.md

Search is not told when a question is asked, so relative expressions ("last Saturday", "three
weeks ago") are resolved against an ``anchor`` the caller supplies (the tenant's latest stored
memory time in the service). Absolute expressions ("in March 2023", "on 7 May 2023") need none.
The earliest expression in the question wins; a question with none gets ``None``, and the leg does
not run. See docs/REFERENCE_TIME_DESIGN.md for why the range is only ever used to ADD candidates,
never to filter or demote: stored time is when something was said, not when it happened.
"""

from __future__ import annotations

from calendar import monthrange
from collections.abc import Callable
from datetime import date, timedelta
import re

_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a couple of": 2, "a few": 3,
}
_MONTH = r"(?P<month>" + "|".join(m.capitalize() for m in _MONTHS) + r"|Sept|" + "|".join(
    m.capitalize()[:3] for m in _MONTHS if m != "may"
) + r")\.?"
_COUNT = r"(?P<count>\d{1,3}|a couple of|a few|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

Range = tuple[date, date]


def _month_number(name: str) -> int:
    lowered = name.lower().rstrip(".")
    for index, month in enumerate(_MONTHS, start=1):
        if month.startswith(lowered):
            return index
    raise ValueError(name)


def _month_span(year: int, month: int) -> Range:
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _shift_months(day: date, months: int) -> tuple[int, int]:
    index = day.year * 12 + (day.month - 1) + months
    return index // 12, index % 12 + 1


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _iso(m: re.Match[str], anchor: date) -> Range:
    day = date(int(m["y"]), int(m["m"]), int(m["d"]))
    return day, day


def _day_month_year(m: re.Match[str], anchor: date) -> Range:
    day = date(int(m["year"]), _month_number(m["month"]), int(m["day"]))
    return day, day


def _month_year(m: re.Match[str], anchor: date) -> Range:
    return _month_span(int(m["year"]), _month_number(m["month"]))


def _month_only(m: re.Match[str], anchor: date) -> Range:
    month = _month_number(m["month"])
    year = anchor.year if month <= anchor.month else anchor.year - 1
    return _month_span(year, month)


def _year(m: re.Match[str], anchor: date) -> Range:
    year = int(m["year"])
    return date(year, 1, 1), date(year, 12, 31)


def _yesterday(m: re.Match[str], anchor: date) -> Range:
    day = anchor - timedelta(days=1)
    return day, day


def _last_weekday(m: re.Match[str], anchor: date) -> Range:
    target = _WEEKDAYS.index(m["weekday"].lower())
    back = (anchor.weekday() - target) % 7 or 7
    day = anchor - timedelta(days=back)
    return day, day


def _last_span(m: re.Match[str], anchor: date) -> Range:
    span = m["span"].lower()
    if span == "week":
        start = _monday(anchor) - timedelta(weeks=1)
        return start, start + timedelta(days=6)
    if span == "weekend":
        saturday = _monday(anchor) - timedelta(weeks=1) + timedelta(days=5)
        return saturday, saturday + timedelta(days=1)
    if span == "month":
        return _month_span(*_shift_months(anchor, -1))
    return date(anchor.year - 1, 1, 1), date(anchor.year - 1, 12, 31)


def _ago(m: re.Match[str], anchor: date) -> Range:
    raw = m["count"].lower()
    count = int(raw) if raw.isdigit() else _NUMBERS[raw]
    unit = m["unit"].lower()
    if unit == "day":
        centre = anchor - timedelta(days=count)
        return centre - timedelta(days=1), centre + timedelta(days=1)
    if unit == "week":
        centre = anchor - timedelta(weeks=count)
        return centre - timedelta(weeks=1), centre + timedelta(weeks=1)
    if unit == "month":
        start = _month_span(*_shift_months(anchor, -count - 1))[0]
        end = _month_span(*_shift_months(anchor, -count + 1))[1]
        return start, end
    return date(anchor.year - count - 1, 1, 1), date(anchor.year - count + 1, 12, 31)


#: (pattern, resolver), in no particular order: the earliest match in the question wins.
_RULES: tuple[tuple[re.Pattern[str], Callable[[re.Match[str], date], Range]], ...] = (
    (re.compile(r"\b(?P<y>(?:19|20)\d\d)-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])\b"), _iso),
    (
        re.compile(r"\b(?P<day>[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+" + _MONTH + r",?\s+(?P<year>(?:19|20)\d\d)\b"),
        _day_month_year,
    ),
    (
        re.compile(r"\b" + _MONTH + r"\s+(?P<day>[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(?P<year>(?:19|20)\d\d)\b"),
        _day_month_year,
    ),
    (re.compile(r"\b" + _MONTH + r",?\s+(?P<year>(?:19|20)\d\d)\b"), _month_year),
    (
        # The prefix is case-insensitive ("In June"); the month is not, so the verb "may" is never
        # read as the month May.
        re.compile(r"\b(?i:in|during|since|until|by|before|after|of|early|late|mid)[- ](?i:early |late |mid-?)?" + _MONTH + r"\b(?!\.?,?\s*(?:[1-9]|[12]\d|3[01]|(?:19|20)\d\d)\b)"),
        _month_only,
    ),
    (re.compile(r"\b(?i:in|during|of|throughout|since)\s+(?P<year>(?:19|20)\d\d)\b"), _year),
    (re.compile(r"\b(?:yesterday|last night)\b", re.IGNORECASE), _yesterday),
    (re.compile(r"\b(?:last|this past)\s+(?P<weekday>" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE), _last_weekday),
    (re.compile(r"\b(?:last|this past)\s+(?P<span>weekend|week|month|year)\b", re.IGNORECASE), _last_span),
    (re.compile(r"\b" + _COUNT + r"\s+(?P<unit>day|week|month|year)s?\s+ago\b", re.IGNORECASE), _ago),
)


def query_time_range(query: str, anchor: date) -> Range | None:
    """The inclusive date range the earliest time expression in ``query`` names, or ``None``."""
    best: tuple[int, Range] | None = None
    for pattern, resolve in _RULES:
        match = pattern.search(query)
        if match is None:
            continue
        try:
            resolved = resolve(match, anchor)
        except ValueError:
            continue
        if best is None or match.start() < best[0]:
            best = (match.start(), resolved)
    return None if best is None else best[1]
