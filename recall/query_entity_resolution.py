"""Deterministic query entity resolution for semantic graph activation.

This module only reads graph identifiers and labels. It never turns a graph label into evidence.
The serving layer still requires a trusted retrieval hit before it can traverse an authored edge.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import re
from typing import Literal

from recall.semantic_graph import SemanticGraphProjection, normalize_entity_name

DatePrecision = Literal["day", "month", "year", "relative"]
EntityMatchKind = Literal["exact", "date"]

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_NAMES = "|".join(_MONTHS)
_ISO_DATE_RE = re.compile(r"(?<![\d])(?P<year>\d{4})[/. -](?P<month>\d{1,2})(?:[/. -](?P<day>\d{1,2}))?(?![\d])")
_D_MONTH_DATE_RE = re.compile(
    rf"(?<!\w)(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_NAMES})\s*,?\s*(?P<year>\d{{4}})(?!\w)",
    re.IGNORECASE,
)
_MONTH_D_DATE_RE = re.compile(
    rf"(?<!\w)(?P<month>{_MONTH_NAMES})\s+(?P<day>\d{{1,2}}),?\s*(?P<year>\d{{4}})(?!\w)",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    rf"(?<!\w)(?P<month>{_MONTH_NAMES})\s+(?P<year>\d{{4}})(?!\w)",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<![\d/-])(?P<year>\d{4})(?![\d/-])")
_RELATIVE_RE = re.compile(
    r"(?<!\w)(?P<value>today|yesterday|tomorrow|this week|last week|next week|"
    r"this month|last month|next month|this year|last year|next year)(?!\w)",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(
    r"(?<=[?!.;])\s+|\s+(?:and|or|but|then|also|while|versus|vs\.?)(?:\s+|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryDate:
    """One date or bounded date range found in a query or entity label."""

    text: str
    start: date
    end: date
    precision: DatePrecision


@dataclass(frozen=True)
class ResolvedQueryEntity:
    """A graph entity selected by a bounded lexical or date match."""

    entity_id: str
    matched_label: str
    clause_index: int
    match_kind: EntityMatchKind


@dataclass(frozen=True)
class QueryEntityResolution:
    """The safe, inspectable output of deterministic query parsing."""

    query: str
    clauses: tuple[str, ...]
    entities: tuple[ResolvedQueryEntity, ...] = ()
    dates: tuple[QueryDate, ...] = ()
    ambiguous_labels: tuple[str, ...] = ()

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.entity_id for item in self.entities))


def _month_range(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        following = date(year + 1, 1, 1)
    else:
        following = date(year, month + 1, 1)
    return start, following - timedelta(days=1)


def _year_range(year: int) -> tuple[date, date]:
    return date(year, 1, 1), date(year, 12, 31)


def _date_from_match(match: re.Match[str]) -> QueryDate | None:
    groups = match.groupdict()
    try:
        year = int(groups["year"])
        month = int(groups["month"]) if groups.get("month") else None
        day = int(groups["day"]) if groups.get("day") else None
        if month is None:
            start, end = _year_range(year)
            precision: DatePrecision = "year"
        elif day is None:
            start, end = _month_range(year, month)
            precision = "month"
        else:
            start = end = date(year, month, day)
            precision = "day"
    except (TypeError, ValueError):
        return None
    return QueryDate(match.group(0), start, end, precision)


def _named_date_from_match(match: re.Match[str]) -> QueryDate | None:
    groups = match.groupdict()
    try:
        year = int(groups["year"])
        month = _MONTHS[groups["month"].casefold()]
        day = int(groups["day"]) if groups.get("day") else None
        if day is None:
            start, end = _month_range(year, month)
            precision: DatePrecision = "month"
        else:
            start = end = date(year, month, day)
            precision = "day"
    except (KeyError, TypeError, ValueError):
        return None
    return QueryDate(match.group(0), start, end, precision)


def _relative_range(value: str, reference: date) -> tuple[date, date] | None:
    normalized = value.casefold()
    if normalized == "today":
        return reference, reference
    if normalized == "yesterday":
        yesterday = reference - timedelta(days=1)
        return yesterday, yesterday
    if normalized == "tomorrow":
        tomorrow = reference + timedelta(days=1)
        return tomorrow, tomorrow
    if normalized in {"this week", "last week", "next week"}:
        offset = {"last week": -7, "this week": 0, "next week": 7}[normalized]
        start = reference - timedelta(days=reference.weekday()) + timedelta(days=offset)
        return start, start + timedelta(days=6)
    if normalized in {"this month", "last month", "next month"}:
        month_offset = {"last month": -1, "this month": 0, "next month": 1}[normalized]
        month_index = reference.year * 12 + reference.month - 1 + month_offset
        year, month_index = divmod(month_index, 12)
        return _month_range(year, month_index + 1)
    if normalized in {"this year", "last year", "next year"}:
        year_offset = {"last year": -1, "this year": 0, "next year": 1}[normalized]
        return _year_range(reference.year + year_offset)
    return None


def extract_query_dates(value: str, *, reference_time: datetime | None = None) -> tuple[QueryDate, ...]:
    """Extract bounded absolute and relative dates without a model call."""

    reference = (reference_time or datetime.now(UTC)).astimezone(UTC).date()
    found: list[tuple[int, int, QueryDate]] = []
    occupied: list[tuple[int, int]] = []

    def add(match: re.Match[str], parsed: QueryDate | None) -> None:
        if parsed is None:
            return
        span = match.span()
        if any(span[0] < end and start < span[1] for start, end in occupied):
            return
        occupied.append(span)
        found.append((span[0], span[1], parsed))

    for pattern, parser in (
        (_ISO_DATE_RE, _date_from_match),
        (_D_MONTH_DATE_RE, _named_date_from_match),
        (_MONTH_D_DATE_RE, _named_date_from_match),
        (_MONTH_YEAR_RE, _named_date_from_match),
    ):
        for match in pattern.finditer(value):
            add(match, parser(match))
    for match in _RELATIVE_RE.finditer(value):
        bounds = _relative_range(match.group("value"), reference)
        if bounds is not None:
            add(match, QueryDate(match.group(0), *bounds, "relative"))
    for match in _YEAR_RE.finditer(value):
        add(
            match,
            QueryDate(match.group(0), *_year_range(int(match.group("year"))), "year"),
        )
    unique: list[QueryDate] = []
    seen: set[tuple[date, date]] = set()
    for _start, _end, item in sorted(found, key=lambda item: item[0]):
        key = (item.start, item.end)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return tuple(unique)


def decompose_query(query: str) -> tuple[str, ...]:
    """Split a query into bounded lexical clauses while retaining the full query first."""

    whole = " ".join(query.split())
    if not whole:
        return ()
    clauses = [whole]
    clauses.extend(part.strip(" ,") for part in _CLAUSE_RE.split(whole))
    return tuple(dict.fromkeys(part for part in clauses if part))


def _contains_label(text: str, label: str) -> bool:
    normalized_text = normalize_entity_name(text)
    normalized_label = normalize_entity_name(label)
    return bool(normalized_label) and f" {normalized_label} " in f" {normalized_text} "


def _without_dates(value: str, *, reference_time: datetime | None) -> str:
    dates = extract_query_dates(value, reference_time=reference_time)
    if not dates:
        return normalize_entity_name(value)
    masked = value
    for item in sorted(dates, key=lambda item: value.find(item.text), reverse=True):
        start = value.rfind(item.text)
        if start >= 0:
            masked = masked[:start] + " " * len(item.text) + masked[start + len(item.text) :]
    return normalize_entity_name(masked)


def _dates_overlap(left: QueryDate, right: QueryDate) -> bool:
    return left.start <= right.end and right.start <= left.end


def _match_label(
    query: str,
    clauses: Sequence[str],
    label: str,
    *,
    reference_time: datetime | None,
    query_dates: Sequence[QueryDate],
) -> tuple[int, EntityMatchKind] | None:
    label_dates = extract_query_dates(label, reference_time=reference_time)
    clause_candidates = tuple(enumerate(clauses[1:], start=1))
    clause_candidates += ((0, clauses[0]),) if clauses else ()
    for clause_index, clause in clause_candidates:
        if _contains_label(clause, label):
            return clause_index, "exact"
    if not label_dates or not query_dates:
        return None
    if not any(_dates_overlap(label_date, query_date) for label_date in label_dates for query_date in query_dates):
        return None
    remaining = _without_dates(label, reference_time=reference_time)
    if remaining and not _contains_label(query, remaining):
        return None
    clause_index = next(
        (index for index, clause in clause_candidates if not remaining or _contains_label(clause, remaining)),
        0,
    )
    return clause_index, "date"


def resolve_query_entities(
    graph: SemanticGraphProjection,
    query: str,
    *,
    reference_time: datetime | None = None,
) -> QueryEntityResolution:
    """Resolve canonical names and explicit aliases, including equivalent date spellings.

    A label shared by multiple graph entities is rejected. Date matching is bounded by the
    non-date words in the label, so a query for one date cannot activate every dated entity.
    """

    clauses = decompose_query(query)
    query_dates = extract_query_dates(query, reference_time=reference_time)
    ambiguous_entity_ids = {
        entity_id
        for diagnostic in graph.diagnostics
        if diagnostic.kind == "ambiguous_entity"
        for entity_id in diagnostic.entity_ids
    }
    all_entities = {entity.id: entity for entity in graph.entities}
    entities = {
        entity.id: entity
        for entity in all_entities.values()
        if entity.id not in ambiguous_entity_ids
    }
    label_entity_ids: dict[str, set[str]] = defaultdict(set)
    labels_by_entity: dict[str, tuple[str, ...]] = {}
    for entity in all_entities.values():
        labels = tuple(dict.fromkeys((entity.canonical_name, *entity.aliases)))
        for label in labels:
            normalized = normalize_entity_name(label)
            if normalized:
                label_entity_ids[normalized].add(entity.id)
        if entity.id in entities:
            labels_by_entity[entity.id] = labels

    ambiguous_labels = {
        label for label, entity_ids in label_entity_ids.items() if len(entity_ids) > 1
    }
    ambiguous_labels.update(
        normalize_entity_name(diagnostic.reference)
        for diagnostic in graph.diagnostics
        if diagnostic.kind == "ambiguous_entity"
        and isinstance(diagnostic.reference, str)
        and normalize_entity_name(diagnostic.reference)
    )
    matches: list[tuple[int, int, str, ResolvedQueryEntity]] = []
    for entity_id, entity in entities.items():
        for label in labels_by_entity[entity_id]:
            normalized_label = normalize_entity_name(label)
            if not normalized_label or normalized_label in ambiguous_labels:
                continue
            label_match = _match_label(
                query,
                clauses,
                label,
                reference_time=reference_time,
                query_dates=query_dates,
            )
            if label_match is None:
                continue
            clause_index, match_kind = label_match
            matches.append(
                (
                    clause_index,
                    0 if match_kind == "exact" else 1,
                    entity_id,
                    ResolvedQueryEntity(entity_id, label, clause_index, match_kind),
                )
            )

    selected: list[ResolvedQueryEntity] = []
    seen_entities: set[str] = set()
    for _clause, _kind, _entity_id, match in sorted(
        matches,
        key=lambda item: (
            item[0],
            item[1],
            -len(normalize_entity_name(item[3].matched_label)),
            item[2],
            item[3].matched_label,
        ),
    ):
        if match.entity_id not in seen_entities:
            selected.append(match)
            seen_entities.add(match.entity_id)
    return QueryEntityResolution(
        query=query,
        clauses=clauses,
        entities=tuple(selected),
        dates=query_dates,
        ambiguous_labels=tuple(sorted(ambiguous_labels)),
    )


__all__ = [
    "QueryDate",
    "QueryEntityResolution",
    "ResolvedQueryEntity",
    "decompose_query",
    "extract_query_dates",
    "resolve_query_entities",
]
