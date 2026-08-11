from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

CONTACT_FIELD_TOKENS = {
    "email",
    "e_mail",
    "mail",
    "contact_email",
    "contact",
    "phone",
    "phone_number",
    "mobile",
    "whatsapp",
    "telegram",
    "discord_tag",
    "discord_handle",
    "linkedin",
}

DEFAULT_ALLOWED_FIELDS = {
    "platform",
    "community",
    "url",
    "title",
    "body",
    "author_handle",
    "posted_at",
    "replies",
    "reactions",
    "upvotes",
    "comments",
    "tags",
}


@dataclass(frozen=True)
class RecruitmentRecord:
    platform: str
    community: str
    url: str
    title: str
    body: str
    author_handle: str
    posted_at: str
    replies: int
    reactions: int
    tags: str


@dataclass(frozen=True)
class RankedLead:
    record: RecruitmentRecord
    score: float
    matched_terms: tuple[str, ...]
    action: str
    reason: str
    message_angle: str


def _normalise_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def detect_contact_fields(fieldnames: Iterable[str]) -> set[str]:
    blocked: set[str] = set()
    for name in fieldnames:
        token = _normalise_key(name)
        if token in CONTACT_FIELD_TOKENS or token.endswith("_email") or token.endswith("_phone"):
            blocked.add(name)
    return blocked


def _coerce_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _row_text(row: RecruitmentRecord) -> str:
    return " ".join(
        part for part in (row.community, row.title, row.body, row.tags, row.platform) if part
    ).lower()


def _engagement_score(row: RecruitmentRecord) -> float:
    return min(math.log1p(max(row.replies, 0) + max(row.reactions, 0)), 4.0)


def _recency_score(posted_at: str, now: datetime) -> float:
    if not posted_at:
        return 0.0
    stamp = posted_at.strip().replace("Z", "+00:00")
    when = datetime.fromisoformat(stamp)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age_days = max((now - when.astimezone(timezone.utc)).total_seconds() / 86400.0, 0.0)
    return max(0.0, 2.0 - min(age_days / 30.0, 2.0))


def _message_angle(matched_terms: tuple[str, ...]) -> str:
    if not matched_terms:
        return "Ask an open question about current memory and retrieval pain."
    if any("hallucin" in term for term in matched_terms):
        return "Lead with abstention and trust boundaries instead of more generation."
    if any("stale" in term or "outdated" in term for term in matched_terms):
        return "Lead with supersession, freshness, and validity-aware retrieval."
    if any("tenant" in term or "compliance" in term or "security" in term for term in matched_terms):
        return "Lead with tenant isolation, provenance, and policy controls."
    return "Lead with trustworthy memory, provenance, and explicit abstention."


def load_records(path: Path) -> list[RecruitmentRecord]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return _records_from_rows(reader.fieldnames or [], reader)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        fieldnames = list(rows[0].keys()) if rows else []
        return _records_from_rows(fieldnames, rows)
    raise ValueError(f"unsupported input format: {path.suffix}")


def _records_from_rows(
    fieldnames: Iterable[str], rows: Iterable[dict[str, object]]
) -> list[RecruitmentRecord]:
    blocked = detect_contact_fields(fieldnames)
    if blocked:
        names = ", ".join(sorted(blocked))
        raise ValueError(
            f"contact fields are not allowed in outreach research inputs: {names}. "
            "Remove direct contact columns and rerun."
        )
    records: list[RecruitmentRecord] = []
    for raw in rows:
        row = {_normalise_key(str(key)): value for key, value in raw.items()}
        records.append(
            RecruitmentRecord(
                platform=str(row.get("platform", "")).strip(),
                community=str(row.get("community", "")).strip(),
                url=str(row.get("url", "")).strip(),
                title=str(row.get("title", "")).strip(),
                body=str(row.get("body", "")).strip(),
                author_handle=str(row.get("author_handle", "")).strip(),
                posted_at=str(row.get("posted_at", "")).strip(),
                replies=_coerce_int(str(row.get("replies", row.get("comments", "0")))),
                reactions=_coerce_int(str(row.get("reactions", row.get("upvotes", "0")))),
                tags=str(row.get("tags", "")).strip(),
            )
        )
    return records


def rank_leads(
    records: Iterable[RecruitmentRecord],
    terms: Iterable[str],
    now: datetime | None = None,
) -> list[RankedLead]:
    now = now or datetime.now(timezone.utc)
    canonical_terms = tuple(term.strip().lower() for term in terms if term.strip())
    ranked: list[RankedLead] = []
    for record in records:
        haystack = _row_text(record)
        matched_terms = tuple(term for term in canonical_terms if term in haystack)
        if not matched_terms:
            continue
        keyword_score = len(matched_terms) * 2.5
        score = keyword_score + _engagement_score(record) + _recency_score(record.posted_at, now)
        if score <= 0:
            continue
        action = "public_reply"
        if record.platform.lower() == "discord":
            action = "community_post"
        if record.platform.lower() in {"forum", "site"}:
            action = "thread_reply"
        reason = (
            f"{len(matched_terms)} topic match(es), engagement {record.replies + record.reactions}, "
            f"community {record.community or 'unknown'}"
        )
        ranked.append(
            RankedLead(
                record=record,
                score=round(score, 2),
                matched_terms=matched_terms,
                action=action,
                reason=reason,
                message_angle=_message_angle(matched_terms),
            )
        )
    ranked.sort(key=lambda lead: (-lead.score, lead.record.platform, lead.record.url))
    return ranked


def write_ranked_csv(path: Path, leads: Iterable[RankedLead]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "score",
                "platform",
                "community",
                "url",
                "title",
                "author_handle",
                "posted_at",
                "matched_terms",
                "action",
                "reason",
                "message_angle",
            ],
        )
        writer.writeheader()
        for lead in leads:
            writer.writerow(
                {
                    "score": f"{lead.score:.2f}",
                    "platform": lead.record.platform,
                    "community": lead.record.community,
                    "url": lead.record.url,
                    "title": lead.record.title,
                    "author_handle": lead.record.author_handle,
                    "posted_at": lead.record.posted_at,
                    "matched_terms": ", ".join(lead.matched_terms),
                    "action": lead.action,
                    "reason": lead.reason,
                    "message_angle": lead.message_angle,
                }
            )


def input_field_guide() -> tuple[str, ...]:
    return tuple(sorted(DEFAULT_ALLOWED_FIELDS))
