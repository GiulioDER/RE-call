from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.beta_recruiting import (
    RecruitmentRecord,
    detect_contact_fields,
    input_field_guide,
    load_records,
    rank_leads,
)


def test_detect_contact_fields_flags_email_columns() -> None:
    blocked = detect_contact_fields(["platform", "title", "email", "contact_email"])
    assert blocked == {"email", "contact_email"}


def test_input_field_guide_excludes_contact_fields() -> None:
    fields = input_field_guide()
    assert "email" not in fields
    assert "phone" not in fields
    assert "author_handle" in fields


def test_load_records_rejects_contact_columns(tmp_path) -> None:
    path = tmp_path / "leads.csv"
    path.write_text(
        "platform,title,email\nreddit,Need help with memory,user@example.com\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contact fields are not allowed"):
        load_records(path)


def test_rank_leads_prefers_fresh_topic_matches() -> None:
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    records = [
        RecruitmentRecord(
            platform="reddit",
            community="r/LocalLLaMA",
            url="https://example.com/reddit-1",
            title="Agent memory keeps going stale",
            body="Our agent hallucinates after long chats and loses provenance.",
            author_handle="user1",
            posted_at="2026-08-10T10:00:00Z",
            replies=12,
            reactions=35,
            tags="agent-memory, provenance",
        ),
        RecruitmentRecord(
            platform="reddit",
            community="r/python",
            url="https://example.com/reddit-2",
            title="Packaging question",
            body="Need help with pyproject extras.",
            author_handle="user2",
            posted_at="2026-05-01T10:00:00Z",
            replies=1,
            reactions=1,
            tags="packaging",
        ),
    ]
    ranked = rank_leads(records, ["agent memory", "hallucinates", "provenance", "stale"], now=now)
    assert [lead.record.url for lead in ranked] == ["https://example.com/reddit-1"]
    assert ranked[0].action == "public_reply"
    assert "abstention" in ranked[0].message_angle.lower()


def test_rank_leads_uses_platform_specific_action() -> None:
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    record = RecruitmentRecord(
        platform="discord",
        community="AI founders",
        url="https://example.com/discord-1",
        title="Memory layer pain",
        body="Looking for better retrieval and compliance guardrails.",
        author_handle="builder",
        posted_at="2026-08-11T08:00:00Z",
        replies=4,
        reactions=6,
        tags="retrieval, compliance",
    )
    ranked = rank_leads([record], ["retrieval", "compliance"], now=now)
    assert ranked[0].action == "community_post"
