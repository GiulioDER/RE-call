"""Strict public and internal models for the hosted API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Message(StrictModel):
    role: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=200_000)
    timestamp: datetime | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_aml_timestamp(cls, value: object) -> object:
        """Accept AML's optional Unix millisecond timestamp without loose coercion."""
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("timestamp must be Unix milliseconds")
        try:
            return datetime.fromtimestamp(value / 1_000, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError("timestamp is outside the supported range") from exc

    @field_validator("role", "content")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class AddRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=512)
    messages: list[Message] = Field(min_length=1, max_length=256)
    user_id: str = Field(min_length=1, max_length=1024)
    session_id: str = Field(min_length=1, max_length=1024)

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class AddResponse(StrictModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str
    status: Literal["stored"] = "stored"
    raw_count: int = Field(ge=1)
    compiled_count: int = Field(ge=0)
    compiler_fallback: bool = False


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(min_length=1, max_length=1024)
    top_k: int = Field(default=100, ge=1, le=100)
    options: list[str] | None = Field(default=None, max_length=20)

    @field_validator("query", "user_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class SearchItem(StrictModel):
    id: str
    content: str
    created_at: datetime | None = None
    source: str
    session_id: str
    kind: str
    score: float


class SearchResponse(StrictModel):
    data: list[SearchItem]


class DeleteRequest(StrictModel):
    user_id: str = Field(min_length=1, max_length=1024)

    @field_validator("user_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


MemoryKind = Literal[
    "symptom",
    "root cause",
    "failed attempt",
    "successful repair",
    "architectural decision",
    "procedure",
    "validation",
    "constraint",
    "repository fact",
]


class CodingMemoryRecord(StrictModel):
    kind: MemoryKind
    task_shape: str = ""
    problem: str = ""
    action: str = ""
    outcome: str = ""
    validation: str = ""
    entities: list[str] = Field(default_factory=list, max_length=32)
    evidence_quotes: list[str] = Field(default_factory=list, max_length=8)
    event_time: datetime | None = None
    source_session_id: str
    supersedes: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def require_substance(self) -> "CodingMemoryRecord":
        if not any(
            value.strip()
            for value in (self.task_shape, self.problem, self.action, self.outcome, self.validation)
        ):
            raise ValueError("a coding memory record must contain substantive evidence")
        return self

    def rendered(self, max_chars: int = 1_200) -> str:
        fields: list[tuple[str, Any]] = [
            ("kind", self.kind),
            ("entities", ", ".join(self.entities)),
            ("evidence", " | ".join(self.evidence_quotes)),
            ("task", self.task_shape),
            ("problem", self.problem),
            ("action", self.action),
            ("outcome", self.outcome),
            ("validation", self.validation),
        ]
        text = "\n".join(f"{key}: {value}" for key, value in fields if value)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"


class CompilerPayload(StrictModel):
    records: list[CodingMemoryRecord] = Field(default_factory=list, max_length=8)


class FacetPayload(StrictModel):
    facets: list[str] = Field(default_factory=list, max_length=4)
