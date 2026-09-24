"""Strict public and internal models for the hosted API."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from datetime import datetime, timezone
import json
import math
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TaskType = Literal["feature", "bugfix", "unknown"]
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_MEDIA_BYTES = 30 * 1024 * 1024
MAX_DATA_URL_CHARS = 14_000_000
_DATA_URL = re.compile(
    r"^data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]*={0,2})$",
    re.ASCII,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextContentPart(StrictModel):
    type: Literal["text"]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ImageURLValue(StrictModel):
    url: str = Field(min_length=1, max_length=MAX_DATA_URL_CHARS)

    @field_validator("url")
    @classmethod
    def validate_inline_image(cls, value: str) -> str:
        decode_image_data_url(value)
        return value


class ImageContentPart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageURLValue


ContentPart: TypeAlias = Annotated[
    TextContentPart | ImageContentPart,
    Field(discriminator="type"),
]
ContentValue: TypeAlias = str | list[ContentPart]


def decode_image_data_url(value: str) -> tuple[str, bytes]:
    """Decode one AML inline image after MIME, size, and signature validation."""
    match = _DATA_URL.fullmatch(value)
    if match is None:
        raise ValueError("image_url.url must be an inline JPEG, PNG, or WebP Base64 Data URI")
    media_type, encoded = match.groups()
    try:
        payload = b64decode(encoded, validate=True)
    except (Base64Error, ValueError) as exc:
        raise ValueError("image_url.url contains invalid Base64") from exc
    if not payload:
        raise ValueError("image payload must not be empty")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(f"decoded image exceeds {MAX_IMAGE_BYTES} bytes")
    signatures = {
        "image/jpeg": payload.startswith(b"\xff\xd8\xff"),
        "image/png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": len(payload) >= 12
        and payload.startswith(b"RIFF")
        and payload[8:12] == b"WEBP",
    }
    if not signatures[media_type]:
        raise ValueError(f"decoded bytes do not match declared media type {media_type}")
    return media_type, payload


def content_media_bytes(value: ContentValue) -> int:
    if isinstance(value, str):
        return 0
    return sum(
        len(decode_image_data_url(part.image_url.url)[1])
        for part in value
        if isinstance(part, ImageContentPart)
    )


MAX_QUERY_CHARS = 20_000
QUERY_HEAD_CHARS = 10_000


def bounded_query(value: str) -> str:
    """Fit a Search question to 20,000 characters instead of refusing it.

    A query within the bound is returned unchanged. A longer one keeps its first 10,000 and
    its last 9,999 characters, joined by a newline: a coding task statement may lead with the
    question or end with it, and both ends survive. The bound itself stays, because the query
    router's code-signal regex is quadratic on a long line and every Search stage (router,
    lexical, dense, graph) was sized against it.
    """
    if len(value) <= MAX_QUERY_CHARS:
        return value
    tail = MAX_QUERY_CHARS - QUERY_HEAD_CHARS - 1
    return value[:QUERY_HEAD_CHARS] + "\n" + value[-tail:]


UNKNOWN_ROLE = "unknown"
MAX_ROLE_CHARS = 64
MAX_TOP_K = 100
MAX_OPTIONS = 20


def _iso_timestamp(text: str) -> datetime | None:
    """An ISO 8601 date, read as UTC when it names no zone; None when it is not one."""
    if not text:
        return None
    if text[-1] in "Zz":
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _usable_parts(parts: list[Any]) -> list[Any]:
    """Keep the parts this service stores: nonblank text, as a text part, and images as sent.

    A part list that was accepted before is returned unchanged: its text parts are already
    `{"type": "text", "text": ...}` and its image parts are untouched.
    """
    kept: list[Any] = []
    for part in parts:
        if isinstance(part, BaseModel):
            part = part.model_dump(mode="python")
        if isinstance(part, str):
            if part.strip():
                kept.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            kept.append(part)
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            kept.append({"type": "text", "text": text})
    return kept


def _blank_content(content: ContentValue) -> bool:
    return not content.strip() if isinstance(content, str) else not content


def _option_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)


class Message(StrictModel):
    # AML trajectories may carry tool call fields (`name`, `tool_call_id`, ...) beside role and
    # content. Rejecting them turns a whole Coding Add into a permanent 422 that no retry can
    # repair, so a message ignores what it does not store while the request stays strict.
    model_config = ConfigDict(extra="ignore", strict=True)

    role: str = Field(default=UNKNOWN_ROLE, min_length=1, max_length=MAX_ROLE_CHARS)
    content: ContentValue = ""
    timestamp: datetime | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_aml_timestamp(cls, value: object) -> object:
        """Read AML's Unix millisecond timestamp, and whatever else a trajectory carries.

        An integer is Unix milliseconds, exactly as before. A float or a numeric string is read
        the same way, and an ISO 8601 string as a date. A value none of those can read, or one
        outside the supported range, becomes no timestamp: an optional field must never refuse
        the whole Add.
        """
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            text = value.strip()
            try:
                value = float(text)
            except ValueError:
                return _iso_timestamp(text)
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            try:
                return datetime.fromtimestamp(value / 1_000, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        return None

    @field_validator("role", mode="before")
    @classmethod
    def usable_role(cls, value: object) -> object:
        """A missing or blank role reads as `unknown`, and a long one keeps its first 64 chars."""
        if not isinstance(value, str) or not value.strip():
            return UNKNOWN_ROLE
        return value[:MAX_ROLE_CHARS]

    @field_validator("content", mode="before")
    @classmethod
    def usable_content(cls, value: object) -> object:
        """Reduce content to what this service stores, instead of refusing the Add.

        `null` (an assistant turn that only calls a tool) reads as blank text, which the Add
        then drops. In a part list, a text-bearing part of any type becomes a text part, a
        blank text part is dropped, and a part carrying neither text nor an image is dropped;
        an image part is kept as sent and validated as strictly as before.
        """
        if value is None:
            return ""
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            return _usable_parts(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: ContentValue) -> ContentValue:
        # Blank text, and a part list left empty, are legal here and dropped by
        # `AddRequest.drop_blank_messages`, so they never reach a window or the compiler.
        # There is no length or part-count cap: a Coding trajectory can carry a whole file or
        # log in one message, and a cap refused the entire Add permanently. The request body
        # limit (`MAX_BODY_BYTES` in app.py) bounds the total, as it always did.
        return value


class AddRequest(StrictModel):
    # A field this service does not read is ignored rather than refused, so a platform that
    # starts sending `metadata` or `app_id` cannot turn every Add into a 422.
    model_config = ConfigDict(extra="ignore", strict=True)

    request_id: str = Field(min_length=1, max_length=512)
    # No message-count cap, for the same reason as the content cap above: a long Coding session
    # can exceed 256 steps, and the body limit already bounds the total volume. An empty list
    # is a durable, empty Add, like a list whose messages are all blank.
    messages: list[Message]
    user_id: str = Field(min_length=1, max_length=1024)
    session_id: str = Field(min_length=1, max_length=1024)

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def drop_blank_messages(self) -> "AddRequest":
        """Store nothing for a blank text message instead of refusing the whole Add.

        Every Add that was accepted before this validator existed had no blank message, so
        dropping them changes nothing for those. An Add whose messages are all blank keeps an
        empty list, which the service answers as a durable, empty Add.
        """
        kept = [message for message in self.messages if not _blank_content(message.content)]
        if len(kept) != len(self.messages):
            self.messages = kept
        return self

    @model_validator(mode="after")
    def validate_media_budget(self) -> "AddRequest":
        media_bytes = sum(content_media_bytes(message.content) for message in self.messages)
        if media_bytes > MAX_REQUEST_MEDIA_BYTES:
            raise ValueError(
                f"decoded Add images exceed {MAX_REQUEST_MEDIA_BYTES} aggregate bytes"
            )
        return self


class AddResponse(StrictModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str
    status: Literal["stored"] = "stored"
    raw_count: int = Field(ge=0)
    compiled_count: int = Field(ge=0)
    compiler_fallback: bool = False


class SearchRequest(StrictModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    query: ContentValue = ""
    user_id: str = Field(min_length=1, max_length=1024)
    # 0 is legal and means "return nothing"; the app answers it, and a blank query, with an
    # empty `data` list without running retrieval.
    top_k: int = Field(default=MAX_TOP_K, ge=0, le=MAX_TOP_K)
    options: list[str] | None = Field(default=None, max_length=MAX_OPTIONS)

    @field_validator("user_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("top_k", mode="before")
    @classmethod
    def usable_top_k(cls, value: object) -> object:
        """`null` or an unreadable value means the formal 100; any number is clamped to 0..100.

        Returning fewer items than asked for is inside the contract, and more is outside it,
        so a request above 100 is answered with at most 100.
        """
        if value is None or isinstance(value, bool):
            return MAX_TOP_K
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                return MAX_TOP_K
        if isinstance(value, float):
            if not math.isfinite(value):
                return MAX_TOP_K
            value = int(value)
        if not isinstance(value, int):
            return MAX_TOP_K
        return max(0, min(value, MAX_TOP_K))

    @field_validator("options", mode="before")
    @classmethod
    def usable_options(cls, value: object) -> object:
        """Keep the first 20 choices as text, whatever shape they arrive in.

        Only facet variants read `options`, and C9 is not one; refusing a choice list with a
        21st entry, or a mapping of labels to answers, failed the Search for nothing.
        """
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, dict):
            value = [f"{key}: {item}" for key, item in value.items()]
        elif not isinstance(value, list):
            value = [value]
        return [_option_text(item) for item in value[:MAX_OPTIONS]]

    @field_validator("query", mode="before")
    @classmethod
    def usable_query(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            return _usable_parts(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: ContentValue) -> ContentValue:
        if isinstance(value, str):
            return bounded_query(value)
        media_bytes = content_media_bytes(value)
        if media_bytes > MAX_REQUEST_MEDIA_BYTES:
            raise ValueError(
                f"decoded Search images exceed {MAX_REQUEST_MEDIA_BYTES} aggregate bytes"
            )
        return value


class SearchItem(StrictModel):
    id: str
    content: ContentValue
    created_at: datetime | None = None
    source: str
    session_id: str
    kind: str
    score: float


class SearchResponse(StrictModel):
    data: list[SearchItem]
    facet_fallback: bool = Field(default=False, exclude=True)
    reranker_fallback: bool = Field(default=False, exclude=True)
    task_type: TaskType = Field(default="unknown", exclude=True)
    specialist_route: str = Field(default="code", exclude=True)
    specialist_embedding_profile: str = Field(default="none", exclude=True)
    reranker_attempted: bool = Field(default=False, exclude=True)
    reranker_completed: bool = Field(default=False, exclude=True)
    reranker_provider: str = Field(default="none", exclude=True)
    reranker_model: str = Field(default="none", exclude=True)
    candidate_input_count: int = Field(default=0, ge=0, exclude=True)
    candidate_output_count: int = Field(default=0, ge=0, exclude=True)
    candidate_permutation_valid: bool = Field(default=True, exclude=True)
    top_10_order_changed: bool = Field(default=False, exclude=True)
    top_10_membership_changed: bool = Field(default=False, exclude=True)
    top_100_order_changed: bool = Field(default=False, exclude=True)
    top_100_membership_changed: bool = Field(default=False, exclude=True)
    candidate_character_count: int = Field(default=0, ge=0, exclude=True)
    rerank_ms: float = Field(default=0.0, ge=0, exclude=True)
    estimated_reranker_cost_usd: float = Field(default=0.0, ge=0, exclude=True)
    generation_id: str = Field(default="unknown", exclude=True)
    corpus_sha256: str = Field(default="", exclude=True)
    code_aware_attempted: bool = Field(default=False, exclude=True)
    code_aware_fallback: bool = Field(default=False, exclude=True)
    code_profile: str = Field(default="none", exclude=True)
    code_rrf_weight: float = Field(default=0.0, ge=0, exclude=True)
    code_query_token_count: int = Field(default=0, ge=0, exclude=True)
    code_match_candidate_count: int = Field(default=0, ge=0, exclude=True)
    code_top_10_order_changed: bool = Field(default=False, exclude=True)
    code_top_10_membership_changed: bool = Field(default=False, exclude=True)
    code_top_100_order_changed: bool = Field(default=False, exclude=True)
    code_top_100_membership_changed: bool = Field(default=False, exclude=True)
    neighbour_seed_limit: int = Field(default=0, ge=0, exclude=True)
    neighbour_seed_count: int = Field(default=0, ge=0, exclude=True)
    neighbour_activated_seed_count: int = Field(default=0, ge=0, exclude=True)
    neighbour_ineligible_seed_count: int = Field(default=0, ge=0, exclude=True)
    neighbour_restored_count: int = Field(default=0, ge=0, exclude=True)
    neighbour_invalid_count: int = Field(default=0, ge=0, exclude=True)
    code_duplicate_output_count: int = Field(default=0, ge=0, exclude=True)
    graph_attempted: bool = Field(default=False, exclude=True)
    graph_fallback: bool = Field(default=False, exclude=True)
    graph_profile: str = Field(default="none", exclude=True)
    graph_relation_hits: int = Field(default=0, ge=0, exclude=True)
    graph_candidate_count: int = Field(default=0, ge=0, exclude=True)
    graph_promoted_count: int = Field(default=0, ge=0, exclude=True)
    graph_invalid_relation_count: int = Field(default=0, ge=0, exclude=True)
    graph_top_10_order_changed: bool = Field(default=False, exclude=True)
    graph_top_100_membership_changed: bool = Field(default=False, exclude=True)
    atomic_rescue_attempted: bool = Field(default=False, exclude=True)
    atomic_rescue_active: bool = Field(default=False, exclude=True)
    atomic_rescue_fallback: bool = Field(default=False, exclude=True)
    atomic_rescue_candidate_available: bool = Field(default=False, exclude=True)


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


class EvidenceSpan(StrictModel):
    """One byte-for-byte quote from a message supplied to Add."""

    message_ordinal: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=4_500)

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError("evidence span end must be greater than start")
        return self


class CodingMemoryRecord(StrictModel):
    kind: MemoryKind
    task_shape: str = ""
    problem: str = ""
    action: str = ""
    outcome: str = ""
    validation: str = ""
    entities: list[str] = Field(default_factory=list, max_length=32)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list, max_length=8)
    # Compatibility with records written before exact source offsets were introduced. New compiler
    # output is normalized into evidence_spans before persistence.
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
        evidence = list(
            dict.fromkeys([span.quote for span in self.evidence_spans] + list(self.evidence_quotes))
        )
        fields: list[tuple[str, Any]] = [
            ("kind", self.kind),
            ("entities", ", ".join(self.entities)),
            ("evidence", " | ".join(evidence)),
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


MAX_COMPILED_RECORDS = 8
MAX_PROPOSAL_ENTITIES = 32
MAX_PROPOSAL_REFERENCES = 8


class CompilerPayload(StrictModel):
    records: list[CodingMemoryRecord] = Field(default_factory=list, max_length=8)


class AnchoredCodingMemoryProposal(StrictModel):
    """A typed record proposal that cites server-created evidence anchors only."""

    kind: MemoryKind
    task_shape: str = ""
    problem: str = ""
    action: str = ""
    outcome: str = ""
    validation: str = ""
    entities: list[str] = Field(default_factory=list, max_length=MAX_PROPOSAL_ENTITIES)
    evidence_anchor_ids: list[str] = Field(
        default_factory=list, min_length=1, max_length=MAX_PROPOSAL_REFERENCES
    )
    event_time: datetime | None = None
    source_session_id: str
    supersedes: list[str] = Field(default_factory=list, max_length=MAX_PROPOSAL_REFERENCES)

    # The model decides how many entities and references to propose; the caps above are ours.
    # Rejecting an over-long list failed the WHOLE compiler answer, every record in it, and the
    # retries resend the identical prompt, so the same answer could fail all three attempts.
    # Keeping the first entries is what the compiler does with the records list anyway.
    @field_validator("entities", mode="before")
    @classmethod
    def keep_first_entities(cls, value: object) -> object:
        return value[:MAX_PROPOSAL_ENTITIES] if isinstance(value, list) else value

    @field_validator("evidence_anchor_ids", "supersedes", mode="before")
    @classmethod
    def keep_first_references(cls, value: object) -> object:
        return value[:MAX_PROPOSAL_REFERENCES] if isinstance(value, list) else value


class AnchoredCompilerPayload(StrictModel):
    records: list[AnchoredCodingMemoryProposal] = Field(
        default_factory=list, max_length=MAX_COMPILED_RECORDS
    )

    @field_validator("records", mode="before")
    @classmethod
    def keep_first_records(cls, value: object) -> object:
        """Keep the first eight proposals instead of refusing a ninth.

        `_compile_anchored` already reads only `records[:8]`, but this cap ran first, so a
        ninth proposal made pydantic reject the answer and C9 stored no compiled record at
        all for that Add. An answer of eight or fewer is validated exactly as before.
        """
        return value[:MAX_COMPILED_RECORDS] if isinstance(value, list) else value


class FacetPayload(StrictModel):
    facets: list[str] = Field(default_factory=list, max_length=4)
    task_type: TaskType = "unknown"
