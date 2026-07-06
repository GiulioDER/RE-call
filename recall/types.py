from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class StalenessReport:
    stale: bool
    newest_indexed_at: datetime | None
    age: timedelta | None
    max_age: timedelta


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    hits: list[ScoredChunk]
    gap_warning: bool
    staleness: StalenessReport
